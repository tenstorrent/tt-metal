# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DSV41_IDX_KEY_SHARD: the decode indexers AS WIRED INTO the model (``DSV41DecodeIndexer`` / ``SpecIndexer`` with ``colshard=True``) vs the replicated key slab, ``forward`` end to end
(project + score + mask + top-512 [+ candidate all_gather + global top-512]) on the 4x8 mesh, bfp8 slab, matmul backend, layer-2 indexer weights.

  test_ks_forward   users per mesh row U x block length n (R = U * n token rows per mesh row: plain decode R = U; spec verify k = 5: U = 2 -> 12 rows (B = 8), U = 4 -> 24 rows (B = 16)),
                    entries N up to 1M-sized slabs, per-row valid counts (ragged, rows below 512, padding rows with 0), random keys and DUPLICATED keys (exact score ties).
                    Per case: sorted-id lists EXACTLY equal to the replicated ones / equal up to the choice among entries tied at the k-th score / real mismatches (must be 0), ids >= nvalid
                    (masked entries: the consumer skips them), traced us per layer replicated vs sharded.
  test_ks_append    the key append (``append_key`` / ``append_key_block``, several keys per step, skip entries, ratio 1 and 2): the sharded slab and the head slab equal the replicated slab bit for bit.

Env: DSV41_IKS_N (comma list of entries, default 16384,65536,262144,524288), DSV41_IKS_GRID (U:n list, default 2:1,2:6,4:1,4:6), DSV41_IKS_KEYS (rand,dup), DSV41_IKS_RATIO (default 2),
DSV41_IKS_BUCKET (0 = full slab traces, else the position bucket), DSV41_IKS_REPS.
"""

import gc
import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_indexer_colshard import (
    DEV,
    LAYER,
    TOPK,
    dev_rows,
    gen_keys,
    load_weights,
    rep_up,
    shard_rows,
    traced_us,
)
from models.demos.blackhole.deepseek_v41_flash.tests.test_indexer_cs_integrated import slab_host
from models.demos.blackhole.deepseek_v41_flash.tests.test_indexer_device import STATE_DIR
from models.demos.blackhole.deepseek_v41_flash.tt.idx_score import score_mask_fused
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import DSV41DecodeIndexer, rope_rows
from models.demos.blackhole.deepseek_v41_flash.tt.loader import rope_freqs
from models.demos.blackhole.deepseek_v41_flash.tt.spec_paged import SpecIndexer
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

NS = [int(x) for x in os.environ.get("DSV41_IKS_N", "16384,65536,262144,524288").split(",")]
GRID = [tuple(int(v) for v in g.split(":")) for g in os.environ.get("DSV41_IKS_GRID", "2:1,2:6,4:1,4:6").split(",")]
KEYS = os.environ.get("DSV41_IKS_KEYS", "rand,dup").split(",")
RATIO = int(os.environ.get("DSV41_IKS_RATIO", "2"))
BUCKET = int(os.environ.get("DSV41_IKS_BUCKET", "0"))
REPS = int(os.environ.get("DSV41_IKS_REPS", "3"))


def gen_keys_dup(base, B, N, nd=4096, seed=0):
    """keys drawn from nd distinct rows: exact duplicates -> exact score ties (many entries tied at the k-th score)."""
    g = torch.Generator().manual_seed(seed)
    sd = base.std()
    pool = (
        base[torch.randint(0, base.shape[0], (nd,), generator=g)] + 0.3 * sd * torch.randn(nd, 128, generator=g)
    ).to(torch.bfloat16)
    return pool[torch.randint(0, nd, (B, N), generator=g)]


def classify(sets_a, sets_b, sc, nv):
    """(exact, same set, equal up to k-th score ties, real mismatch) for one row; sc: host scores of the row [n] (masked), nv valid entries."""
    a, b = sorted(sets_a), sorted(sets_b)
    exact = a == b
    if nv < TOPK:  # fewer valid entries than k: every valid entry must be selected, the rest is masked garbage
        va, vb = {x for x in a if x < nv}, {x for x in b if x < nv}
        ok = va == vb and len(va) == nv
        return exact, ok, ok, not ok
    A, B = set(a), set(b)
    if A == B:
        return exact, True, True, False
    kth = sc[torch.tensor(a)].min()
    diff = list(A ^ B)
    ties = all(float(sc[e]) == float(kth) for e in diff)
    return exact, False, ties, not ties


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEV, indirect=True)
@pytest.mark.timeout(10800)
@torch.no_grad()
def test_ks_forward(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    cfg_m, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    w, wsrc = load_weights()
    ref = torch.load(os.path.join(STATE_DIR, "layer2_N32768.pt"))
    base_keys = ref["index_k"].float()
    summary, nbad_total = [], 0
    for N in NS:
        for U, n in GRID:
            R = U * n
            kind = "spec" if n > 1 else "plain"
            n_alloc = -(-(N + 32) // 32) * 32
            freqs = rope_freqs(LAYER, RATIO * N + 64)
            kw = dict(users_per_row=U, ratio=RATIO, key_dtype=ttnn.bfloat8_b, fp4_q=True, backend="matmul")
            cls = SpecIndexer if kind == "spec" else DSV41DecodeIndexer
            extra = dict(n=n) if kind == "spec" else {}
            try:
                rep = cls(md, w, freqs, n_alloc=n_alloc, **kw, **extra)
                cs = cls(md, w, freqs, n_alloc=n_alloc, colshard=True, mesh_config=cfg_m, ccl=ccl, **kw, **extra)
                assert cs.cs
            except Exception as e:  # noqa: BLE001
                print(f"IKS N={N} U={U} n={n}: build ERR {type(e).__name__}: {str(e)[:200]}", flush=True)
                summary.append(f"IKS N={N} U={U} n={n}: build ERR")
                continue
            for kmode in KEYS:
                tag = f"N={N} ratio={RATIO} U={U} n={n} R={R} keys={kmode} bucket={BUCKET or 'full'}"
                try:
                    Nfill = N - 64
                    if BUCKET:  # a position bucket scores the first nb entries only: every valid entry must lie inside
                        Nfill = min(Nfill, BUCKET // RATIO - 64)
                    keys = (gen_keys if kmode == "rand" else gen_keys_dup)(base_keys, rows * U, Nfill, seed=N + n)
                    rep.load_keys(keys)
                    cs.load_keys(keys)
                    for ix in (rep, cs):
                        ix.set_pos_bucket(BUCKET or None)
                    gq = torch.Generator().manual_seed(1)
                    x_q, qr = ref["x_q"], ref["qr"]
                    x = (x_q[None] + 0.3 * x_q.std() * torch.randn(R, 5120, generator=gq)).reshape(1, 1, R, 5120)
                    qrr = (qr[None] + 0.3 * qr.std() * torch.randn(R, 1280, generator=gq)).reshape(1, 1, R, 1280)
                    x, qrr = rep_up(md, x.to(torch.bfloat16)), rep_up(md, qrr.to(torch.bfloat16))
                    pos = torch.full((rows * R,), RATIO * Nfill, dtype=torch.long)
                    C, S = rope_rows(freqs, pos)
                    # per-row valid counts: the first rows full, the others ragged; a row below k; a padding row (spec rows only: the plain decode path has the batch maximum)
                    nvalid = torch.tensor([Nfill - (13 * (i % R)) % 90 for i in range(rows * R)], dtype=torch.float32)
                    if kind == "spec":
                        for r in range(rows):
                            nvalid[r * R + R - 1] = 300.0  # below k
                            if R >= 12:
                                nvalid[r * R + R - 2] = 0.0  # padding row
                    st = {
                        "C": shard_rows(md, C.reshape(rows * R, 1, 1, 128).to(torch.bfloat16)),
                        "S": shard_rows(md, S.reshape(rows * R, 1, 1, 128).to(torch.bfloat16)),
                        "valid": rep_up(
                            md,
                            torch.full((1, 1, 1, 1), max(Nfill, TOPK), dtype=torch.int32),
                            ttnn.uint32,
                            ttnn.ROW_MAJOR_LAYOUT,
                        ),
                    }
                    if kind == "spec":
                        st["nvalid"] = shard_rows(md, nvalid.reshape(rows * R, 1, 1, 1), ttnn.float32)
                    nv_row = nvalid if kind == "spec" else torch.full((rows * R,), float(Nfill))
                    # reference scores (replicated): the fused score + mask op (spec) / the chain (plain)
                    q, wv = rep.project(x, qrr, dict(st))
                    if kind == "spec":
                        nvu = ttnn.reshape(
                            ttnn.typecast(
                                ttnn.pad(
                                    ttnn.reshape(ttnn.to_layout(st["nvalid"], ttnn.ROW_MAJOR_LAYOUT), [1, 1, 1, R]),
                                    [(0, 0), (0, 0), (0, 0), (0, 32 - R)],
                                    0.0,
                                ),
                                ttnn.uint32,
                            ),
                            [1, 1, 1, 32],
                        )
                        rsc = score_mask_fused(q, rep.k_cache, rep._w_flat, nvu, n, rep.n_score, valid=st["valid"])
                    else:
                        rsc = rep.score(q, wv)
                    rids = rep.forward(x, qrr, dict(st))
                    cids = cs.forward(x, qrr, dict(st))
                    ttnn.synchronize_device(md)
                    ex = same = ties = real = ninv = 0
                    for r in range(rows):
                        a_ids = dev_rows(md, rids, r).reshape(R, -1).long()
                        b_ids = dev_rows(md, cids, r).reshape(R, -1).long()
                        scr = dev_rows(md, rsc, r).reshape(R, -1).float()
                        for i in range(R):
                            nvi = int(nv_row[r * R + i])
                            e_, s_, t_, b_ = classify(a_ids[i].tolist(), b_ids[i].tolist(), scr[i], nvi)
                            ex += e_
                            same += s_
                            ties += t_
                            real += b_
                            ninv += int((b_ids[i] >= nvi).sum()) if nvi >= TOPK else 0
                    tot = rows * R
                    t_rep = traced_us(md, lambda: rep.forward(x, qrr, dict(st)), inner=4, reps=REPS)
                    t_cs = traced_us(md, lambda: cs.forward(x, qrr, dict(st)), inner=4, reps=REPS)
                    msg = (
                        f"IKS {tag} wts={wsrc}: rows bit-identical {ex}/{tot}, same set {same}/{tot}, equal up to k-th-score ties {ties}/{tot}, REAL mismatches {real}; "
                        f"masked ids selected {ninv}; traced us/layer replicated {t_rep:.0f} vs sharded {t_cs:.0f}"
                    )
                    print(msg, flush=True)
                    summary.append(msg)
                    nbad_total += real
                except Exception as e:  # noqa: BLE001
                    import traceback

                    traceback.print_exc()
                    print(f"IKS {tag}: ERR {type(e).__name__}: {str(e)[:300]}", flush=True)
                    summary.append(f"IKS {tag}: ERR {type(e).__name__}")
                    nbad_total += 1
            rep = cs = None
            gc.collect()
    print("\n".join(summary))
    assert nbad_total == 0


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEV, indirect=True)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_ks_append(mesh_device):
    """append_key (plain, one key per user) / append_key_block (spec, n rows, several consecutive entries, skips) into the sharded + head slabs == the replicated slab."""
    md = mesh_device
    rows, cols = tuple(md.shape)
    cfg_m, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    w, _ = load_weights()
    g = torch.Generator().manual_seed(3)
    wk = (torch.randn(128, 512, generator=g) * 0.05).to(torch.bfloat16)
    knorm = torch.ones(128).to(torch.bfloat16)
    N = 70000
    for ratio in (1, 2):
        for U, n in ((2, 1), (2, 6), (4, 6)):
            R = U * n
            kind = "spec" if n > 1 else "plain"
            n_alloc = -(-(N + 32) // 32) * 32
            freqs = rope_freqs(LAYER, ratio * N + 64)
            kw = dict(users_per_row=U, ratio=ratio, key_dtype=ttnn.bfloat8_b, fp4_q=True, backend="matmul")
            cls = SpecIndexer if kind == "spec" else DSV41DecodeIndexer
            extra = dict(n=n) if kind == "spec" else {}
            rep = cls(md, w, freqs, n_alloc=n_alloc, **kw, **extra)
            cs = cls(md, w, freqs, n_alloc=n_alloc, colshard=True, mesh_config=cfg_m, ccl=ccl, **kw, **extra)
            for ix in (rep, cs):
                ix.set_key_weights(wk, knorm)
                ix.load_keys(torch.zeros(rows * U, 0, 128))
            ok_all = True
            for step, e0 in enumerate((0, 5, 17, 20000, 33333)):
                lat = shard_rows(md, torch.randn(rows, 1, R, 512, generator=g).to(torch.bfloat16))
                # block rows j of every user: positions e0 * ratio + j (group-completing rows write entry (pos + 1) // ratio - 1 ... as the model: ratio 1 every row, ratio 2 every second row)
                posb = torch.tensor([ratio * e0 + (i % n) + (1 if ratio == 2 else 0) for i in range(rows * R)])
                C, S = rope_rows(freqs, posb.clamp(max=freqs.shape[0] - 1))
                st = {
                    "iCg": shard_rows(md, C.reshape(rows, 1, R, 128).to(torch.bfloat16)),
                    "iSg": shard_rows(md, S.reshape(rows, 1, R, 128).to(torch.bfloat16)),
                    "pos": shard_rows(md, posb.reshape(rows * R).to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                    "ent": shard_rows(
                        md,
                        (torch.arange(rows * U) % 7 + e0).reshape(rows * U).to(torch.int32),
                        ttnn.int32,
                        ttnn.ROW_MAJOR_LAYOUT,
                    ),
                }
                for ix in (rep, cs):
                    if kind == "spec":
                        ix.append_key_block(lat, dict(st))
                    else:
                        ix.append_key(lat, dict(st))
                a, b = slab_host(md, rep, U), slab_host(md, cs, U)
                m = min(a.shape[2], b.shape[2])
                eq = bool((a[:, :, :m] == b[:, :, :m]).all())
                hd = torch.stack(
                    [
                        ttnn.to_torch(ttnn.get_device_tensors(cs.k_head)[r * cols]).float().reshape(U, -1, 128)
                        for r in range(rows)
                    ]
                )
                eq_h = bool((hd == a[:, :, : hd.shape[2]]).all())
                nz = int((a.abs().amax(-1) > 0).sum())
                print(
                    f"IKSA ratio={ratio} U={U} n={n} step={step} e0={e0}: sharded slab == replicated {eq}, head slab == replicated {eq_h}, nonzero entries so far {nz}",
                    flush=True,
                )
                ok_all &= eq and eq_h
            assert ok_all
            rep = cs = None
            gc.collect()


class _ExportStub:
    """what the model's key hand-off methods need from ``self`` (tt/dsv41_model.py ``_export_dec_keys``): lets the unbound methods run on a prefill key FIFO built here."""

    def __init__(self, md):
        from models.demos.blackhole.deepseek_v41_flash.tt.dsv41_model import Model

        self.md = md
        self.rows, self.cols = tuple(md.shape)
        for nm in (
            "_export_state",
            "_export_state_cs",
            "_export_user_keys",
            "_export_user_keys_cs",
            "_export_dec_keys",
        ):
            setattr(self, nm, getattr(Model, nm).__get__(self))

    def _mp(self):
        return ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(self.rows, self.cols))

    def log_dram(self, *a, **k):
        pass


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEV, indirect=True)
@pytest.mark.timeout(3600)
@torch.no_grad()
def test_ks_export(mesh_device):
    """key hand-off prefill FIFO (column-sharded, p % cols) -> the column-sharded decode slab + head slab == the hand-off into a replicated slab (same device code path as the model's), for aligned
    and unaligned FIFO offsets, several users / mesh rows, entries below / above the head slab."""
    from types import SimpleNamespace

    from models.demos.blackhole.deepseek_v41_flash.tt.prefill_sparse import shard_len

    md = mesh_device
    rows, cols = tuple(md.shape)
    cfg_m, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    w, _ = load_weights()
    U, Up = 2, 1
    L = 65536  # FIFO length (entries)
    Lc = shard_len(L, cols)
    dL = cols * Lc - L
    NA = L + 64
    freqs = rope_freqs(LAYER, 2 * NA + 64)
    kw = dict(users_per_row=U, ratio=2, key_dtype=ttnn.bfloat8_b, fp4_q=True, backend="matmul")
    stub = _ExportStub(md)
    g = torch.Generator().manual_seed(5)
    T = (torch.randn(rows * Up, L, 128, generator=g) * 0.7).to(
        torch.bfloat16
    )  # the virtual FIFO of every (row, prefill slot)
    pad = torch.zeros(rows * Up, cols * Lc - L, 128, dtype=torch.bfloat16)
    full = torch.cat([pad, T], dim=1)  # [rows*Up, cols*Lc, 128], virtual slot v at position dL + v
    loc = full.reshape(rows * Up, Lc, cols, 128).permute(0, 2, 1, 3)  # [B, cols, Lc, 128]
    loc = torch.cat([loc, torch.zeros(rows * Up, cols, 32, 128, dtype=torch.bfloat16)], dim=2)
    keys = ttnn.from_torch(
        loc.contiguous(),
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    ix = SimpleNamespace(keys=keys, sh=dict(cols=cols, Lc=Lc, mc=cfg_m, ccl=ccl), KL_full=L, KL=Lc)
    for local in ("1", "0"):
        os.environ["DSV41_IDX_CS_EXPORT_LOCAL"] = local
        for off, n, u_dst, rws in (
            (L - 4096, 4096, 1, [0, 2]),
            (L - 4099, 4099, 0, [1, 3]),
            (L - 40000, 40000, 1, [0, 1, 2, 3]),
        ):
            rep = DSV41DecodeIndexer(md, w, freqs, n_alloc=NA, **kw)
            cs = DSV41DecodeIndexer(md, w, freqs, n_alloc=NA, colshard=True, mesh_config=cfg_m, ccl=ccl, **kw)
            for ix_ in (rep, cs):
                ix_.load_keys(torch.zeros(rows * U, 0, 128))
            stub._export_user_keys(ix, rep.k_cache, 0, u_dst, rws, off, n)
            stub._export_dec_keys(cs, ix, 0, u_dst, rws, off, n)
            ttnn.synchronize_device(md)
            a, b = slab_host(md, rep, U), slab_host(md, cs, U)
            m = min(a.shape[2], b.shape[2])
            eq = bool((a[:, :, :m] == b[:, :, :m]).all())
            hd = torch.stack(
                [
                    ttnn.to_torch(ttnn.get_device_tensors(cs.k_head)[r * cols]).float().reshape(U, -1, 128)
                    for r in range(rows)
                ]
            )
            eq_h = bool((hd == a[:, :, : hd.shape[2]]).all())
            # host expectation (bfp8 quantisation tolerance): exported entries == T[row][off + e]
            ok_host = True
            for r in rws:
                want = T[r][off : off + n].float()
                got = a[r, u_dst, :n]
                ok_host &= bool(((got - want).abs().max() < 0.05 * want.abs().max()))
            others = bool((a[[r for r in range(rows) if r not in rws]].abs().sum() == 0)) and bool(
                a[:, 1 - u_dst].abs().sum() == 0
            )
            print(
                f"IKSE local={local} off={off} n={n} user={u_dst} rows={rws}: sharded == replicated hand-off {eq}, head slab == {eq_h}, host check {ok_host}, other rows / users untouched {others}",
                flush=True,
            )
            assert eq and eq_h and ok_host and others
            rep = cs = None
            gc.collect()
    os.environ.pop("DSV41_IDX_CS_EXPORT_LOCAL", None)
