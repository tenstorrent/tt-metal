# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The column-sharded decode key slab AS WIRED INTO the model's indexer classes (``DSV41DecodeIndexer(colshard=True)`` / ``SpecIndexer``) vs the replicated slab (4x8 mesh, 4 users per row, bfp8 slab,
matmul backend, layer-2 indexer weights).

  * ``append_key`` / ``append_key_block``: the same key written through the in-trace local-slot computation (``_cs_ent``) lands on the owner column; the whole sharded slab equals the replicated one (bit equal),
  * selection: top-512 SETS equal the replicated ones up to bf16 ties, plain decode (``st["valid"]`` only: local valid lengths computed on the device) and verify (per-row ``st["nvalid"]``, R = 4 * 6 rows),
    at the full slab and at a position bucket (``set_pos_bucket``),
  * the stages are captured in a trace (no allocation under trace) and the replay gives the same ids.

Env: DSV41_ICS_CTX (comma list, max context of the slab, default 65536,262144), DSV41_ICS_REPS.
"""

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
from models.demos.blackhole.deepseek_v41_flash.tests.test_indexer_device import STATE_DIR
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import DSV41DecodeIndexer, rope_rows
from models.demos.blackhole.deepseek_v41_flash.tt.loader import rope_freqs
from models.demos.blackhole.deepseek_v41_flash.tt.spec_paged import SpecIndexer
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CTX = [int(x) for x in os.environ.get("DSV41_ICS_CTX", "65536,262144").split(",")]
NSPEC = 6


def slab_host(md, ix, U):
    """[rows, U, n_global (or n_alloc), 128] float of the slab of every mesh row, un-interleaved."""
    rows, cols = tuple(md.shape)
    ttnn.synchronize_device(md)
    devs = ttnn.get_device_tensors(ix.k_cache)
    if not ix.cs:
        return torch.stack([ttnn.to_torch(devs[r * cols]).float().reshape(U, -1, 128) for r in range(rows)])
    out = []
    for r in range(rows):
        per_c = torch.stack(
            [ttnn.to_torch(devs[r * cols + c]).float().reshape(U, -1, 128) for c in range(cols)]
        )  # [cols,U,nl,128]
        out.append(per_c.permute(1, 2, 0, 3).reshape(U, -1, 128))  # entry = cols * slot + c
    return torch.stack(out)


def set_diff(md, a, b, sc, R, rows):
    """number of rows whose id sets differ, of those pure ties at the k-th score, real mismatches (a: replicated ids, b: sharded ids, sc: replicated scores [R,1,1,n])."""
    nmis = ntie = nbad = 0
    for r in range(rows):
        ai = dev_rows(md, a, r).reshape(R, -1).long()
        bi = dev_rows(md, b, r).reshape(R, -1).long()
        sr = dev_rows(md, sc, r).reshape(R, -1).float()
        for i in range(R):
            A, B = set(ai[i].tolist()), set(bi[i].tolist())
            if A != B:
                nmis += 1
                kth = sr[i, ai[i]].min()
                if all(abs(float(sr[i, e]) - float(kth)) == 0.0 for e in A ^ B):
                    ntie += 1
                else:
                    nbad += 1
    return nmis, ntie, nbad


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEV, indirect=True)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_cs_integrated(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    U = 4
    cfg_m, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    w, wsrc = load_weights()
    ref = torch.load(os.path.join(STATE_DIR, "layer2_N32768.pt"))
    base_keys = ref["index_k"].float()
    g = torch.Generator().manual_seed(3)
    wk = (torch.randn(128, 512, generator=g) * 0.05).to(torch.bfloat16)
    knorm = torch.ones(128).to(torch.bfloat16)
    summary = []
    for ctx in CTX:
        for kind in ("plain", "spec"):
            n = NSPEC if kind == "spec" else 1
            R = U * n
            Nmax = ctx // 2
            n_alloc = -(-(Nmax + 32) // 32) * 32
            kw = dict(users_per_row=U, ratio=2, key_dtype=ttnn.bfloat8_b, fp4_q=True, backend="matmul")
            cls = SpecIndexer if kind == "spec" else DSV41DecodeIndexer
            extra = dict(n=n) if kind == "spec" else {}
            freqs = rope_freqs(LAYER, ctx + 64)
            rep = cls(md, w, freqs, n_alloc=n_alloc, **kw, **extra)
            cs = cls(md, w, freqs, n_alloc=n_alloc, colshard=True, mesh_config=cfg_m, ccl=ccl, **kw, **extra)
            assert cs.cs and cs.n_alloc * cols == cs.n_global >= n_alloc
            for ix in (rep, cs):
                ix.set_key_weights(wk, knorm)
            tag = f"ctx={ctx} {kind}"
            try:
                # --- keys, an append, slab equality ------------------------------------------------------------------------------------
                for Nfill, bucket in (
                    (Nmax - 200, None),
                    (min(Nmax - 200, 15000), 32768),
                    (min(Nmax - 200, 3500), 8192),
                ):
                    if bucket is not None and bucket >= ctx:
                        continue
                    keys = gen_keys(base_keys, rows * U, Nfill, seed=ctx + n)
                    rep.load_keys(keys)
                    cs.load_keys(keys)
                    # one append step (the model's append_key / append_key_block): lat_pre, rope rows, entry indices
                    lat = shard_rows(md, torch.randn(rows, 1, R, 512).to(torch.bfloat16))
                    posb = (Nfill + 5) * 2 + (
                        torch.arange(rows * R) % n
                    )  # positions of the rows; the first row of a pair completes entry Nfill + 5
                    C, S = rope_rows(freqs, torch.full((rows * R,), 2 * (Nfill + 5), dtype=torch.long))
                    mk_st = lambda: {
                        "iCg": shard_rows(md, C.reshape(rows, 1, R, 128)[:, :, :, :].to(torch.bfloat16)),
                        "iSg": shard_rows(md, S.reshape(rows, 1, R, 128).to(torch.bfloat16)),
                        "C": shard_rows(md, C.reshape(rows * R, 1, 1, 128).to(torch.bfloat16)),
                        "S": shard_rows(md, S.reshape(rows * R, 1, 1, 128).to(torch.bfloat16)),
                        "pos": shard_rows(
                            md, posb.reshape(rows * R).to(torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
                        ),
                        "ent": shard_rows(
                            md,
                            (torch.arange(rows * U) % 3 + Nfill + 5).reshape(rows * U).to(torch.int32),
                            ttnn.int32,
                            ttnn.ROW_MAJOR_LAYOUT,
                        ),
                        "valid": rep_up(
                            md,
                            torch.full((1, 1, 1, 1), max(Nfill + 8, TOPK), dtype=torch.int32),
                            ttnn.uint32,
                            ttnn.ROW_MAJOR_LAYOUT,
                        ),
                    }
                    if kind == "spec":
                        for ix in (rep, cs):
                            ix.append_key_block(lat, mk_st())
                    else:
                        lat_u = shard_rows(md, torch.randn(rows, 1, U, 512).to(torch.bfloat16))
                        for ix in (rep, cs):
                            ix.append_key(lat_u, mk_st())
                    a, b = slab_host(md, rep, U), slab_host(md, cs, U)
                    m = min(a.shape[2], b.shape[2])
                    slab_equal = bool((a[:, :, :m] == b[:, :, :m]).all())
                    hd = torch.stack(
                        [
                            ttnn.to_torch(ttnn.get_device_tensors(cs.k_head)[r * cols]).float().reshape(U, -1, 128)
                            for r in range(rows)
                        ]
                    )
                    slab_equal &= bool(
                        (hd == a[:, :, : hd.shape[2]]).all()
                    )  # the replicated head slab = the first n_head entries
                    # --- selection ---------------------------------------------------------------------------------------------------------
                    st = mk_st()
                    nvalid = torch.tensor([Nfill - (7 * (i % R) % 50) for i in range(rows * R)], dtype=torch.float32)
                    if kind == "spec":
                        st["nvalid"] = shard_rows(md, nvalid.reshape(rows * R, 1, 1, 1), ttnn.float32)
                    st["valid"] = rep_up(
                        md,
                        torch.full((1, 1, 1, 1), max(Nfill, TOPK), dtype=torch.int32),
                        ttnn.uint32,
                        ttnn.ROW_MAJOR_LAYOUT,
                    )
                    gq = torch.Generator().manual_seed(1)
                    x_q, qr = ref["x_q"], ref["qr"]
                    x = (x_q[None] + 0.3 * x_q.std() * torch.randn(R, 5120, generator=gq)).reshape(1, 1, R, 5120)
                    qrr = (qr[None] + 0.3 * qr.std() * torch.randn(R, 1280, generator=gq)).reshape(1, 1, R, 1280)
                    x, qrr = rep_up(md, x.to(torch.bfloat16)), rep_up(md, qrr.to(torch.bfloat16))
                    for ix in (rep, cs):
                        ix.set_pos_bucket(bucket)
                    q, wv = rep.project(x, qrr, st)
                    rsc = rep.score(q, wv)
                    rids = rep.select(rsc, dict(st))
                    csc = cs.score(q, wv)
                    cids = cs.select(csc, dict(st))
                    ttnn.synchronize_device(md)
                    nmis, ntie, nbad = set_diff(md, rids, cids, rsc, R, rows)
                    # --- trace: capture select under a trace (fresh st dict per call: the derived valid lengths are part of the trace) ---
                    t_rep = traced_us(md, lambda: rep.select(rep.score(q, wv), dict(st)), inner=4, reps=3)
                    t_cs = traced_us(md, lambda: cs.select(cs.score(q, wv), dict(st)), inner=4, reps=3)
                    msg = (
                        f"ICSI {tag} {'bucket ' + str(bucket) if bucket else 'full slab'} Nfill={Nfill} wts={wsrc}: slab equal after append {slab_equal}; "
                        f"id sets != replicated {nmis}/{rows * R} (ties {ntie}, REAL {nbad}); traced us/layer score+select: replicated {t_rep:.0f} vs column-sharded {t_cs:.0f}"
                    )
                    print(msg, flush=True)
                    summary.append(msg)
                    assert slab_equal and nbad == 0, msg
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                print(f"ICSI {tag}: ERR {type(e).__name__}: {str(e)[:300]}", flush=True)
                summary.append(f"ICSI {tag}: ERR {type(e).__name__}")
                raise
            rep = cs = None
            import gc

            gc.collect()
    print("\n".join(summary))
