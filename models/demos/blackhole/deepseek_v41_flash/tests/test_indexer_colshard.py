# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode indexer with the key slab column-sharded (entry interleave, tt/indexer_colshard.py) vs the CURRENT replicated implementation (4x8 mesh, 4 users per row, bfp8 slab, matmul backend).

Layer-2 indexer weights (real if loadable, else random), queries from the real layer-2 state (tests/indexer_ref_state.py), keys: the real 32768 layer-2 keys re-sampled with per-key noise (a synthetic slab of the
requested length, distinct keys; real keys are only available up to 32768 entries). Context length = 2 * entries (ratio 2). Env: DSV41_ICS_CTX (comma list, default 8192,32768,65536,131072,262144),
DSV41_ICS_MODES (comma list of T1 | T1nv | T16nv: tokens per step per user, nv = per-row valid counts as in the ragged / spec-verify path), DSV41_ICS_REPS.
Prints ICS lines: exact top-k SET agreement (ties classified), score diff, traced time per layer (score + select incl. candidate all_gather + merge), and per-chip slab bytes.
"""

import os
import statistics
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_indexer_device import STATE_DIR, dq
from models.demos.blackhole.deepseek_v41_flash.tt.indexer import DSV41DecodeIndexer, rope_rows
from models.demos.blackhole.deepseek_v41_flash.tt.indexer_colshard import ColShardSelect
from models.demos.blackhole.deepseek_v41_flash.tt.loader import rope_freqs
from models.demos.blackhole.deepseek_v41_flash.tt.spec_paged import SpecIndexer
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

LAYER, TOPK, NSPEC = 2, 512, 16
CTX = [int(x) for x in os.environ.get("DSV41_ICS_CTX", "8192,32768,65536,131072,262144").split(",")]
MODES = os.environ.get("DSV41_ICS_MODES", "T1,T1nv,T16nv").split(",")
BACKEND = os.environ.get(
    "DSV41_ICS_BACKEND", "matmul"
)  # matmul | fused (the model uses fused above 65536 entries, tt/indexer.py default_backend; T1 modes only)
REPS = int(os.environ.get("DSV41_ICS_REPS", "9"))

DEV = [
    pytest.param(
        {"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 200_000_000},
        id="ring",
    )
]


def traced_us(md, fn, inner=10, reps=REPS):
    """warm (one eager call), capture ``inner`` calls in a trace, median of ``reps`` replays, us per call."""
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(inner):
        fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t) / inner * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts)


def load_weights():
    try:
        from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards

        sh = _Shards()
        p = f"layers.{LAYER}.attn.indexer."
        return {
            "wq_b": dq(sh, p + "wq_b"),
            "weights_proj": sh.get(p + "weights_proj.weight").to(torch.bfloat16),
        }, "real"
    except Exception as e:  # noqa: BLE001
        print(f"ICS weights: real load failed ({str(e)[:100]}), random", flush=True)
        g = torch.Generator().manual_seed(0)
        return {
            "wq_b": (torch.randn(4096, 1280, generator=g) * 0.03).to(torch.bfloat16),
            "weights_proj": (torch.randn(32, 5120, generator=g) * 0.03).to(torch.bfloat16),
        }, "random"


def gen_keys(base, B, N, seed=0):
    g = torch.Generator().manual_seed(seed)
    sd = base.std()
    out = torch.empty(B, N, 128, dtype=torch.bfloat16)
    for u in range(B):
        idx = torch.randint(0, base.shape[0], (N,), generator=g)
        k = base[idx] * (1 + 0.15 * torch.randn(N, 1, generator=g)) + 0.15 * sd * torch.randn(N, 128, generator=g)
        out[u] = k.to(torch.bfloat16)
    return out


def shard_rows(md, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    rows, cols = tuple(md.shape)
    return ttnn.from_torch(
        t.contiguous(),
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=(rows, cols)),
    )


def rep_up(md, t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        t.contiguous(),
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def rep_select_flat(rep, scores, st, R):
    """the CURRENT select (full slab, per-row valid counts) with the [1,1,R,n] row layout instead of [R,1,1,n] (same ops otherwise): what the replicated path costs once its select is flattened too."""
    n = rep.n_score
    sr = ttnn.reshape(scores, [1, 1, R, n])
    if "nvalid" in st:
        keep = ttnn.lt(rep._arange_for(n), ttnn.reshape(st["nvalid"], [1, 1, R, 1]))
        m = ttnn.where(keep, ttnn.typecast(ttnn.to_layout(sr, ttnn.TILE_LAYOUT), ttnn.float32), -1e30)
        sr = ttnn.to_layout(ttnn.typecast(m, ttnn.bfloat16), ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.reshape(
        ttnn.experimental.topk_large_indices(sr, k=TOPK, valid_length_tensor=st["valid"]), [R, 1, 1, TOPK]
    )


def rows_of(idx, scores):
    """fused backend: list of per-user [1,1,32,n] score tiles -> [U,1,1,n] row-major (row 0 = the query), as ``DSV41DecodeIndexer.select`` does."""
    if idx.backend != "fused":
        return scores
    rows = [ttnn.slice(t, [0, 0, 0, 0], [1, 1, 1, idx.n_alloc]) for t in scores]
    return rows[0] if len(rows) == 1 else ttnn.concat(rows, dim=0)


def dev_rows(md, t, r):
    """host tensor of mesh row r (column 0)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[r * md.shape[1]])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEV, indirect=True)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_colshard_select(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    U = 4
    cfg_m, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    w, wsrc = load_weights()
    ref = torch.load(os.path.join(STATE_DIR, "layer2_N32768.pt"))
    base_keys = ref["index_k"].float()
    pos0 = ref["pos"]
    summary = []
    for ctx in CTX:
        N = ctx // 2
        n_alloc = -(-(N + 32) // (32 * cols)) * (32 * cols)
        n_local = n_alloc // cols
        keys = gen_keys(base_keys, rows * U, N, seed=ctx)
        for mode in MODES:
            n = NSPEC if mode.startswith("T16") else 1
            nv = mode.endswith("nv")
            R = U * n
            kw = dict(users_per_row=U, ratio=2, key_dtype=ttnn.bfloat8_b, fp4_q=True, backend=BACKEND)
            mk = (
                (lambda na: SpecIndexer(md, w, rope_freqs(LAYER, 2 * N + 64), n_alloc=na, n=n, **kw))
                if n > 1
                else (lambda na: DSV41DecodeIndexer(md, w, rope_freqs(LAYER, 2 * N + 64), n_alloc=na, **kw))
            )
            tag = f"ctx={ctx} N={N} {mode} {BACKEND}"
            try:
                rep, loc = mk(n_alloc), mk(n_local)
                rep.load_keys(keys)
                cs = ColShardSelect(loc, R, ccl, cfg_m, n_alloc, TOPK)
                cs.load_keys(keys)
                # step inputs
                gq = torch.Generator().manual_seed(1)
                pos = torch.full((rows * R,), 2 * N, dtype=torch.long) + (torch.arange(rows * R) % n)
                C, S = rope_rows(rep.freqs_cis, pos)
                st = {
                    "C": shard_rows(md, C.reshape(rows * R, 1, 1, 128).to(torch.bfloat16)),
                    "S": shard_rows(md, S.reshape(rows * R, 1, 1, 128).to(torch.bfloat16)),
                    "valid": rep_up(
                        md,
                        torch.full((1, 1, 1, 1), max(N, TOPK), dtype=torch.int32),
                        ttnn.uint32,
                        ttnn.ROW_MAJOR_LAYOUT,
                    ),
                }
                nvalid = torch.tensor([N - (7 * (i % R) % 50) for i in range(rows * R)], dtype=torch.float32)
                st_sh = dict(st)
                if nv:
                    st["nvalid"] = st_sh["nvalid"] = shard_rows(md, nvalid.reshape(rows * R, 1, 1, 1), ttnn.float32)
                x_q, qr = ref["x_q"], ref["qr"]
                x = (x_q[None] + 0.3 * x_q.std() * torch.randn(R, 5120, generator=gq)).reshape(1, 1, R, 5120)
                qrr = (qr[None] + 0.3 * qr.std() * torch.randn(R, 1280, generator=gq)).reshape(1, 1, R, 1280)
                x, qrr = rep_up(md, x.to(torch.bfloat16)), rep_up(md, qrr.to(torch.bfloat16))
                q, wv = rep.project(x, qrr, st)
                vl_col = cs.valid_per_column(max(N, TOPK))
                # correctness
                rs_raw = rep.score(q, wv)
                rids = rep.select(rs_raw, st)
                rs = rows_of(rep, rs_raw)
                ls = rows_of(loc, loc.score(q, wv))
                sids, svals = cs.select(ls, st_sh, vl_col, want_vals=True)
                ttnn.synchronize_device(md)
                nmis = ntie = nbad = ninv = 0
                maxd = 0.0
                col_ok = True
                for r in range(rows):
                    a_ids = dev_rows(md, rids, r).reshape(R, -1).long()
                    b_ids = dev_rows(md, sids, r).reshape(R, -1).long()
                    rsc = dev_rows(md, rs, r).reshape(R, -1).float()
                    bv = dev_rows(md, svals, r).reshape(R, -1).float()
                    b_alt = ttnn.to_torch(ttnn.get_device_tensors(sids)[r * cols + cols - 1]).reshape(R, -1).long()
                    col_ok &= bool((b_alt == b_ids).all())
                    for i in range(R):
                        A, B = set(a_ids[i].tolist()), set(b_ids[i].tolist())
                        nv_i = int(nvalid[r * R + i]) if nv else N
                        ninv += sum(1 for e in B if e >= nv_i)
                        if A != B:
                            nmis += 1
                            kth = rsc[i, a_ids[i]].min()
                            diff = list(A ^ B)
                            if all(abs(float(rsc[i, e]) - float(kth)) == 0.0 for e in diff):
                                ntie += 1
                            else:
                                nbad += 1
                        common = list(A & B)
                        pos_b = {int(e): j for j, e in enumerate(b_ids[i].tolist())}
                        d = max([abs(float(rsc[i, e]) - float(bv[i, pos_b[e]])) for e in common] or [0.0])
                        maxd = max(maxd, d)
                tot = rows * R
                # timing
                t_rep = traced_us(md, lambda: rep.select(rep.score(q, wv), st))
                t_sh = traced_us(md, lambda: cs.select(rows_of(loc, loc.score(q, wv)), st_sh, vl_col))
                t_rs = traced_us(md, lambda: rows_of(rep, rep.score(q, wv)))
                t_ls = traced_us(md, lambda: rows_of(loc, loc.score(q, wv)))
                t_rsel = traced_us(md, lambda: rep.select(rs_raw, st))
                fids = rep_select_flat(rep, rs, st, R)
                flat_same = all(
                    set(dev_rows(md, fids, r).reshape(R, -1)[i].tolist())
                    == set(dev_rows(md, rids, r).reshape(R, -1)[i].tolist())
                    for r in range(rows)
                    for i in range(R)
                )
                t_rflat = traced_us(md, lambda: rep_select_flat(rep, rs, st, R))
                t_ssel = traced_us(md, lambda: cs.select(ls, st_sh, vl_col))
                slab_rep = U * n_alloc * 136
                msg = (
                    f"ICS {tag} wts={wsrc}: rows with set != replicated {nmis}/{tot} (pure ties at the k-th score {ntie}, REAL mismatches {nbad}), ids beyond valid {ninv}, "
                    f"cols agree {col_ok}, max |score diff| at common ids {maxd:.4g}; traced us/layer (score+select): replicated {t_rep:.0f} "
                    f"(score {t_rs:.0f} + select {t_rsel:.0f}; flat-layout select {t_rflat:.0f}, same sets {flat_same}) vs sharded {t_sh:.0f} (score {t_ls:.0f} + select+gather+merge {t_ssel:.0f}); "
                    f"slab per chip per owner layer {slab_rep / 2**20:.1f} MiB -> {slab_rep / cols / 2**20:.1f} MiB"
                )
                print(msg, flush=True)
                summary.append(msg)
                assert nbad == 0
            except Exception as e:  # noqa: BLE001
                print(f"ICS {tag}: ERR {type(e).__name__}: {str(e)[:300]}", flush=True)
                summary.append(f"ICS {tag}: ERR")
            for o in ("rep", "loc", "cs", "rs", "rids", "ls", "sids", "svals", "q", "wv"):
                locals().pop(o, None)
            rep = loc = cs = None
            import gc

            gc.collect()
    print("\n".join(summary))


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEV, indirect=True)
@pytest.mark.timeout(1800)
@torch.no_grad()
def test_colshard_write(mesh_device):
    """per-step key write of the sharded slab under a trace: the update index tensor (host-updated each step, as ``st["ent"]``) carries the local slot on the owner column and -1 elsewhere."""
    md = mesh_device
    rows, cols = tuple(md.shape)
    U, N = 4, 4096
    n_alloc = -(-(N + 32) // 256) * 256
    w, _ = load_weights()
    loc = DSV41DecodeIndexer(
        md,
        w,
        rope_freqs(LAYER, 128),
        users_per_row=U,
        n_alloc=n_alloc // cols,
        key_dtype=ttnn.bfloat8_b,
        backend="matmul",
    )
    cs = ColShardSelect(loc, U, None, None, n_alloc, TOPK)
    cs.load_keys(torch.zeros(rows * U, 0, 128))
    kcfg = ttnn.create_sharded_memory_config(
        shape=(32, 128),
        core_grid=ttnn.num_cores_to_corerangeset(U, ttnn.CoreCoord(8, 8), row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    kt = torch.zeros(1, U, 32, 128)
    kt[0, :, 0] = torch.randn(U, 128)
    k = rep_up(md, kt.to(torch.bfloat16))
    k = ttnn.to_memory_config(k, kcfg)
    ent0 = torch.tensor([[10 + 8 * r + u for u in range(U)] for r in range(rows)])
    idx = cs.write_index(cs.ent_to_local(ent0, rows, cols))
    fn = lambda: ttnn.experimental.paged_update_cache(loc.k_cache, k, update_idxs_tensor=idx, page_table=None)
    t_us = traced_us(md, fn)
    # trace with changed index tensors: owner column follows the entry
    tid_ents = [
        torch.tensor([[(1000 + 997 * s + 3 * r + u) % N for u in range(U)] for r in range(rows)]) for s in range(3)
    ]
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    full = torch.zeros(rows, cols, U, n_alloc // cols, 128)
    for ent in tid_ents:
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                cs.ent_to_local(ent, rows, cols).reshape(-1).contiguous(),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensorToMesh(md, dim=0),
            ),
            idx,
        )
        ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
        for r in range(rows):
            for c in range(cols):
                t = ttnn.to_torch(ttnn.get_device_tensors(loc.k_cache)[r * cols + c]).float().reshape(U, -1, 128)
                full[r, c] = t
        for r in range(rows):
            for u in range(U):
                e = int(ent[r, u])
                c, j = e % cols, e // cols
                got = full[r, c, u, j]
                want = kt[0, u, 0]
                assert torch.corrcoef(torch.stack([got, want]))[0, 1] > 0.99, (r, u, e)
    ttnn.release_trace(md, tid)
    # exactly one nonzero slot per (row, user) on the whole slab per write since ents of earlier steps persist: count nonzero slots
    nz = (full.abs().amax(-1) > 0).sum(dim=(1, 3))  # [rows, U] slots written over the 4 writes (initial write + 3)
    print(
        f"ICS write: traced paged_update_cache {t_us:.1f} us/call; nonzero slots per (row,user) over 4 distinct writes: {nz.tolist()}",
        flush=True,
    )
    assert int(nz.max()) <= 4 and int(nz.min()) >= 1


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", DEV, indirect=True)
@pytest.mark.timeout(1800)
@torch.no_grad()
def test_colshard_breakdown(mesh_device):
    """traced time of every stage of ``ColShardSelect.select`` (T=1, per-user valid counts) at DSV41_ICS_BD_CTX (default 65536)."""
    md = mesh_device
    rows, cols = tuple(md.shape)
    U, R = 4, 4
    cfg_m, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    ctx = int(os.environ.get("DSV41_ICS_BD_CTX", "65536"))
    N = ctx // 2
    n_alloc = -(-(N + 32) // 256) * 256
    nl = n_alloc // cols
    k = TOPK
    sc = rep_up(md, torch.randn(R, 1, 1, nl).to(torch.bfloat16), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    sc_t = rep_up(md, torch.randn(R, 1, 1, nl).to(torch.bfloat16))
    gid = rep_up(md, torch.arange(nl, dtype=torch.float32).reshape(1, 1, 1, nl), ttnn.float32)
    nvl = rep_up(md, torch.full((R, 1, 1, 1), nl - 5.0), ttnn.float32)
    vl = rep_up(md, torch.full((1, 1, 1, 1), nl, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    ids = ttnn.experimental.topk_large_indices(sc, k=k, valid_length_tensor=vl)
    ids_t = ttnn.to_layout(ttnn.typecast(ids, ttnn.uint32), ttnn.TILE_LAYOUT)
    vals = ttnn.gather(sc_t, 3, index=ids_t)
    cv_in = ttnn.to_layout(vals, ttnn.ROW_MAJOR_LAYOUT)
    ci_in = ttnn.to_layout(ttnn.typecast(ids_t, ttnn.float32), ttnn.ROW_MAJOR_LAYOUT)
    cv = cfg_m.allgather(cv_in, ccl, axis=1, dim=3)
    ci = cfg_m.allgather(ci_in, ccl, axis=1, dim=3)
    pos = ttnn.experimental.topk_large_indices(cv, k=k)
    pos_t = ttnn.to_layout(ttnn.typecast(pos, ttnn.uint32), ttnn.TILE_LAYOUT)
    ci_t = ttnn.to_layout(ci, ttnn.TILE_LAYOUT)
    cv_t = ttnn.to_layout(cv, ttnn.TILE_LAYOUT)
    gid_c = rep_up(md, torch.zeros(1, 1, 1, 1), ttnn.float32)
    stages = {
        "to_layout RM->TILE scores": lambda: ttnn.to_layout(sc, ttnn.TILE_LAYOUT),
        "mask (lt + where f32 + casts)": lambda: ttnn.typecast(
            ttnn.where(ttnn.lt(gid, nvl), ttnn.typecast(sc_t, ttnn.float32), -1e30), ttnn.bfloat16
        ),
        "to_layout TILE->RM scores": lambda: ttnn.to_layout(sc_t, ttnn.ROW_MAJOR_LAYOUT),
        "local topk_large_indices": lambda: ttnn.experimental.topk_large_indices(sc, k=k, valid_length_tensor=vl),
        "ids typecast+to TILE": lambda: ttnn.to_layout(ttnn.typecast(ids, ttnn.uint32), ttnn.TILE_LAYOUT),
        "gather local vals": lambda: ttnn.gather(sc_t, 3, index=ids_t),
        "gid math": lambda: ttnn.typecast(
            ttnn.add(ttnn.multiply(ttnn.typecast(ids_t, ttnn.float32), 8.0), gid_c),
            ttnn.float32,
        ),
        "all_gather vals": lambda: cfg_m.allgather(cv_in, ccl, axis=1, dim=3),
        "all_gather ids": lambda: cfg_m.allgather(ci_in, ccl, axis=1, dim=3),
        "merge topk_large_indices (4096)": lambda: ttnn.experimental.topk_large_indices(cv, k=k),
        "merge gather ids (f32)": lambda: ttnn.gather(ci_t, 3, index=pos_t),
        "merge gather vals": lambda: ttnn.gather(cv_t, 3, index=pos_t),
        "merge layout conversions": lambda: (
            ttnn.to_layout(ci, ttnn.TILE_LAYOUT),
            ttnn.to_layout(cv, ttnn.TILE_LAYOUT),
        ),
    }
    for name, fn in stages.items():
        try:
            print(f"ICS_BD ctx={ctx} {name}: {traced_us(md, fn):.0f} us", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"ICS_BD {name}: ERR {str(e)[:150]}", flush=True)
