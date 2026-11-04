# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device checks for QWEN36_SPEC_SDPA_PACKED: the fused spec-verify SDPA with spec_pack_heads=NH (each user group's
Tg=4 candidates x NH=3 local q heads in ONE packed q tile, a causal bound per row) against today's unpacked call
(spec_multi_pos_tiles=4: 4 padded q row-tiles per group).

  test_spec_sdpa_packed_op               the op alone at the verify's shapes (8 groups x Tg=4, NH=3, head_dim 256, bf8
                                         paged KV in 64-row blocks, 15 cores per group, dynamic k chunk): the valid
                                         output rows must be bit-identical (NaN == NaN), and both close to a torch
                                         reference. Position sets cover 32-row tile and 128-position chunk edges, mid
                                         context and 32K. The 32K case also times both calls back to back.
  test_spec_sdpa_packed_verify_attention TPAttention.forward_decode in spec-verify mode (the lean verify body, real
                                         layer weights) with the flag off and on: the layer output and the written KV
                                         caches must be bit-identical.

Run (inside the tt-metal container, on a Blackhole Galaxy / P150x8 TP mesh):
    source /tmp/penv.sh
    pytest models/demos/blackhole/qwen36/tests/test_spec_sdpa_packed.py -v -s
"""
import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import (
    load_attn_layer,
    model_path,
    parametrize_mesh_tp,
    replicate_to_device,
)

BLOCK = 64  # demo/vLLM paged block
TG = 4  # candidates per group (T = K+1 = 4)
NH = 3  # local q heads per device at TP8
HD = 256

# Per-group base positions; group g's candidate j sits at base[g] + j (a G-group call takes the first G).
POSITION_SETS = {
    # 29..32 / 30..33 straddle a 32-row tile, 125..128 the first 128-position chunk edge; 127..130 leaves candidate 0
    # with a fully masked last chunk; 0..3 is the shortest scan.
    "edges": [127, 125, 28, 29, 30, 124, 1, 0],
    "mid": [2047, 4093, 517, 1000, 2048, 3001, 4094, 1919],
    "32k": [32765, 32768, 32766, 32767, 32289, 30000, 31999, 32700],
}
# (position set, groups): 8 groups is the 8-user verify (15 cores per group); 4 / 2 / 1 groups are the smaller row
# buckets (30 / 60 / 64 cores per group, deeper reduction trees, masked chunks spread over more cores).
OP_CASES = [("edges", 8), ("mid", 8), ("32k", 8), ("edges", 4), ("edges", 2), ("mid", 2), ("mid", 1), ("32k", 1)]


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def _rep(mesh, t, dtype, layout, mem=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=mem
    )


def _dev0(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[: t.shape[0]]


def _spec_cfg(mesh, groups):
    """The verify's program config (TPAttention._spec_sdpa_plan at T=4, 8 users): cores per group = grid // groups,
    capped at 64; dynamic k chunk; exp_approx off."""
    grid = mesh.compute_with_storage_grid_size()
    max_cores = min(64, max(1, (grid.x * grid.y) // groups))
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        exp_approx_mode=False,
        q_chunk_size=0,
        k_chunk_size=0,
        max_cores_per_head_batch=max_cores,
    )


def _valid_rows(out, groups):
    """[1, G, Tg*32, HD] -> [G, Tg, NH, HD]: rows 0..NH-1 of each candidate row-tile."""
    return out.reshape(groups, TG, 32, HD)[:, :, :NH, :].float()


def _same(a, b):
    """Bit-identical as values, NaN == NaN."""
    return bool(((a == b) | (torch.isnan(a) & torch.isnan(b))).all())


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("pos_set, groups", OP_CASES, ids=[f"{p}-g{g}" for p, g in OP_CASES])
def test_spec_sdpa_packed_op(mesh_device, pos_set, groups, reset_seeds, ensure_gc):
    GROUPS = groups
    base = POSITION_SETS[pos_set][:GROUPS]
    nb = math.ceil((max(base) + TG) / BLOCK)  # blocks per group (user)
    nblocks = GROUPS * nb
    g = torch.Generator().manual_seed(11)
    k_init = torch.randn(nblocks, 1, BLOCK, HD, generator=g).to(torch.bfloat16)
    v_init = torch.randn(nblocks, 1, BLOCK, HD, generator=g).to(torch.bfloat16)
    # Each user owns a random set of blocks (as a real allocator hands them out).
    perm = torch.randperm(nblocks, generator=g)
    pt = perm.reshape(GROUPS, nb).to(torch.int32)
    pos = torch.tensor([base[gi] + j for gi in range(GROUPS) for j in range(TG)], dtype=torch.int32)
    # Q: every row random, the padding rows NH..31 of each candidate row-tile too (they must not matter).
    q = torch.randn(1, GROUPS, TG * 32, HD, generator=g).to(torch.bfloat16)

    kc = _rep(mesh_device, k_init, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    vc = _rep(mesh_device, v_init, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    pt_tt = _rep(mesh_device, pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    pos_tt = _rep(mesh_device, pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    q_tt = _rep(mesh_device, q, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    cfg = _spec_cfg(mesh_device, GROUPS)
    scale = HD**-0.5

    def call(pack):
        return ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt,
            kc,
            vc,
            page_table_tensor=pt_tt,
            cur_pos_tensor=pos_tt,
            scale=scale,
            program_config=cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            spec_multi_pos_tiles=TG,
            **({"spec_pack_heads": NH} if pack else {}),
        )

    outs = {}
    for pack in (False, True):
        o = call(pack)
        outs[pack] = _dev0(mesh_device, o)
        ttnn.deallocate(o)
    ref_u, got_p = _valid_rows(outs[False], GROUPS), _valid_rows(outs[True], GROUPS)

    # Torch reference from the device's own (bf8-rounded) cache.
    kd = _dev0(mesh_device, kc).float()
    vd = _dev0(mesh_device, vc).float()
    qv = q.reshape(GROUPS, TG, 32, HD)[:, :, :NH, :].float()
    ref = torch.zeros(GROUPS, TG, NH, HD)
    for gi in range(GROUPS):
        k_u = kd[pt[gi].long()].reshape(nb * BLOCK, HD)
        v_u = vd[pt[gi].long()].reshape(nb * BLOCK, HD)
        for j in range(TG):
            p = int(pos[gi * TG + j])
            s = (qv[gi, j] @ k_u[: p + 1].T) * scale
            ref[gi, j] = torch.softmax(s, dim=-1) @ v_u[: p + 1]

    nan_u, nan_p = int(torch.isnan(ref_u).sum()), int(torch.isnan(got_p).sum())
    same = _same(ref_u, got_p)
    diff = int((~((ref_u == got_p) | (torch.isnan(ref_u) & torch.isnan(got_p)))).sum())
    pcc_up = _pcc(torch.nan_to_num(ref_u), torch.nan_to_num(got_p))
    per_group = []
    for gi in range(GROUPS):
        per_group.append(
            (
                base[gi],
                round(_pcc(torch.nan_to_num(ref_u[gi]), ref[gi]), 5),
                round(_pcc(torch.nan_to_num(got_p[gi]), ref[gi]), 5),
            )
        )
    tag = f"{pos_set}-g{GROUPS}"
    logger.info(
        f"[{tag}] packed vs unpacked: bit-identical={same} ({diff} differing values), PCC={pcc_up:.6f}, "
        f"NaN unpacked={nan_u} packed={nan_p}; per group (base, PCC unpacked vs torch, PCC packed vs torch): "
        f"{per_group}"
    )
    assert same, f"[{tag}] packed spec SDPA differs from the unpacked call in {diff} values (PCC {pcc_up:.6f})"
    for b, pu, pp in per_group:
        if not math.isnan(pu) and pu >= 0.99:
            assert pp >= 0.99, f"[{tag}] base {b}: packed PCC vs torch {pp} (unpacked {pu})"

    if pos_set == "32k" and GROUPS == 8:
        # Per-call device time at the production shape, calls back to back (each is 400+ us, well above dispatch).
        n = int(os.environ.get("SPEC_SDPA_TIME_ITERS", "40"))
        times = {}
        for pack in (False, True, False, True):
            for _ in range(3):
                ttnn.deallocate(call(pack))
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(n):
                ttnn.deallocate(call(pack))
            ttnn.synchronize_device(mesh_device)
            times.setdefault(pack, []).append((time.perf_counter() - t0) / n * 1e6)
        logger.info(
            f"[32k] spec SDPA per call (8 groups x Tg=4, 32K): unpacked {[round(t, 1) for t in times[False]]} us, "
            f"packed {[round(t, 1) for t in times[True]]} us; x16 layers: unpacked "
            f"{16 * min(times[False]) / 1000:.2f} ms, packed {16 * min(times[True]) / 1000:.2f} ms"
        )

    for t in (kc, vc, pt_tt, pos_tt, q_tt):
        ttnn.deallocate(t)


@torch.no_grad()
@parametrize_mesh_tp()
def test_spec_sdpa_packed_verify_attention(mesh_device, reset_seeds, ensure_gc, monkeypatch):
    from models.demos.blackhole.qwen36.tt.attention import tp as attn_tp
    from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode
    from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL

    assert attn_tp.SPEC_SDPA_PACK_SUPPORTED, "this ttnn build has no spec_pack_heads"
    os.environ.setdefault("HF_MODEL", model_path())
    users, T = 8, TG
    R = users * T
    base = [2047, 2048, 3001, 127, 4093, 1000, 517, 3500]
    nb = math.ceil((max(base) + T) / BLOCK)
    args = Qwen36ModelArgs(mesh_device, max_batch_size=8, max_seq_len=nb * BLOCK)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
    NKV, HDm = args.n_local_kv_heads, args.head_dim
    tt_ccl = TT_CCL(mesh_device)
    tw = load_attention_weights_tp(mesh_device, load_attn_layer(args.CKPT_DIR, li), args)

    g = torch.Generator().manual_seed(5)
    nblocks = users * nb
    init_k = torch.randn(nblocks, NKV, BLOCK, HDm, generator=g).to(torch.bfloat16)
    init_v = torch.randn(nblocks, NKV, BLOCK, HDm, generator=g).to(torch.bfloat16)
    x = torch.randn(1, 1, R, args.dim, generator=g).to(torch.bfloat16)
    pos = torch.tensor([base[r // T] + r % T for r in range(R)], dtype=torch.int32)
    users_pt = torch.tensor([list(range(u * nb, (u + 1) * nb)) for u in range(users)], dtype=torch.int32)
    rows_pt = users_pt.repeat_interleave(T, dim=0)

    def rep_rm(t):
        return _rep(mesh_device, t, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, mem=None)

    def run(packed):
        monkeypatch.setenv("QWEN36_SPEC_SDPA_PACKED", "1" if packed else "0")
        a = TPAttention(mesh_device, args, tw, tt_ccl)
        assert a.spec_sdpa_pack_heads(T) == (a.NH if packed else 0)
        kc = _rep(mesh_device, init_k, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
        vc = _rep(mesh_device, init_v, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
        a.set_paged_kv_cache(kc, vc)
        groups = a.spec_sdpa_groups(T, users)
        assert groups == users
        cos, sin = rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, pos)
        out = a.forward_decode(
            replicate_to_device(mesh_device, x),
            rep_rm(pos),
            cos,
            sin,
            page_table=rep_rm(rows_pt),
            alias_kv_write=True,
            spec_verify_mode=True,
            spec_page_table=rep_rm(users_pt),
            spec_n_users=users,
            spec_user_page_table=rep_rm(users_pt),
        )
        ttnn.synchronize_device(mesh_device)
        comp = ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        got = tuple(ttnn.to_torch(t, mesh_composer=comp) for t in (out, kc, vc))
        for t in (out, kc, vc):
            ttnn.deallocate(t)
        return got

    off_out, off_k, off_v = run(False)
    on_out, on_k, on_v = run(True)
    off_f, on_f = off_out.float(), on_out.float()
    same = _same(off_f, on_f)
    pcc = _pcc(torch.nan_to_num(off_f), torch.nan_to_num(on_f))
    nan_rows = sorted({int(r) for r in torch.nonzero(torch.isnan(off_f).reshape(-1, R, off_f.shape[-1]).any(-1))[:, 1]})
    logger.info(
        f"verify attention, 8 users x T=4 at {base}: flag on vs off output bit-identical={same} (PCC {pcc:.6f}); "
        f"K cache identical={torch.equal(off_k, on_k)}, V cache identical={torch.equal(off_v, on_v)}; "
        f"flag-off rows with NaN: {nan_rows}, flag-on NaN values: {int(torch.isnan(on_f).sum())}"
    )
    assert torch.equal(off_k, on_k) and torch.equal(off_v, on_v), "the KV write changed"
    assert same, f"packed verify attention differs: PCC {pcc:.6f}"
