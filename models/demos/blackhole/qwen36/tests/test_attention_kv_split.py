# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests for QWEN36_ATTN_KV_SPLIT=user (KV-pair user split of the spec verify / drafter SDPA, tt/kv_split.py).

* test_kv_split_sdpa_op: the fused packed spec-verify SDPA at 4 groups x 6 heads (the split's shape, 27 cores per group)
  against 8 groups x 3 heads (today's shape, 13 cores per group) on the same paged KV, and both against a torch fp32
  reference, at 32K-class positions. Also the drafter's plain decode SDPA at 4 users x 6 heads vs 8 users x 3.
* test_kv_split_select: the per-device row selection (fill padding + one-hot matmul) and scatter are exact, also
  with NaN in the tile padding rows; the in-trace own-position gather equals the host reference on every device.
* test_kv_split_verify / test_kv_split_draft: one real attention layer (prefilled paged cache), one verify step of
  n users x T = 4 rows / one drafter step of B users, split vs unsplit (PCC, finite, per-row worst PCC).

Run (8-chip row, prod flags): QWEN36_SPEC_SDPA_PACKED=1 QWEN36_KV_WRITE_MULTIROW=1 QWEN36_VERIFY_HEADS_DENSE=1
  pytest models/demos/blackhole/qwen36/tests/test_attention_kv_split.py -s
"""
import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import (
    compute_pcc,
    load_attn_layer,
    model_path,
    parametrize_mesh_tp,
    replicate_to_device,
    shard_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt import kv_split as ks
from models.demos.blackhole.qwen36.tt import tp_common as tpc

HD = 256
BLOCK = 64


def _dev0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _per_dev(t):
    return [ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)]


def _rep(mesh, t, dtype, layout):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _sdpa_ref(q, kc, vc, pt, pos, scale):
    """fp32 torch decode attention: q [H, HD] of a row at position pos over the paged cache rows of table pt."""
    nb = math.ceil((pos + 1) / BLOCK)
    k = torch.cat([kc[pt[i], 0] for i in range(nb)], dim=0)[: pos + 1].float()
    v = torch.cat([vc[pt[i], 0] for i in range(nb)], dim=0)[: pos + 1].float()
    return torch.softmax((q.float() @ k.T) * scale, dim=-1) @ v


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("ctx", [pytest.param(32768, id="32k"), pytest.param(4096, id="4k")])
def test_kv_split_sdpa_op(mesh_device, ctx, reset_seeds):
    """Packed spec SDPA: 4 groups x 6 heads (split) vs 8 groups x 3 heads (today), same KV; and the drafter SDPA."""
    from models.demos.blackhole.qwen36.tt.attention.tp import SPEC_SDPA_PACK_SUPPORTED

    if not SPEC_SDPA_PACK_SUPPORTED:
        pytest.skip("ttnn build without spec_pack_heads")
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    ncores = grid.x * grid.y
    B, T, NH = 8, 4, 3
    nb = ctx // BLOCK + 1
    g = torch.Generator().manual_seed(5)
    kc = (torch.randn(B * nb, 1, BLOCK, HD, generator=g) * 0.5).to(torch.bfloat16)
    vc = torch.randn(B * nb, 1, BLOCK, HD, generator=g).to(torch.bfloat16)
    kc_tt = _rep(mesh, kc, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    vc_tt = _rep(mesh, vc, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    # the bf8 cache as the device holds it (the reference reads the same bytes)
    kc_q, vc_q = _dev0(kc_tt), _dev0(vc_tt)
    perm = torch.randperm(B * nb, generator=g).reshape(B, nb).to(torch.int32)
    base = [ctx - T - 1 - 97 * u for u in range(B)]
    base[1], base[2] = 63 - T + 1, 64  # a user at the first block's end and one at the second block's start
    pos_rows = torch.tensor([base[u] + j for u in range(B) for j in range(T)], dtype=torch.int32)
    q = (torch.randn(B, T, 2 * NH, HD, generator=g)).to(torch.bfloat16)  # [user, cand, pair head (6), HD]
    scale = HD**-0.5

    def spec_call(qv, groups, cores, table, pos, nh):
        q_tt = _rep(mesh, qv, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        shp = ttnn.Shape([1, groups, T * 32, HD])
        q_tt = ttnn.reshape(q_tt, shp, shp)
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
            max_cores_per_head_batch=cores,
        )
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt,
            kc_tt,
            vc_tt,
            page_table_tensor=_rep(mesh, table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            cur_pos_tensor=_rep(mesh, pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            scale=scale,
            program_config=cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            spec_multi_pos_tiles=T,
            spec_pack_heads=nh,
        )
        o = ttnn.reshape(o, ttnn.Shape([1, groups * T, nh, HD]), ttnn.Shape([1, groups * T, 32, HD]))
        return _dev0(o).float().reshape(groups, T, nh, HD)

    # today: 8 users x 3 heads (the pair's first 3 heads, then its last 3), 13 cores per group
    c8 = min(64, ncores // B)
    lo = spec_call(q[:, :, :NH].reshape(1, B * T, NH, HD), B, c8, perm, pos_rows, NH)
    hi = spec_call(q[:, :, NH:].reshape(1, B * T, NH, HD), B, c8, perm, pos_rows, NH)
    today = torch.cat([lo, hi], dim=2)  # [B, T, 6, HD]
    # split: the 4 users of rank 0 (0, 2, 4, 6) x 6 heads, 27 cores per group
    own = ks.own_users(B, 0)
    c4 = min(64, ncores // len(own))
    split = spec_call(
        q[own].reshape(1, len(own) * T, 2 * NH, HD),
        len(own),
        c4,
        perm[own],
        pos_rows[ks.own_rows(B, T, 0)].contiguous(),
        2 * NH,
    )
    worst = 1.0
    for i, u in enumerate(own):
        for j in range(T):
            ref = _sdpa_ref(q[u, j], kc_q, vc_q, perm[u], int(pos_rows[u * T + j]), scale)
            p_split = compute_pcc(ref, split[i, j])
            p_today = compute_pcc(ref, today[u, j])
            p_cross = compute_pcc(today[u, j], split[i, j])
            worst = min(worst, p_split, p_cross)
            assert torch.isfinite(split[i, j]).all()
            # The two runs reduce over 2x different core counts (bf8 KV, bf16 partials): judge both against the fp32
            # reference -- the split must be as close to it as today's call -- and each other at bf16-noise level.
            assert p_split >= min(0.9995, p_today - 1e-4) and p_cross >= 0.9997, (u, j, p_split, p_today, p_cross)
    logger.info(f"packed spec SDPA split (4x6, {c4} cores) vs today (8x3, {c8} cores) ctx={ctx}: worst PCC {worst:.6f}")

    # drafter: plain paged decode SDPA, 8 users x 3 heads vs the rank-0 users x 6 heads (max_cores_per_head_batch 64)
    qd = torch.randn(B, 2 * NH, HD, generator=g).to(torch.bfloat16)
    pos_d = torch.tensor([base[u] + T for u in range(B)], dtype=torch.int32)

    def dec_call(qv, table, pos):
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
            max_cores_per_head_batch=64,
        )
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            _rep(mesh, qv.unsqueeze(0), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            kc_tt,
            vc_tt,
            page_table_tensor=_rep(mesh, table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            cur_pos_tensor=_rep(mesh, pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            scale=scale,
            program_config=cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return _dev0(o).float()[0]

    d_lo = dec_call(qd[:, :NH], perm, pos_d)
    d_hi = dec_call(qd[:, NH:], perm, pos_d)
    d_split = dec_call(qd[own], perm[own], pos_d[own])
    worst_d = 1.0
    for i, u in enumerate(own):
        ref = _sdpa_ref(qd[u], kc_q, vc_q, perm[u], int(pos_d[u]), scale)
        today_u = torch.cat([d_lo[u, :NH], d_hi[u, :NH]], dim=0)
        p = compute_pcc(ref, d_split[i, : 2 * NH])
        p_t = compute_pcc(ref, today_u)
        pc = compute_pcc(today_u, d_split[i, : 2 * NH])
        worst_d = min(worst_d, p, pc)
        assert p >= min(0.9995, p_t - 1e-4) and pc >= 0.9997, (u, p, p_t, pc)
    logger.info(f"drafter decode SDPA split (4 users x 6) vs today (8 x 3) ctx={ctx}: worst PCC {worst_d:.6f}")


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("n_users,T", [(8, 4), (4, 4), (2, 4), (8, 1), (4, 1)])
def test_kv_split_select(mesh_device, n_users, T, reset_seeds, monkeypatch):
    """Selection / scatter exactness (NaN-poisoned tile padding) and the own-position gather, per device."""
    mesh = mesh_device
    tp = mesh.get_num_devices()
    if tp % 2:
        pytest.skip("needs KV pairs")
    nh2 = 6
    R = n_users * T
    Ro = (n_users // 2) * T
    plan = ks.KVSplitPlan(mesh, tp, nh2 // 2)
    plan.prepare(n_users, T)
    c = plan.consts(n_users, T)
    # q in the dense head layout, NaN in its tile padding rows (R..31): the selection must not see them
    qh = torch.randn(1, nh2, R, HD).to(torch.bfloat16)
    qh_tt = _rep(mesh, qh, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    if R % 32:
        qh_tt = ttnn.fill_implicit_tile_padding(qh_tt, float("nan"))
    from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention

    sel = TPAttention._kv_split_select_q(object.__new__(TPAttention), qh_tt, R, c)  # [1, Ro, nh2, HD] per device
    for d, got in enumerate(_per_dev(sel)):
        rows = ks.own_rows(n_users, T, d % 2)
        want = qh[0][:, rows, :].transpose(0, 1)  # [Ro, nh2, HD]
        assert torch.equal(got[0, :Ro, :nh2].to(torch.bfloat16), want), f"q selection not exact on device {d}"
    # scatter: own rows back to their global rows, zeros elsewhere (exact)
    flat = torch.randn(1, 1, Ro, nh2 * HD).to(torch.bfloat16)
    flat_tt = _rep(mesh, flat, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    full = ttnn.matmul(c["scatter"], flat_tt, compute_kernel_config=tpc.COMPUTE_SELECT)
    for d, got in enumerate(_per_dev(full)):
        want = torch.zeros(R, nh2 * HD, dtype=torch.bfloat16)
        want[ks.own_rows(n_users, T, d % 2)] = flat[0, 0]
        assert torch.equal(got[0, 0, :R].to(torch.bfloat16), want), f"scatter not exact on device {d}"
    # own positions (both gather layouts)
    pos = torch.tensor([1000 + 37 * u + j for u in range(n_users) for j in range(T)], dtype=torch.int32)
    for mode in ("rm", "tile"):
        monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT_POS", mode)
        pos_tt = ttnn.from_torch(
            pos,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        own = plan.own_positions(pos_tt, n_users, T)
        for d, got in enumerate(_per_dev(own)):
            want = ks.own_positions_host(pos, n_users, T, d % 2)
            assert torch.equal(
                got.reshape(-1)[:Ro].to(torch.int32), want
            ), f"own positions ({mode}) wrong on device {d}"
    logger.info(f"split select / scatter / positions exact ({n_users} users x T={T}, R={R})")


def _attn_setup(mesh_device, monkeypatch, max_batch):
    monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT", "user")
    monkeypatch.setenv("QWEN36_VERIFY_ATTN_LEAN", "1")
    os.environ.setdefault("HF_MODEL", model_path())
    from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp
    from models.demos.blackhole.qwen36.tt.kv_split import kv_split_supported
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.ccl import TT_CCL

    args = Qwen36ModelArgs(mesh_device, max_batch_size=max_batch, max_seq_len=4096)
    if not kv_split_supported(args):
        pytest.skip(f"no KV pairs at TP={args.num_devices}, n_kv_heads={args.n_kv_heads}")
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "full_attention")
    sd = load_attn_layer(args.CKPT_DIR, li)
    tw = load_attention_weights_tp(mesh_device, sd, args)
    a = TPAttention(mesh_device, args, tw, TT_CCL(mesh_device))
    assert a._kv_split_ready, "QWEN36_ATTN_KV_SPLIT=user must load the pair weights"
    return args, a


def _prefill_users(mesh_device, args, a, n, bpu, lens):
    from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_prefill

    cache_dtype = ttnn.bfloat8_b if tpc.kv_cache_bf8_enabled() else ttnn.bfloat16

    def mk_cache():
        return _rep(
            mesh_device, torch.zeros(n * bpu, 1, BLOCK, HD, dtype=torch.bfloat16), cache_dtype, ttnn.TILE_LAYOUT
        )

    a.set_paged_kv_cache(mk_cache(), mk_cache())
    for u in range(n):
        L = lens[u]
        cos_p, sin_p = rot_mats_prefill(mesh_device, args.rope_head_dim, L, args.rope_theta)
        pt = _rep(
            mesh_device,
            torch.tensor([list(range(u * bpu, (u + 1) * bpu))], dtype=torch.int32),
            ttnn.int32,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        a.forward_prefill_paged(
            shard_to_device(mesh_device, torch.randn(1, 1, L, args.dim, dtype=torch.bfloat16), dim=-1),
            cos_p,
            sin_p,
            pt,
            chunk_page_table=pt,
            chunk_start_idx=0,
            user_id=0,
        )


def _ctx(mesh, plan, n, T, pos_tt, pt_users_host):
    plan.prepare(n, T)
    own_pt = ttnn.from_torch(
        ks.own_page_table_host(pt_users_host, mesh.get_num_devices()),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    return {"n_users": n, "T": T, "consts": plan.consts(n, T), "pos": plan.own_positions(pos_tt, n, T), "pt": own_pt}


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("n_users", [pytest.param(8, id="U8"), pytest.param(4, id="U4")])
def test_kv_split_verify(mesh_device, n_users, reset_seeds, ensure_gc, monkeypatch):
    """One lean verify step (n users x T = 4, fused spec SDPA), split vs unsplit, on a prefilled real layer."""
    from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode

    args, a = _attn_setup(mesh_device, monkeypatch, 8)
    n, T = n_users, 4
    rows = n * T
    lens = [700 + 211 * u for u in range(n)]
    bpu = math.ceil((max(lens) + T + 1) / BLOCK) + 1  # every row's position must stay inside its own table
    _prefill_users(mesh_device, args, a, n, bpu, lens)
    pos_rows = torch.tensor([lens[u] + j for u in range(n) for j in range(T)], dtype=torch.int32)
    cur = ttnn.from_torch(
        pos_rows, dtype=ttnn.int32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    cos_v, sin_v = rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, pos_rows)
    cos_v = ttnn.reshape(cos_v, (1, 1, rows, args.rope_head_dim))  # the verify body stages [1,1,R,rd]
    sin_v = ttnn.reshape(sin_v, (1, 1, rows, args.rope_head_dim))
    blocks = [list(range(u * bpu, (u + 1) * bpu)) for u in range(n)]
    rm = lambda tb: _rep(mesh_device, torch.tensor(tb, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    pt_rows, pt_users = rm([blocks[u] for u in range(n) for _ in range(T)]), rm(blocks)
    G = a.spec_sdpa_groups(T, n)
    pt_groups = rm([blocks[gi // (G // n)] for gi in range(G)])
    x_tt = replicate_to_device(mesh_device, torch.randn(1, 1, rows, args.dim, dtype=torch.bfloat16))
    comp = tp_composer(mesh_device)
    plan = ks.KVSplitPlan(mesh_device, mesh_device.get_num_devices(), args.n_local_heads)

    def verify(split):
        a._kv_split_ctx = _ctx(mesh_device, plan, n, T, cur, torch.tensor(blocks, dtype=torch.int32)) if split else None
        try:
            if split:
                assert a._kv_split_verify_ok(rows, n, pt_users), "the verify split did not engage"
            out = a.forward_decode(
                x_tt,
                cur,
                cos_v,
                sin_v,
                page_table=pt_rows,
                alias_kv_write=True,
                spec_verify_mode=True,
                spec_page_table=pt_groups,
                spec_n_users=n,
                spec_user_page_table=pt_users,
            )
            return ttnn.to_torch(out, mesh_composer=comp).float().reshape(-1, args.dim)[:rows]
        finally:
            a._kv_split_ctx = None

    base, split = verify(False), verify(True)
    again = verify(False)  # the split step rewrote the candidate K/V: the unsplit result must not move
    assert torch.isfinite(split).all() and torch.isfinite(base).all()
    per_row = [compute_pcc(base[r], split[r]) for r in range(rows)]
    pcc_k = compute_pcc(base, again)
    logger.info(
        f"verify split vs unsplit ({n} users x T={T}): PCC {compute_pcc(base, split):.6f}, worst row "
        f"{min(per_row):.6f}, max |diff| {(base - split).abs().max().item():.3e}; unsplit re-run PCC {pcc_k:.6f} "
        f"(equal={torch.equal(base, again)})"
    )
    assert min(per_row) >= 0.999, per_row
    assert pcc_k >= 0.99999


@torch.no_grad()
@parametrize_mesh_tp()
def test_kv_split_draft(mesh_device, reset_seeds, ensure_gc, monkeypatch):
    """One drafter-shaped lean decode step (B = 8 users, one row each, max_cores_per_head_batch 64), split vs unsplit."""
    from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode

    args, a = _attn_setup(mesh_device, monkeypatch, 8)
    a.decode_sdpa_max_cores = 64
    B = 8
    # every user's prompt stays <= 2048 tokens: a single forward_prefill_paged call longer than that leaves rows of junk
    # in this test's KV cache when earlier tests left non-zero DRAM behind (seen on device for 2079 / 2111, both paths)
    lens = [900 + 140 * u for u in range(B)]
    bpu = math.ceil((max(lens) + 2) / BLOCK) + 1
    _prefill_users(mesh_device, args, a, B, bpu, lens)
    pos = torch.tensor(lens, dtype=torch.int32)
    cur = ttnn.from_torch(
        pos, dtype=ttnn.int32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    cos_d, sin_d = rot_mats_decode(mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, pos)
    blocks = [list(range(u * bpu, (u + 1) * bpu)) for u in range(B)]
    pt = _rep(mesh_device, torch.tensor(blocks, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    x_tt = replicate_to_device(mesh_device, torch.randn(1, 1, B, args.dim, dtype=torch.bfloat16))
    comp = tp_composer(mesh_device)
    plan = ks.KVSplitPlan(mesh_device, mesh_device.get_num_devices(), args.n_local_heads)

    def step(split):
        a._kv_split_ctx = _ctx(mesh_device, plan, B, 1, cur, torch.tensor(blocks, dtype=torch.int32)) if split else None
        try:
            if split:
                assert a._kv_split_draft_ok(B, pt), "the drafter split did not engage"
            out = a.forward_decode(x_tt, cur, cos_d, sin_d, page_table=pt)
            return ttnn.to_torch(out, mesh_composer=comp).float().reshape(-1, args.dim)[:B]
        finally:
            a._kv_split_ctx = None

    base, split = step(False), step(True)
    assert torch.isfinite(split).all()
    per_row = [compute_pcc(base[r], split[r]) for r in range(B)]
    logger.info(
        f"drafter split vs unsplit (B={B}): PCC {compute_pcc(base, split):.6f}, worst row {min(per_row):.6f}, "
        f"max |diff| {(base - split).abs().max().item():.3e}"
    )
    assert min(per_row) >= 0.999, per_row
