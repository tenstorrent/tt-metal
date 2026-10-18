# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_SPEC_FAST_JOIN=1 on device, layer level (random GDN weights, TP mesh, config.json via HF_MODEL only):
TPGatedDeltaNet.seed_spec_rows_select against the whole-slot seeders it replaces, for both GDN verify paths
(QWEN36_VERIFY_SPEC_STEP=1: gdn_spec_step with the window ping-pong pair; =0: the composite verify).

  * twin layers with identical weights and durable state run one verify (so every user has ring / window content),
    then "prefill" fresh users with write_slot; twin A seeds them with seed_spec_rows (ring slot 0 + E_prev rebuild),
    twin B with seed_spec_rows_select (per-row slice_write + one-hot window select): ring token slot 0 torch.equal for
    every user, the window rows a mi = 0 verify reads (1 .. K-1) torch.equal for every user, the other users' rows
    untouched; a verify from both states gives torch.equal ring contents and outputs;
  * the all-users variant against seed_spec_state;
  * the two taps sources (fused-decode RM history vs conv_states -> window) agree bit for bit;
  * warmup_fast_seed compiles every program: the seeds under the compile guard add no program-cache entry.

The engine-level A/B (tokens, acceptance, step times, flag 0 vs 1 on the checkpoint) is
tests/test_spec_continuous.py::test_fast_join_vs_eager_seed.

  QWEN36_FABRIC_RING=1 MESH_DEVICE=P150x8 HF_MODEL=... pytest models/demos/blackhole/qwen36/tests/test_spec_fastjoin.py -sv
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import (
    model_path,
    parametrize_mesh_tp,
    random_gdn_state_dict,
    replicate_to_device,
    tp_composer,
)
from models.demos.blackhole.qwen36.tt.gdn.tp import (
    TPGatedDeltaNet,
    fresh_row_select,
    load_gdn_weights_tp,
    spec_conv_sel,
    spec_ctrl_page,
    spec_state_blk_idx,
)
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


def _cat0(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()


def _rep_u32(mesh, t):
    return ttnn.from_torch(
        t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _rep_f32(mesh, t):
    return ttnn.from_torch(
        t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _build_twins(mesh, B, fused, monkeypatch):
    """Two GDN layers sharing one random weight set, both on the same spec verify path (``fused``: gdn_spec_step,
    else the composite chain), with the fused batched decode / gdn_decode_step op on (the serving configuration), so
    write_slot maintains the RM conv history the fast seed reads."""
    os.environ.setdefault("HF_MODEL", model_path())
    args = Qwen36ModelArgs(mesh, max_batch_size=B, max_seq_len=256)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    sd = random_gdn_state_dict(args, seed=li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh) if mesh.get_num_devices() > 1 else None
    monkeypatch.setenv("QWEN36_GDN_FUSED_DECODE", "1")
    monkeypatch.setenv("QWEN36_GDN_DECODE_STEP_OP", "1")
    monkeypatch.setenv("QWEN36_GDN_STEP_FP32_STATE", "1")
    tw = load_gdn_weights_tp(mesh, sd, args)
    monkeypatch.setenv("QWEN36_VERIFY_SPEC_STEP", "1" if fused else "0")
    ga, gb = TPGatedDeltaNet(mesh, args, tw, tt_ccl), TPGatedDeltaNet(mesh, args, tw, tt_ccl)
    assert ga._spec_fused is fused and gb._spec_fused is fused
    if not fused:
        for g in (ga, gb):
            g.use_fused_recurrent_decode = True  # what set_gdn_fused_decode(True) does for the composite path
    return args, ga, gb


def _seed_durable(mesh, g, rec0, hist0):
    g._stable_state = True
    g.reset_state()
    g.assemble_batched_state([_rep_f32(mesh, r) for r in rec0], [replicate_to_device(mesh, h) for h in hist0])


def _ring(mesh, g, nd, T, B, Nv, Dk, Dv):
    return _cat0(mesh, g._spec_ring).reshape(nd, T, B, Nv, Dk, Dv)


def _win(mesh, g, nd, B):
    w = g.verify_win_cur()
    return _cat0(mesh, w).reshape(nd, B, w.shape[1], w.shape[2])


def _verify(mesh, g, x, mi, B, T, Nv, K, comp):
    """One verify at the accepted indices ``mi`` (per user) on either verify path; returns the layer output."""
    if g._spec_fused:
        ctrl = _rep_u32(mesh, spec_ctrl_page(mi, B, Nv, parity=g._spec_win_par))
        out = g.forward_verify_recurrent(
            replicate_to_device(mesh, x), B * T, pre_gathered=True, n_users=B, spec_ctrl=ctrl
        )
        g._spec_win_par ^= 1  # what model.verify_traced does after a replay
        ttnn.deallocate(ctrl)
    else:
        idx = _rep_u32(mesh, spec_state_blk_idx(mi, B, Nv))
        sel = replicate_to_device(mesh, spec_conv_sel(mi, B, T, K))
        out = g.forward_verify_recurrent(
            replicate_to_device(mesh, x), B * T, pre_gathered=True, n_users=B, state_blk_idx=idx, conv_sel=sel
        )
        ttnn.deallocate(idx)
        ttnn.deallocate(sel)
    return ttnn.to_torch(out, mesh_composer=comp).float()


def _prefill_fresh(mesh, g, u, rec_u, taps_u):
    """write_slot of user u from a torch state: rec [1, Nv, Dk, Dv] fp32, taps [K] x [1, 1, C] bf16."""
    g.write_slot(u, _rep_f32(mesh, rec_u), [replicate_to_device(mesh, t) for t in taps_u])


def _eq(tag, a, b):
    eq = torch.equal(a, b)
    logger.info(f"[fastjoin] {tag}: torch.equal={eq} max|d|={(a - b).abs().max().item():.3e}")
    assert eq, f"{tag}: not bit-identical"


def _guarded(mesh, fn):
    """Run ``fn`` with program-cache misses forbidden; assert it added no program-cache entry."""
    n_prog = mesh.num_program_cache_entries()
    mesh.set_program_cache_misses_allowed(False)
    try:
        fn()
    finally:
        mesh.set_program_cache_misses_allowed(True)
    assert mesh.num_program_cache_entries() == n_prog, "the fast seed compiled a program after warmup"


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("B,T", [(8, 4), (4, 8), (1, 4)])
def test_layer_fast_seed_matches_slow_seed(mesh_device, fused, B, T, reset_seeds, ensure_gc, monkeypatch):
    mesh = mesh_device
    mesh.enable_program_cache()
    args, ga, gb = _build_twins(mesh, B, fused, monkeypatch)
    nd = mesh.get_num_devices()
    Nv, Dk, Dv, C, K = ga.Nv, ga.Dk, ga.Dv, ga.qkv_dim_tp, ga.K
    comp = tp_composer(mesh)
    g_t = torch.Generator().manual_seed(B * 100 + T)
    rec0 = [0.1 * torch.randn(1, Nv, Dk, Dv, generator=g_t) for _ in range(B)]
    hist0 = [torch.randn(1, 3, C, generator=g_t).to(torch.bfloat16) for _ in range(B)]
    for g in (ga, gb):
        _seed_durable(mesh, g, rec0, hist0)
        g.prepare_spec_verify(B, T)
        g.seed_spec_state()
    fresh = sorted({2, 5} & set(range(B))) if B > 1 else [0]
    sel = replicate_to_device(mesh, fresh_row_select(fresh, B, T, K))
    sel_all = replicate_to_device(mesh, fresh_row_select(range(B), B, T, K))
    # Warmup (what MTPSpecEngine.prepare_decode runs): compiles every fast-seed program; it writes warmup content into
    # B's ring / window, which the seed_spec_state below re-establishes from the untouched durable state.
    gb.warmup_fast_seed(sel)
    gb.seed_spec_state()
    # every user gets ring / window content of its own (running rows)
    x0 = torch.randn(1, 1, B * T, args.dim, generator=g_t).to(torch.bfloat16)
    for g in (ga, gb):
        _verify(mesh, g, x0, [0] * B, B, T, Nv, K, comp)
    _eq("after verify 1: ring twins", _ring(mesh, ga, nd, T, B, Nv, Dk, Dv), _ring(mesh, gb, nd, T, B, Nv, Dk, Dv))
    _eq("after verify 1: window twins", _win(mesh, ga, nd, B), _win(mesh, gb, nd, B))

    # ---- 1. fresh users join: seed_spec_rows (A) vs seed_spec_rows_select (B) ----
    new_rec = {u: 0.1 * torch.randn(1, Nv, Dk, Dv, generator=g_t) for u in fresh}
    new_taps = {u: [torch.randn(1, 1, C, generator=g_t).to(torch.bfloat16) for _ in range(K)] for u in fresh}
    for g in (ga, gb):
        for u in fresh:
            _prefill_fresh(mesh, g, u, new_rec[u], new_taps[u])
    ring_before = _ring(mesh, gb, nd, T, B, Nv, Dk, Dv)
    win_before = _win(mesh, gb, nd, B)
    ga.seed_spec_rows(fresh)
    _guarded(mesh, lambda: gb.seed_spec_rows_select(fresh, sel))
    ring_a, ring_b = _ring(mesh, ga, nd, T, B, Nv, Dk, Dv), _ring(mesh, gb, nd, T, B, Nv, Dk, Dv)
    win_a, win_b = _win(mesh, ga, nd, B), _win(mesh, gb, nd, B)
    _eq("fresh seed: ring slot 0 A vs B", ring_a[:, 0], ring_b[:, 0])
    _eq("fresh seed: ring slots 1.. untouched (B)", ring_b[:, 1:], ring_before[:, 1:])
    for u in fresh:
        _eq(
            f"fresh seed: ring slot 0 of user {u} = its prefill state",
            ring_b[:, 0, u],
            new_rec[u][0].expand(nd, -1, -1, -1),
        )
        want = torch.cat(new_taps[u][1:], dim=1)[0].float().expand(nd, -1, -1)  # taps 1 .. K-1
        _eq(f"fresh seed: window rows 1..K-1 of user {u} = taps 1..K-1", win_b[:, u, 1:K], want)
        assert not bool(win_b[:, u, 0].any()) and not bool(
            win_b[:, u, K:].any()
        ), "unread rows of a fresh user are zero"
    _eq("fresh seed: window rows 1..K-1 A vs B (every user)", win_a[:, :, 1:K], win_b[:, :, 1:K])
    keep = [u for u in range(B) if u not in fresh]
    if keep:
        _eq("fresh seed: running users' window untouched", win_b[:, keep], win_before[:, keep])
        _eq("fresh seed: running users' window A vs B", win_a[:, keep], win_b[:, keep])
    # the two taps sources agree
    t_hist, t_win = gb._fresh_taps_tile(route="hist"), gb._fresh_taps_tile(route="win")
    assert t_hist is not None, "the fused-decode RM history is not maintained: _build_twins expects it"
    _eq("taps tile: RM history route vs conv_states route", _cat0(mesh, t_hist), _cat0(mesh, t_win))
    ttnn.deallocate(t_hist)
    ttnn.deallocate(t_win)

    # ---- 2. the next verify from both states: fresh users at mi = 0, running users at mi = 1 ----
    mi = [0 if u in fresh else min(1, T - 1) for u in range(B)]
    x1 = torch.randn(1, 1, B * T, args.dim, generator=g_t).to(torch.bfloat16)
    out_a = _verify(mesh, ga, x1, mi, B, T, Nv, K, comp)
    out_b = _verify(mesh, gb, x1, mi, B, T, Nv, K, comp)
    _eq("verify after the seeds: layer output A vs B", out_a, out_b)
    _eq(
        "verify after the seeds: ring A vs B",
        _ring(mesh, ga, nd, T, B, Nv, Dk, Dv),
        _ring(mesh, gb, nd, T, B, Nv, Dk, Dv),
    )
    _eq("verify after the seeds: new window A vs B", _win(mesh, ga, nd, B), _win(mesh, gb, nd, B))

    # ---- 3. every user fresh: seed_spec_state (A) vs seed_spec_rows_select(all) (B) ----
    all_rec = [0.1 * torch.randn(1, Nv, Dk, Dv, generator=g_t) for _ in range(B)]
    all_taps = [[torch.randn(1, 1, C, generator=g_t).to(torch.bfloat16) for _ in range(K)] for _ in range(B)]
    for g in (ga, gb):
        for u in range(B):
            _prefill_fresh(mesh, g, u, all_rec[u], all_taps[u])
    ga.seed_spec_state()
    _guarded(mesh, lambda: gb.seed_spec_rows_select(range(B), sel_all))
    _eq(
        "all fresh: ring slot 0 A vs B",
        _ring(mesh, ga, nd, T, B, Nv, Dk, Dv)[:, 0],
        _ring(mesh, gb, nd, T, B, Nv, Dk, Dv)[:, 0],
    )
    _eq("all fresh: window rows 1..K-1 A vs B", _win(mesh, ga, nd, B)[:, :, 1:K], _win(mesh, gb, nd, B)[:, :, 1:K])
    x2 = torch.randn(1, 1, B * T, args.dim, generator=g_t).to(torch.bfloat16)
    out_a = _verify(mesh, ga, x2, [0] * B, B, T, Nv, K, comp)
    out_b = _verify(mesh, gb, x2, [0] * B, B, T, Nv, K, comp)
    _eq("all fresh: verify output A vs B", out_a, out_b)
    _eq("all fresh: verify ring A vs B", _ring(mesh, ga, nd, T, B, Nv, Dk, Dv), _ring(mesh, gb, nd, T, B, Nv, Dk, Dv))
    assert torch.isfinite(out_b).all()
    for t in (sel, sel_all):
        ttnn.deallocate(t)
