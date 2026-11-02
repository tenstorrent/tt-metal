# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops/router_topk`` (the fused exact-fp32 router top-k, one generic op) vs the composite
``TtRouter.forward`` and a torch golden, on REAL router inputs: the layer-4 residual streams of the T256 HF dump
through the real layer-4 gate weights and bias (random logits make the router degenerate). Checks per mode
(default: w fp32 / idx fp32 / dense bf16; ``ep_rows``: w bf16 / idx uint32; ``dense_only``): ids bit-exact vs the
composite and the golden, weights to 1e-6 relative, the dense matrix equal (bf16); T = 32 and T = 4 (padded rows
zero); a tie case vs the stable golden (the composite's ``ttnn.topk`` orders ties differently: README); trace
capture + replay on refreshed scores; eager timing of the two paths.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/custom_ops/test_router_topk.py -s
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.router_topk.op import RouterTopk
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_tensors, shard_headers
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.moe import TtRouter

DUMP = Path("bringup/references/glm53_flash/T256")
LAYER = 4
TILE = 32


def _gate(layer: int):
    snapshot = default_snapshot_dir()
    headers = shard_headers(snapshot)
    prefix = f"model.language_model.layers.{layer}.mlp.gate."
    t = load_tensors(snapshot, [prefix + "weight", prefix + "e_score_correction_bias"], headers)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    return cfg, t[prefix + "weight"], t[prefix + "e_score_correction_bias"]


def _real_x(T: int, offset: int = 0) -> torch.Tensor:
    """[1,1,T,H] bf16: the mean over the 4 residual streams of tokens offset..offset+T of the layer-4 prefill dump."""
    pre = torch.load(DUMP / f"layer{LAYER}_prefill.pt")
    x = pre["x_in"][0, offset : offset + T].float().mean(1)  # [T, H]
    return x.reshape(1, 1, T, -1).bfloat16()


def _composite(router: TtRouter, x: ttnn.Tensor, **kw):
    router.fused = False
    out = router.forward(x, **kw)
    router.fused = True
    return out


def _fused(router: TtRouter, x: ttnn.Tensor, **kw):
    router.fused = True
    return router.forward(x, **kw)


def _host(t):
    return None if t is None else ttnn.to_torch(t)


def _check_mode(router, x, T, scores_host, mode: str):
    """Run both paths in ``mode`` and compare: ids bit-exact (fused == composite == golden), w <= 1e-6 rel,
    dense equal (bf16). Returns a summary dict (numbers from the device tensors)."""
    E, k, scale = router.E, router.k, router.cfg.routed_scaling_factor
    bias = ttnn.to_torch(router.bias).float().reshape(-1)[:E]
    g_idx, g_w, g_dense = RouterTopk.golden(scores_host[:T], bias, k, scale)
    kw = {"dense_only": mode == "dense_only", "ep_rows": mode == "ep_rows"}
    c_w, c_idx, c_dense = (_host(t) for t in _composite(router, x, **kw))
    f_w, f_idx, f_dense = (_host(t) for t in _fused(router, x, **kw))
    res = {"mode": mode, "T": T}
    if mode != "dense_only":
        f_i = f_idx[0, 0, :T, :k].long()
        c_i = c_idx[0, 0, :T, :k].long()
        res["ids_fused_eq_composite"] = bool(torch.equal(f_i, c_i))
        res["ids_fused_eq_golden"] = bool(torch.equal(f_i, g_idx))
        res["ids_composite_eq_golden"] = bool(torch.equal(c_i, g_idx))
        fw = f_w[0, 0, :T, :k].float()
        cw = c_w[0, 0, :T, :k].float()
        gw = g_w if mode != "ep_rows" else g_w.bfloat16().float()  # ep_rows: bf16 weights
        res["w_fused_vs_golden_rel"] = float(((fw - gw).abs() / gw.abs()).max())
        res["w_composite_vs_golden_rel"] = float(((cw - gw).abs() / gw.abs()).max())
        res["w_fused_bitexact_golden"] = (
            int((fw.view(torch.int32) == gw.view(torch.int32)).sum()) if mode != "ep_rows" else None
        )
        if mode == "ep_rows":
            res["w_fused_eq_composite_bf16"] = bool(torch.equal(fw, cw))
        if f_idx.shape[-2] > T:  # padded rows of the fused outputs are zero
            res["pad_rows_zero"] = bool((f_idx[0, 0, T:] == 0).all() and (f_w[0, 0, T:] == 0).all())
    if mode != "ep_rows":
        fd = f_dense[0, 0, :T, :E].float()
        cd = c_dense[0, 0, :T, :E].float()
        gd = g_dense.float()
        res["dense_fused_eq_golden"] = bool(torch.equal(fd, gd))
        res["dense_fused_eq_composite"] = bool(torch.equal(fd, cd))
        res["dense_fused_vs_golden_maxabs"] = float((fd - gd).abs().max())
        res["dense_composite_vs_golden_maxabs"] = float((cd - gd).abs().max())
        res["dense_fused_nonzero"] = int((fd != 0).sum())
        if f_dense.shape[-2] > T:
            res["dense_pad_rows_zero"] = bool((f_dense[0, 0, T:] == 0).all())
    logger.info(f"router_topk {res}")
    return res


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("T", [32, 4])
def test_router_topk_real(mesh_device, T):
    """Real layer-4 inputs through the real gate: both paths from the SAME device scores (the fused-sigmoid gate
    matmul), ids bit-exact, weights to 1e-6, dense equal; three output modes."""
    cfg, gate_w, bias = _gate(LAYER)
    router = TtRouter(mesh_device, cfg, gate_w, bias)
    x = ttnn.from_torch(_real_x(T), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    scores_host = ttnn.to_torch(router._scores(x, T))[0, 0].float()  # [T, E] exactly what both paths rank
    assert scores_host.shape == (T, router.E)
    choice = scores_host + ttnn.to_torch(router.bias).float().reshape(1, -1)[:, : router.E]
    top = torch.sort(choice, -1, descending=True).values[:, : router.k + 1]
    logger.info(
        f"T={T}: scores in [{scores_host.min():.4f}, {scores_host.max():.4f}], min gap between the 8th and 9th "
        f"choice {(top[:, 7] - top[:, 8]).min():.3e}, exact ties in the top-9: "
        f"{int((top[:, :-1] == top[:, 1:]).sum())}"
    )
    results = [_check_mode(router, x, T, scores_host, mode) for mode in ("default", "ep_rows", "dense_only")]
    for r in results:
        if r["mode"] != "dense_only":
            assert r["ids_fused_eq_composite"], r
            assert r["ids_fused_eq_golden"], r
            assert r["w_fused_vs_golden_rel"] <= 1e-6, r
            if "pad_rows_zero" in r:
                assert r["pad_rows_zero"], r
        if r["mode"] != "ep_rows":
            assert r["dense_fused_eq_golden"], r
            if "dense_pad_rows_zero" in r:
                assert r["dense_pad_rows_zero"], r


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_router_topk_ties(mesh_device):
    """Scores from a few distinct levels (exact ties in every row, including across the 8th/9th boundary), zero
    bias: the fused op must reproduce the stable golden (descending value, lowest column first) exactly; the
    composite's ``ttnn.topk`` tie order is reported, not asserted (README: bitonic network)."""
    E, k, scale = 288, 8, 2.5
    op = RouterTopk(mesh_device, E=E, k=k, scale=scale)
    g = torch.Generator().manual_seed(5)
    levels = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    scores = levels[torch.randint(0, len(levels), (TILE, E), generator=g)]
    scores[0] = 0.5  # one row entirely tied
    scores[1] = levels[torch.arange(E) % 5]  # periodic
    bias = torch.zeros(E)
    s_t = ttnn.from_torch(
        scores.reshape(1, 1, TILE, E), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    b_t = ttnn.from_torch(bias.reshape(1, 1, 1, E), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    dense, w, idx = op.allocate_dense(TILE), op.allocate_w(TILE, ttnn.float32), op.allocate_idx(TILE, ttnn.uint32)
    op(s_t, b_t, TILE, dense=dense, w=w, idx=idx)
    g_idx, g_w, g_dense = RouterTopk.golden(scores, bias, k, scale)
    f_idx = ttnn.to_torch(idx)[0, 0, :, :k].long()
    f_w = ttnn.to_torch(w)[0, 0, :, :k].float()
    f_dense = ttnn.to_torch(dense)[0, 0, :, :E].float()
    rows_eq = int((f_idx == g_idx).all(-1).sum())
    logger.info(
        f"ties: fused ids == stable golden on {rows_eq}/{TILE} rows; row 0 {f_idx[0].tolist()}, row 1 {f_idx[1].tolist()}"
    )
    _, t_idx = ttnn.topk(ttnn.add(s_t, b_t), k, dim=-1, largest=True, sorted=True)
    t_idx = ttnn.to_torch(t_idx)[0, 0, :, :k].long()
    logger.info(
        f"ties: ttnn.topk ids == stable golden on {int((t_idx == g_idx).all(-1).sum())}/{TILE} rows, same SETS on "
        f"{sum(set(t_idx[r].tolist()) == set(g_idx[r].tolist()) for r in range(TILE))}/{TILE} rows (not asserted)"
    )
    assert rows_eq == TILE
    assert float(((f_w - g_w).abs() / g_w.abs()).max()) <= 1e-6
    assert torch.equal(f_dense, g_dense.float())


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 8 * 1024 * 1024}], indirect=True)
def test_router_topk_trace_replay(mesh_device):
    """The op inside a trace: outputs allocated in the capture, scores refreshed in place between replays."""
    cfg, gate_w, bias = _gate(LAYER)
    E, k, scale = cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.routed_scaling_factor
    op = RouterTopk(mesh_device, E=E, k=k, scale=scale)
    b_t = ttnn.from_torch(
        bias.float().reshape(1, 1, 1, E), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    pre = torch.load(DUMP / f"layer{LAYER}_prefill.pt")["x_in"][0].float().mean(1)  # [256, H]
    wg = gate_w.float()

    def scores_of(offset):
        return torch.sigmoid(pre[offset : offset + TILE] @ wg.t())  # [32, E] fp32 host-side sigmoid scores

    s_host = scores_of(0)
    s_t = ttnn.from_torch(
        s_host.reshape(1, 1, TILE, E), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )

    def step():
        dense, w, idx = op.allocate_dense(TILE), op.allocate_w(TILE, ttnn.float32), op.allocate_idx(TILE, ttnn.float32)
        op(s_t, b_t, TILE, dense=dense, w=w, idx=idx)
        return dense, w, idx

    step()  # compile
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outs = step()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    for rep, offset in enumerate((32, 64, 96)):
        s_host = scores_of(offset)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(s_host.reshape(1, 1, TILE, E), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), s_t
        )
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        g_idx, g_w, g_dense = RouterTopk.golden(s_host, bias.float(), k, scale)
        f_dense, f_w, f_idx = (ttnn.to_torch(t)[0, 0] for t in outs)
        ok_i = torch.equal(f_idx[:, :k].long(), g_idx)
        rel = float(((f_w[:, :k].float() - g_w).abs() / g_w.abs()).max())
        ok_d = torch.equal(f_dense[:, :E].float(), g_dense.float())
        logger.info(f"trace replay {rep} (tokens {offset}..): ids exact {ok_i}, w rel {rel:.2e}, dense equal {ok_d}")
        assert ok_i and rel <= 1e-6 and ok_d
    ttnn.release_trace(mesh_device, tid)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_router_topk_timing(mesh_device):
    """Eager wall time of the whole router forward (gate matmul included), composite vs fused, T = 32 (dense_only,
    the hybrid decode mode). Device time per op comes from the Tracy A/B (tests/test_perf_layer.py)."""
    cfg, gate_w, bias = _gate(LAYER)
    router = TtRouter(mesh_device, cfg, gate_w, bias)
    x = ttnn.from_torch(_real_x(TILE), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    def bench(fn, n=20):
        for _ in range(3):
            out = fn()
            [ttnn.deallocate(t) for t in out if t is not None]
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(n):
            out = fn()
            [ttnn.deallocate(t) for t in out if t is not None]
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) / n * 1e6

    t_c = bench(lambda: _composite(router, x, dense_only=True))
    t_f = bench(lambda: _fused(router, x, dense_only=True))
    logger.info(f"router forward eager wall (T=32, dense_only): composite {t_c:.0f} us, fused {t_f:.0f} us")
