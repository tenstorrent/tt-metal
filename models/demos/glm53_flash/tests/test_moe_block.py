# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The per-chip routed-expert block of the 8x4 hybrid placement on ONE chip with real layer weights:
chip (0, 0)'s slice = column 0's 72 experts x row 0's 256-wide intermediate slice (W13 [1, 72, 4096, 512],
W2 [1, 72, 256, 4096]), real tokens and the real router's weights. Compares the sparse block path with the
legacy program config (``GLM53_SPARSE_CONFIG=legacy``: most cores, 4-tile K blocks), the tuned config
(``tt/moe.py::sparse_matmul_config``) and the dense layout (``TtRoutedExperts(dense=True)``, all tokens in
one pass): PCC against a torch reference and against each other, and the warmed device time per call
(captured trace, replayed) at M = 32 (4 users/row x 8 rows gathered) and M = 256 (32 users/row).

    TT_VISIBLE_DEVICES=2 pytest models/demos/glm53_flash/tests/test_moe_block.py -s
"""

from __future__ import annotations

import dataclasses
import os
import time

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.glm53_flash.tests.test_moe import LAYER, realistic_mlp_input, weights  # noqa: F401 (fixture)
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.moe import TtRoutedExperts

ROWS, COLS = 8, 4  # the emulated placement: 8x4 Galaxy, chip (0, 0)
STEPS = int(os.environ.get("GLM53_BLOCK_STEPS", "20"))
# reference outputs of an earlier run (the "current block"): saved when the file does not exist, else loaded and
# every path is compared against its entry (PCC >= 0.9999 = the exactness bar for a block rewrite)
REF = os.environ.get("GLM53_BLOCK_REF", "")
DENSE_W_DTYPE = (
    ttnn.float32 if os.environ.get("GLM53_BLOCK_W_FP32", "0") == "1" else ttnn.bfloat16
)  # the router's dtype


def count_ops(device, fn) -> tuple[list[str], list[str]]:
    """(top-level ttnn ops, device operations) of one call, from a graph capture with the fast runtime mode off
    (so the C++-internal ops, e.g. the sparse matmul's output zero-fill, are visible)."""
    with ttnn.manage_config("enable_fast_runtime_mode", False):
        ttnn.graph.begin_graph_capture()
        y = fn()
        ttnn.synchronize_device(device)
        graph = ttnn.graph.end_graph_capture()
    ttnn.deallocate(y)
    depth, top, dev = 0, [], []
    for node in graph:
        kind = node.get("node_type")
        if kind == "function_start":
            name = str(node.get("params", {}).get("name", "?"))
            if depth == 0:
                top.append(name)
            if "::prim::" in name or "DeviceOperation" in name or name.endswith("operation::run"):
                dev.append(name)
            depth += 1
        elif kind == "function_end":
            depth -= 1
    return top, dev


def local_slices(sd, cfg: GLM53FlashConfig, row: int = 0, col: int = 0):
    """Chip (row, col)'s expert slices of the hybrid placement: [E_loc, 2 I_loc, H] (gate rows then up rows,
    HF stacked order) and [E_loc, H, I_loc]."""
    E_loc, I_loc, I = cfg.n_routed_experts // COLS, cfg.moe_intermediate_size // ROWS, cfg.moe_intermediate_size
    gu, dn = sd["experts.gate_up_proj"], sd["experts.down_proj"]  # [E, 2I, H], [E, H, I]
    es, isl = slice(col * E_loc, (col + 1) * E_loc), slice(row * I_loc, (row + 1) * I_loc)
    gate_up_loc = torch.cat([gu[es, :I, :][:, isl, :], gu[es, I:, :][:, isl, :]], dim=1)
    return gate_up_loc.contiguous(), dn[es][:, :, isl].contiguous()


def ref_block(x: torch.Tensor, w_loc: torch.Tensor, gate_up_loc, down_loc, limit: float) -> torch.Tensor:
    """Torch fp32 reference of the chip's contribution: sum_e w[t, e] * expert_e(x[t]) on the slices."""
    E_loc, I2, _ = gate_up_loc.shape
    I_loc = I2 // 2
    xf = x.float()
    y = torch.zeros(x.shape[0], down_loc.shape[1])
    for e in range(E_loc):
        g = xf @ gate_up_loc[e, :I_loc].float().t()
        u = xf @ gate_up_loc[e, I_loc:].float().t()
        h = F.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)
        y += w_loc[:, e : e + 1] * (h @ down_loc[e].float().t())
    return y


def timed_trace(device, fn, steps: int):
    """Compile, capture one call as a trace, warm replay, then ``steps`` replays: (output, seconds per call)."""
    y = fn()
    ttnn.synchronize_device(device)
    ttnn.deallocate(y)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    y = fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(steps):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    dt = (time.perf_counter() - t0) / steps
    out = ttnn.to_torch(y).float()
    ttnn.release_trace(device, tid)
    ttnn.deallocate(y)
    return out, dt


@pytest.mark.timeout(2400)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 160 * 1024 * 1024}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_expert_block_paths(mesh_device, weights):
    snapshot, text, raw, hf, sd = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    E_loc, I_loc = cfg.n_routed_experts // COLS, cfg.moe_intermediate_size // ROWS
    cfg_loc = dataclasses.replace(cfg, n_routed_experts=E_loc, moe_intermediate_size=I_loc)
    gate_up_loc, down_loc = local_slices(sd, cfg)
    logger.info(f"layer {LAYER} chip (0,0) slices: gate_up {tuple(gate_up_loc.shape)} down {tuple(down_loc.shape)}")

    T_max = 256
    x_all = realistic_mlp_input(snapshot, text, raw, T_max).reshape(T_max, -1)  # real tokens [256, H] bf16
    with torch.no_grad():
        _, w_hf, idx_hf = hf.gate(x_all)
    dense_all = torch.zeros(T_max, cfg.n_routed_experts).scatter_(
        1, idx_hf.reshape(T_max, -1), w_hf.reshape(T_max, -1).float()
    )
    w_loc_all = dense_all[:, :E_loc].contiguous()  # column 0's experts (zero rows for tokens routed elsewhere)

    sparse = TtRoutedExperts(mesh_device, cfg_loc, gate_up_loc, down_loc)
    dense = TtRoutedExperts(mesh_device, cfg_loc, gate_up_loc, down_loc, dense=True)
    results = {}
    ref_outs = None
    if REF and os.path.isfile(REF):
        ref_outs = {(int(k.split("|")[0]), k.split("|")[1]): v for k, v in torch.load(REF).items()}
        logger.info(f"reference outputs loaded from {REF}: {sorted(ref_outs)}")
    all_outs = {}
    for T in (32, 256):
        x, w_loc = x_all[:T], w_loc_all[:T]
        active = int((w_loc.abs().sum(0) > 0).sum())
        ref = ref_block(x, w_loc, gate_up_loc, down_loc, cfg_loc.swiglu_limit)
        x_dev = ttnn.from_torch(
            x.reshape(1, 1, T, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        w_dev = ttnn.from_torch(
            w_loc.reshape(1, 1, T, E_loc), dtype=DENSE_W_DTYPE, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        outs = {}
        for name, mod, env in (
            ("sparse legacy", sparse, "legacy"),
            ("sparse tuned", sparse, ""),
            ("dense", dense, ""),
        ):
            os.environ["GLM53_SPARSE_CONFIG"] = env
            top, dev_ops = count_ops(mesh_device, lambda: mod.forward(x_dev, w_dev))
            y, dt = timed_trace(mesh_device, lambda: mod.forward(x_dev, w_dev), STEPS)
            os.environ.pop("GLM53_SPARSE_CONFIG", None)
            y = y.reshape(T, -1)
            outs[name] = y
            all_outs[(T, name)] = y
            p = pcc(ref, y)
            p_vs = pcc(outs["sparse legacy"], y)
            p_ref = pcc(ref_outs[(T, name)], y) if ref_outs is not None and (T, name) in ref_outs else float("nan")
            blocks = 1 if mod.dense else -(-T // 32)
            results[(T, name)] = (dt, p, p_vs, active, len(top), len(dev_ops), p_ref)
            logger.info(
                f"M={T} ({active} of {E_loc} experts active) {name:14s}: {dt*1e6:8.1f} us/call "
                f"({dt*1e6/blocks:7.1f} us per 32-token block x {blocks}), {len(top)} ttnn ops / {len(dev_ops)} device ops, "
                f"pcc vs torch {p:.6f}, vs sparse-legacy {p_vs:.6f}, vs reference run {p_ref:.6f}"
            )
            logger.info(f"    ttnn ops: {top}")
            logger.info(f"    device ops: {dev_ops}")
        ttnn.deallocate(x_dev)
        ttnn.deallocate(w_dev)
    if REF and ref_outs is None:
        torch.save({f"{T}|{name}": y for (T, name), y in all_outs.items()}, REF)
        logger.info(f"reference outputs saved to {REF}")
    logger.info(
        "| M | active | path | us/call | us per 32-token block | ttnn ops | device ops | pcc vs torch | "
        "pcc vs sparse-legacy | pcc vs reference run |"
    )
    for (T, name), (dt, p, p_vs, active, n_top, n_dev, p_ref) in results.items():
        blocks = 1 if name == "dense" else -(-T // 32)
        logger.info(
            f"| {T} | {active}/{E_loc} | {name} | {dt*1e6:.1f} | {dt*1e6/blocks:.1f} | {n_top} | {n_dev} | {p:.6f} | "
            f"{p_vs:.6f} | {p_ref:.6f} |"
        )
    for (T, name), (dt, p, p_vs, active, n_top, n_dev, p_ref) in results.items():
        assert p > 0.999, (T, name, p)
        assert p_vs > 0.9999, (T, name, p_vs)
        assert p_ref != p_ref or p_ref > 0.9999, (T, name, p_ref)
