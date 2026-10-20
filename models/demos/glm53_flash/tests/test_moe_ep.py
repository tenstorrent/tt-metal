# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The EP (expert-parallel) routed-expert path (``TtEPExperts``, doc/mesh/orchestration_v2.md) on one chip
(D = 1: no dispatch / combine) against the hybrid stream path and the torch reference, on real layer-4 weights
and routing: a chip's share of full-width experts (9 = the 8x4, 36 = a (2,4) block), the tokens' top-8 lists over
all 288 experts (the ids outside the chip's share are ignored, as after a dispatch), T = 32 / 256 tokens, traced.
Also the dispatched-row imbalance of the real routing (rows per device, D = 32 / 8) and the gate|up bfp4 knob
(GLM53_EXPERT_W13_DTYPE=bfp4: block-level PCC vs the bfp8 paths).

    TT_VISIBLE_DEVICES=2 pytest models/demos/glm53_flash/tests/test_moe_ep.py -s
"""

from __future__ import annotations

import dataclasses
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tests.test_moe_block import (  # noqa: F401 (fixtures)
    LAYER,
    count_ops,
    realistic_mlp_input,
    ref_block,
    timed_trace,
    weights,
)
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.moe import TtEPExperts, TtRoutedExperts

STEPS = int(os.environ.get("GLM53_BLOCK_STEPS", "20"))
W13_DTYPE = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}[os.environ.get("GLM53_EXPERT_W13_DTYPE", "bfp8")]


def imbalance(dense_all: torch.Tensor, D: int) -> str:
    """Dispatched rows per device (chip d owns experts [d E/D, (d+1) E/D)) of a [T, E] routing matrix."""
    E = dense_all.shape[1]
    n_e = (dense_all != 0).sum(0).float()
    per_dev = n_e.reshape(D, E // D).sum(1)
    return (
        f"D={D}: rows per device max {per_dev.max():.0f} / mean {per_dev.mean():.1f} / min {per_dev.min():.0f} "
        f"(max/mean {per_dev.max() / per_dev.mean():.2f}), active experts {(n_e > 0).sum()} of {E}"
    )


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 160 * 1024 * 1024}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("E_loc", [9, 36])
@pytest.mark.parametrize("a2a", [False, True], ids=["direct", "a2a"])
def test_moe_ep_single_chip(mesh_device, weights, E_loc, a2a, monkeypatch):
    """``a2a``: the D = 1 dispatch / combine (GLM53_EP_FORCE_A2A=1, the identity on one chip) around the EP op --
    the mesh plumbing checked before a block is available (runs only with GLM53_EP_A2A_TESTS=1 until then)."""
    if a2a:
        if os.environ.get("GLM53_EP_A2A_TESTS", "0") != "1":
            pytest.skip("D = 1 all_to_all plumbing: GLM53_EP_A2A_TESTS=1 (device_params need the fabric)")
        monkeypatch.setenv("GLM53_EP_FORCE_A2A", "1")
    snapshot, text, raw, hf, sd = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    E, I = cfg.n_routed_experts, cfg.moe_intermediate_size
    cfg_loc = dataclasses.replace(cfg, n_routed_experts=E_loc)  # the chip's experts at full width
    gate_up_loc = sd["experts.gate_up_proj"][:E_loc].contiguous()  # [E_loc, 2I, H]
    down_loc = sd["experts.down_proj"][:E_loc].contiguous()  # [E_loc, H, I]

    T_max = 256
    x_all = realistic_mlp_input(snapshot, text, raw, T_max).reshape(T_max, -1)
    with torch.no_grad():
        _, w_hf, idx_hf = hf.gate(x_all)
    idx_all = idx_hf.reshape(T_max, -1).long()  # [256, 8] global expert ids
    w_all = w_hf.reshape(T_max, -1).float()
    dense_all = torch.zeros(T_max, E).scatter_(1, idx_all, w_all)
    for D in (32, 8):
        for T in (32, 256):
            logger.info(f"real routing T={T}: {imbalance(dense_all[:T], D)}")

    hybrid = TtRoutedExperts(mesh_device, cfg_loc, gate_up_loc, down_loc, stream=True)
    ep = TtEPExperts(mesh_device, cfg_loc, gate_up_loc, down_loc, expert_dtype=W13_DTYPE)
    results = []
    for T in (32, 256):
        x, w_loc = x_all[:T], dense_all[:T, :E_loc]
        active = int((w_loc.abs().sum(0) > 0).sum())
        rows = int((w_loc != 0).sum())
        ref = ref_block(x, w_loc, gate_up_loc, down_loc, cfg.swiglu_limit)
        x_dev = ttnn.from_torch(
            x.reshape(1, 1, T, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        dense_dev = ttnn.from_torch(
            w_loc.reshape(1, 1, T, E_loc), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        idx_dev = ttnn.from_torch(
            idx_all[:T].float().reshape(1, 1, T, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        w_dev = ttnn.from_torch(
            w_all[:T].reshape(1, 1, T, -1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        y_h, dt_h = timed_trace(mesh_device, lambda: hybrid.forward(x_dev, dense_dev), STEPS)
        top, dev_ops = count_ops(mesh_device, lambda: ep.forward(x_dev, w_dev, idx_dev))
        y_e, dt_e = timed_trace(mesh_device, lambda: ep.forward(x_dev, w_dev, idx_dev), STEPS)
        y_h, y_e = y_h.reshape(T, -1), y_e.reshape(T, -1)
        p_ref, p_h, p_he = pcc(ref, y_e), pcc(ref, y_h), pcc(y_h, y_e)
        logger.info(
            f"E_loc {E_loc} T={T} ({active} active, {rows} routed rows): hybrid stream {dt_h*1e6:.1f} us, "
            f"EP {dt_e*1e6:.1f} us ({len(dev_ops)} device ops; w13 {W13_DTYPE.name}); pcc EP vs torch {p_ref:.6f}, "
            f"hybrid vs torch {p_h:.6f}, EP vs hybrid {p_he:.6f}"
        )
        logger.info(f"    EP ttnn ops: {top}")
        results.append((E_loc, T, active, rows, dt_h * 1e6, dt_e * 1e6, len(dev_ops), p_ref, p_h, p_he))
        if W13_DTYPE == ttnn.bfloat8_b:
            assert p_he > 0.9999, p_he
            assert p_ref > 0.9999, p_ref
        else:
            assert p_he > 0.98, p_he
        for t in (x_dev, dense_dev, idx_dev, w_dev):
            ttnn.deallocate(t)
    logger.info(
        "| E_loc | T | active | rows | hybrid us | EP us | EP device ops | EP vs torch | hybrid vs torch | EP vs hybrid |"
    )
    logger.info("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        logger.info(
            f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.1f} | {r[5]:.1f} | {r[6]} | {r[7]:.6f} | {r[8]:.6f} | {r[9]:.6f} |"
        )
