# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase 2: the MoE layer on a (2, 4) chip block vs the HF MoE module, real layer-4 weights, realistic inputs; each
row carries its own tokens (decode shape 2 users/row and a 32-token block/row). GLM53_MOE_LAYOUT=hybrid (default:
experts over the mesh columns, intermediate over the rows) or ep (every chip 36 full-width experts, the tokens
exchanged with all_to_all_dispatch / combine along the rows; the fabric defaults to FABRIC_1D for it).

    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=.../p150_x8_mesh_graph_descriptor.textproto \
    MESH_DEVICE=P150x8 [GLM53_MOE_LAYOUT=ep] pytest models/demos/glm53_flash/tests/test_mesh_moe.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tests.test_moe import LAYER, realistic_mlp_input, weights  # noqa: F401 (fixture)
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params
from models.demos.glm53_flash.tt.moe import TtMoE

ROWS, COLS = 2, 4


def l1_high_water(mesh_device) -> int:
    """The address of a fresh one-tile L1 probe (freed again): the L1 allocator is top-down, so a lower address
    after a forward means the module left persistent L1 buffers behind (lazily built constants). A per-layer L1
    constant is a model-level bug even when one layer passes: on the 45-layer 8x4 EP e2e the per-layer zeroed slot
    constant of the EP combine shrank the L1 left for the static CBs until a later program clashed
    (doc/mesh/README.md, EP32 L1 clash)."""
    probe = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device, ttnn.L1_MEMORY_CONFIG
    )
    addr = probe.buffer_address()
    ttnn.deallocate(probe)
    return addr


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(ROWS, COLS)], indirect=True)
@pytest.mark.parametrize("rows_T", [2, 32])
def test_mesh_moe_vs_hf(mesh_device, weights, rows_T):
    snapshot, text, raw, hf, sd = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    mlp = {k[len("mlp.") :]: v for k, v in raw.items() if "experts." not in k or "shared_experts" in k}
    mlp["experts.gate_up_proj"] = sd["experts.gate_up_proj"]
    mlp["experts.down_proj"] = sd["experts.down_proj"]
    mesh = GLMMesh.from_device(mesh_device)
    t0 = time.perf_counter()
    tt = TtMoE(mesh_device, cfg, mlp, mesh=mesh)
    layout = os.environ.get("GLM53_MOE_LAYOUT", "hybrid")
    logger.info(
        f"{layout} MoE layer {LAYER} on {ROWS}x{COLS} (GLM53_FABRIC={os.environ.get('GLM53_FABRIC', 'default')}): "
        f"experts built + uploaded in {time.perf_counter() - t0:.1f} s"
    )
    x = realistic_mlp_input(snapshot, text, raw, ROWS * 32).reshape(1, 1, ROWS * 32, -1)  # row r: rows [32r, 32r+T)
    xs = torch.cat([x[:, :, 32 * r : 32 * r + rows_T] for r in range(ROWS)], dim=2)  # [1,1,ROWS*T,4096]
    with torch.no_grad():
        y_hf = hf(xs.reshape(1, ROWS * rows_T, -1)).reshape(1, 1, ROWS * rows_T, -1)
    tt_x = mesh.upload(xs.bfloat16(), ttnn.bfloat16, rows=2)
    l1_before = l1_high_water(mesh_device)
    for it in range(2):
        logger.info(f"{layout} MoE forward iter {it} (T={rows_T}/row) starting")
        t0 = time.perf_counter()
        y = tt(tt_x)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"{layout} MoE forward iter {it} done in {time.perf_counter() - t0:.2f} s")
        shards = [ttnn.to_torch(d).float() for d in mesh.device_tensors(y)]
        worst = 1.0
        for r in range(ROWS):
            for c in range(COLS):
                p = pcc(y_hf[:, :, r * rows_T : (r + 1) * rows_T], shards[r * COLS + c][:, :, :rows_T])
                worst = min(worst, p)
                if c:
                    assert torch.equal(shards[r * COLS + c], shards[r * COLS]), (r, c)
        logger.info(
            f"{layout} MoE layer {LAYER} on {ROWS}x{COLS}, T={rows_T}/row, iter {it}: worst pcc vs HF {worst:.6f}"
        )
        assert worst > 0.995
        ttnn.deallocate(y)
        l1_after = l1_high_water(mesh_device)
        logger.info(
            f"{layout} MoE layer: L1 high-water mark {l1_before} before the forward, {l1_after} after iter {it}"
        )
        assert l1_after == l1_before, (
            f"the {layout} MoE layer left {l1_before - l1_after} B of persistent L1 buffers behind (lazily built L1 "
            "constants: they pile up per layer and clash with later programs' static CBs on the 45-layer model)"
        )
