# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase 2: the hybrid MoE (experts over the mesh columns, intermediate over the rows) on a (2, 4)
chip block vs the HF MoE module, real layer-3 weights, realistic inputs; each row carries its own
tokens (decode shape 2 users/row and a 32-token block/row).

    TT_VISIBLE_DEVICES=27,26,31,30,23,22,19,18 TT_MESH_GRAPH_DESC_PATH=.../p150_x8_mesh_graph_descriptor.textproto \
    MESH_DEVICE=P150x8 pytest models/demos/glm53_flash/tests/test_mesh_moe.py -s
"""

from __future__ import annotations

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
    tt = TtMoE(mesh_device, cfg, mlp, mesh=mesh)
    x = realistic_mlp_input(snapshot, text, raw, ROWS * 32).reshape(1, 1, ROWS * 32, -1)  # row r: rows [32r, 32r+T)
    xs = torch.cat([x[:, :, 32 * r : 32 * r + rows_T] for r in range(ROWS)], dim=2)  # [1,1,ROWS*T,4096]
    with torch.no_grad():
        y_hf = hf(xs.reshape(1, ROWS * rows_T, -1)).reshape(1, 1, ROWS * rows_T, -1)
    tt_x = mesh.upload(xs.bfloat16(), ttnn.bfloat16, rows=2)
    for it in range(2):
        y = tt(tt_x)
        shards = [ttnn.to_torch(d).float() for d in mesh.device_tensors(y)]
        worst = 1.0
        for r in range(ROWS):
            for c in range(COLS):
                p = pcc(y_hf[:, :, r * rows_T : (r + 1) * rows_T], shards[r * COLS + c][:, :, :rows_T])
                worst = min(worst, p)
                if c:
                    assert torch.equal(shards[r * COLS + c], shards[r * COLS]), (r, c)
        logger.info(
            f"hybrid MoE layer {LAYER} on {ROWS}x{COLS}, T={rows_T}/row, iter {it}: worst pcc vs HF {worst:.6f}"
        )
        assert worst > 0.995
