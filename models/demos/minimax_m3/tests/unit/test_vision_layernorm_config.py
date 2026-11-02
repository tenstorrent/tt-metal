# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-only check of the compute config of every LayerNorm the vision encoder builds (tt/vision/layernorm.py). No
device: the encoder is built with stand-ins for the mesh device and ttnn.from_torch. test_vision_layernorm_vs_ref.py
and the block and tower tests check the LayerNorm's numerics on the device.

The 1-card leg selects this test by name (test_vision_), as it has no mesh id.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_layernorm_config.py
"""
from __future__ import annotations

from types import SimpleNamespace

import ttnn
from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionConfig, MiniMaxM3VisionEncoder
from models.demos.minimax_m3.tt.vision.layernorm import M3VLLayerNorm
from models.demos.minimax_m3.tt.vision.model import M3VLVisionModel


def test_vision_layernorm_compute_config(monkeypatch):
    """Every LayerNorm the encoder builds, pre_layrnorm and the two of each block (there is none after the blocks and
    none in the projector), holds a compute config from __init__, HiFi4 with exact math, fp32 accumulation and no L1
    accumulation, and gives that same config to ttnn.layer_norm on every call. The mesh device, ttnn.from_torch and
    ttnn.layer_norm are stand-ins, and the encoder has 2 narrow layers."""
    built, calls = [], []
    init = M3VLLayerNorm.__init__

    def record_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(M3VLLayerNorm, "__init__", record_init)
    monkeypatch.setattr(ttnn, "from_torch", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh_device: None)
    monkeypatch.setattr(ttnn, "layer_norm", lambda x, **kwargs: calls.append((x, kwargs)))
    config = MiniMaxM3VisionConfig(
        hidden_size=160,  # 2 heads of the checkpoint's head size, 80
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        projector_hidden_size=32,
        text_hidden_size=32,
    )
    mesh_device = SimpleNamespace(compute_with_storage_grid_size=lambda: ttnn.CoreCoord(12, 10))
    model = M3VLVisionModel.from_torch(mesh_device, MiniMaxM3VisionEncoder(config))

    norms = [model.pre_layrnorm] + [norm for block in model.blocks for norm in (block.layer_norm1, block.layer_norm2)]
    assert len(built) == len(norms), f"{len(built)} LayerNorms built, {len(norms)} in pre_layrnorm and the blocks"
    assert set(map(id, built)) == set(map(id, norms)), "pre_layrnorm and the blocks do not hold the LayerNorms built"
    for norm in norms:
        kernel_config = norm.compute_kernel_config
        fields = (
            kernel_config.math_fidelity,
            kernel_config.math_approx_mode,
            kernel_config.fp32_dest_acc_en,
            kernel_config.packer_l1_acc,
        )
        assert fields == (ttnn.MathFidelity.HiFi4, False, True, False), fields
        for x in ("first input", "second input"):
            norm(x)
            x_in, kwargs = calls.pop()
            assert x_in == x and kwargs["compute_kernel_config"] is kernel_config
