# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-only checks of the compute configs that the vision encoder's modules hold (tt/vision/common.py and the modules
that build theirs with it), and of the RoPE theta and LayerNorm eps that the model takes from its config. No device:
the encoder is built with stand-ins for the mesh device and ttnn.from_torch. The device tests check the numerics these
settings give; this file pins the settings, so that a lower fidelity, approximate math or bf16 accumulation fails here
even where it would stay within a device test's bound.

The 1-card leg selects these tests by name (test_vision_), as they have no mesh id.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_compute_config.py
"""
from __future__ import annotations

import dataclasses

import torch

import ttnn
from models.demos.minimax_m3.reference.vision import MiniMaxM3VisionEncoder
from models.demos.minimax_m3.tt.vision.config import MiniMaxM3VisionConfig
from models.demos.minimax_m3.tt.vision.layernorm import M3VLLayerNorm
from models.demos.minimax_m3.tt.vision.model import M3VLVisionModel
from models.demos.minimax_m3.tt.vision.rope import rope_cos_sin_padded

from .vision_helpers import grid_device

# 2 narrow layers, with 2 heads of the checkpoint's head size, 80.
CONFIG = MiniMaxM3VisionConfig(
    hidden_size=160,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=2,
    projector_hidden_size=32,
    text_hidden_size=32,
)
# (math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc) of the weight matmuls, whose partial sums
# accumulate in L1, and of LayerNorm, RoPE and SDPA.
WEIGHT_MATMUL_FIELDS = (ttnn.MathFidelity.HiFi4, False, True, True)
OTHER_OP_FIELDS = (ttnn.MathFidelity.HiFi4, False, True, False)


def _fields(kernel_config) -> tuple:
    return (
        kernel_config.math_fidelity,
        kernel_config.math_approx_mode,
        kernel_config.fp32_dest_acc_en,
        kernel_config.packer_l1_acc,
    )


def _build(monkeypatch, config: MiniMaxM3VisionConfig = CONFIG) -> M3VLVisionModel:
    """M3VLVisionModel.from_torch of a reference with config on a stand-in 12x10 device; ttnn.from_torch keeps the
    host tensor."""
    monkeypatch.setattr(ttnn, "from_torch", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh_device: None)
    return M3VLVisionModel.from_torch(grid_device(12, 10), MiniMaxM3VisionEncoder(config))


def test_vision_layernorm_compute_config(monkeypatch):
    """Every LayerNorm the encoder builds, pre_layrnorm and the two of each block (there is none after the blocks and
    none in the projector), holds a compute config from __init__, HiFi4 with exact math, fp32 accumulation and no L1
    accumulation, and gives that same config to ttnn.layer_norm on every call. ttnn.layer_norm is a stand-in too."""
    built, calls = [], []
    init = M3VLLayerNorm.__init__

    def record_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(M3VLLayerNorm, "__init__", record_init)
    monkeypatch.setattr(ttnn, "layer_norm", lambda x, **kwargs: calls.append((x, kwargs)))
    model = _build(monkeypatch)

    norms = [model.pre_layrnorm] + [norm for block in model.blocks for norm in (block.layer_norm1, block.layer_norm2)]
    assert len(built) == len(norms), f"{len(built)} LayerNorms built, {len(norms)} in pre_layrnorm and the blocks"
    assert set(map(id, built)) == set(map(id, norms)), "pre_layrnorm and the blocks do not hold the LayerNorms built"
    for norm in norms:
        kernel_config = norm.compute_kernel_config
        assert _fields(kernel_config) == OTHER_OP_FIELDS, _fields(kernel_config)
        for x in ("first input", "second input"):
            norm(x)
            x_in, kwargs = calls.pop()
            assert x_in == x and kwargs["compute_kernel_config"] is kernel_config


def test_vision_compute_configs(monkeypatch):
    """Every weight matmul of the encoder, the patch embed, the attention's QKV and output projection (one config) and
    the MLPs of the blocks and the projector, holds HiFi4 with exact math, fp32 accumulation and L1 accumulation of
    its partial sums. The attention's RoPE and SDPA hold the same without L1 accumulation."""
    model = _build(monkeypatch)
    weight_matmuls = {
        "patch_embed": model.patch_embed.compute_kernel_config,
        "projector.mlp": model.projector.mlp.compute_kernel_config,
        "projector.merge_mlp": model.projector.merge_mlp.compute_kernel_config,
    }
    other_ops = {}
    for i, block in enumerate(model.blocks):
        weight_matmuls[f"blocks.{i}.attention"] = block.attention.compute_kernel_config
        weight_matmuls[f"blocks.{i}.mlp"] = block.mlp.compute_kernel_config
        other_ops[f"blocks.{i}.attention.rope"] = block.attention.rope_compute_kernel_config
        other_ops[f"blocks.{i}.attention.sdpa"] = block.attention.sdpa_compute_kernel_config
    for name, kernel_config in weight_matmuls.items():
        assert _fields(kernel_config) == WEIGHT_MATMUL_FIELDS, (name, _fields(kernel_config))
    for name, kernel_config in other_ops.items():
        assert _fields(kernel_config) == OTHER_OP_FIELDS, (name, _fields(kernel_config))


def test_vision_model_takes_rope_theta_and_eps(monkeypatch):
    """The model stages the RoPE tables with its config's rope_theta, and each of its LayerNorms takes the config's
    layer_norm_eps: a config with theta 5000 and eps 1e-6, unlike the checkpoint's 10000 and 1e-5, gets both."""
    config = dataclasses.replace(CONFIG, rope_theta=5000.0, layer_norm_eps=1e-6)
    model = _build(monkeypatch, config)
    norms = [model.pre_layrnorm] + [norm for block in model.blocks for norm in (block.layer_norm1, block.layer_norm2)]
    assert [norm.eps for norm in norms] == [1e-6] * len(norms)

    grid = torch.tensor([[1, 4, 6]])
    tables = model._stage_rope(grid)
    for theta, same in ((5000.0, True), (10000.0, False)):
        expected = rope_cos_sin_padded(grid, config.head_dim, model.padded_head_dim, theta, config.spatial_merge_size)
        for table, table_ref in zip(tables, expected):
            assert torch.equal(table, table_ref.to(torch.bfloat16)[None, None]) == same, f"theta {theta}"
