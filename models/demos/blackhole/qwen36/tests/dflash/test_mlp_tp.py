# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M5: the drafter's SwiGLU MLP vs ``Qwen3MLP``, real weights.

Input is the fixture's real ``noise_embedding`` pushed through the real
``post_attention_layernorm`` -- i.e. an activation with the scale and distribution the MLP
actually sees in layer 0, rather than random values. (The exact layer-0 MLP input is
``post_attention_layernorm(x + attn_out)``, which needs attention; this stands in for it at
the right magnitude and is graded against the same tensor on both sides, so it measures the
MLP alone.)

Run:
    MESH_DEVICE=T3K pytest models/demos/blackhole/qwen36/tests/dflash/test_mlp_tp.py -v -s
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.dflash.conftest import load_fixture
from models.demos.blackhole.qwen36.tests.test_factory import get_pcc_threshold, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.dflash.ccl import ccl_topology
from models.demos.blackhole.qwen36.tt.dflash.mlp import DFlashMLP
from models.demos.blackhole.qwen36.tt.dflash.weights import load_layer_weights, read_state_dict
from models.tt_transformers.tt.ccl import TT_CCL


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("layer_idx", [0, 4], ids=["layer0", "layer4"])
def test_mlp_tp(mesh_device, layer_idx, reset_seeds, ensure_gc, request, drafter_cfg):
    cfg = drafter_cfg
    fx = load_fixture(512)

    keys = [
        f"layers.{layer_idx}.mlp.gate_proj.weight",
        f"layers.{layer_idx}.mlp.up_proj.weight",
        f"layers.{layer_idx}.mlp.down_proj.weight",
        f"layers.{layer_idx}.post_attention_layernorm.weight",
    ]
    sd = read_state_dict(keys=keys)

    from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm

    # Realistic input: real embeddings through the real pre-MLP norm.
    pre_norm = Qwen3RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    with torch.no_grad():
        pre_norm.weight.copy_(sd[keys[3]])
    x = pre_norm(fx["noise_embedding"].float())  # [1, 16, 5120]

    # ---- oracle: stock Qwen3MLP with the real weights --------------------------------
    class _Cfg:
        hidden_size = cfg.hidden_size
        intermediate_size = cfg.intermediate_size
        hidden_act = "silu"
        mlp_bias = False

    ref = Qwen3MLP(_Cfg())
    with torch.no_grad():
        ref.gate_proj.weight.copy_(sd[keys[0]])
        ref.up_proj.weight.copy_(sd[keys[1]])
        ref.down_proj.weight.copy_(sd[keys[2]])
    expected = ref(x)

    # ---- device ----------------------------------------------------------------------
    weights = load_layer_weights(mesh_device, cfg, layer_idx, state_dict=read_state_dict(keys=_layer_keys(layer_idx)))
    tt_ccl = TT_CCL(mesh_device)
    mlp = DFlashMLP(mesh_device, cfg, weights, tt_ccl, topology=ccl_topology(mesh_device))

    tt_x = ttnn.from_torch(
        x.reshape(1, 1, x.shape[1], cfg.hidden_size).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = mlp.forward(tt_x)

    tp = mesh_device.get_num_devices()
    stacked = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    actual = stacked[:1].reshape(1, x.shape[1], cfg.hidden_size).float()
    spread = max((stacked[d : d + 1].float() - stacked[:1].float()).abs().max().item() for d in range(1, tp))
    logger.info(f"layer {layer_idx} replication spread: {spread:.3e}")

    passing, pcc = comp_pcc(expected, actual, get_pcc_threshold(request))
    logger.info(f"layer {layer_idx} MLP PCC {pcc}")
    assert passing, f"layer {layer_idx} MLP PCC {pcc}"


def _layer_keys(layer_idx: int):
    from models.demos.blackhole.qwen36.tt.dflash.weights import layer_keys

    return layer_keys(layer_idx)
