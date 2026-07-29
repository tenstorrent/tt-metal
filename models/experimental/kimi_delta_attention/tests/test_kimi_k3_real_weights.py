# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 layer-1 KDA PCC with the pinned Hugging Face weights."""

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.checkpoint import load_kda_layer_state_dict
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.kimi_k3_config import (
    KimiK3Config,
    kimi_k3_kda_config,
    kimi_k3_program_config,
)
from models.experimental.kimi_delta_attention.reference import kda_forward_reference
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention
from models.tt_transformers.tt.ccl import TT_CCL

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
        indirect=True,
    ),
]


def _checkpoint_dir() -> Path:
    value = os.getenv("KIMI_K3_CKPT")
    if value is None:
        pytest.skip("set KIMI_K3_CKPT to the pinned Kimi-K3 checkpoint subset")
    return Path(value)


def _host_shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def test_kimi_k3_layer_1_real_weights_pcc(mesh_device: ttnn.MeshDevice) -> None:
    checkpoint_dir = _checkpoint_dir()
    downloaded_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    config = kimi_k3_kda_config()
    assert KDAConfig.from_model_config(downloaded_config) == config

    state_dict = load_kda_layer_state_dict(checkpoint_dir, KimiK3Config.FIRST_KDA_LAYER, config)
    hidden = torch.randn(
        1,
        32,
        config.hidden_size,
        generator=torch.Generator().manual_seed(1607),
        dtype=torch.bfloat16,
    )
    golden_output, golden_state = kda_forward_reference(hidden, state_dict, config)

    tensor_cache_path = checkpoint_dir / "ttnn_cache" / "layer_1"
    tensor_cache_path.mkdir(parents=True, exist_ok=True)
    layer = KimiDeltaAttention(
        mesh_device,
        config,
        state_dict,
        tensor_cache_path=tensor_cache_path,
        tt_ccl=TT_CCL(mesh_device),
        program_config=kimi_k3_program_config(),
    )
    layer.reset_state(batch_size=1)
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output = layer.forward(hidden_tt)

    actual_output = ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    actual_recurrent = torch.cat(_host_shards(layer.recurrent_state), dim=1)
    convolution_shards = _host_shards(layer.convolution_state)
    local_heads = config.num_heads // 8
    local_key_width = local_heads * config.head_k_dim
    local_value_width = local_heads * config.head_v_dim
    actual_convolution = torch.cat(
        (
            torch.cat([shard[..., :local_key_width] for shard in convolution_shards], dim=-1),
            torch.cat([shard[..., local_key_width : 2 * local_key_width] for shard in convolution_shards], dim=-1),
            torch.cat(
                [
                    shard[..., 2 * local_key_width : 2 * local_key_width + local_value_width]
                    for shard in convolution_shards
                ],
                dim=-1,
            ),
        ),
        dim=-1,
    )
    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    )

    for name, golden, actual in (
        ("output", golden_output, actual_output),
        ("recurrent state", golden_state.recurrent, actual_recurrent),
        ("convolution state", golden_convolution, actual_convolution),
    ):
        passed, pcc = comp_pcc(golden, actual, pcc=0.98)
        print(f"Kimi-K3 layer 1 {name}: PCC={pcc:.6f}")
        assert passed, f"Kimi-K3 layer 1 {name} PCC {pcc:.6f} < 0.98"
