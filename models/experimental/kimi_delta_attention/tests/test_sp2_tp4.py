# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness test for the LoudBox SP=2, TP=4 KDA path."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.reference import kda_forward_reference
from models.experimental.kimi_delta_attention.tests.test_factory import random_weights
from models.experimental.kimi_delta_attention.tt.sp_layer import SP2TP4KimiDeltaAttention


def _host_shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_2D}],
        indirect=True,
    ),
]


def test_sp2_tp4_layer_pcc(mesh_device: ttnn.MeshDevice) -> None:
    """The second span must consume the recurrent and short-conv carry on fabric."""
    config = KDAConfig(
        hidden_size=256,
        num_heads=8,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        norm_eps=1e-5,
        chunk_size=32,
    )
    state_dict = random_weights(config)
    sequence = 64
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(6081)).to(
        torch.bfloat16
    )
    golden_output, golden_state = kda_forward_reference(hidden, state_dict, config)

    layer = SP2TP4KimiDeltaAttention(mesh_device, config, state_dict)
    layer.reset_state(batch_size=1)
    span_inputs = []
    for span, span_device in enumerate(layer.span_devices):
        span_inputs.append(
            ttnn.from_torch(
                hidden[:, span * (sequence // 2) : (span + 1) * (sequence // 2)],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=span_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(span_device),
            )
        )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        first_output, second_output = layer.forward(*span_inputs)

    actual_output = torch.cat(
        (
            ttnn.to_torch(first_output, mesh_composer=ttnn.ConcatMeshToTensor(layer.span_devices[0], dim=-1)),
            ttnn.to_torch(second_output, mesh_composer=ttnn.ConcatMeshToTensor(layer.span_devices[1], dim=-1)),
        ),
        dim=1,
    )
    assert layer.second_layer.recurrent_state is not None
    assert layer.second_layer.convolution_state is not None
    actual_recurrent = torch.cat(_host_shards(layer.second_layer.recurrent_state), dim=1)
    convolution_shards = _host_shards(layer.second_layer.convolution_state)
    local_key_width = config.head_k_dim * config.num_heads // 4
    local_value_width = config.head_v_dim * config.num_heads // 4
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
        assert passed, f"SP=2 TP=4 {name} PCC {pcc:.6f} < 0.98"
