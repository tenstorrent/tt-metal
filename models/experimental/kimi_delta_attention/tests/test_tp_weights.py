# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Eight-device whole-head KDA weight-layout tests."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.tests.test_factory import random_weights
from models.experimental.kimi_delta_attention.tt.weights import load_kda_weights

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True),
]


def _host_shards(tensor: ttnn.Tensor) -> list[torch.Tensor]:
    return [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]


def test_tp_weight_layout(mesh_device: ttnn.MeshDevice) -> None:
    config = KDAConfig(
        hidden_size=64,
        num_heads=8,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )
    state_dict = random_weights(config)
    weights = load_kda_weights(mesh_device, config, state_dict)

    assert weights.tensor_parallel_size == 8
    qkv_shards = _host_shards(weights.qkv_projection)
    auxiliary_shards = _host_shards(weights.auxiliary_projection)
    output_shards = _host_shards(weights.output_projection)
    tap_shards = _host_shards(weights.convolution_taps[0])

    for device_index in range(8):
        head_start = device_index * config.head_k_dim
        head_end = head_start + config.head_k_dim
        expected_qkv = torch.cat(
            (
                state_dict["q_proj.weight"][head_start:head_end],
                state_dict["k_proj.weight"][head_start:head_end],
                state_dict["v_proj.weight"][head_start:head_end],
            ),
            dim=0,
        ).T
        expected_auxiliary = torch.cat(
            (
                state_dict["f_a_proj.weight"],
                state_dict["g_a_proj.weight"],
                state_dict["b_proj.weight"][device_index : device_index + 1],
            ),
            dim=0,
        ).T
        expected_output = state_dict["o_proj.weight"][:, head_start:head_end].T
        expected_tap = torch.cat(
            (
                state_dict["q_conv1d.weight"][head_start:head_end, 0, 0],
                state_dict["k_conv1d.weight"][head_start:head_end, 0, 0],
                state_dict["v_conv1d.weight"][head_start:head_end, 0, 0],
            )
        ).reshape(1, 1, -1)

        torch.testing.assert_close(qkv_shards[device_index], expected_qkv.to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(
            auxiliary_shards[device_index], expected_auxiliary.to(torch.bfloat16), rtol=0, atol=0
        )
        torch.testing.assert_close(output_shards[device_index], expected_output.to(torch.bfloat16), rtol=0, atol=0)
        torch.testing.assert_close(tap_shards[device_index], expected_tap.to(torch.bfloat16), rtol=0, atol=0)
