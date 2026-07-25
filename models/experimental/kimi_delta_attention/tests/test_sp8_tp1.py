# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end seven-boundary KDA sequence-parallel protocol probe."""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.reference import kda_forward_reference
from models.experimental.kimi_delta_attention.tests.test_factory import random_weights
from models.experimental.kimi_delta_attention.tt.sp_layer import SP8TP1KimiDeltaAttention


pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_2D}],
        indirect=True,
    ),
]


def _probe_shape() -> tuple[KDAConfig, int]:
    if os.getenv("KDA_SP8_TARGET_SHAPE", "0") == "1":
        return (
            KDAConfig(
                hidden_size=2304,
                num_heads=32,
                head_k_dim=128,
                head_v_dim=128,
                conv_kernel_size=4,
                norm_eps=1e-5,
                chunk_size=32,
            ),
            int(os.getenv("KDA_SP8_TEST_SEQ", "5120")),
        )
    return (
        KDAConfig(
            hidden_size=256,
            num_heads=8,
            head_k_dim=128,
            head_v_dim=128,
            conv_kernel_size=4,
            norm_eps=1e-5,
            chunk_size=32,
        ),
        256,
    )


def test_sp8_tp1_layer_pcc(mesh_device: ttnn.MeshDevice) -> None:
    """All seven fabric boundaries preserve output and both causal caches."""
    config, sequence = _probe_shape()
    if sequence % (8 * config.chunk_size):
        raise ValueError(f"KDA_SP8_TEST_SEQ must be divisible by {8 * config.chunk_size}, got {sequence}")
    state_dict = random_weights(config)
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(8117)).to(
        torch.bfloat16
    )
    golden_output, golden_state = kda_forward_reference(hidden, state_dict, config)

    layer = SP8TP1KimiDeltaAttention(mesh_device, config, state_dict)
    layer.reset_state(batch_size=1)
    span = sequence // 8
    span_inputs = tuple(
        ttnn.from_torch(
            hidden[:, span_index * span : (span_index + 1) * span],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=span_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(span_device),
        )
        for span_index, span_device in enumerate(layer.span_devices)
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        outputs = layer.forward(*span_inputs)

    actual_output = torch.cat(
        [
            ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(span_device, dim=-1))
            for output, span_device in zip(outputs, layer.span_devices, strict=True)
        ],
        dim=1,
    )
    final_layer = layer.layers[-1]
    assert final_layer.recurrent_state is not None
    assert final_layer.convolution_state is not None
    actual_recurrent = ttnn.to_torch(final_layer.recurrent_state)
    actual_convolution = ttnn.to_torch(final_layer.convolution_state)
    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    )
    for name, golden, actual in (
        ("output", golden_output, actual_output),
        ("recurrent state", golden_state.recurrent, actual_recurrent),
        ("convolution state", golden_convolution, actual_convolution),
    ):
        passed, pcc = comp_pcc(golden, actual, pcc=0.98)
        assert passed, f"SP=8 TP=1 {name} PCC {pcc:.6f} < 0.98"

    for boundary in range(span, sequence, span):
        passed, pcc = comp_pcc(
            golden_output[:, boundary : boundary + 1], actual_output[:, boundary : boundary + 1], pcc=0.98
        )
        assert passed, f"SP=8 TP=1 boundary {boundary} PCC {pcc:.6f} < 0.98"
