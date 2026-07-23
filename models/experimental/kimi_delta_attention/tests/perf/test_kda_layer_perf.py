# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-profiler harness for the target-shape single-device KDA layer."""

import os

import pytest
import torch
from tracy import signpost

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.tests.test_factory import random_weights
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True),
]


def test_kda_layer_device_perf(device: ttnn.Device) -> None:
    """Profile warm target-shape prefill; invoke through Tracy, not as a latency test."""
    sequence = int(os.getenv("PERF_SEQ", "640"))
    if sequence % 32:
        raise ValueError(f"PERF_SEQ must be divisible by 32, got {sequence}")
    config = KDAConfig(
        hidden_size=2304,
        num_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        norm_eps=1e-5,
        chunk_size=32,
    )
    hidden = torch.randn(
        1,
        sequence,
        config.hidden_size,
        generator=torch.Generator().manual_seed(607),
    ).to(torch.bfloat16)
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    layer = KimiDeltaAttention(device, config, random_weights(config))
    layer.reset_state(batch_size=1)
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    layer.set_external_state(layer.recurrent_state, layer.convolution_state)

    warm_output = layer.forward(hidden_tt, mode="chunk")
    ttnn.synchronize_device(device)
    ttnn.deallocate(warm_output)

    outputs: list[ttnn.Tensor] = []
    repetitions = int(os.getenv("PERF_REPS", "3"))
    signpost(header="start")
    for _ in range(repetitions):
        outputs.append(layer.forward(hidden_tt, mode="chunk"))
    ttnn.synchronize_device(device)
    signpost(header="stop")

    for output in outputs:
        ttnn.deallocate(output)
