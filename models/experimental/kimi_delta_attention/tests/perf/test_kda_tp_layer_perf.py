# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-profiler harness for the target-shape TP=8 KDA layer."""

import os

import pytest
import torch
from tracy import signpost

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.tests.test_factory import random_weights
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention
from models.tt_transformers.tt.ccl import TT_CCL

pytestmark = [
    run_for_blackhole(),
    pytest.mark.perf,
    pytest.mark.timeout(0),
    pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
        indirect=True,
    ),
]


def test_kda_tp_layer_device_perf(mesh_device: ttnn.MeshDevice) -> None:
    """Profile warm target-shape TP=8 prefill; invoke through Tracy for attribution."""
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
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(1607)).to(
        torch.bfloat16
    )
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    layer = KimiDeltaAttention(mesh_device, config, random_weights(config), tt_ccl=TT_CCL(mesh_device))
    layer.reset_state(batch_size=1)
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    layer.set_external_state(layer.recurrent_state, layer.convolution_state)

    warm_output = layer.forward(hidden_tt, mode="chunk")
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm_output)

    outputs: list[ttnn.Tensor] = []
    repetitions = int(os.getenv("PERF_REPS", "3"))
    signpost(header="start")
    for _ in range(repetitions):
        outputs.append(layer.forward(hidden_tt, mode="chunk"))
    ttnn.synchronize_device(mesh_device)
    signpost(header="stop")

    for output in outputs:
        ttnn.deallocate(output)
