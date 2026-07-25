# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-profiler control for a single TP=4 KDA span on LoudBox.

Use ``PERF_SEQ=640`` for the production-rank local-work control and
``PERF_SEQ=1280`` for the equal-global-sequence SP=2/TP=4 comparison.
"""

from __future__ import annotations

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
        [
            {
                "l1_small_size": 24576,
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "trace_region_size": 256 * 1024 * 1024,
            }
        ],
        indirect=True,
    ),
]


def _profile_eager(
    mesh_device: ttnn.MeshDevice,
    layer: KimiDeltaAttention,
    hidden: ttnn.Tensor,
    repetitions: int,
) -> None:
    outputs: list[ttnn.Tensor] = []
    signpost(header="tp4_start")
    for _ in range(repetitions):
        outputs.append(layer.forward(hidden, mode="chunk"))
    ttnn.synchronize_device(mesh_device)
    signpost(header="tp4_stop")
    for output in outputs:
        ttnn.deallocate(output)


def _profile_trace(
    mesh_device: ttnn.MeshDevice,
    layer: KimiDeltaAttention,
    hidden: ttnn.Tensor,
    repetitions: int,
) -> None:
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = layer.forward(hidden, mode="chunk")
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    signpost(header="tp4_start")
    for _ in range(repetitions):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    signpost(header="tp4_stop")

    ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(output)


def test_kda_tp4_layer_device_perf(mesh_device: ttnn.MeshDevice) -> None:
    """Profile one TP=4 KDA span using the same model as the SP experiment."""
    sequence = int(os.getenv("PERF_SEQ", "1280"))
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
    # LoudBox fabric initialization requires the physical 1x8 board. Profile
    # only its first logical TP=4 group after opening that parent mesh, exactly
    # as the SP=2 harness does for each child trace.
    tp4_device = mesh_device.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=tp4_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(tp4_device),
    )
    layer = KimiDeltaAttention(tp4_device, config, random_weights(config), tt_ccl=TT_CCL(tp4_device))
    layer.reset_state(batch_size=1)
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    layer.set_external_state(layer.recurrent_state, layer.convolution_state)

    warm_output = layer.forward(hidden_tt, mode="chunk")
    ttnn.synchronize_device(tp4_device)
    ttnn.deallocate(warm_output)

    repetitions = int(os.getenv("PERF_REPS", "3"))
    if os.getenv("PERF_TRACE", "0") == "1":
        _profile_trace(tp4_device, layer, hidden_tt, repetitions)
    else:
        _profile_eager(tp4_device, layer, hidden_tt, repetitions)
