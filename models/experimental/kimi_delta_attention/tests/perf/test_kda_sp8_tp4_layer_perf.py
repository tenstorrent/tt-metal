# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""E2E slowest-device profiler for the native Galaxy SP=8, TP=4 KDA layer.

This is intentionally separate from the LoudBox SP2×TP4 profiler.  Each
Galaxy span owns the same TP4/T640 local work, but this harness captures all
eight child meshes so the real four TP-rank affine trees and output CCLs are
measured together.
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
from models.experimental.kimi_delta_attention.tt.sp_layer import (
    SP8AffineTP4KimiDeltaAttention,
    SP8TP4KimiDeltaAttention,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.perf,
    pytest.mark.timeout(0),
    pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True),
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


def _synchronize_spans(layer: SP8TP4KimiDeltaAttention) -> None:
    for span_device in layer.span_devices:
        ttnn.synchronize_device(span_device)


def _release_outputs(outputs: tuple[ttnn.Tensor, ...]) -> None:
    for output in outputs:
        ttnn.deallocate(output)


def _profile_eager(
    layer: SP8TP4KimiDeltaAttention,
    span_inputs: tuple[ttnn.Tensor, ...],
    repetitions: int,
) -> None:
    outputs: list[tuple[ttnn.Tensor, ...]] = []
    signpost(header="sp8_tp4_start")
    for _ in range(repetitions):
        outputs.append(layer.forward(*span_inputs))
    _synchronize_spans(layer)
    signpost(header="sp8_tp4_stop")
    for layer_outputs in outputs:
        _release_outputs(layer_outputs)


def _profile_child_traces(
    layer: SP8TP4KimiDeltaAttention,
    span_inputs: tuple[ttnn.Tensor, ...],
    repetitions: int,
) -> None:
    """Capture one command stream per SP rank and replay them concurrently."""
    traces = tuple(ttnn.begin_trace_capture(device, cq_id=0) for device in layer.span_devices)
    outputs = layer.forward(*span_inputs)
    for device, trace_id in zip(layer.span_devices, traces, strict=True):
        ttnn.end_trace_capture(device, trace_id, cq_id=0)

    for device, trace_id in zip(layer.span_devices, traces, strict=True):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    _synchronize_spans(layer)

    signpost(header="sp8_tp4_start")
    for _ in range(repetitions):
        for device, trace_id in zip(layer.span_devices, traces, strict=True):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    _synchronize_spans(layer)
    signpost(header="sp8_tp4_stop")

    for device, trace_id in zip(layer.span_devices, traces, strict=True):
        ttnn.release_trace(device, trace_id)
    _release_outputs(outputs)


def test_kda_sp8_tp4_layer_device_perf(mesh_device: ttnn.MeshDevice) -> None:
    """Profile a production-shape local workload at global T=5120 by default."""
    sequence = int(os.getenv("PERF_SEQ", "5120"))
    if sequence % (8 * 128):
        raise ValueError(f"PERF_SEQ must give 128-token-aligned SP8 spans, got {sequence}")
    config = KDAConfig(
        hidden_size=2304,
        num_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        norm_eps=1e-5,
        chunk_size=32,
    )
    affine = os.getenv("PERF_SP8TP4_AFFINE", "1") == "1"
    layer_class = SP8AffineTP4KimiDeltaAttention if affine else SP8TP4KimiDeltaAttention
    layer = layer_class(mesh_device, config, random_weights(config))
    layer.reset_state(batch_size=1)
    layer.enable_trace_stable_state()

    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(9182)).to(
        torch.bfloat16
    )
    span = sequence // 8
    span_inputs = tuple(
        ttnn.from_torch(
            hidden[:, rank * span : (rank + 1) * span],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=span_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(span_device),
        )
        for rank, span_device in enumerate(layer.span_devices)
    )

    warm_outputs = layer.forward(*span_inputs)
    _synchronize_spans(layer)
    _release_outputs(warm_outputs)

    repetitions = int(os.getenv("PERF_REPS", "10"))
    if repetitions <= 0:
        raise ValueError(f"PERF_REPS must be positive, got {repetitions}")
    if os.getenv("PERF_TRACE", "0") == "1":
        if affine and os.getenv("KDA_SP8_TRACE_SCHEDULE", "0") != "1":
            raise ValueError("affine SP8TP4 trace requires KDA_SP8_TRACE_SCHEDULE=1")
        _profile_child_traces(layer, span_inputs, repetitions)
    else:
        _profile_eager(layer, span_inputs, repetitions)
