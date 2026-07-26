# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-profiler harness for the LoudBox SP=2, TP=4 KDA layer.

The measured sequence length is the *global* sequence length.  Each four-chip
TP group receives one half, so ``PERF_SEQ=5120`` gives each chip the same
20,480 head-token workload as the TP=8 control at global length 5,120.
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
from models.experimental.kimi_delta_attention.tt.sp_layer import SP2TP4KimiDeltaAttention

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


def _release_outputs(*outputs: ttnn.Tensor) -> None:
    """Release ordinary outputs, but preserve MRS-owned trace outputs.

    ``KDA_MRS_DIRECT_OUTPUT=1`` returns the fused MRS persistent buffer
    directly.  It is intentionally retained for the layer lifetime so trace
    replay cannot observe a freed address; the normal path remains unchanged.
    """
    if os.getenv("KDA_MRS_DIRECT_OUTPUT", "0") == "1":
        return
    for output in outputs:
        ttnn.deallocate(output)


def _synchronize_spans(layer: SP2TP4KimiDeltaAttention) -> None:
    """Wait on the queues that own the SP=2 x TP=4 work.

    The parent 1x8 mesh owns the child submeshes but does not execute their
    KDA operations. Synchronizing it would retain its command queue while the
    pytest fixture closes the children.
    """
    for span_device in layer.span_devices:
        ttnn.synchronize_device(span_device)


def _profile_eager(
    layer: SP2TP4KimiDeltaAttention,
    first_span: ttnn.Tensor,
    second_span: ttnn.Tensor,
    repetitions: int,
) -> None:
    outputs: list[tuple[ttnn.Tensor, ttnn.Tensor]] = []
    signpost(header="sp2_tp4_start")
    for _ in range(repetitions):
        outputs.append(layer.forward(first_span, second_span, mode="chunk"))
    _synchronize_spans(layer)
    signpost(header="sp2_tp4_stop")
    for first_output, second_output in outputs:
        _release_outputs(first_output, second_output)


def _profile_trace(
    mesh_device: ttnn.MeshDevice,
    layer: SP2TP4KimiDeltaAttention,
    first_span: ttnn.Tensor,
    second_span: ttnn.Tensor,
    repetitions: int,
) -> None:
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    first_output, second_output = layer.forward(first_span, second_span, mode="chunk")
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)

    signpost(header="sp2_tp4_start")
    for _ in range(repetitions):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    signpost(header="sp2_tp4_stop")

    ttnn.release_trace(mesh_device, trace_id)
    _release_outputs(first_output, second_output)


def _profile_child_traces(
    layer: SP2TP4KimiDeltaAttention,
    first_span: ttnn.Tensor,
    second_span: ttnn.Tensor,
    repetitions: int,
) -> None:
    """Replay independently captured TP=4 child-mesh command streams.

    The parent 1x8 mesh owns no KDA work. Capturing it therefore loses the
    child queues at teardown and can add host scheduling between the two SP
    spans. Each child trace owns one endpoint of the socket handoff.
    """
    first_device, second_device = layer.span_devices
    first_trace = ttnn.begin_trace_capture(first_device, cq_id=0)
    second_trace = ttnn.begin_trace_capture(second_device, cq_id=0)
    first_output, second_output = layer.forward(first_span, second_span, mode="chunk")
    ttnn.end_trace_capture(first_device, first_trace, cq_id=0)
    ttnn.end_trace_capture(second_device, second_trace, cq_id=0)

    ttnn.execute_trace(first_device, first_trace, cq_id=0, blocking=False)
    ttnn.execute_trace(second_device, second_trace, cq_id=0, blocking=False)
    _synchronize_spans(layer)

    signpost(header="sp2_tp4_start")
    for _ in range(repetitions):
        ttnn.execute_trace(first_device, first_trace, cq_id=0, blocking=False)
        ttnn.execute_trace(second_device, second_trace, cq_id=0, blocking=False)
    _synchronize_spans(layer)
    signpost(header="sp2_tp4_stop")

    ttnn.release_trace(first_device, first_trace)
    ttnn.release_trace(second_device, second_trace)
    _release_outputs(first_output, second_output)


def test_kda_sp2_tp4_layer_device_perf(mesh_device: ttnn.MeshDevice) -> None:
    """Profile SP=2, TP=4 at the TP=8-comparable global sequence target."""
    sequence = int(os.getenv("PERF_SEQ", "5120"))
    if sequence % 64:
        raise ValueError(f"PERF_SEQ must be divisible by 64 for two aligned spans, got {sequence}")
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
    layer = SP2TP4KimiDeltaAttention(mesh_device, config, random_weights(config))
    layer.reset_state(batch_size=1)
    layer.enable_trace_stable_state()
    span = sequence // 2
    first_span = ttnn.from_torch(
        hidden[:, :span],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=layer.span_devices[0],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(layer.span_devices[0]),
    )
    second_span = ttnn.from_torch(
        hidden[:, span:],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=layer.span_devices[1],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(layer.span_devices[1]),
    )

    warm_first, warm_second = layer.forward(first_span, second_span, mode="chunk")
    _synchronize_spans(layer)
    _release_outputs(warm_first, warm_second)

    repetitions = int(os.getenv("PERF_REPS", "3"))
    if os.getenv("PERF_CHILD_TRACE", "0") == "1":
        _profile_child_traces(layer, first_span, second_span, repetitions)
    elif os.getenv("PERF_TRACE", "0") == "1":
        _profile_trace(mesh_device, layer, first_span, second_span, repetitions)
    else:
        _profile_eager(layer, first_span, second_span, repetitions)
