# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Galaxy micro-gates of the SP-sharded KDA state path (synthetic carry, no checkpoint).

Around its forward, a K3 KDA layer imports its slot's carry (``KdaStates.import_layer``: the recurrent and conv-tail
slab -> staging copies, then their SP all-gathers into the 24-head carry) and exports the new one
(``KdaStates.export_layer``: two copies into the slab, each selecting the chip's heads on device). Each chain is
captured and timed by ``_trace_samples_ms``: host wall time per call over back-to-back trace replays through device
synchronization, the median over samples. A chain alone pays the cross-chip start skew of its gathers, which inside
the layer the TP input all-gather absorbs, so these references are not in-layer advances.

Two layouts: the 8x4 Galaxy (SP8 on axis 0, 3 heads stored per chip) and one 4x4 stage (a Galaxy tray half,
``run_4x4_stage_tests.sh``: SP4 on the axis-1 ring, 6 heads stored per chip).
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Callable

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.kda_test_utils import assert_bit_identical
from models.demos.deepseek_v3_d_p.tests.kda.utils import assert_device_time, kda_device_params
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import (
    KdaContractGeometry,
    KdaStates,
    allocate_native_state,
    deallocate_state,
    device_scalar,
)

pytestmark = [run_for_blackhole(), pytest.mark.perf]

_DEVICE_OVERRIDES = {"l1_small_size": 24576, "trace_region_size": 256 * 1024 * 1024}
_TIMING_SAMPLES = int(os.getenv("KDA_ADAPTER_TIMING_SAMPLES", "20"))
_TIMING_REPETITIONS = int(os.getenv("KDA_ADAPTER_TIMING_REPS", "100"))
_LAYER_IDX = 1

# Replaying a trace costs the host a launch floor on a 32-chip mesh (~12.5 or ~18 us per replay on one Galaxy,
# switching between the two within a process). The import's device time is above it; one export is below it, so a
# trace of one export timed that floor (14 or 17-30 us) and a trace of two tiny copies timed the same. The export trace
# therefore captures _STATE_PATH_CALLS_PER_REPLAY exports, so the device, not the host, sets the wall.
# References in us per call (K3 payload). SP8xTP4 (TorusXY): session medians on one Galaxy stayed within 0.7%
# (import); the export's medians spanned 4.90-4.93 us per call over 8 sessions (16 per replay). SP4xTP4 (one tray
# half, TorusX; 6 heads stored per chip): import 41.39 / 41.36 us, export 5.91 / 5.92 us per call, two sessions on
# one Galaxy (bh-glx-110-c10u20, 2026-11-19). Per-op times on two Blackhole Galaxies have differed by up to ~6%, so
# the margin leaves room for that machine offset, as select_final_carry's Galaxy gate does (8%).
_STATE_PATH_REFERENCE_US = {
    ("SP8xTP4", "import"): 49.7,
    ("SP8xTP4", "export"): 4.9,
    ("SP4xTP4", "import"): 41.4,
    ("SP4xTP4", "export"): 5.9,
}
_STATE_PATH_CALLS_PER_REPLAY = {"import": 1, "export": 16}
_STATE_PATH_MARGIN = 0.08
# The gate's chunk start: non-zero, so the import reads the slot rather than zero-filling it.
_STATE_PATH_CHUNK_START = 2880


def _trace_samples_ms(
    mesh_device: ttnn.MeshDevice, operation: Callable[[], object], calls_per_replay: int = 1
) -> list[float]:
    """Per-call wall of ``operation`` over back-to-back replays of a trace that captures it ``calls_per_replay`` times."""
    operation()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(calls_per_replay):
        operation()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    samples_ms = []
    for _ in range(_TIMING_SAMPLES):
        start = time.perf_counter()
        for _ in range(_TIMING_REPETITIONS):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        samples_ms.append((time.perf_counter() - start) * 1e3 / (_TIMING_REPETITIONS * calls_per_replay))
    ttnn.release_trace(mesh_device, trace_id)
    return samples_ms


def _eager_ms(mesh_device: ttnn.MeshDevice, operation: Callable[[], object]) -> float:
    start = time.perf_counter()
    operation()
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - start) * 1e3


def _assert_mesh_equal(expected: ttnn.Tensor, actual: ttnn.Tensor, *, name: str) -> None:
    expected_shards = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(expected)]
    actual_shards = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(actual)]
    assert len(expected_shards) == len(actual_shards)
    for index, pair in enumerate(zip(expected_shards, actual_shards)):
        assert_bit_identical(*pair, name=f"{name} device {index}")


def _patterned_state(mesh_device: ttnn.MeshDevice, geometry: KdaContractGeometry) -> KdaState:
    recurrent = torch.arange(torch.tensor(geometry.recurrent_shape).prod().item(), dtype=torch.float32).reshape(
        geometry.recurrent_shape
    )
    convolution = (
        torch.arange(torch.tensor(geometry.convolution_shape).prod().item(), dtype=torch.int32)
        .remainder(251)
        .to(torch.bfloat16)
        .reshape(geometry.convolution_shape)
    )
    mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    return KdaState(
        recurrent=ttnn.from_torch(
            recurrent,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        ),
        convolution=ttnn.from_torch(
            convolution,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        ),
    )


def _replay_once(mesh_device: ttnn.MeshDevice, operation: Callable[[], object]) -> None:
    """Capture ``operation`` (already warm) in a trace, replay it once, and release the trace."""
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    operation()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize(
    "mesh_device,device_params,sp_axis,stored_heads",
    [
        pytest.param(
            shape,
            kda_device_params(shape, **_DEVICE_OVERRIDES),
            sp_axis,
            stored_heads,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=f"mesh-{shape[0]}x{shape[1]}"),
            id=layout,
        )
        for shape, sp_axis, stored_heads, layout in (((8, 4), 0, 3, "SP8xTP4"), ((4, 4), 1, 6, "SP4xTP4"))
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("operation", ["import", "export"])
def test_kimi_k3_state_path_perf(mesh_device: ttnn.MeshDevice, sp_axis: int, stored_heads: int, operation: str) -> None:
    """Gate the sharded slab import (copies + SP all-gathers) and export (head-selecting copies) on Galaxy."""
    geometry = KdaContractGeometry.from_kda_config(
        kimi_k3_kda_config(), mesh_shape=tuple(mesh_device.shape), sp_axis=sp_axis, tp_axis=1 - sp_axis
    )
    layout = f"SP{geometry.sequence_parallel_size}xTP{geometry.tensor_parallel_size}"
    assert geometry.stored_heads == stored_heads, f"{layout} stores {stored_heads} heads per chip"
    slabs = KdaStates.allocate(mesh_device, geometry, layer_ids=(_LAYER_IDX,), num_slots=1)
    slot = device_scalar(mesh_device, 0)
    chunk_start = device_scalar(mesh_device, _STATE_PATH_CHUNK_START)
    request_head = device_scalar(mesh_device, 0)
    carry = _patterned_state(mesh_device, geometry)
    imported = allocate_native_state(mesh_device, geometry)
    try:
        operations = {
            "import": lambda: slabs.import_layer(imported, slot, _LAYER_IDX, zero_fill_at_zero=chunk_start),
            "export": lambda: slabs.export_layer(carry, slot, _LAYER_IDX),
        }
        slabs.export_layer(carry, slot, _LAYER_IDX)
        calls_per_replay = _STATE_PATH_CALLS_PER_REPLAY[operation]
        traced_us = [ms * 1e3 for ms in _trace_samples_ms(mesh_device, operations[operation], calls_per_replay)]
        eager_us = [_eager_ms(mesh_device, operations[operation]) * 1e3 for _ in range(_TIMING_SAMPLES)]

        # A traced replay of the timed chain moves the carry bit-exactly into a destination cleared beforehand: the
        # import's into a zero-filled carry, the export's into a zeroed slab read back by an eager import.
        def clear_carry() -> None:
            slabs.import_layer(imported, slot, _LAYER_IDX, zero_fill_at_zero=request_head)

        clear_carry()
        if operation == "export":
            slabs.export_layer(imported, slot, _LAYER_IDX)
            operations["import"]()  # reads the slab back: zero only if the zero carry reached it
        ttnn.synchronize_device(mesh_device)
        for tensor in (imported.recurrent, imported.convolution):
            assert not any(ttnn.to_torch(shard).any() for shard in ttnn.get_device_tensors(tensor)), "not cleared"
        _replay_once(mesh_device, operations[operation])
        if operation == "export":
            operations["import"]()
        ttnn.synchronize_device(mesh_device)
        _assert_mesh_equal(carry.recurrent, imported.recurrent, name=f"traced {operation} recurrent")
        _assert_mesh_equal(carry.convolution, imported.convolution, name=f"traced {operation} conv")
    finally:
        deallocate_state(carry)
        deallocate_state(imported)
        slabs.deallocate()
        for scalar in (slot, chunk_start, request_head):
            ttnn.deallocate(scalar)

    reference_us = _STATE_PATH_REFERENCE_US[(layout, operation)]
    median_us = statistics.median(traced_us)
    result = {
        "layout": layout,
        "operation": operation,
        "fabric_config": ttnn.get_fabric_config().name,
        "stored_heads": geometry.stored_heads,
        "replays_per_sample": _TIMING_REPETITIONS,
        "calls_per_replay": calls_per_replay,
        "traced_samples_us": traced_us,
        "median_traced_us": median_us,
        # Untraced single calls through device synchronization: reported, not gated; their host dispatch depends on
        # the host (see the untraced layer gate).
        "eager_samples_us": eager_us,
        "median_eager_us": statistics.median(eager_us),
        "reference_traced_us": reference_us,
        "margin_pct": _STATE_PATH_MARGIN * 100.0,
    }
    print("KDA_STATE_PATH_PERF=" + json.dumps(result, sort_keys=True))
    assert_device_time(median_us, reference_us, _STATE_PATH_MARGIN, f"{layout} traced state {operation} wall per call")
