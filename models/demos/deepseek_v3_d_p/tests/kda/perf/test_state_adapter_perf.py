# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real Kimi-K3 correctness and cost of the KDA state migration adapter.

Two paths are measured against the layer they follow: the production per-layer slab export/import
(`KdaStates.export_layer` / `import_layer`, what a K3 KDA layer runs inside the trace) and the
whole-tensor native <-> single-layer contract conversion. Both must round-trip bit-identically.

``test_kimi_k3_state_path_perf`` gates the SP-sharded slab import and export on Galaxy with a synthetic carry.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import KDAReferenceState
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _trace_wall_samples_ms
from models.demos.deepseek_v3_d_p.tests.kda.reference_cache import load_or_compute_cpu_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KIMI_K3_CONVOLUTION_NRMSE,
    KimiK3TestCase,
    LayerNrmseThresholds,
    assert_device_time,
    check_kimi_k3_accuracy,
    kda_device_params,
    kda_mesh_param,
    make_kimi_k3_device_case,
    make_kimi_k3_test_case,
)
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import (
    KdaContractGeometry,
    KdaStates,
    allocate_contract_state,
    allocate_native_state,
    contract_memory_configs,
    deallocate_state,
    device_scalar,
    export_state,
    import_state,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical, make_actual_start

pytestmark = [run_for_blackhole(), pytest.mark.perf]

_DEVICE_OVERRIDES = {"l1_small_size": 24576, "trace_region_size": 256 * 1024 * 1024}
# LoudBox shapes on the 2D fabric of their shape and the 8x4 galaxy on TorusXY, as Kimi-K3 runs. A galaxy
# runs the 8x4 case.
_LAYOUTS = [
    pytest.param((1, 8), 1, kda_device_params((1, 8), **_DEVICE_OVERRIDES), id="SP1xTP8"),
    pytest.param((2, 4), 1, kda_device_params((2, 4), **_DEVICE_OVERRIDES), id="SP2xTP4"),
    pytest.param((2, 4), 0, kda_device_params((2, 4), **_DEVICE_OVERRIDES), id="SP4xTP2"),
    pytest.param(
        (8, 4),
        1,
        kda_device_params((8, 4), **_DEVICE_OVERRIDES),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="SP8xTP4",
    ),
]

_SEQUENCE = 5120
_TIMING_SAMPLES = int(os.getenv("KDA_ADAPTER_TIMING_SAMPLES", "20"))
_TIMING_REPETITIONS = int(os.getenv("KDA_ADAPTER_TIMING_REPS", "100"))
_LAYER_REPETITIONS = int(os.getenv("PERF_REPS", "10"))
_LAYER_IDX = 1
_PCC_THRESHOLD = 0.9995


@pytest.fixture(scope="session")
def kimi_k3_production_reference(
    kimi_k3_checkpoint_dir: Path,
) -> Callable[[], tuple[KimiK3TestCase, torch.Tensor, KDAReferenceState, float]]:
    """Return a lazy loader for the session-cached production-length CPU oracle."""
    cached_reference: tuple[KimiK3TestCase, torch.Tensor, KDAReferenceState, float] | None = None

    def load() -> tuple[KimiK3TestCase, torch.Tensor, KDAReferenceState, float]:
        nonlocal cached_reference
        if cached_reference is None:
            case = make_kimi_k3_test_case(kimi_k3_checkpoint_dir, sequence=_SEQUENCE)
            golden_output, golden_state, elapsed = load_or_compute_cpu_reference(case)
            cached_reference = case, golden_output, golden_state, elapsed
        return cached_reference

    return load


def _summary(samples_ms: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(samples_ms)
    p95_index = max(0, min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1))
    return {
        "samples_ms": samples_ms,
        "min_ms": min(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p95_ms": ordered[p95_index],
        "max_ms": max(samples_ms),
    }


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


def _physical_contract(state: KdaState, slabs: KdaStates, geometry: KdaContractGeometry) -> dict[str, object]:
    configs = contract_memory_configs(state.recurrent.device(), geometry)
    assert state.recurrent.memory_config() == configs.recurrent
    assert state.convolution.memory_config() == configs.convolution
    recurrent_pages = {tensor.buffer_aligned_page_size() for tensor in ttnn.get_device_tensors(state.recurrent)}
    convolution_pages = {tensor.buffer_aligned_page_size() for tensor in ttnn.get_device_tensors(state.convolution)}
    assert recurrent_pages == {4096}
    assert convolution_pages == {geometry.convolution_segment_bytes}
    assert slabs.convolution.buffer_aligned_page_size() == geometry.convolution_segment_bytes
    return {
        "recurrent_nd_shard_shape": [1, 1, geometry.head_dim, 32],
        "convolution_nd_shard_shape": [1, 1, geometry.convolution_slab_row_width],
        "recurrent_page_bytes": 4096,
        "convolution_page_bytes": geometry.convolution_segment_bytes,
        "convolution_slab_page_bytes": slabs.convolution.buffer_aligned_page_size(),
        "recurrent_pages_per_segment": geometry.recurrent_segment_bytes // 4096,
        "convolution_pages_per_segment": 1,
        "recurrent_segments_per_device": geometry.recurrent_shards_per_layer,
        "convolution_segments_per_device": geometry.convolution_shards_per_layer,
    }


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis,device_params", _LAYOUTS, indirect=["mesh_device", "device_params"]
)
def test_kimi_k3_state_adapter_cost(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    kimi_k3_production_reference: Callable[[], tuple[KimiK3TestCase, torch.Tensor, KDAReferenceState, float]],
) -> None:
    case, golden_output, golden_state, _ = kimi_k3_production_reference()
    layer, hidden = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=tensor_parallel_axis)
    sp_axis = 1 - tensor_parallel_axis
    geometry = KdaContractGeometry.from_kda_config(
        case.config, mesh_shape=tuple(mesh_device.shape), sp_axis=sp_axis, tp_axis=tensor_parallel_axis
    )
    layout = f"SP{geometry.sequence_parallel_size}xTP{geometry.tensor_parallel_size}"

    actual_start = make_actual_start(mesh_device)
    initial_state = layer.allocate_state(batch_size=1)
    output, real_state = layer.forward(hidden, initial_state, actual_start)
    ttnn.synchronize_device(mesh_device)
    try:
        pcc = check_kimi_k3_accuracy(
            f"Kimi-K3 state adapter T={_SEQUENCE} {layout}",
            case,
            golden_output,
            golden_state,
            real_state,
            output,
            mesh_device,
            tensor_parallel_axis,
            page_channels=layer.convolution_page_channels,
            pcc_threshold=_PCC_THRESHOLD,
            nrmse=LayerNrmseThresholds(output=0.03, recurrent=0.04, convolution=KIMI_K3_CONVOLUTION_NRMSE),
        )
    finally:
        ttnn.deallocate(output)

    allocation_start = time.perf_counter()
    slabs = KdaStates.allocate(mesh_device, geometry, sp_axis=sp_axis, layer_ids=(_LAYER_IDX,), num_slots=1)
    contract = allocate_contract_state(mesh_device, geometry)
    imported = allocate_native_state(mesh_device, geometry)
    ttnn.synchronize_device(mesh_device)
    allocation_ms = (time.perf_counter() - allocation_start) * 1e3
    physical = _physical_contract(contract, slabs, geometry)
    slot = device_scalar(mesh_device, 0)  # the export reads its slot on device, as the traced commit does

    # Production path, cold: one export at the layer's commit and one import back.
    cold = {
        "slab_export": _eager_ms(mesh_device, lambda: slabs.export_layer(real_state, slot, _LAYER_IDX)),
        "slab_import": _eager_ms(mesh_device, lambda: slabs.import_layer(imported, slot, _LAYER_IDX)),
        "contract_export": _eager_ms(mesh_device, lambda: export_state(real_state, contract, geometry)),
        "contract_import": _eager_ms(mesh_device, lambda: import_state(contract, imported, geometry)),
    }
    slabs.import_layer(imported, slot, _LAYER_IDX)
    ttnn.synchronize_device(mesh_device)
    _assert_mesh_equal(real_state.recurrent, imported.recurrent, name=f"{layout} slab S round trip")
    _assert_mesh_equal(real_state.convolution, imported.convolution, name=f"{layout} slab conv round trip")
    import_state(contract, imported, geometry)
    ttnn.synchronize_device(mesh_device)
    _assert_mesh_equal(real_state.recurrent, imported.recurrent, name=f"{layout} contract S round trip")
    _assert_mesh_equal(real_state.convolution, imported.convolution, name=f"{layout} contract conv round trip")

    patterned = _patterned_state(mesh_device, geometry)
    slabs.export_layer(patterned, slot, _LAYER_IDX)
    slabs.import_layer(imported, slot, _LAYER_IDX)
    ttnn.synchronize_device(mesh_device)
    _assert_mesh_equal(patterned.recurrent, imported.recurrent, name=f"{layout} patterned slab S")
    _assert_mesh_equal(patterned.convolution, imported.convolution, name=f"{layout} patterned slab conv")

    slabs.export_layer(real_state, slot, _LAYER_IDX)
    export_state(real_state, contract, geometry)
    operations = {
        "slab_export": lambda: slabs.export_layer(real_state, slot, _LAYER_IDX),
        "slab_import": lambda: slabs.import_layer(imported, slot, _LAYER_IDX),
        "contract_export": lambda: export_state(real_state, contract, geometry),
        "contract_import": lambda: import_state(contract, imported, geometry),
    }
    timing = {name: _summary(_trace_samples_ms(mesh_device, operation)) for name, operation in operations.items()}
    layer_samples_ms, _ = _trace_wall_samples_ms(mesh_device, layer, hidden, _LAYER_REPETITIONS)
    layer_timing = _summary(layer_samples_ms)
    layer_median_ms = float(layer_timing["median_ms"])
    for name, entry in timing.items():
        entry["layer_overhead_pct"] = 100.0 * float(entry["median_ms"]) / layer_median_ms

    result = {
        "layout": layout,
        "sequence": _SEQUENCE,
        "fabric_config": ttnn.get_fabric_config().name,
        "pcc": pcc,
        "bit_identical_round_trips": True,
        "geometry": {
            "local_heads": geometry.local_heads,
            "unique_recurrent_segments": geometry.recurrent_segments_per_layer,
            "unique_convolution_segments": geometry.convolution_segments_per_layer,
            **physical,
        },
        "allocation_ms": allocation_ms,
        "cold_eager_ms": cold,
        "timing_repetitions": _TIMING_REPETITIONS,
        "timing_sample_count": _TIMING_SAMPLES,
        "timing": timing,
        "layer_trace_wall": layer_timing,
    }
    print("KDA_STATE_ADAPTER_PERF=" + json.dumps(result, sort_keys=True))

    deallocate_state(patterned)
    deallocate_state(contract)
    deallocate_state(imported)
    deallocate_state(initial_state)
    deallocate_state(real_state)
    slabs.deallocate()
    ttnn.deallocate(slot)
    ttnn.deallocate(actual_start)


# --- Galaxy micro-gates of the SP-sharded state path (synthetic carry, no checkpoint) ---
# Around its forward, a K3 KDA layer on Galaxy SP8xTP4 imports its slot's carry (KdaStates.import_layer: the
# recurrent and conv-tail slab -> staging copies, then their SP all-gathers into the 24-head carry) and exports the
# new one (KdaStates.export_layer: two copies into the slab, each selecting the chip's heads on device). Each chain is
# captured and timed by _trace_samples_ms: host wall time per call over back-to-back trace replays through device
# synchronization, the median over samples. A chain alone pays the cross-chip start skew of its gathers, which inside
# the layer the TP input all-gather absorbs, so these references are not in-layer advances.
# Replaying a trace costs the host a launch floor on this 32-chip mesh (~12.5 or ~18 us per replay on one Galaxy,
# switching between the two within a process). The import's device time is above it; one export is below it, so a
# trace of one export timed that floor (14 or 17-30 us) and a trace of two tiny copies timed the same. The export trace
# therefore captures _STATE_PATH_CALLS_PER_REPLAY exports, so the device, not the host, sets the wall (~78 us).
# Galaxy references in us per call (TorusXY, K3 payload). Session medians on one Galaxy stayed within 0.7% (import);
# the export's medians spanned 4.90-4.93 us per call over 8 sessions (16 per replay), 4.89-4.92 per sample in both
# host floor states; the gate has not yet run on a CI runner, and per-op times on two Blackhole Galaxies have differed
# by up to ~6%, so the margin leaves room for that machine offset, as select_final_carry's Galaxy gate does (8%).
_STATE_PATH_REFERENCE_US = {"import": 49.7, "export": 4.9}
_STATE_PATH_CALLS_PER_REPLAY = {"import": 1, "export": 16}
_STATE_PATH_MARGIN = 0.08
# The layer gate's chunk start: non-zero, so the import reads the slot rather than zero-filling it.
_STATE_PATH_CHUNK_START = 2880


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
    "mesh_device,device_params",
    [kda_mesh_param((8, 4), "mesh-8x4", "SP8xTP4", kda_device_params((8, 4), **_DEVICE_OVERRIDES))],
    indirect=True,
)
@pytest.mark.parametrize("operation", ["import", "export"])
def test_kimi_k3_state_path_perf(mesh_device: ttnn.MeshDevice, operation: str) -> None:
    """Gate the sharded slab import (copies + SP all-gathers) and export (head-selecting copies) on Galaxy."""
    geometry = KdaContractGeometry.from_kda_config(
        kimi_k3_kda_config(), mesh_shape=tuple(mesh_device.shape), sp_axis=0, tp_axis=1
    )
    assert geometry.stored_heads == 3, "Galaxy SP8xTP4 stores 3 heads per chip"
    slabs = KdaStates.allocate(mesh_device, geometry, sp_axis=0, layer_ids=(_LAYER_IDX,), num_slots=1)
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

    reference_us = _STATE_PATH_REFERENCE_US[operation]
    median_us = statistics.median(traced_us)
    result = {
        "layout": "SP8xTP4",
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
    assert_device_time(median_us, reference_us, _STATE_PATH_MARGIN, f"SP8xTP4 traced state {operation} wall per call")
