# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real Kimi-K3 correctness and cost of the KDA state migration adapter.

Two paths are measured against the layer they follow: the production per-layer slab export/import
(`KdaStates.export_layer` / `import_layer`, what `KdaStateCache.commit` runs inside the trace) and the
whole-tensor native <-> single-layer contract conversion. Both must round-trip bit-identically.
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
from models.demos.deepseek_v3_d_p.reference.kda import KDAReferenceState
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, fabric_1d_device_params
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _PCC_THRESHOLD, _trace_wall_samples_ms
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KimiK3TestCase,
    check_kimi_k3_accuracy,
    make_kimi_k3_device_case,
)
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import (
    KdaContractGeometry,
    KdaStates,
    allocate_contract_state,
    allocate_native_state,
    contract_memory_configs,
    deallocate_state,
    export_state,
    import_state,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_bit_identical, make_actual_start

pytest_plugins = ("models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf",)

pytestmark = [run_for_blackhole(), pytest.mark.perf]

_DEVICE_OVERRIDES = {"l1_small_size": 24576, "trace_region_size": 256 * 1024 * 1024}
# LoudBox shapes on the unwrapped 1D fabric (the KDA perf leg's profile) and the 8x4 galaxy on the 2D
# fabric the Kimi-K3 chunked tests run on. A 1D fabric on a 2x4 sub-mesh of a galaxy fails router sync,
# so a galaxy runs the 8x4 case.
_LAYOUTS = [
    pytest.param((1, 8), 1, fabric_1d_device_params(**_DEVICE_OVERRIDES), id="SP1xTP8"),
    pytest.param((2, 4), 1, fabric_1d_device_params(**_DEVICE_OVERRIDES), id="SP2xTP4"),
    pytest.param((2, 4), 0, fabric_1d_device_params(**_DEVICE_OVERRIDES), id="SP4xTP2"),
    pytest.param(
        (8, 4),
        1,
        fabric2d_device_params(**_DEVICE_OVERRIDES),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="SP8xTP4",
    ),
]

_SEQUENCE = 5120
_TIMING_SAMPLES = int(os.getenv("KDA_ADAPTER_TIMING_SAMPLES", "20"))
_TIMING_REPETITIONS = int(os.getenv("KDA_ADAPTER_TIMING_REPS", "100"))
_LAYER_REPETITIONS = int(os.getenv("PERF_REPS", "10"))
_LAYER_IDX = 1


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


def _trace_samples_ms(mesh_device: ttnn.MeshDevice, operation: Callable[[], object]) -> list[float]:
    operation()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
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
        samples_ms.append((time.perf_counter() - start) * 1e3 / _TIMING_REPETITIONS)
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
    assert convolution_pages == {128}
    assert slabs.convolution.buffer_aligned_page_size() == geometry.convolution_segment_bytes
    return {
        "recurrent_nd_shard_shape": [1, 1, geometry.head_dim, 32],
        "convolution_nd_shard_shape": [1, geometry.conv_history, 64],
        "recurrent_page_bytes": 4096,
        "convolution_page_bytes": 128,
        "convolution_slab_page_bytes": slabs.convolution.buffer_aligned_page_size(),
        "recurrent_pages_per_segment": geometry.recurrent_segment_bytes // 4096,
        "convolution_pages_per_segment": geometry.convolution_segment_bytes // 128,
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
            pcc_threshold=_PCC_THRESHOLD,
        )
    finally:
        ttnn.deallocate(output)

    allocation_start = time.perf_counter()
    slabs = KdaStates.allocate(mesh_device, geometry, layer_ids=(_LAYER_IDX,), num_slots=1)
    contract = allocate_contract_state(mesh_device, geometry)
    imported = allocate_native_state(mesh_device, geometry)
    ttnn.synchronize_device(mesh_device)
    allocation_ms = (time.perf_counter() - allocation_start) * 1e3
    physical = _physical_contract(contract, slabs, geometry)

    # Production path, cold: one export at the layer's commit and one import back.
    cold = {
        "slab_export": _eager_ms(mesh_device, lambda: slabs.export_layer(real_state, 0, _LAYER_IDX)),
        "slab_import": _eager_ms(mesh_device, lambda: slabs.import_layer(imported, 0, _LAYER_IDX)),
        "contract_export": _eager_ms(mesh_device, lambda: export_state(real_state, contract, geometry)),
        "contract_import": _eager_ms(mesh_device, lambda: import_state(contract, imported, geometry)),
    }
    slabs.import_layer(imported, 0, _LAYER_IDX)
    ttnn.synchronize_device(mesh_device)
    _assert_mesh_equal(real_state.recurrent, imported.recurrent, name=f"{layout} slab S round trip")
    _assert_mesh_equal(real_state.convolution, imported.convolution, name=f"{layout} slab conv round trip")
    import_state(contract, imported, geometry)
    ttnn.synchronize_device(mesh_device)
    _assert_mesh_equal(real_state.recurrent, imported.recurrent, name=f"{layout} contract S round trip")
    _assert_mesh_equal(real_state.convolution, imported.convolution, name=f"{layout} contract conv round trip")

    patterned = _patterned_state(mesh_device, geometry)
    slabs.export_layer(patterned, 0, _LAYER_IDX)
    slabs.import_layer(imported, 0, _LAYER_IDX)
    ttnn.synchronize_device(mesh_device)
    _assert_mesh_equal(patterned.recurrent, imported.recurrent, name=f"{layout} patterned slab S")
    _assert_mesh_equal(patterned.convolution, imported.convolution, name=f"{layout} patterned slab conv")

    slabs.export_layer(real_state, 0, _LAYER_IDX)
    export_state(real_state, contract, geometry)
    operations = {
        "slab_export": lambda: slabs.export_layer(real_state, 0, _LAYER_IDX),
        "slab_import": lambda: slabs.import_layer(imported, 0, _LAYER_IDX),
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
    ttnn.deallocate(actual_start)
