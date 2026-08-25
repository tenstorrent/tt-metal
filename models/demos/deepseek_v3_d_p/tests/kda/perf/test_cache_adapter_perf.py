# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real Kimi-K3 correctness and performance ablation for cache adapters."""

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
from models.demos.deepseek_v3_d_p.tests.kda.cache_adapters import (
    KDA_CONV_SEGMENT_BYTES,
    KDA_S_SEGMENT_BYTES,
    KdaCacheGeometry,
    allocate_contract_state,
    allocate_native_state,
    contract_memory_configs,
    deallocate_state,
    export_convolution,
    export_recurrent,
    import_convolution,
    import_recurrent,
)
from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _PCC_THRESHOLD, _trace_wall_samples_ms
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KimiK3TestCase,
    assert_bit_identical,
    check_kimi_k3_accuracy,
    make_kimi_k3_device_case,
)
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState

pytest_plugins = ("models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf",)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.perf,
    pytest.mark.parametrize(
        "device_params",
        [
            pytest.param(
                {
                    "l1_small_size": 24576,
                    "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                    "trace_region_size": 256 * 1024 * 1024,
                },
                id="fabric_1d",
            )
        ],
        indirect=True,
    ),
]

_SEQUENCE = 5120
_TIMING_SAMPLES = int(os.getenv("KDA_ADAPTER_TIMING_SAMPLES", "20"))
_TIMING_REPETITIONS = int(os.getenv("KDA_ADAPTER_TIMING_REPS", "100"))
_LAYER_REPETITIONS = int(os.getenv("PERF_REPS", "10"))


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


def _patterned_state(mesh_device: ttnn.MeshDevice, geometry: KdaCacheGeometry) -> KdaState:
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


def _physical_contract(state: KdaState, geometry: KdaCacheGeometry) -> dict[str, object]:
    configs = contract_memory_configs(state.recurrent.device())
    assert state.recurrent.memory_config() == configs.recurrent
    assert state.convolution.memory_config() == configs.convolution
    recurrent_pages = [tensor.buffer_aligned_page_size() for tensor in ttnn.get_device_tensors(state.recurrent)]
    convolution_pages = [tensor.buffer_aligned_page_size() for tensor in ttnn.get_device_tensors(state.convolution)]
    assert set(recurrent_pages) == {4096}
    assert set(convolution_pages) == {128}
    return {
        "recurrent_nd_shard_shape": [1, 1, 128, 32],
        "convolution_nd_shard_shape": [1, 3, 64],
        "recurrent_page_bytes": recurrent_pages[0],
        "convolution_page_bytes": convolution_pages[0],
        "recurrent_pages_per_segment": KDA_S_SEGMENT_BYTES // recurrent_pages[0],
        "convolution_pages_per_segment": KDA_CONV_SEGMENT_BYTES // convolution_pages[0],
        "recurrent_segments_per_device": geometry.recurrent_segments_per_device,
        "convolution_segments_per_device": geometry.convolution_segments_per_device,
    }


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis",
    [((1, 8), 1), ((2, 4), 1), ((2, 4), 0)],
    indirect=["mesh_device"],
    ids=["SP1xTP8", "SP2xTP4", "SP4xTP2"],
)
def test_kimi_k3_cache_adapter_ablation(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    kimi_k3_production_reference: tuple[KimiK3TestCase, torch.Tensor, KDAReferenceState, float],
) -> None:
    case, golden_output, golden_state, _ = kimi_k3_production_reference
    layer, hidden = make_kimi_k3_device_case(mesh_device, case, tensor_parallel_axis=tensor_parallel_axis)
    mesh_shape = tuple(mesh_device.shape)
    sp_axis = 1 - tensor_parallel_axis
    geometry = KdaCacheGeometry(mesh_shape[sp_axis], mesh_shape[tensor_parallel_axis])
    layout = f"SP{geometry.sequence_parallel_size}xTP{geometry.tensor_parallel_size}"

    initial_state = layer.allocate_state(batch_size=1)
    output, real_state = layer.forward(hidden, initial_state)
    ttnn.synchronize_device(mesh_device)
    try:
        pcc = check_kimi_k3_accuracy(
            f"Kimi-K3 cache adapter T={_SEQUENCE} {layout}",
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
    contract = allocate_contract_state(mesh_device, geometry)
    imported = allocate_native_state(mesh_device, geometry)
    ttnn.synchronize_device(mesh_device)
    allocation_ms = (time.perf_counter() - allocation_start) * 1e3
    physical = _physical_contract(contract, geometry)

    cold = {
        "export_s": _eager_ms(mesh_device, lambda: export_recurrent(real_state.recurrent, contract.recurrent)),
        "export_convolution": _eager_ms(
            mesh_device, lambda: export_convolution(real_state.convolution, contract.convolution)
        ),
        "import_s": _eager_ms(mesh_device, lambda: import_recurrent(contract.recurrent, imported.recurrent)),
        "import_convolution": _eager_ms(
            mesh_device, lambda: import_convolution(contract.convolution, imported.convolution)
        ),
    }
    _assert_mesh_equal(real_state.recurrent, imported.recurrent, name=f"{layout} real S round trip")
    _assert_mesh_equal(real_state.convolution, imported.convolution, name=f"{layout} real conv round trip")

    patterned = _patterned_state(mesh_device, geometry)
    export_recurrent(patterned.recurrent, contract.recurrent)
    export_convolution(patterned.convolution, contract.convolution)
    import_recurrent(contract.recurrent, imported.recurrent)
    import_convolution(contract.convolution, imported.convolution)
    ttnn.synchronize_device(mesh_device)
    _assert_mesh_equal(patterned.recurrent, imported.recurrent, name=f"{layout} patterned S")
    _assert_mesh_equal(patterned.convolution, imported.convolution, name=f"{layout} patterned conv")

    export_recurrent(real_state.recurrent, contract.recurrent)
    export_convolution(real_state.convolution, contract.convolution)
    operations = {
        "export_s": lambda: export_recurrent(real_state.recurrent, contract.recurrent),
        "export_convolution": lambda: export_convolution(real_state.convolution, contract.convolution),
        "export_combined": lambda: (
            export_recurrent(real_state.recurrent, contract.recurrent),
            export_convolution(real_state.convolution, contract.convolution),
        ),
        "import_s": lambda: import_recurrent(contract.recurrent, imported.recurrent),
        "import_convolution": lambda: import_convolution(contract.convolution, imported.convolution),
        "import_combined": lambda: (
            import_recurrent(contract.recurrent, imported.recurrent),
            import_convolution(contract.convolution, imported.convolution),
        ),
    }
    timing = {name: _summary(_trace_samples_ms(mesh_device, operation)) for name, operation in operations.items()}
    layer_timing = _summary(_trace_wall_samples_ms(mesh_device, layer, hidden, _LAYER_REPETITIONS))
    layer_median_ms = float(layer_timing["median_ms"])
    for direction in ("export", "import"):
        combined = timing[f"{direction}_combined"]
        combined["layer_overhead_pct"] = 100.0 * float(combined["median_ms"]) / layer_median_ms

    result = {
        "layout": layout,
        "sequence": _SEQUENCE,
        "fabric_config": ttnn.get_fabric_config().name,
        "pcc": pcc,
        "bit_identical_real_round_trip": True,
        "bit_identical_patterned_round_trip": True,
        "geometry": {
            "local_heads": geometry.local_heads,
            "unique_recurrent_segments": geometry.unique_recurrent_segments,
            "unique_convolution_segments": geometry.unique_convolution_segments,
            "physical_recurrent_bytes": geometry.physical_recurrent_bytes,
            "physical_convolution_bytes": geometry.physical_convolution_bytes,
            **physical,
        },
        "allocation_ms": allocation_ms,
        "cold_eager_ms": cold,
        "timing_repetitions": _TIMING_REPETITIONS,
        "timing_sample_count": _TIMING_SAMPLES,
        "timing": timing,
        "layer_trace_wall": layer_timing,
        "direct_layout": {
            "recurrent_producer": "unsupported: recurrent scan output must be interleaved",
            "recurrent_consumer": "unsupported: recurrent scan input must be interleaved",
            "convolution_consumer": "unsupported: qkv causal convolution input must be interleaved",
            "convolution_final_tail_producer": "operator path accepts contract output layout; not end-to-end",
        },
    }
    print("KDA_CACHE_ADAPTER_ABLATION=" + json.dumps(result, sort_keys=True))

    deallocate_state(patterned)
    deallocate_state(contract)
    deallocate_state(imported)
    deallocate_state(initial_state)
    deallocate_state(real_state)
