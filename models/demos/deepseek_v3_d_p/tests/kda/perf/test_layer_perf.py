# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Local real-weight and CI-synthetic performance acceptance for Kimi-K3 KDA."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import KDAReferenceState, kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import KIMI_K3_FIRST_KDA_LAYER
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KimiK3TestCase,
    check_kimi_k3_accuracy,
    make_kimi_k3_device_case,
    make_kimi_k3_test_case,
    make_synthetic_kimi_k3_test_case,
)
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

pytestmark = [
    run_for_blackhole(),
    pytest.mark.perf,
    pytest.mark.timeout(900),
]

_SEQUENCE = 5120
_REPETITIONS = 10
_TIMING_SAMPLES = 5
_PCC_THRESHOLD = 0.9995
_CPU_REFERENCE_CACHE_VERSION = 4
_PERF_SKU = "bh_loudbox"
_PERF_MARGIN = 0.03
# LoudBox calibration at 350413d7a98e (2026-08-31): median across five independent
# sessions, each using the median of five warm synchronized 10-replay samples.
_PERF_REFERENCE_MS = {
    "SP1xTP8": 9.597,
    "SP2xTP4": 9.539,
    "SP4xTP2": 9.991,
}
_GALAXY_PERF_REFERENCE_MS: float | None = None
_GALAXY_CALIBRATION_ENV = "KDA_GALAXY_PERF_CALIBRATE"


def _tensor_sha256(tensor: torch.Tensor) -> str:
    storage = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(storage)).hexdigest()


def _cpu_reference_cache_path(case: KimiK3TestCase) -> Path:
    reference_dir = Path(inspect.getfile(kda_forward_reference)).parent
    fingerprint = hashlib.sha256()
    fingerprint.update(f"v{_CPU_REFERENCE_CACHE_VERSION}".encode())
    fingerprint.update(str(KIMI_K3_FIRST_KDA_LAYER).encode())
    fingerprint.update(str(case.hidden.shape[1]).encode())
    fingerprint.update(case.weights_identity.encode())
    fingerprint.update(json.dumps(asdict(case.config), sort_keys=True).encode())
    fingerprint.update(_tensor_sha256(case.hidden).encode())
    for source_path in sorted(reference_dir.glob("*.py")):
        fingerprint.update(source_path.name.encode())
        fingerprint.update(source_path.read_bytes())
    return (
        Path(ttnn.CONFIG.model_cache_path)
        / "kimi_k3"
        / case.weights_identity
        / "cpu_reference"
        / f"layer_{KIMI_K3_FIRST_KDA_LAYER}_t{case.hidden.shape[1]}_{fingerprint.hexdigest()[:20]}.pt"
    )


def _reference_tensors(output: torch.Tensor, state: KDAReferenceState) -> dict[str, torch.Tensor]:
    return {
        "output": output.detach().clone(),
        "recurrent": state.recurrent.detach().clone(),
        "q_convolution": state.q_convolution.detach().clone(),
        "k_convolution": state.k_convolution.detach().clone(),
        "v_convolution": state.v_convolution.detach().clone(),
    }


def _validate_cached_reference(case: KimiK3TestCase, payload: dict[str, Any]) -> tuple[torch.Tensor, KDAReferenceState]:
    tensors = {name: payload[name] for name in payload["digests"]}
    expected_shapes = {
        "output": (1, case.hidden.shape[1], case.config.hidden_size),
        "recurrent": (1, case.config.num_heads, case.config.head_k_dim, case.config.head_v_dim),
        "q_convolution": (1, case.config.conv_kernel_size - 1, case.config.q_dim),
        "k_convolution": (1, case.config.conv_kernel_size - 1, case.config.k_dim),
        "v_convolution": (1, case.config.conv_kernel_size - 1, case.config.v_dim),
    }
    assert set(tensors) == set(expected_shapes), f"unexpected CPU-reference cache tensors: {set(tensors)}"
    for name, tensor in tensors.items():
        assert isinstance(tensor, torch.Tensor), f"cached {name} is not a tensor"
        assert (
            tuple(tensor.shape) == expected_shapes[name]
        ), f"cached {name} shape {tuple(tensor.shape)} != {expected_shapes[name]}"
        assert _tensor_sha256(tensor) == payload["digests"][name], f"cached {name} checksum mismatch"
    return tensors["output"], KDAReferenceState(
        recurrent=tensors["recurrent"],
        q_convolution=tensors["q_convolution"],
        k_convolution=tensors["k_convolution"],
        v_convolution=tensors["v_convolution"],
    )


def _load_or_compute_cpu_reference(case: KimiK3TestCase) -> tuple[torch.Tensor, KDAReferenceState, float]:
    cache_path = _cpu_reference_cache_path(case)
    start = time.perf_counter()
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        output, state = _validate_cached_reference(case, payload)
        elapsed = time.perf_counter() - start
        logger.info(f"KDA T=5120 CPU reference cache hit: {cache_path}")
        logger.info(f"KDA T=5120 CPU reference load completed in {elapsed:.3f} seconds")
        return output, state, elapsed

    output, state = kda_forward_reference(case.hidden, case.state_dict, case.config)
    tensors = _reference_tensors(output, state)
    payload = {**tensors, "digests": {name: _tensor_sha256(tensor) for name, tensor in tensors.items()}}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - start
    logger.info(f"KDA T=5120 CPU reference cache miss: {cache_path}")
    logger.info(f"KDA T=5120 CPU reference computation completed in {elapsed:.3f} seconds")
    return output, state, elapsed


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
            golden_output, golden_state, elapsed = _load_or_compute_cpu_reference(case)
            cached_reference = case, golden_output, golden_state, elapsed
        return cached_reference

    return load


def _perf_reference_ms(layout: str) -> float:
    if os.environ.get("KDA_PERF_SKU") != _PERF_SKU:
        raise ValueError(f"set KDA_PERF_SKU={_PERF_SKU} to opt in to this hardware-specific performance gate")
    return _PERF_REFERENCE_MS[layout]


def _allocate_state(layer: ttKDA) -> KdaState:
    return layer.allocate_state(batch_size=1)


def _deallocate_state(state: KdaState) -> None:
    ttnn.deallocate(state.recurrent)
    ttnn.deallocate(state.convolution)


def _device_program_label(kernel_sources: tuple[str, ...]) -> str:
    names = set()
    for source in kernel_sources:
        parts = source.replace("\\", "/").split("/")
        if "operations" not in parts:
            continue
        index = parts.index("operations") + 1
        experimental = index < len(parts) and parts[index] == "experimental"
        if experimental:
            index += 1
        if index < len(parts):
            name = f"{'experimental.' if experimental else ''}{parts[index]}"
            if name.endswith(("kda", "ccl")) and index + 1 < len(parts):
                name = f"{name}.{parts[index + 1]}"
            names.add(name)
    if names:
        return "+".join(sorted(names))
    basenames = {Path(source).stem for source in kernel_sources}
    return "+".join(sorted(basenames)) if basenames else "unknown"


def _log_device_program_times(mesh_device: ttnn.MeshDevice, layer: ttKDA, hidden: ttnn.Tensor, layout: str) -> None:
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        raise RuntimeError(f"real-time profiler is inactive for the {layout} KDA e2e device-time breakdown")
    state = _allocate_state(layer)
    output = None
    next_state = None
    profiled_results: list[tuple[ttnn.Tensor, KdaState]] = []

    def run_profiled_forward() -> tuple[ttnn.Tensor, KdaState]:
        result = layer.forward(hidden, state)
        profiled_results.append(result)
        return result

    try:
        (output, next_state), records = profile_realtime_program(
            mesh_device,
            run_profiled_forward,
            collect_all=True,
            record_timeout_seconds=30.0,
        )
        per_program: dict[int, dict[str, Any]] = {}
        for record in records:
            runtime_id = record["runtime_id"]
            if not runtime_id:
                continue
            entry = per_program.setdefault(
                runtime_id,
                {
                    "duration_ns": 0.0,
                    "kernel_sources": record["kernel_sources"],
                    "chip_ids": set(),
                    "record_count": 0,
                },
            )
            entry["duration_ns"] = max(entry["duration_ns"], record["duration_ns"])
            entry["chip_ids"].add(record["chip_id"])
            entry["record_count"] += 1
        if not per_program:
            raise RuntimeError("real-time profiler returned no KDA program records")
        expected_chip_count = mesh_device.get_num_devices()
        programs: list[dict[str, Any]] = [
            {
                "sequence": sequence,
                "name": _device_program_label(info["kernel_sources"]),
                "device_time_ns": round(float(info["duration_ns"]), 3),
                "chip_count": len(info["chip_ids"]),
                "record_count": int(info["record_count"]),
                "complete": len(info["chip_ids"]) == expected_chip_count,
            }
            for sequence, info in enumerate(per_program.values())
        ]
        incomplete_program_sequences = [program["sequence"] for program in programs if not program["complete"]]
        durations_by_name: dict[str, list[float]] = {}
        for program in programs:
            durations_by_name.setdefault(program["name"], []).append(program["device_time_ns"])
        operation_summary = [
            {
                "name": name,
                "program_count": len(durations),
                "median_device_time_ns": round(statistics.median(durations), 3),
                "max_device_time_ns": round(max(durations), 3),
            }
            for name, durations in durations_by_name.items()
        ]
        print(
            "KDA_LAYER_DEVICE_TIMES="
            + json.dumps(
                {
                    "layout": layout,
                    "measurement": "one warm eager forward outside gated trace samples",
                    "duration_semantics": (
                        "per-program max across reported chip records; programs may overlap and durations must not be summed"
                    ),
                    "chip_completeness": {
                        "expected_chip_count": expected_chip_count,
                        "incomplete_program_sequences": incomplete_program_sequences,
                    },
                    "operation_summary": operation_summary,
                    "programs": programs,
                },
                sort_keys=True,
            )
        )
    finally:
        if profiled_results and output is None:
            output, next_state = profiled_results[-1]
        if output is not None:
            ttnn.deallocate(output)
        if next_state is not None:
            _deallocate_state(next_state)
        _deallocate_state(state)


def _trace_wall_samples_ms(
    mesh_device: ttnn.MeshDevice,
    layer: ttKDA,
    hidden: ttnn.Tensor,
    repetitions: int,
    case: KimiK3TestCase,
    golden_output: torch.Tensor,
    golden_state: KDAReferenceState,
    tensor_parallel_axis: int,
    layout: str,
) -> tuple[list[float], dict[str, float]]:
    state = _allocate_state(layer)
    warm_output, warm_state = layer.forward(hidden, state)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm_output)
    _deallocate_state(warm_state)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output, next_state = layer.forward(hidden, state)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    trace_pcc = check_kimi_k3_accuracy(
        f"Kimi-K3 layer 1 T={case.hidden.shape[1]} {layout} trace replay",
        case,
        golden_output,
        golden_state,
        next_state,
        output,
        mesh_device,
        tensor_parallel_axis,
        pcc_threshold=_PCC_THRESHOLD,
    )
    samples_ms = []
    for _ in range(_TIMING_SAMPLES):
        start = time.perf_counter()
        for _ in range(repetitions):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        samples_ms.append((time.perf_counter() - start) * 1e3 / repetitions)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(output)
    _deallocate_state(state)
    _deallocate_state(next_state)
    return samples_ms, trace_pcc


def _trace_wall_samples_without_accuracy(
    mesh_device: ttnn.MeshDevice,
    layer: ttKDA,
    hidden: ttnn.Tensor,
    repetitions: int,
) -> list[float]:
    state = _allocate_state(layer)
    warm_output, warm_state = layer.forward(hidden, state)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(warm_output)
    _deallocate_state(warm_state)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output, next_state = layer.forward(hidden, state)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    samples_ms = []
    for _ in range(_TIMING_SAMPLES):
        start = time.perf_counter()
        for _ in range(repetitions):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        samples_ms.append((time.perf_counter() - start) * 1e3 / repetitions)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.deallocate(output)
    _deallocate_state(state)
    _deallocate_state(next_state)
    return samples_ms


@pytest.mark.parametrize(
    "mesh_device,tensor_parallel_axis",
    [((1, 8), 1), ((2, 4), 1), ((2, 4), 0)],
    indirect=["mesh_device"],
    ids=["SP1xTP8", "SP2xTP4", "SP4xTP2"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        # FABRIC_2D follow-up: SP1xTP8 hangs with a device timeout and failed
        # Ethernet-core recovery. SP2xTP4/SP4xTP2 were correct but 0.05%/0.10%
        # slower than FABRIC_1D; do not enable 2D until the SP1 failure is fixed.
        pytest.param(
            {
                "l1_small_size": 24576,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "trace_region_size": 256 * 1024 * 1024,
            },
            id="fabric_1d",
        ),
    ],
    indirect=True,
)
def test_kimi_k3_layer_1_perf(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    kimi_k3_production_reference: Callable[[], tuple[KimiK3TestCase, torch.Tensor, KDAReferenceState, float]],
) -> None:
    """Compare production geometry with an independent CPU oracle before timing it."""
    sequence = _SEQUENCE
    sequence_parallel_axis = 1 - tensor_parallel_axis
    mesh_shape = tuple(mesh_device.shape)
    layout = f"SP{mesh_shape[sequence_parallel_axis]}xTP{mesh_shape[tensor_parallel_axis]}"
    repetitions = _REPETITIONS
    reference_ms = _perf_reference_ms(layout)
    case, golden_output, golden_state, cpu_reference_seconds = kimi_k3_production_reference()
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
    )

    initial_state = _allocate_state(layer)
    start = time.perf_counter()
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output, state = layer.forward(hidden_tt, initial_state)
    ttnn.synchronize_device(mesh_device)
    device_forward_ms = (time.perf_counter() - start) * 1e3
    try:
        pcc = check_kimi_k3_accuracy(
            f"Kimi-K3 layer 1 T={sequence} {layout}",
            case,
            golden_output,
            golden_state,
            state,
            output,
            mesh_device,
            tensor_parallel_axis,
            pcc_threshold=_PCC_THRESHOLD,
        )
    finally:
        ttnn.deallocate(output)
    _deallocate_state(initial_state)
    _deallocate_state(state)

    samples_ms, trace_pcc = _trace_wall_samples_ms(
        mesh_device,
        layer,
        hidden_tt,
        repetitions,
        case,
        golden_output,
        golden_state,
        tensor_parallel_axis,
        layout,
    )
    first_wall_ms = samples_ms[0]
    median_wall_ms = statistics.median(samples_ms)
    tail_wall_ms = max(samples_ms)
    min_wall_ms = reference_ms * (1.0 - _PERF_MARGIN)
    max_wall_ms = reference_ms * (1.0 + _PERF_MARGIN)
    result = {
        "fabric_config": ttnn.get_fabric_config().name,
        "layout": layout,
        "sequence": sequence,
        "repetitions": repetitions,
        "pcc": pcc,
        "trace_pcc": trace_pcc,
        "pcc_reference": "independent pure-Torch FP32 CPU reference",
        "trace_wall_ms": median_wall_ms,
        "first_trace_wall_ms": first_wall_ms,
        "trace_wall_samples_ms": samples_ms,
        "median_trace_wall_ms": median_wall_ms,
        "tail_trace_wall_ms": tail_wall_ms,
        "timing_sample_count": _TIMING_SAMPLES,
        "reference_trace_wall_ms": reference_ms,
        "perf_margin_pct": _PERF_MARGIN * 100.0,
        "min_trace_wall_ms": min_wall_ms,
        "max_trace_wall_ms": max_wall_ms,
        "cpu_reference_seconds": cpu_reference_seconds,
        "device_forward_ms": device_forward_ms,
    }
    print("KDA_LAYER_PERF=" + json.dumps(result, sort_keys=True))
    assert min_wall_ms <= median_wall_ms <= max_wall_ms, (
        f"{layout} median trace wall {median_wall_ms:.3f} ms is outside LoudBox range "
        f"[{min_wall_ms:.3f}, {max_wall_ms:.3f}] ms (reference {reference_ms:.3f} ms ± {_PERF_MARGIN:.0%})"
    )
    if layout == "SP2xTP4":
        # Best-effort diagnostics run once after the five samples and their sole regression assertion.
        try:
            _log_device_program_times(mesh_device, layer, hidden_tt, layout)
        except Exception as error:
            print("KDA_LAYER_DEVICE_TIMES_ERROR=" + json.dumps({"layout": layout, "error": str(error)}, sort_keys=True))


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(l1_small_size=24576, trace_region_size=256 * 1024 * 1024),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="SP8xTP4-torus-xy",
        )
    ],
    indirect=True,
)
def test_synthetic_kimi_k3_perf(
    mesh_device: ttnn.MeshDevice,
    device_params: dict,
) -> None:
    """Measure checkpoint-free production K3 latency; calibrate before enabling its gate."""
    tensor_parallel_axis = 1
    case = make_synthetic_kimi_k3_test_case(sequence=_SEQUENCE)
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
        cache_weights=False,
    )
    samples_ms = _trace_wall_samples_without_accuracy(mesh_device, layer, hidden_tt, _REPETITIONS)
    median_wall_ms = statistics.median(samples_ms)
    result = {
        "fabric_config": ttnn.get_fabric_config().name,
        "layout": "SP8xTP4",
        "sequence": _SEQUENCE,
        "weights": "deterministic synthetic",
        "repetitions": _REPETITIONS,
        "trace_wall_samples_ms": samples_ms,
        "median_trace_wall_ms": median_wall_ms,
        "timing_sample_count": _TIMING_SAMPLES,
        "reference_trace_wall_ms": _GALAXY_PERF_REFERENCE_MS,
        "perf_margin_pct": _PERF_MARGIN * 100.0,
    }
    print("KDA_SYNTHETIC_PERF=" + json.dumps(result, sort_keys=True))
    if _GALAXY_PERF_REFERENCE_MS is None:
        assert os.environ.get(_GALAXY_CALIBRATION_ENV) == "1", (
            "Galaxy KDA perf target is not calibrated; set "
            f"{_GALAXY_CALIBRATION_ENV}=1 only for the first hosted measurement"
        )
    else:
        lower = _GALAXY_PERF_REFERENCE_MS * (1.0 - _PERF_MARGIN)
        upper = _GALAXY_PERF_REFERENCE_MS * (1.0 + _PERF_MARGIN)
        assert lower <= median_wall_ms <= upper, (
            f"SP8xTP4 median trace wall {median_wall_ms:.3f} ms is outside Galaxy range "
            f"[{lower:.3f}, {upper:.3f}] ms "
            f"(reference {_GALAXY_PERF_REFERENCE_MS:.3f} ms ± {_PERF_MARGIN:.0%})"
        )
    try:
        _log_device_program_times(mesh_device, layer, hidden_tt, "SP8xTP4")
    except Exception as error:
        print("KDA_LAYER_DEVICE_TIMES_ERROR=" + json.dumps({"layout": "SP8xTP4", "error": str(error)}, sort_keys=True))
