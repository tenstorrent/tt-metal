# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight correctness and performance acceptance for Kimi-K3 KDA."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import KDAReferenceState, kda_forward_reference
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    KimiK3TestCase,
    check_kimi_k3_accuracy,
    make_kimi_k3_device_case,
    make_kimi_k3_test_case,
)
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA

pytestmark = [
    run_for_blackhole(),
    pytest.mark.perf,
    pytest.mark.parametrize(
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
    ),
]

_SEQUENCE = 5120
_REPETITIONS = 10
_TIMING_SAMPLES = int(os.getenv("KDA_TIMING_SAMPLES", "5"))
if _TIMING_SAMPLES < 1:
    raise ValueError(f"KDA_TIMING_SAMPLES must be positive, got {_TIMING_SAMPLES}")
_PCC_THRESHOLD = 0.9995
_PERF_TARGETS_PATH = Path(__file__).parent / "perf_targets" / "bh_loudbox.json"
_CPU_REFERENCE_CACHE_VERSION = 2


def _tensor_sha256(tensor: torch.Tensor) -> str:
    storage = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(storage)).hexdigest()


def _update_state_dict_fingerprint(fingerprint: Any, state_dict: Mapping[str, torch.Tensor]) -> None:
    for name in sorted(state_dict):
        tensor = state_dict[name]
        metadata = json.dumps(
            [name, str(tensor.dtype), list(tensor.shape)],
            separators=(",", ":"),
        )
        fingerprint.update(metadata.encode())
        fingerprint.update(_tensor_sha256(tensor).encode())


def _cpu_reference_cache_path(case: KimiK3TestCase) -> Path:
    reference_dir = Path(inspect.getfile(kda_forward_reference)).parent
    fingerprint = hashlib.sha256()
    fingerprint.update(f"v{_CPU_REFERENCE_CACHE_VERSION}".encode())
    fingerprint.update(str(KimiK3Config.FIRST_KDA_LAYER).encode())
    fingerprint.update(str(case.hidden.shape[1]).encode())
    fingerprint.update(case.checkpoint_dir.name.encode())
    fingerprint.update((case.checkpoint_dir / "config.json").read_bytes())
    fingerprint.update(_tensor_sha256(case.hidden).encode())
    _update_state_dict_fingerprint(fingerprint, case.state_dict)
    for source_path in (reference_dir / "layer.py", reference_dir / "ops.py"):
        fingerprint.update(source_path.read_bytes())
    return (
        Path(ttnn.CONFIG.model_cache_path)
        / "kimi_k3"
        / case.checkpoint_dir.resolve().name
        / "cpu_reference"
        / f"layer_{KimiK3Config.FIRST_KDA_LAYER}_t{case.hidden.shape[1]}_{fingerprint.hexdigest()[:20]}.pt"
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
) -> tuple[KimiK3TestCase, torch.Tensor, KDAReferenceState, float]:
    """Compute the independent production-length CPU oracle once per test session."""
    case = make_kimi_k3_test_case(kimi_k3_checkpoint_dir, sequence=_SEQUENCE)
    golden_output, golden_state, elapsed = _load_or_compute_cpu_reference(case)
    return case, golden_output, golden_state, elapsed


def _load_perf_target(layout: str, *, sequence: int, repetitions: int) -> tuple[float, float]:
    targets = json.loads(_PERF_TARGETS_PATH.read_text(encoding="utf-8"))
    workload = targets["workload"]
    if sequence != int(workload["sequence"]):
        raise ValueError("LoudBox targets only apply to the required sequence length")
    if repetitions != int(workload["repetitions"]):
        raise ValueError("LoudBox targets require the calibrated replay count; rebaseline to change it")
    target = targets["targets"][layout]
    return float(target["reference_ms"]), float(target["max_regression_pct"])


def _allocate_state(layer: ttKDA) -> KdaState:
    return layer.allocate_state(batch_size=1)


def _deallocate_state(state: KdaState) -> None:
    ttnn.deallocate(state.recurrent)
    ttnn.deallocate(state.convolution)


def _trace_wall_samples_ms(
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
def test_kimi_k3_layer_1_perf(
    mesh_device: ttnn.MeshDevice,
    tensor_parallel_axis: int,
    kimi_k3_production_reference: tuple[KimiK3TestCase, torch.Tensor, KDAReferenceState, float],
) -> None:
    """Compare production geometry with an independent CPU oracle before timing it."""
    sequence = _SEQUENCE
    case, golden_output, golden_state, cpu_reference_seconds = kimi_k3_production_reference
    sequence_parallel_axis = 1 - tensor_parallel_axis
    layer, hidden_tt = make_kimi_k3_device_case(
        mesh_device,
        case,
        tensor_parallel_axis=tensor_parallel_axis,
    )

    state = _allocate_state(layer)
    start = time.perf_counter()
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output, state = layer.forward(hidden_tt, state)
    ttnn.synchronize_device(mesh_device)
    device_forward_ms = (time.perf_counter() - start) * 1e3
    mesh_shape = tuple(mesh_device.shape)
    layout = f"SP{mesh_shape[sequence_parallel_axis]}xTP{mesh_shape[tensor_parallel_axis]}"
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
    _deallocate_state(state)

    repetitions = int(os.getenv("PERF_REPS", str(_REPETITIONS)))
    samples_ms = _trace_wall_samples_ms(mesh_device, layer, hidden_tt, repetitions)
    wall_ms = samples_ms[0]
    median_wall_ms = statistics.median(samples_ms)
    tail_wall_ms = max(samples_ms)
    reference_ms, max_regression_pct = _load_perf_target(layout, sequence=sequence, repetitions=repetitions)
    max_wall_ms = reference_ms * (1.0 + max_regression_pct / 100.0)
    result = {
        "fabric_config": ttnn.get_fabric_config().name,
        "layout": layout,
        "sequence": sequence,
        "repetitions": repetitions,
        "pcc": pcc,
        "pcc_reference": "independent pure-Torch FP32 CPU reference",
        "trace_wall_ms": wall_ms,
        "trace_wall_samples_ms": samples_ms,
        "median_trace_wall_ms": median_wall_ms,
        "tail_trace_wall_ms": tail_wall_ms,
        "timing_sample_count": _TIMING_SAMPLES,
        "reference_trace_wall_ms": reference_ms,
        "max_regression_pct": max_regression_pct,
        "max_trace_wall_ms": max_wall_ms,
        "cpu_reference_seconds": cpu_reference_seconds,
        "device_forward_ms": device_forward_ms,
    }
    print("KDA_LAYER_PERF=" + json.dumps(result, sort_keys=True))
    assert median_wall_ms <= max_wall_ms, (
        f"{layout} median trace wall {median_wall_ms:.3f} ms exceeds LoudBox limit {max_wall_ms:.3f} ms "
        f"(reference {reference_ms:.3f} ms + {max_regression_pct:.1f}%)"
    )
