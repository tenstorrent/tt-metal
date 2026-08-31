# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Independent Python mirror of the KDA theoretical performance model."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

from loguru import logger

CHUNK_SIZE = 32
MATRIX_FLOPS_PER_CORE_CYCLE = 4096
DRAM_BYTES_PER_NS = 512
_PROFILER_INT_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class KdaWork:
    dense_flops: int = 0
    multiply_results: int = 0
    add_results: int = 0
    reduction_input_elements: int = 0
    omitted_sfpu_results: int = 0


@dataclass(frozen=True)
class KdaTensorTraffic:
    buffer_address: int
    physical_bytes: int
    is_dram: bool


@dataclass(frozen=True)
class KdaEstimate:
    valid: bool = False
    ideal_fpu_cycles: int = 0
    ideal_fpu_ns: int = 0
    mandatory_dram_bytes: int = 0
    ideal_dram_ns: int = 0
    ideal_ns: int = 0
    omitted_sfpu_results: int = 0
    input_bytes: tuple[int, ...] = ()
    output_bytes: tuple[int, ...] = ()


@dataclass(frozen=True)
class KdaUtilization:
    fpu_utilization_pct: float
    dram_utilization_pct: float
    roofline_utilization_pct: float


def _sum_nonnegative(*terms: int) -> int:
    assert all(isinstance(term, int) and term >= 0 for term in terms), "terms must be non-negative integers"
    return sum(terms)


def _product_nonnegative(*factors: int) -> int:
    assert all(isinstance(factor, int) and factor >= 0 for factor in factors), "factors must be non-negative integers"
    return math.prod(factors)


def _make_work(*values: int) -> KdaWork:
    assert all(value >= 0 for value in values), "work values must be non-negative"
    return KdaWork(*values)


def sigmoid_gated_rms_norm_work(batch: int, num_heads: int, sequence: int, value_dim: int) -> KdaWork:
    rows = _product_nonnegative(batch, num_heads, sequence)
    elements = _product_nonnegative(rows, value_dim)
    return _make_work(
        0,
        _product_nonnegative(4, elements),
        rows,
        elements,
        _sum_nonnegative(rows, elements),
    )


def qkv_causal_conv1d_silu_work(batch: int, sequence: int, q_width: int, k_width: int, v_width: int) -> KdaWork:
    elements = _product_nonnegative(batch, sequence, _sum_nonnegative(q_width, k_width, v_width))
    return _make_work(
        0,
        _product_nonnegative(4, elements),
        _product_nonnegative(3, elements),
        0,
        elements,
    )


def reduce_affine_transforms_work(batch_heads: int, groups_per_head: int, key_dim: int, value_dim: int) -> KdaWork:
    assert groups_per_head >= 0, "groups_per_head must be non-negative"
    compositions = _product_nonnegative(batch_heads, max(0, groups_per_head - 1))
    key_squared = _product_nonnegative(key_dim, key_dim)
    dense_per_composition = _sum_nonnegative(
        _product_nonnegative(2, key_squared, key_dim),
        _product_nonnegative(2, key_squared, value_dim),
    )
    return _make_work(
        _product_nonnegative(compositions, dense_per_composition),
        0,
        _product_nonnegative(compositions, key_dim, value_dim),
        0,
        0,
    )


def affine_exclusive_scan_work(batch_heads: int, groups_per_head: int, key_dim: int, value_dim: int) -> KdaWork:
    assert groups_per_head >= 0, "groups_per_head must be non-negative"
    transitions = _product_nonnegative(batch_heads, max(0, groups_per_head - 1))
    return _make_work(
        _product_nonnegative(transitions, 2, key_dim, key_dim, value_dim),
        0,
        _product_nonnegative(transitions, key_dim, value_dim),
        0,
        0,
    )


def prepare_chunk_recurrence_work(num_heads: int, num_chunks: int, key_dim: int, value_dim: int) -> KdaWork:
    instances = _product_nonnegative(num_heads, num_chunks)
    inverse_flops = CHUNK_SIZE * (CHUNK_SIZE - 1) * (CHUNK_SIZE + 1) // 3
    return _make_work(
        _product_nonnegative(
            instances, _sum_nonnegative(_product_nonnegative(4, CHUNK_SIZE, CHUNK_SIZE, key_dim), inverse_flops)
        ),
        _product_nonnegative(
            instances,
            _sum_nonnegative(
                _product_nonnegative(10, CHUNK_SIZE, key_dim), _product_nonnegative(CHUNK_SIZE, value_dim)
            ),
        ),
        _product_nonnegative(
            instances,
            _sum_nonnegative(
                2 * CHUNK_SIZE,
                (CHUNK_SIZE - 1) * key_dim,
                CHUNK_SIZE * key_dim,
                CHUNK_SIZE * CHUNK_SIZE,
            ),
        ),
        _product_nonnegative(instances, 2, CHUNK_SIZE, key_dim),
        _product_nonnegative(instances, _sum_nonnegative(2 * CHUNK_SIZE, 3 * CHUNK_SIZE * key_dim, key_dim)),
    )


def recurrent_chunk_scan_work(batch_heads: int, num_chunks: int, key_dim: int, value_dim: int) -> KdaWork:
    instances = _product_nonnegative(batch_heads, num_chunks)
    return _make_work(
        _product_nonnegative(
            instances,
            _sum_nonnegative(
                _product_nonnegative(6, CHUNK_SIZE, key_dim, value_dim),
                _product_nonnegative(4, CHUNK_SIZE, CHUNK_SIZE, value_dim),
            ),
        ),
        _product_nonnegative(instances, key_dim, value_dim),
        _product_nonnegative(instances, _sum_nonnegative(2 * CHUNK_SIZE * value_dim, key_dim * value_dim)),
        0,
        0,
    )


def summarize_chunk_recurrence_work(batch_heads: int, num_chunks: int, key_dim: int, value_dim: int) -> KdaWork:
    instances = _product_nonnegative(batch_heads, num_chunks)
    return _make_work(
        _product_nonnegative(
            instances,
            _sum_nonnegative(
                _product_nonnegative(8, CHUNK_SIZE, key_dim, value_dim),
                _product_nonnegative(4, CHUNK_SIZE, CHUNK_SIZE, value_dim),
            ),
        ),
        _product_nonnegative(instances, 2, key_dim, value_dim),
        _sum_nonnegative(
            _product_nonnegative(instances, _sum_nonnegative(2 * CHUNK_SIZE * value_dim, 2 * key_dim * value_dim)),
            _product_nonnegative(batch_heads, key_dim, value_dim),
        ),
        0,
        0,
    )


def profile_realtime_program(device: Any, run_fn: Any) -> tuple[Any, dict[str, Any]]:
    """Profile one KDA program while retaining the device clock used by its record."""
    import ttnn

    profile_record = None
    dropped = 0

    def collect_records(batch: Any) -> None:
        nonlocal dropped, profile_record
        dropped += int(batch.dropped)
        for record in batch.records:
            if profile_record is not None:
                return
            start_timestamp = int(record.start_timestamp)
            end_timestamp = int(record.end_timestamp)
            frequency = float(record.frequency)
            if frequency > 0 and end_timestamp > start_timestamp:
                profile_record = {
                    "runtime_id": int(record.runtime_id),
                    "chip_id": int(record.chip_id),
                    "duration_ns": (end_timestamp - start_timestamp) / frequency,
                    "frequency_ghz": frequency,
                    "kernel_sources": tuple(str(source) for source in record.kernel_sources),
                }

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect_records)
    try:
        result = run_fn()
        ttnn.synchronize_device(device)
        deadline = time.monotonic() + 1.0
        while profile_record is None and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)

    if dropped:
        raise RuntimeError(f"Real-time profiler dropped {dropped} record(s)")
    if profile_record is None:
        raise RuntimeError("Real-time profiler returned no valid KDA program record")
    return result, profile_record


def clock_mhz_from_frequency_ghz(frequency_ghz: float) -> int:
    if not math.isfinite(frequency_ghz) or frequency_ghz <= 0:
        return 0
    return math.floor(frequency_ghz * 1000 + 0.5)


def math_fidelity_factor(math_fidelity: Any) -> int:
    import ttnn

    return {
        ttnn.MathFidelity.LoFi: 1,
        ttnn.MathFidelity.HiFi2: 2,
        ttnn.MathFidelity.HiFi3: 3,
        ttnn.MathFidelity.HiFi4: 4,
    }.get(math_fidelity, 0)


def tensor_traffic(tensor: Any) -> KdaTensorTraffic | None:
    import ttnn

    if tensor.storage_type() != ttnn.StorageType.DEVICE or not tensor.is_allocated():
        logger.warning("KDA performance model expected an allocated device tensor; returning a zero estimate")
        return None
    physical_bytes = _product_nonnegative(int(tensor.volume()), int(tensor.element_size()))
    return KdaTensorTraffic(
        buffer_address=int(tensor.buffer_address()),
        physical_bytes=physical_bytes,
        is_dram=tensor.memory_config().buffer_type == ttnn.BufferType.DRAM,
    )


def zero_estimate(input_count: int, output_count: int) -> KdaEstimate:
    return KdaEstimate(input_bytes=(0,) * input_count, output_bytes=(0,) * output_count)


def _ceil_div(numerator: int, denominator: int) -> int:
    return numerator // denominator + int(numerator % denominator != 0)


def estimate(
    work: KdaWork,
    inputs: Iterable[KdaTensorTraffic],
    outputs: Iterable[KdaTensorTraffic],
    *,
    core_count: int,
    clock_mhz: int,
    fidelity_factor: int,
) -> KdaEstimate:
    inputs = tuple(inputs)
    outputs = tuple(outputs)
    fallback = zero_estimate(len(inputs), len(outputs))
    if core_count <= 0 or clock_mhz <= 0 or fidelity_factor not in (1, 2, 3, 4):
        logger.warning("KDA performance model received invalid cores, clock, or fidelity; returning a zero estimate")
        return fallback

    cycle_numerator = _sum_nonnegative(
        _product_nonnegative(work.dense_flops, fidelity_factor),
        _product_nonnegative(32, work.multiply_results, fidelity_factor),
        _product_nonnegative(32, work.add_results),
        _product_nonnegative(16, work.reduction_input_elements, fidelity_factor),
    )
    cycle_denominator = _product_nonnegative(MATRIX_FLOPS_PER_CORE_CYCLE, core_count)
    ideal_fpu_cycles = _ceil_div(cycle_numerator, cycle_denominator)
    fpu_time_numerator = _product_nonnegative(ideal_fpu_cycles, 1000)
    ideal_fpu_ns = _ceil_div(fpu_time_numerator, clock_mhz)

    input_bytes = [0] * len(inputs)
    output_bytes = [0] * len(outputs)
    mandatory_dram_bytes = 0
    input_addresses: set[int] = set()
    output_addresses: set[int] = set()
    for index, traffic in enumerate(inputs):
        if traffic.is_dram:
            if traffic.physical_bytes > _PROFILER_INT_MAX:
                raise OverflowError("KDA performance-model input bytes do not fit profiler fields")
            input_bytes[index] = traffic.physical_bytes
            if traffic.buffer_address not in input_addresses:
                input_addresses.add(traffic.buffer_address)
                mandatory_dram_bytes = _sum_nonnegative(mandatory_dram_bytes, traffic.physical_bytes)
    for index, traffic in enumerate(outputs):
        if traffic.is_dram:
            if traffic.physical_bytes > _PROFILER_INT_MAX:
                raise OverflowError("KDA performance-model output bytes do not fit profiler fields")
            output_bytes[index] = traffic.physical_bytes
            if traffic.buffer_address not in output_addresses:
                output_addresses.add(traffic.buffer_address)
                mandatory_dram_bytes = _sum_nonnegative(mandatory_dram_bytes, traffic.physical_bytes)

    ideal_dram_ns = _ceil_div(mandatory_dram_bytes, DRAM_BYTES_PER_NS)
    ideal_ns = max(ideal_fpu_ns, ideal_dram_ns)
    if max(ideal_fpu_cycles, ideal_fpu_ns, ideal_dram_ns, ideal_ns) > _PROFILER_INT_MAX:
        raise OverflowError("KDA performance-model estimate does not fit profiler fields")
    return KdaEstimate(
        valid=True,
        ideal_fpu_cycles=ideal_fpu_cycles,
        ideal_fpu_ns=ideal_fpu_ns,
        mandatory_dram_bytes=mandatory_dram_bytes,
        ideal_dram_ns=ideal_dram_ns,
        ideal_ns=ideal_ns,
        omitted_sfpu_results=work.omitted_sfpu_results,
        input_bytes=tuple(input_bytes),
        output_bytes=tuple(output_bytes),
    )


def estimate_for_tensors(
    work: KdaWork,
    inputs: Iterable[Any],
    outputs: Iterable[Any],
    *,
    device: Any,
    frequency_ghz: float,
    math_fidelity: Any,
) -> KdaEstimate:
    inputs = tuple(inputs)
    outputs = tuple(outputs)
    input_traffic = tuple(tensor_traffic(tensor) for tensor in inputs)
    output_traffic = tuple(tensor_traffic(tensor) for tensor in outputs)
    if any(traffic is None for traffic in (*input_traffic, *output_traffic)):
        return zero_estimate(len(inputs), len(outputs))
    grid = device.compute_with_storage_grid_size()
    return estimate(
        work,
        input_traffic,
        output_traffic,
        core_count=int(grid.x) * int(grid.y),
        clock_mhz=clock_mhz_from_frequency_ghz(frequency_ghz),
        fidelity_factor=math_fidelity_factor(math_fidelity),
    )


def utilization(estimate: KdaEstimate, measured_ns: float) -> KdaUtilization:
    if not estimate.valid or not math.isfinite(measured_ns) or measured_ns <= 0:
        return KdaUtilization(0.0, 0.0, 0.0)
    return KdaUtilization(
        fpu_utilization_pct=100 * estimate.ideal_fpu_ns / measured_ns,
        dram_utilization_pct=100 * estimate.ideal_dram_ns / measured_ns,
        roofline_utilization_pct=100 * estimate.ideal_ns / measured_ns,
    )
