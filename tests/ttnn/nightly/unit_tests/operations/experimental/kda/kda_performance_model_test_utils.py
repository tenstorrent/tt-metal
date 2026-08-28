# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Independent Python mirror of the KDA theoretical performance model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from loguru import logger

CHUNK_SIZE = 32
MATRIX_FLOPS_PER_CORE_CYCLE = 4096
DRAM_BYTES_PER_NS = 512
_UINT64_MAX = (1 << 64) - 1
_UINT128_MAX = (1 << 128) - 1
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


def _checked_sum(*terms: int | None) -> int | None:
    if any(term is None or term < 0 for term in terms):
        return None
    result = sum(term for term in terms if term is not None)
    return result if result <= _UINT128_MAX else None


def _checked_product(*factors: int | None) -> int | None:
    result = 1
    for factor in factors:
        if factor is None or factor < 0 or (result and factor > _UINT128_MAX // result):
            return None
        result *= factor
    return result


def _make_work(operation: str, *values: int | None) -> KdaWork | None:
    if any(value is None or value > _UINT64_MAX for value in values):
        logger.warning(f"KDA {operation} performance-model work overflowed; returning a zero estimate")
        return None
    return KdaWork(*values)


def sigmoid_gated_rms_norm_work(batch: int, num_heads: int, sequence: int, value_dim: int) -> KdaWork | None:
    rows = _checked_product(batch, num_heads, sequence)
    elements = _checked_product(rows, value_dim)
    return _make_work(
        "sigmoid_gated_rms_norm",
        0,
        _checked_product(4, elements),
        rows,
        elements,
        _checked_sum(rows, elements),
    )


def qkv_causal_conv1d_silu_work(batch: int, sequence: int, q_width: int, k_width: int, v_width: int) -> KdaWork | None:
    elements = _checked_product(batch, sequence, _checked_sum(q_width, k_width, v_width))
    return _make_work(
        "qkv_causal_conv1d_silu",
        0,
        _checked_product(4, elements),
        _checked_product(3, elements),
        0,
        elements,
    )


def reduce_affine_transforms_work(
    batch_heads: int, groups_per_head: int, key_dim: int, value_dim: int
) -> KdaWork | None:
    compositions = _checked_product(batch_heads, max(0, groups_per_head - 1))
    key_squared = _checked_product(key_dim, key_dim)
    dense_per_composition = _checked_sum(
        _checked_product(2, key_squared, key_dim),
        _checked_product(2, key_squared, value_dim),
    )
    return _make_work(
        "reduce_affine_transforms",
        _checked_product(compositions, dense_per_composition),
        0,
        _checked_product(compositions, key_dim, value_dim),
        0,
        0,
    )


def affine_exclusive_scan_work(batch_heads: int, groups_per_head: int, key_dim: int, value_dim: int) -> KdaWork | None:
    transitions = _checked_product(batch_heads, max(0, groups_per_head - 1))
    return _make_work(
        "affine_exclusive_scan",
        _checked_product(transitions, 2, key_dim, key_dim, value_dim),
        0,
        _checked_product(transitions, key_dim, value_dim),
        0,
        0,
    )


def prepare_chunk_recurrence_work(num_heads: int, num_chunks: int, key_dim: int, value_dim: int) -> KdaWork | None:
    instances = _checked_product(num_heads, num_chunks)
    inverse_flops = CHUNK_SIZE * (CHUNK_SIZE - 1) * (CHUNK_SIZE + 1) // 3
    return _make_work(
        "prepare_chunk_recurrence",
        _checked_product(instances, _checked_sum(_checked_product(4, CHUNK_SIZE, CHUNK_SIZE, key_dim), inverse_flops)),
        _checked_product(
            instances,
            _checked_sum(_checked_product(10, CHUNK_SIZE, key_dim), _checked_product(CHUNK_SIZE, value_dim)),
        ),
        _checked_product(
            instances,
            _checked_sum(
                2 * CHUNK_SIZE,
                (CHUNK_SIZE - 1) * key_dim,
                CHUNK_SIZE * key_dim,
                CHUNK_SIZE * CHUNK_SIZE,
            ),
        ),
        _checked_product(instances, 2, CHUNK_SIZE, key_dim),
        _checked_product(instances, _checked_sum(2 * CHUNK_SIZE, 3 * CHUNK_SIZE * key_dim, key_dim)),
    )


def recurrent_chunk_scan_work(batch_heads: int, num_chunks: int, key_dim: int, value_dim: int) -> KdaWork | None:
    instances = _checked_product(batch_heads, num_chunks)
    return _make_work(
        "recurrent_chunk_scan",
        _checked_product(
            instances,
            _checked_sum(
                _checked_product(6, CHUNK_SIZE, key_dim, value_dim),
                _checked_product(4, CHUNK_SIZE, CHUNK_SIZE, value_dim),
            ),
        ),
        _checked_product(instances, key_dim, value_dim),
        _checked_product(instances, _checked_sum(2 * CHUNK_SIZE * value_dim, key_dim * value_dim)),
        0,
        0,
    )


def summarize_chunk_recurrence_work(batch_heads: int, num_chunks: int, key_dim: int, value_dim: int) -> KdaWork | None:
    instances = _checked_product(batch_heads, num_chunks)
    return _make_work(
        "summarize_chunk_recurrence",
        _checked_product(
            instances,
            _checked_sum(
                _checked_product(8, CHUNK_SIZE, key_dim, value_dim),
                _checked_product(4, CHUNK_SIZE, CHUNK_SIZE, value_dim),
            ),
        ),
        _checked_product(instances, 2, key_dim, value_dim),
        _checked_sum(
            _checked_product(instances, _checked_sum(2 * CHUNK_SIZE * value_dim, 2 * key_dim * value_dim)),
            _checked_product(batch_heads, key_dim, value_dim),
        ),
        0,
        0,
    )


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
    physical_bytes = _checked_product(int(tensor.volume()), int(tensor.element_size()))
    if physical_bytes is None or physical_bytes > _UINT64_MAX:
        logger.warning("KDA performance-model physical byte count overflowed; returning a zero estimate")
        return None
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

    cycle_numerator = _checked_sum(
        _checked_product(work.dense_flops, fidelity_factor),
        _checked_product(32, work.multiply_results, fidelity_factor),
        _checked_product(32, work.add_results),
        _checked_product(16, work.reduction_input_elements, fidelity_factor),
    )
    cycle_denominator = _checked_product(MATRIX_FLOPS_PER_CORE_CYCLE, core_count)
    if cycle_numerator is None or cycle_denominator is None:
        logger.warning("KDA performance-model cycle arithmetic overflowed; returning a zero estimate")
        return fallback
    ideal_fpu_cycles = _ceil_div(cycle_numerator, cycle_denominator)
    fpu_time_numerator = _checked_product(ideal_fpu_cycles, 1000)
    if fpu_time_numerator is None:
        logger.warning("KDA performance-model time arithmetic overflowed; returning a zero estimate")
        return fallback
    ideal_fpu_ns = _ceil_div(fpu_time_numerator, clock_mhz)

    input_bytes = [0] * len(inputs)
    output_bytes = [0] * len(outputs)
    mandatory_dram_bytes = 0
    input_addresses: set[int] = set()
    output_addresses: set[int] = set()
    for index, traffic in enumerate(inputs):
        if traffic.is_dram:
            if traffic.physical_bytes > _PROFILER_INT_MAX:
                logger.warning("KDA performance-model input bytes do not fit profiler fields; returning zero")
                return fallback
            input_bytes[index] = traffic.physical_bytes
            if traffic.buffer_address not in input_addresses:
                input_addresses.add(traffic.buffer_address)
                total = _checked_sum(mandatory_dram_bytes, traffic.physical_bytes)
                if total is None:
                    logger.warning("KDA performance-model DRAM input bytes overflowed; returning a zero estimate")
                    return fallback
                mandatory_dram_bytes = total
    for index, traffic in enumerate(outputs):
        if traffic.is_dram:
            if traffic.physical_bytes > _PROFILER_INT_MAX:
                logger.warning("KDA performance-model output bytes do not fit profiler fields; returning zero")
                return fallback
            output_bytes[index] = traffic.physical_bytes
            if traffic.buffer_address not in output_addresses:
                output_addresses.add(traffic.buffer_address)
                total = _checked_sum(mandatory_dram_bytes, traffic.physical_bytes)
                if total is None:
                    logger.warning("KDA performance-model DRAM output bytes overflowed; returning a zero estimate")
                    return fallback
                mandatory_dram_bytes = total

    ideal_dram_ns = _ceil_div(mandatory_dram_bytes, DRAM_BYTES_PER_NS)
    ideal_ns = max(ideal_fpu_ns, ideal_dram_ns)
    if max(ideal_fpu_cycles, ideal_fpu_ns, ideal_dram_ns, ideal_ns) > _PROFILER_INT_MAX:
        logger.warning("KDA performance-model estimate does not fit profiler fields; returning a zero estimate")
        return fallback
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
