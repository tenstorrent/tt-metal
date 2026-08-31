# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Independent Python mirror of the KDA theoretical performance model."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

CHUNK_SIZE = 32
MATRIX_FLOPS_PER_CORE_CYCLE = 4096
DRAM_BYTES_PER_NS = 512


@dataclass(frozen=True)
class KdaWork:
    fpu_matrix_flops: int = 0
    fpu_multiply_ops: int = 0
    fpu_add_ops: int = 0
    fpu_reduction_ops: int = 0
    sfpu_exp_ops: int = 0
    sfpu_rsqrt_ops: int = 0
    sfpu_sigmoid_ops: int = 0
    sfpu_silu_ops: int = 0
    dram_bytes: int = 0


@dataclass(frozen=True)
class KdaPerformance:
    work: KdaWork
    ideal_fpu_ns: float
    ideal_dram_ns: float
    ideal_ns: float
    fpu_utilization_pct: float
    dram_utilization_pct: float
    utilization_pct: float


@dataclass(frozen=True)
class _TensorTraffic:
    buffer_address: int
    physical_bytes: int
    is_dram: bool


def _sum_nonnegative(*terms: int) -> int:
    assert all(isinstance(term, int) and term >= 0 for term in terms), "terms must be non-negative integers"
    return sum(terms)


def _product_nonnegative(*factors: int) -> int:
    assert all(isinstance(factor, int) and factor >= 0 for factor in factors), "factors must be non-negative integers"
    return math.prod(factors)


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in tensor.shape)


def _base_work(**values: int) -> KdaWork:
    assert all(value >= 0 for value in values.values()), "work values must be non-negative"
    return KdaWork(**values)


def _sigmoid_gated_rms_norm_work(batch: int, num_heads: int, sequence: int, value_dim: int) -> KdaWork:
    rows = _product_nonnegative(batch, num_heads, sequence)
    elements = _product_nonnegative(rows, value_dim)
    return _base_work(
        fpu_multiply_ops=_product_nonnegative(4, elements),
        fpu_add_ops=rows,
        fpu_reduction_ops=_product_nonnegative(rows, max(0, value_dim - 1)),
        sfpu_rsqrt_ops=rows,
        sfpu_sigmoid_ops=elements,
    )


def _qkv_causal_conv1d_silu_work(batch: int, sequence: int, width: int) -> KdaWork:
    elements = _product_nonnegative(batch, sequence, width)
    return _base_work(
        fpu_multiply_ops=_product_nonnegative(4, elements),
        fpu_add_ops=_product_nonnegative(3, elements),
        sfpu_silu_ops=elements,
    )


def _reduce_affine_transforms_work(batch_heads: int, groups_per_head: int, key_dim: int, value_dim: int) -> KdaWork:
    assert groups_per_head >= 0, "groups_per_head must be non-negative"
    compositions = _product_nonnegative(batch_heads, max(0, groups_per_head - 1))
    return _base_work(
        fpu_matrix_flops=_product_nonnegative(
            compositions, _sum_nonnegative(2 * key_dim**3, 2 * key_dim**2 * value_dim)
        ),
        fpu_add_ops=_product_nonnegative(compositions, key_dim, value_dim),
    )


def _affine_exclusive_scan_work(batch_heads: int, groups_per_head: int, key_dim: int, value_dim: int) -> KdaWork:
    assert groups_per_head >= 0, "groups_per_head must be non-negative"
    transitions = _product_nonnegative(batch_heads, max(0, groups_per_head - 1))
    return _base_work(
        fpu_matrix_flops=_product_nonnegative(transitions, 2, key_dim, key_dim, value_dim),
        fpu_add_ops=_product_nonnegative(transitions, key_dim, value_dim),
    )


def _prepare_chunk_recurrence_work(num_heads: int, num_chunks: int, key_dim: int, value_dim: int) -> KdaWork:
    instances = _product_nonnegative(num_heads, num_chunks)
    inverse_flops = CHUNK_SIZE * (CHUNK_SIZE - 1) * (CHUNK_SIZE + 1) // 3
    return _base_work(
        fpu_matrix_flops=_product_nonnegative(
            instances, _sum_nonnegative(4 * CHUNK_SIZE**2 * key_dim, inverse_flops)
        ),
        fpu_multiply_ops=_product_nonnegative(instances, 10 * CHUNK_SIZE * key_dim + CHUNK_SIZE * value_dim),
        fpu_add_ops=_product_nonnegative(
            instances, 2 * CHUNK_SIZE + (CHUNK_SIZE - 1) * key_dim + CHUNK_SIZE * key_dim + CHUNK_SIZE**2
        ),
        fpu_reduction_ops=_product_nonnegative(instances, 2, CHUNK_SIZE, max(0, key_dim - 1)),
        sfpu_exp_ops=_product_nonnegative(instances, 3 * CHUNK_SIZE * key_dim + key_dim),
        sfpu_rsqrt_ops=_product_nonnegative(instances, 2, CHUNK_SIZE),
    )


def _recurrent_chunk_scan_work(batch_heads: int, num_chunks: int, key_dim: int, value_dim: int) -> KdaWork:
    instances = _product_nonnegative(batch_heads, num_chunks)
    return _base_work(
        fpu_matrix_flops=_product_nonnegative(
            instances, 6 * CHUNK_SIZE * key_dim * value_dim + 4 * CHUNK_SIZE**2 * value_dim
        ),
        fpu_multiply_ops=_product_nonnegative(instances, key_dim, value_dim),
        fpu_add_ops=_product_nonnegative(instances, 2 * CHUNK_SIZE * value_dim + key_dim * value_dim),
    )


def _summarize_chunk_recurrence_work(batch_heads: int, num_chunks: int, key_dim: int, value_dim: int) -> KdaWork:
    instances = _product_nonnegative(batch_heads, num_chunks)
    return _base_work(
        fpu_matrix_flops=_product_nonnegative(
            instances, 8 * CHUNK_SIZE * key_dim * value_dim + 4 * CHUNK_SIZE**2 * value_dim
        ),
        fpu_multiply_ops=_product_nonnegative(instances, 2, key_dim, value_dim),
        fpu_add_ops=_sum_nonnegative(
            _product_nonnegative(instances, 2 * CHUNK_SIZE * value_dim + 2 * key_dim * value_dim),
            _product_nonnegative(batch_heads, key_dim, value_dim),
        ),
    )


def _math_fidelity_factor(math_fidelity: Any) -> int:
    import ttnn

    try:
        return {
            ttnn.MathFidelity.LoFi: 1,
            ttnn.MathFidelity.HiFi2: 2,
            ttnn.MathFidelity.HiFi3: 3,
            ttnn.MathFidelity.HiFi4: 4,
        }[math_fidelity]
    except KeyError as error:
        raise ValueError(f"unsupported math fidelity: {math_fidelity}") from error


def _tensor_traffic(tensor: Any) -> _TensorTraffic:
    import ttnn

    if tensor.storage_type() != ttnn.StorageType.DEVICE or not tensor.is_allocated():
        raise ValueError("KDA performance model requires allocated device tensors")
    return _TensorTraffic(
        buffer_address=int(tensor.buffer_address()),
        physical_bytes=_product_nonnegative(int(tensor.volume()), int(tensor.element_size())),
        is_dram=tensor.memory_config().buffer_type == ttnn.BufferType.DRAM,
    )


def _mandatory_dram_bytes(inputs: Iterable[Any], outputs: Iterable[Any]) -> int:
    total = 0
    for tensors in (tuple(inputs), tuple(outputs)):
        addresses: set[int] = set()
        for tensor in tensors:
            traffic = _tensor_traffic(tensor)
            if traffic.is_dram and traffic.buffer_address not in addresses:
                addresses.add(traffic.buffer_address)
                total = _sum_nonnegative(total, traffic.physical_bytes)
    return total


def _performance(
    work: KdaWork,
    inputs: Iterable[Any],
    outputs: Iterable[Any],
    *,
    measured_ns: float,
    core_count: int,
    frequency_ghz: float,
    math_fidelity: Any,
) -> KdaPerformance:
    if not isinstance(core_count, int) or core_count <= 0:
        raise ValueError("core_count must be a positive integer")
    if not math.isfinite(frequency_ghz) or frequency_ghz <= 0:
        raise ValueError("frequency_ghz must be finite and positive")
    if not math.isfinite(measured_ns) or measured_ns <= 0:
        raise ValueError("measured_ns must be finite and positive")
    work = replace(work, dram_bytes=_mandatory_dram_bytes(inputs, outputs))
    fidelity_factor = _math_fidelity_factor(math_fidelity)
    cycle_numerator = _sum_nonnegative(
        _product_nonnegative(work.fpu_matrix_flops, fidelity_factor),
        _product_nonnegative(32, work.fpu_multiply_ops, fidelity_factor),
        _product_nonnegative(32, work.fpu_add_ops),
        _product_nonnegative(16, work.fpu_reduction_ops, fidelity_factor),
    )
    ideal_fpu_ns = cycle_numerator / (MATRIX_FLOPS_PER_CORE_CYCLE * core_count * frequency_ghz)
    ideal_dram_ns = work.dram_bytes / DRAM_BYTES_PER_NS
    ideal_ns = max(ideal_fpu_ns, ideal_dram_ns)
    return KdaPerformance(
        work=work,
        ideal_fpu_ns=ideal_fpu_ns,
        ideal_dram_ns=ideal_dram_ns,
        ideal_ns=ideal_ns,
        fpu_utilization_pct=100 * ideal_fpu_ns / measured_ns,
        dram_utilization_pct=100 * ideal_dram_ns / measured_ns,
        utilization_pct=100 * ideal_ns / measured_ns,
    )


def sigmoid_gated_rms_norm_performance(
    input_tensor: Any,
    gate: Any,
    weight: Any,
    output: Any,
    *,
    measured_ns: float,
    core_count: int,
    frequency_ghz: float,
    math_fidelity: Any,
) -> KdaPerformance:
    input_shape, gate_shape, output_shape, weight_shape = map(_shape, (input_tensor, gate, output, weight))
    if len(input_shape) != 3 or len(gate_shape) != 3 or output_shape != gate_shape or len(weight_shape) != 1:
        raise ValueError("sigmoid-gated RMSNorm tensor shapes are inconsistent")
    batch, sequence, hidden = gate_shape
    value_dim = input_shape[-1]
    if value_dim <= 0 or hidden % value_dim or weight_shape != (value_dim,):
        raise ValueError("sigmoid-gated RMSNorm tensor shapes are inconsistent")
    num_heads = hidden // value_dim
    if input_shape != (batch * num_heads, sequence, value_dim):
        raise ValueError("sigmoid-gated RMSNorm tensor shapes are inconsistent")
    return _performance(
        _sigmoid_gated_rms_norm_work(batch, num_heads, sequence, value_dim),
        (input_tensor, gate, weight),
        (output,),
        measured_ns=measured_ns,
        core_count=core_count,
        frequency_ghz=frequency_ghz,
        math_fidelity=math_fidelity,
    )


def qkv_causal_conv1d_silu_performance(
    input_tensor: Any,
    history: Any,
    taps: Sequence[Any],
    outputs: Sequence[Any],
    *,
    measured_ns: float,
    core_count: int,
    frequency_ghz: float,
    math_fidelity: Any,
) -> KdaPerformance:
    input_shape = _shape(input_tensor)
    output_shapes = tuple(_shape(output) for output in outputs)
    if len(input_shape) != 3 or len(output_shapes) != 3 or any(len(shape) != 3 for shape in output_shapes):
        raise ValueError("QKV causal Conv1D plus SiLU tensor shapes are inconsistent")
    batch, sequence, width = input_shape
    if (
        any(shape[:2] != (batch, sequence) for shape in output_shapes)
        or sum(shape[-1] for shape in output_shapes) != width
    ):
        raise ValueError("QKV causal Conv1D plus SiLU tensor shapes are inconsistent")
    return _performance(
        _qkv_causal_conv1d_silu_work(batch, sequence, width),
        (input_tensor, history, *taps),
        outputs,
        measured_ns=measured_ns,
        core_count=core_count,
        frequency_ghz=frequency_ghz,
        math_fidelity=math_fidelity,
    )


def reduce_affine_transforms_performance(
    a: Any,
    b: Any,
    outputs: Sequence[Any],
    *,
    measured_ns: float,
    core_count: int,
    frequency_ghz: float,
    math_fidelity: Any,
) -> KdaPerformance:
    a_shape, b_shape = _shape(a), _shape(b)
    output_shapes = tuple(_shape(output) for output in outputs)
    if len(a_shape) != 3 or len(b_shape) != 3 or len(output_shapes) != 2:
        raise ValueError("affine-transform reduction tensor shapes are inconsistent")
    batch_heads, key_dim, output_key_dim = output_shapes[0]
    value_dim = output_shapes[1][-1]
    if (
        batch_heads <= 0
        or key_dim != output_key_dim
        or output_shapes[1][:2] != (batch_heads, key_dim)
        or a_shape[0] % batch_heads
    ):
        raise ValueError("affine-transform reduction tensor shapes are inconsistent")
    groups_per_head = a_shape[0] // batch_heads
    if a_shape != (batch_heads * groups_per_head, key_dim, key_dim) or b_shape != (a_shape[0], key_dim, value_dim):
        raise ValueError("affine-transform reduction tensor shapes are inconsistent")
    return _performance(
        _reduce_affine_transforms_work(batch_heads, groups_per_head, key_dim, value_dim),
        (a, b),
        outputs,
        measured_ns=measured_ns,
        core_count=core_count,
        frequency_ghz=frequency_ghz,
        math_fidelity=math_fidelity,
    )


def affine_exclusive_scan_performance(
    a: Any,
    b: Any,
    initial_state: Any,
    output: Any,
    *,
    measured_ns: float,
    core_count: int,
    frequency_ghz: float,
    math_fidelity: Any,
) -> KdaPerformance:
    a_shape, b_shape, state_shape, output_shape = map(_shape, (a, b, initial_state, output))
    if any(len(shape) != 3 for shape in (a_shape, b_shape, state_shape, output_shape)):
        raise ValueError("affine exclusive-scan tensor shapes are inconsistent")
    batch_heads, key_dim, value_dim = state_shape
    if batch_heads <= 0 or a_shape[0] % batch_heads:
        raise ValueError("affine exclusive-scan tensor shapes are inconsistent")
    groups_per_head = a_shape[0] // batch_heads
    if (
        a_shape != (a_shape[0], key_dim, key_dim)
        or b_shape != (a_shape[0], key_dim, value_dim)
        or output_shape != b_shape
    ):
        raise ValueError("affine exclusive-scan tensor shapes are inconsistent")
    return _performance(
        _affine_exclusive_scan_work(batch_heads, groups_per_head, key_dim, value_dim),
        (a, b, initial_state),
        (output,),
        measured_ns=measured_ns,
        core_count=core_count,
        frequency_ghz=frequency_ghz,
        math_fidelity=math_fidelity,
    )


def prepare_chunk_recurrence_performance(
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    *,
    measured_ns: float,
    core_count: int,
    frequency_ghz: float,
    math_fidelity: Any,
) -> KdaPerformance:
    if len(inputs) != 5 or len(outputs) != 7:
        raise ValueError("chunk-recurrence preparation requires five inputs and seven outputs")
    q_shape, v_shape, beta_shape = map(_shape, (inputs[0], inputs[2], inputs[4]))
    if len(q_shape) != 3 or len(v_shape) != 3 or len(beta_shape) != 4:
        raise ValueError("chunk-recurrence preparation tensor shapes are inconsistent")
    num_heads, num_chunks, chunk_size, trailing = beta_shape
    if (
        num_heads <= 0
        or chunk_size != CHUNK_SIZE
        or trailing != 1
        or q_shape[-1] % num_heads
        or v_shape[-1] % num_heads
    ):
        raise ValueError("chunk-recurrence preparation tensor shapes are inconsistent")
    key_dim, value_dim = q_shape[-1] // num_heads, v_shape[-1] // num_heads
    if q_shape[-2] != num_chunks * CHUNK_SIZE or v_shape[-2] != num_chunks * CHUNK_SIZE:
        raise ValueError("chunk-recurrence preparation tensor shapes are inconsistent")
    return _performance(
        _prepare_chunk_recurrence_work(num_heads, num_chunks, key_dim, value_dim),
        inputs,
        outputs,
        measured_ns=measured_ns,
        core_count=core_count,
        frequency_ghz=frequency_ghz,
        math_fidelity=math_fidelity,
    )


def recurrent_chunk_scan_performance(
    inputs: Sequence[Any],
    state: Any,
    outputs: Sequence[Any],
    *,
    measured_ns: float,
    core_count: int,
    frequency_ghz: float,
    math_fidelity: Any,
) -> KdaPerformance:
    if len(inputs) != 7 or len(outputs) != 2:
        raise ValueError("recurrent chunk scan requires seven protocol inputs and two outputs")
    batch_heads, num_chunks, chunk_size, value_dim = _shape(inputs[0])
    key_dim = _shape(inputs[1])[-1]
    if chunk_size != CHUNK_SIZE or _shape(state) != (batch_heads, key_dim, value_dim):
        raise ValueError("recurrent chunk-scan tensor shapes are inconsistent")
    return _performance(
        _recurrent_chunk_scan_work(batch_heads, num_chunks, key_dim, value_dim),
        (*inputs, state),
        outputs,
        measured_ns=measured_ns,
        core_count=core_count,
        frequency_ghz=frequency_ghz,
        math_fidelity=math_fidelity,
    )


def summarize_chunk_recurrence_performance(
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    *,
    measured_ns: float,
    core_count: int,
    frequency_ghz: float,
    math_fidelity: Any,
) -> KdaPerformance:
    if len(inputs) != 7 or len(outputs) != 2:
        raise ValueError("chunk-recurrence summary requires seven inputs and two outputs")
    batch_heads, num_chunks, chunk_size, value_dim = _shape(inputs[0])
    key_dim = _shape(inputs[1])[-1]
    if chunk_size != CHUNK_SIZE:
        raise ValueError("chunk-recurrence summary tensor shapes are inconsistent")
    return _performance(
        _summarize_chunk_recurrence_work(batch_heads, num_chunks, key_dim, value_dim),
        inputs,
        outputs,
        measured_ns=measured_ns,
        core_count=core_count,
        frequency_ghz=frequency_ghz,
        math_fidelity=math_fidelity,
    )
