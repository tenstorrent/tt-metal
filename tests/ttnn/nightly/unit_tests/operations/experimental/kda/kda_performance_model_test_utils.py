# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Independent Python mirror of the KDA theoretical performance model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

CHUNK_SIZE = 32
MATRIX_FLOPS_PER_CORE_CYCLE = 4096
# Nominal clock used by the repository's Blackhole realtime utilization models.
_BLACKHOLE_CLOCK_GHZ = 1.35
# Blackhole ceiling used by the canonical operation model: ttnn/core/operation.cpp.
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


def _dram_bytes(inputs: Sequence[Any], outputs: Sequence[Any]) -> int:
    import ttnn

    tensors = (*inputs, *outputs)
    if any(tensor.storage_type() != ttnn.StorageType.DEVICE or not tensor.is_allocated() for tensor in tensors):
        raise ValueError("KDA performance model requires allocated device tensors")

    input_addresses = tuple(int(tensor.buffer_address()) for tensor in inputs)
    if len(input_addresses) != len(set(input_addresses)):
        raise ValueError("KDA performance model does not support aliased inputs")

    return sum(
        int(tensor.volume()) * int(tensor.element_size())
        for tensor in tensors
        if tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    )


def _performance(
    work: KdaWork,
    *,
    measured_ns: float,
    core_count: int,
    math_fidelity: Any,
) -> KdaPerformance:
    if not isinstance(core_count, int) or core_count <= 0:
        raise ValueError("core_count must be a positive integer")
    if not math.isfinite(measured_ns) or measured_ns <= 0:
        raise ValueError("measured_ns must be finite and positive")

    fidelity_factor = _math_fidelity_factor(math_fidelity)
    cycle_numerator = (
        work.fpu_matrix_flops * fidelity_factor
        + 32 * work.fpu_multiply_ops * fidelity_factor
        + 32 * work.fpu_add_ops
        + 16 * work.fpu_reduction_ops * fidelity_factor
    )
    ideal_fpu_ns = cycle_numerator / (MATRIX_FLOPS_PER_CORE_CYCLE * core_count * _BLACKHOLE_CLOCK_GHZ)
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
    math_fidelity: Any,
) -> KdaPerformance:
    tensors = (input_tensor, gate, weight, output)
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in tensors):
        raise ValueError("sigmoid-gated RMSNorm tensor shapes must be positive")
    if len(input_tensor.shape) != 3 or len(gate.shape) != 3 or len(weight.shape) != 1 or output.shape != gate.shape:
        raise ValueError("sigmoid-gated RMSNorm tensor shapes are inconsistent")

    batch, sequence, hidden = gate.shape
    value_dim = input_tensor.shape[-1]
    if hidden % value_dim or weight.shape != (value_dim,):
        raise ValueError("sigmoid-gated RMSNorm tensor shapes are inconsistent")
    num_heads = hidden // value_dim
    if input_tensor.shape != (batch * num_heads, sequence, value_dim):
        raise ValueError("sigmoid-gated RMSNorm tensor shapes are inconsistent")

    rows = batch * num_heads * sequence
    elements = rows * value_dim
    work = KdaWork(
        fpu_multiply_ops=4 * elements,
        fpu_add_ops=rows,
        fpu_reduction_ops=rows * (value_dim - 1),
        sfpu_rsqrt_ops=rows,
        sfpu_sigmoid_ops=elements,
        dram_bytes=_dram_bytes((input_tensor, gate, weight), (output,)),
    )
    return _performance(
        work,
        measured_ns=measured_ns,
        core_count=core_count,
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
    math_fidelity: Any,
) -> KdaPerformance:
    if len(taps) != 4 or len(outputs) != 3:
        raise ValueError("QKV causal Conv1D plus SiLU requires four taps and three outputs")
    tensors = (input_tensor, history, *taps, *outputs)
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in tensors):
        raise ValueError("QKV causal Conv1D plus SiLU tensor shapes must be positive")
    if (
        len(input_tensor.shape) != 3
        or len(history.shape) != 3
        or any(len(tap.shape) == 0 for tap in taps)
        or any(len(output.shape) != 3 for output in outputs)
    ):
        raise ValueError("QKV causal Conv1D plus SiLU tensor shapes are inconsistent")

    batch, sequence, width = input_tensor.shape
    if (
        history.shape != (batch, 3, width)
        or any(tap.shape[-1] != width or math.prod(tap.shape) != width for tap in taps)
        or any(output.shape != (batch, sequence, output.shape[-1]) for output in outputs)
        or sum(output.shape[-1] for output in outputs) != width
    ):
        raise ValueError("QKV causal Conv1D plus SiLU tensor shapes are inconsistent")

    elements = batch * sequence * width
    work = KdaWork(
        fpu_multiply_ops=4 * elements,
        fpu_add_ops=3 * elements,
        sfpu_silu_ops=elements,
        dram_bytes=_dram_bytes((input_tensor, history, *taps), outputs),
    )
    return _performance(
        work,
        measured_ns=measured_ns,
        core_count=core_count,
        math_fidelity=math_fidelity,
    )


def reduce_affine_transforms_performance(
    a: Any,
    b: Any,
    outputs: Sequence[Any],
    *,
    measured_ns: float,
    core_count: int,
    math_fidelity: Any,
) -> KdaPerformance:
    if len(outputs) != 2:
        raise ValueError("affine-transform reduction requires two outputs")
    tensors = (a, b, *outputs)
    if any(len(tensor.shape) != 3 for tensor in tensors):
        raise ValueError("affine-transform reduction tensor shapes are inconsistent")
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in tensors):
        raise ValueError("affine-transform reduction tensor shapes must be positive")

    batch_heads, key_dim, output_key_dim = outputs[0].shape
    value_dim = outputs[1].shape[-1]
    if key_dim != output_key_dim or outputs[1].shape != (batch_heads, key_dim, value_dim) or a.shape[0] % batch_heads:
        raise ValueError("affine-transform reduction tensor shapes are inconsistent")
    groups_per_head = a.shape[0] // batch_heads
    if a.shape != (batch_heads * groups_per_head, key_dim, key_dim) or b.shape != (
        a.shape[0],
        key_dim,
        value_dim,
    ):
        raise ValueError("affine-transform reduction tensor shapes are inconsistent")

    compositions = batch_heads * (groups_per_head - 1)
    work = KdaWork(
        fpu_matrix_flops=compositions * (2 * key_dim**3 + 2 * key_dim**2 * value_dim),
        fpu_add_ops=compositions * key_dim * value_dim,
        dram_bytes=_dram_bytes((a, b), outputs),
    )
    return _performance(
        work,
        measured_ns=measured_ns,
        core_count=core_count,
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
    math_fidelity: Any,
) -> KdaPerformance:
    tensors = (a, b, initial_state, output)
    if any(len(tensor.shape) != 3 for tensor in tensors):
        raise ValueError("affine exclusive-scan tensor shapes are inconsistent")
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in tensors):
        raise ValueError("affine exclusive-scan tensor shapes must be positive")

    batch_heads, key_dim, value_dim = initial_state.shape
    if a.shape[0] % batch_heads:
        raise ValueError("affine exclusive-scan tensor shapes are inconsistent")
    groups_per_head = a.shape[0] // batch_heads
    if (
        a.shape != (batch_heads * groups_per_head, key_dim, key_dim)
        or b.shape != (a.shape[0], key_dim, value_dim)
        or output.shape != b.shape
    ):
        raise ValueError("affine exclusive-scan tensor shapes are inconsistent")

    transitions = batch_heads * (groups_per_head - 1)
    work = KdaWork(
        fpu_matrix_flops=transitions * 2 * key_dim**2 * value_dim,
        fpu_add_ops=transitions * key_dim * value_dim,
        dram_bytes=_dram_bytes((a, b, initial_state), (output,)),
    )
    return _performance(
        work,
        measured_ns=measured_ns,
        core_count=core_count,
        math_fidelity=math_fidelity,
    )


def prepare_chunk_recurrence_performance(
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    *,
    measured_ns: float,
    core_count: int,
    math_fidelity: Any,
) -> KdaPerformance:
    if len(inputs) != 5 or len(outputs) != 7:
        raise ValueError("chunk-recurrence preparation requires five inputs and seven outputs")
    tensors = (*inputs, *outputs)
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in tensors):
        raise ValueError("chunk-recurrence preparation tensor shapes must be positive")

    q, k, v, g, beta = inputs
    if len(q.shape) != 3 or len(v.shape) != 3 or len(beta.shape) != 4:
        raise ValueError("chunk-recurrence preparation tensor shapes are inconsistent")
    num_heads, num_chunks, chunk_size, trailing = beta.shape
    if chunk_size != CHUNK_SIZE or trailing != 1 or q.shape[-1] % num_heads or v.shape[-1] % num_heads:
        raise ValueError("chunk-recurrence preparation tensor shapes are inconsistent")
    key_dim = q.shape[-1] // num_heads
    value_dim = v.shape[-1] // num_heads
    if (
        q.shape != (1, num_chunks * CHUNK_SIZE, num_heads * key_dim)
        or k.shape != q.shape
        or g.shape != q.shape
        or v.shape != (1, num_chunks * CHUNK_SIZE, num_heads * value_dim)
    ):
        raise ValueError("chunk-recurrence preparation tensor shapes are inconsistent")
    expected_output_shapes = (
        (num_heads, num_chunks, CHUNK_SIZE, value_dim),
        (num_heads, num_chunks, CHUNK_SIZE, key_dim),
        (num_heads, num_chunks, CHUNK_SIZE, key_dim),
        (num_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
        (num_heads, num_chunks, key_dim, CHUNK_SIZE),
        (num_heads, num_chunks, key_dim, 1),
        (num_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
    )
    if any(output.shape != expected for output, expected in zip(outputs, expected_output_shapes, strict=True)):
        raise ValueError("chunk-recurrence preparation tensor shapes are inconsistent")

    instances = num_heads * num_chunks
    inverse_flops = CHUNK_SIZE * (CHUNK_SIZE - 1) * (CHUNK_SIZE + 1) // 3
    work = KdaWork(
        fpu_matrix_flops=instances * (4 * CHUNK_SIZE**2 * key_dim + inverse_flops),
        fpu_multiply_ops=instances * (10 * CHUNK_SIZE * key_dim + CHUNK_SIZE * value_dim),
        fpu_add_ops=instances * (2 * CHUNK_SIZE + (CHUNK_SIZE - 1) * key_dim + CHUNK_SIZE * key_dim + CHUNK_SIZE**2),
        fpu_reduction_ops=instances * 2 * CHUNK_SIZE * (key_dim - 1),
        sfpu_exp_ops=instances * (3 * CHUNK_SIZE * key_dim + key_dim),
        sfpu_rsqrt_ops=instances * 2 * CHUNK_SIZE,
        dram_bytes=_dram_bytes(inputs, outputs),
    )
    return _performance(
        work,
        measured_ns=measured_ns,
        core_count=core_count,
        math_fidelity=math_fidelity,
    )


def recurrent_chunk_scan_performance(
    inputs: Sequence[Any],
    state: Any,
    outputs: Sequence[Any],
    *,
    measured_ns: float,
    core_count: int,
    math_fidelity: Any,
) -> KdaPerformance:
    if len(inputs) != 7 or len(outputs) != 2:
        raise ValueError("recurrent chunk scan requires seven protocol inputs and two outputs")
    tensors = (*inputs, state, *outputs)
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in tensors):
        raise ValueError("recurrent chunk-scan tensor shapes must be positive")
    if any(len(tensor.shape) != 4 for tensor in inputs) or len(state.shape) != 3:
        raise ValueError("recurrent chunk-scan tensor shapes are inconsistent")

    batch_heads, num_chunks, chunk_size, value_dim = inputs[0].shape
    key_dim = inputs[1].shape[-1]
    expected_input_shapes = (
        (batch_heads, num_chunks, CHUNK_SIZE, value_dim),
        (batch_heads, num_chunks, CHUNK_SIZE, key_dim),
        (batch_heads, num_chunks, CHUNK_SIZE, key_dim),
        (batch_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
        (batch_heads, num_chunks, key_dim, CHUNK_SIZE),
        (batch_heads, num_chunks, key_dim, 1),
        (batch_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
    )
    if (
        chunk_size != CHUNK_SIZE
        or any(tensor.shape != expected for tensor, expected in zip(inputs, expected_input_shapes, strict=True))
        or state.shape != (batch_heads, key_dim, value_dim)
        or outputs[0].shape != (batch_heads, num_chunks, CHUNK_SIZE, value_dim)
        or outputs[1].shape != state.shape
    ):
        raise ValueError("recurrent chunk-scan tensor shapes are inconsistent")

    instances = batch_heads * num_chunks
    work = KdaWork(
        fpu_matrix_flops=instances * (6 * CHUNK_SIZE * key_dim * value_dim + 4 * CHUNK_SIZE**2 * value_dim),
        fpu_multiply_ops=instances * key_dim * value_dim,
        fpu_add_ops=instances * (2 * CHUNK_SIZE * value_dim + key_dim * value_dim),
        dram_bytes=_dram_bytes((*inputs, state), outputs),
    )
    return _performance(
        work,
        measured_ns=measured_ns,
        core_count=core_count,
        math_fidelity=math_fidelity,
    )


def summarize_chunk_recurrence_performance(
    inputs: Sequence[Any],
    outputs: Sequence[Any],
    *,
    measured_ns: float,
    core_count: int,
    math_fidelity: Any,
) -> KdaPerformance:
    if len(inputs) != 7 or len(outputs) != 2:
        raise ValueError("chunk-recurrence summary requires seven inputs and two outputs")
    tensors = (*inputs, *outputs)
    if any(any(dimension <= 0 for dimension in tensor.shape) for tensor in tensors):
        raise ValueError("chunk-recurrence summary tensor shapes must be positive")
    if any(len(tensor.shape) != 4 for tensor in inputs):
        raise ValueError("chunk-recurrence summary tensor shapes are inconsistent")

    batch_heads, num_chunks, chunk_size, value_dim = inputs[0].shape
    key_dim = inputs[1].shape[-1]
    expected_input_shapes = (
        (batch_heads, num_chunks, CHUNK_SIZE, value_dim),
        (batch_heads, num_chunks, CHUNK_SIZE, key_dim),
        (batch_heads, num_chunks, CHUNK_SIZE, key_dim),
        (batch_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
        (batch_heads, num_chunks, key_dim, CHUNK_SIZE),
        (batch_heads, num_chunks, key_dim, 1),
        (batch_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
    )
    if (
        chunk_size != CHUNK_SIZE
        or key_dim != value_dim
        or any(tensor.shape != expected for tensor, expected in zip(inputs, expected_input_shapes, strict=True))
        or outputs[0].shape != (batch_heads, key_dim, key_dim)
        or outputs[1].shape != (batch_heads, key_dim, value_dim)
    ):
        raise ValueError("chunk-recurrence summary tensor shapes are inconsistent")

    instances = batch_heads * num_chunks
    work = KdaWork(
        fpu_matrix_flops=instances * (8 * CHUNK_SIZE * key_dim * value_dim + 4 * CHUNK_SIZE**2 * value_dim),
        fpu_multiply_ops=instances * 2 * key_dim * value_dim,
        fpu_add_ops=instances * (2 * CHUNK_SIZE * value_dim + 2 * key_dim * value_dim)
        + batch_heads * key_dim * value_dim,
        dram_bytes=_dram_bytes(inputs, outputs),
    )
    return _performance(
        work,
        measured_ns=measured_ns,
        core_count=core_count,
        math_fidelity=math_fidelity,
    )
