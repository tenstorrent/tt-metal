# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA sigmoid-gated RMSNorm."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda.ops import sigmoid_gated_rms_norm_reference
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_bit_identical

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True),
]

_BATCH = 1
_SEQUENCE = 64
_NUM_HEADS = 12
_VALUE_DIM = 128
_EPSILON = 1e-5


def _torch_dtype(dtype: ttnn.DataType) -> torch.dtype:
    return torch.float32 if dtype == ttnn.float32 else torch.bfloat16


def _host_inputs(
    *,
    batch: int = _BATCH,
    sequence: int = _SEQUENCE,
    num_heads: int = _NUM_HEADS,
    value_dim: int = _VALUE_DIM,
    input_dtype: ttnn.DataType = ttnn.float32,
    seed: int = 319,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(
        batch * num_heads,
        sequence,
        value_dim,
        generator=generator,
        dtype=_torch_dtype(input_dtype),
    )
    gate = torch.randn(batch, sequence, num_heads * value_dim, generator=generator, dtype=torch.bfloat16)
    weight = torch.randn(value_dim, generator=generator, dtype=torch.bfloat16)
    return inputs, gate, weight


def _to_device(
    tensor: torch.Tensor,
    device: ttnn.Device,
    *,
    dtype: ttnn.DataType,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _device_inputs(
    device: ttnn.Device,
    *,
    batch: int = _BATCH,
    sequence: int = _SEQUENCE,
    num_heads: int = _NUM_HEADS,
    value_dim: int = _VALUE_DIM,
    input_dtype: ttnn.DataType = ttnn.float32,
    seed: int = 319,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]]:
    host = _host_inputs(
        batch=batch,
        sequence=sequence,
        num_heads=num_heads,
        value_dim=value_dim,
        input_dtype=input_dtype,
        seed=seed,
    )
    inputs, gate, weight = host
    device_tensors = (
        _to_device(inputs, device, dtype=input_dtype),
        _to_device(gate, device, dtype=ttnn.bfloat16),
        _to_device(weight, device, dtype=ttnn.bfloat16),
    )
    return host, device_tensors


def _run(
    input_tt: ttnn.Tensor,
    gate_tt: ttnn.Tensor,
    weight_tt: ttnn.Tensor,
    *,
    num_heads: int = _NUM_HEADS,
    epsilon: float = _EPSILON,
    memory_config: ttnn.MemoryConfig | None = None,
    output_dtype: ttnn.DataType = ttnn.float32,
) -> ttnn.Tensor:
    return ttnn.experimental.kda.sigmoid_gated_rms_norm(
        input_tt,
        gate_tt,
        weight_tt,
        num_heads,
        epsilon=epsilon,
        memory_config=memory_config,
        output_dtype=output_dtype,
    )


@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("output_dtype", [ttnn.float32, ttnn.bfloat16])
def test_sigmoid_gated_rms_norm_production_contract(
    device: ttnn.Device, input_dtype: ttnn.DataType, output_dtype: ttnn.DataType
) -> None:
    """Validate production geometry, numerics, cache reuse, trace, residency, and ownership."""
    host, device_tensors = _device_inputs(device, input_dtype=input_dtype)
    inputs, gate, weight = host
    input_tt, gate_tt, weight_tt = device_tensors
    expected = sigmoid_gated_rms_norm_reference(
        inputs.reshape(_BATCH, _NUM_HEADS, _SEQUENCE, _VALUE_DIM).permute(0, 2, 1, 3),
        gate.reshape(_BATCH, _SEQUENCE, _NUM_HEADS, _VALUE_DIM),
        weight,
        eps=_EPSILON,
    ).reshape(_BATCH, _SEQUENCE, _NUM_HEADS * _VALUE_DIM)
    expected = expected.to(_torch_dtype(output_dtype))

    input_snapshots = tuple(ttnn.to_torch(tensor).clone() for tensor in device_tensors)

    def run() -> ttnn.Tensor:
        with ttnn.manage_config("throw_exception_on_fallback", True):
            return _run(input_tt, gate_tt, weight_tt, output_dtype=output_dtype)

    output_tt = run()
    assert output_tt.dtype == output_dtype
    assert output_tt.layout == ttnn.TILE_LAYOUT
    assert output_tt.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert tuple(ttnn.to_torch(output_tt).shape) == (_BATCH, _SEQUENCE, _NUM_HEADS * _VALUE_DIM)
    assert all(output_tt.buffer_address() != tensor.buffer_address() for tensor in device_tensors)

    cache_entries = device.num_program_cache_entries()
    repeated_tt = run()
    ttnn.synchronize_device(device)
    assert device.num_program_cache_entries() == cache_entries

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced_tt = run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for _ in range(2):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    eager = ttnn.to_torch(output_tt)
    repeated = ttnn.to_torch(repeated_tt)
    traced = ttnn.to_torch(traced_tt)
    assert_accurate(expected, eager, name="eager", pcc_threshold=0.999)
    assert_bit_identical(eager, repeated, name="eager repeat")
    assert_bit_identical(eager, traced, name="trace replay")

    for name, before, tensor in zip(("input", "gate", "weight"), input_snapshots, device_tensors, strict=True):
        assert_bit_identical(before, ttnn.to_torch(tensor), name=f"{name} immutability")

    ttnn.release_trace(device, trace_id)


def test_sigmoid_gated_rms_norm_program_key_includes_epsilon(device: ttnn.Device) -> None:
    _, (input_tt, gate_tt, weight_tt) = _device_inputs(
        device, batch=1, sequence=32, num_heads=2, value_dim=64, seed=1321
    )
    _run(input_tt, gate_tt, weight_tt, num_heads=2, epsilon=1e-5)
    entries = device.num_program_cache_entries()
    _run(input_tt, gate_tt, weight_tt, num_heads=2, epsilon=2e-5)
    assert device.num_program_cache_entries() == entries + 1
    _run(input_tt, gate_tt, weight_tt, num_heads=2, epsilon=2e-5)
    assert device.num_program_cache_entries() == entries + 1


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("host_input", "allocated device tensor"),
        ("input_dtype", "input has unsupported dtype"),
        ("gate_dtype", "gate has unsupported dtype"),
        ("weight_dtype", "weight has unsupported dtype"),
        ("input_layout", "input must use TILE layout"),
        ("gate_shape", "gate must have shape"),
        ("weight_shape", r"weight must be \[V\]"),
        ("sequence_alignment", "sequence must be positive and tile aligned"),
        ("value_alignment", "value_dim must be positive and tile aligned"),
        ("sharded_input", "input must use interleaved memory"),
    ],
)
def test_sigmoid_gated_rms_norm_rejects_invalid_tensors(
    device: ttnn.Device, expect_error: Callable, case: str, message: str
) -> None:
    input_dtype = ttnn.float32
    sequence = 33 if case == "sequence_alignment" else 32
    value_dim = 100 if case == "value_alignment" else 128
    host = _host_inputs(sequence=sequence, value_dim=value_dim, input_dtype=input_dtype, seed=9321)
    inputs, gate, weight = host
    input_tt = _to_device(inputs, device, dtype=input_dtype)
    gate_tt = _to_device(gate, device, dtype=ttnn.bfloat16)
    weight_tt = _to_device(weight, device, dtype=ttnn.bfloat16)

    if case == "host_input":
        input_tt = ttnn.from_torch(inputs, dtype=input_dtype, layout=ttnn.TILE_LAYOUT)
    elif case == "input_dtype":
        input_tt = _to_device(inputs, device, dtype=ttnn.bfloat8_b)
    elif case == "gate_dtype":
        gate_tt = _to_device(gate.float(), device, dtype=ttnn.float32)
    elif case == "weight_dtype":
        weight_tt = _to_device(weight.float(), device, dtype=ttnn.float32)
    elif case == "input_layout":
        input_tt = _to_device(inputs, device, dtype=input_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "gate_shape":
        bad_gate = gate[..., :-32]
        gate_tt = _to_device(bad_gate, device, dtype=ttnn.bfloat16)
    elif case == "weight_shape":
        weight_tt = _to_device(weight.reshape(2, -1), device, dtype=ttnn.bfloat16)
    elif case == "sharded_input":
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [inputs.shape[0] * inputs.shape[1], inputs.shape[2]],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        input_tt = _to_device(inputs, device, dtype=input_dtype, memory_config=sharded_config)

    with expect_error(RuntimeError, message):
        _run(input_tt, gate_tt, weight_tt)


@pytest.mark.parametrize(
    ("num_heads", "epsilon", "output_dtype", "message"),
    [
        (0, 1e-5, ttnn.float32, "num_heads must be positive"),
        (5, 1e-5, ttnn.float32, "leading dimension must be divisible"),
        (12, 0.0, ttnn.float32, "epsilon must be finite and positive"),
        (12, float("nan"), ttnn.float32, "epsilon must be finite and positive"),
        (12, 1e-5, ttnn.uint32, "output_dtype must be FLOAT32 or BFLOAT16"),
    ],
)
def test_sigmoid_gated_rms_norm_rejects_invalid_options(
    device: ttnn.Device,
    expect_error: Callable,
    num_heads: int,
    epsilon: float,
    output_dtype: ttnn.DataType,
    message: str,
) -> None:
    _, (input_tt, gate_tt, weight_tt) = _device_inputs(device)
    with expect_error(RuntimeError, message):
        _run(input_tt, gate_tt, weight_tt, num_heads=num_heads, epsilon=epsilon, output_dtype=output_dtype)


def test_sigmoid_gated_rms_norm_rejects_sharded_output(device: ttnn.Device, expect_error: Callable) -> None:
    _, (input_tt, gate_tt, weight_tt) = _device_inputs(device)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [_SEQUENCE, _NUM_HEADS * _VALUE_DIM],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    with expect_error(RuntimeError, "output memory configuration must be interleaved"):
        _run(input_tt, gate_tt, weight_tt, memory_config=sharded_config)
