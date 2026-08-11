# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Direct contract coverage for experimental KDA chunk-recurrence preparation."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_bit_identical

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 2_000_000}], indirect=True),
]

CHUNK_SIZE = 32
OUTPUT_NAMES = ("v_beta", "kd", "q_decay", "intra", "k_dec_t", "final_decay", "t_inv")


def _host_inputs(
    num_heads: int,
    num_chunks: int,
    key_dim: int,
    value_dim: int,
    *,
    seed: int = 1731,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    sequence = num_chunks * CHUNK_SIZE
    q = (0.3 * torch.randn(1, sequence, num_heads * key_dim, generator=generator)).to(torch.bfloat16).float()
    k = (0.3 * torch.randn(1, sequence, num_heads * key_dim, generator=generator)).to(torch.bfloat16).float()
    v = (0.2 * torch.randn(1, sequence, num_heads * value_dim, generator=generator)).to(torch.bfloat16).float()
    g = (-0.001 - 0.05 * torch.rand(1, sequence, num_heads * key_dim, generator=generator)).to(torch.bfloat16).float()
    beta = torch.sigmoid(torch.randn(num_heads, num_chunks, CHUNK_SIZE, 1, generator=generator)).float()
    eye = torch.eye(CHUNK_SIZE, dtype=torch.float32).reshape(1, 1, CHUNK_SIZE, CHUNK_SIZE)
    tril = torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32)).reshape(1, 1, CHUNK_SIZE, CHUNK_SIZE)
    ones = torch.ones(1, 1, CHUNK_SIZE, CHUNK_SIZE, dtype=torch.float32)
    indices = torch.arange(CHUNK_SIZE)
    rows, columns = indices[:, None], indices[None, :]
    lower_rows, lower_columns = rows < CHUNK_SIZE // 2, columns < CHUNK_SIZE // 2
    masks = torch.cat(
        [
            (lower_rows & lower_columns).float(),
            (~lower_rows & ~lower_columns).float(),
            (~lower_rows & lower_columns).float(),
        ],
        dim=1,
    ).reshape(1, 1, CHUNK_SIZE, 3 * CHUNK_SIZE)
    return q, k, v, g, beta, eye, tril, ones, masks


def _reshape_flat(tensor: torch.Tensor, num_heads: int, num_chunks: int, dim: int) -> torch.Tensor:
    return (
        tensor.float()
        .reshape(num_chunks * CHUNK_SIZE, num_heads, dim)
        .permute(1, 0, 2)
        .reshape(num_heads, num_chunks, CHUNK_SIZE, dim)
    )


def _oracle(
    inputs: tuple[torch.Tensor, ...],
    num_heads: int,
    output_bf16_mask: int,
) -> tuple[torch.Tensor, ...]:
    q, k, v, g, beta, *_ = inputs
    num_chunks = beta.shape[1]
    key_dim = q.shape[-1] // num_heads
    value_dim = v.shape[-1] // num_heads
    q = _reshape_flat(q, num_heads, num_chunks, key_dim)
    k = _reshape_flat(k, num_heads, num_chunks, key_dim)
    v = _reshape_flat(v, num_heads, num_chunks, value_dim)
    g = _reshape_flat(g, num_heads, num_chunks, key_dim)
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) + 1e-6) * (key_dim**-0.5)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) + 1e-6)
    cumulative_g = torch.cumsum(g, dim=2)
    decay = torch.exp(cumulative_g)
    inverse_decay = torch.exp(-cumulative_g)
    final_g = cumulative_g[:, :, -1]

    v_beta = beta * v
    kd = beta * k * decay
    q_decay = q * decay
    intra = torch.matmul(q_decay, (k * inverse_decay).transpose(-1, -2)).tril()
    k_dec_t = (k * torch.exp(final_g.unsqueeze(2) - cumulative_g)).transpose(-1, -2)
    final_decay = torch.exp(final_g).unsqueeze(-1)
    akk = torch.matmul(beta * k * decay, (k * inverse_decay).transpose(-1, -2))
    identity = torch.eye(CHUNK_SIZE, dtype=torch.float32).reshape(1, 1, CHUNK_SIZE, CHUNK_SIZE)
    t_inv = torch.linalg.inv(identity + torch.tril(akk, diagonal=-1))
    outputs = (v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv)
    return tuple(
        output.to(torch.bfloat16) if output_bf16_mask & (1 << index) else output.float()
        for index, output in enumerate(outputs)
    )


def _to_device(
    tensor: torch.Tensor,
    device: ttnn.Device,
    dtype: ttnn.DataType,
    *,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _device_inputs(inputs: tuple[torch.Tensor, ...], device: ttnn.Device) -> tuple[ttnn.Tensor, ...]:
    return tuple(
        _to_device(tensor, device, ttnn.bfloat16 if index < 4 else ttnn.float32) for index, tensor in enumerate(inputs)
    )


def _run(
    inputs: tuple[ttnn.Tensor, ...],
    num_heads: int,
    *,
    output_bf16_mask: int = 0,
    memory_config: ttnn.MemoryConfig | None = None,
) -> list[ttnn.Tensor]:
    with ttnn.manage_config("throw_exception_on_fallback", True):
        return ttnn.experimental.kda.prepare_chunk_recurrence(
            *inputs,
            num_heads,
            output_bf16_mask=output_bf16_mask,
            memory_config=memory_config,
        )


@pytest.mark.parametrize(
    ("num_heads", "num_chunks", "key_dim", "value_dim", "output_bf16_mask"),
    [(2, 2, 32, 32, 0), (2, 2, 32, 64, 0x26), (3, 1, 64, 32, 0x11)],
)
def test_prepare_chunk_recurrence_contract_cache_trace_and_determinism(
    device: ttnn.Device,
    num_heads: int,
    num_chunks: int,
    key_dim: int,
    value_dim: int,
    output_bf16_mask: int,
) -> None:
    host_inputs = _host_inputs(num_heads, num_chunks, key_dim, value_dim)
    expected = _oracle(host_inputs, num_heads, output_bf16_mask)
    inputs = _device_inputs(host_inputs, device)
    snapshots = tuple(ttnn.to_torch(tensor).clone() for tensor in inputs)

    first = _run(inputs, num_heads, output_bf16_mask=output_bf16_mask)
    assert len(first) == 7
    expected_shapes = (
        (num_heads, num_chunks, CHUNK_SIZE, value_dim),
        (num_heads, num_chunks, CHUNK_SIZE, key_dim),
        (num_heads, num_chunks, CHUNK_SIZE, key_dim),
        (num_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
        (num_heads, num_chunks, key_dim, CHUNK_SIZE),
        (num_heads, num_chunks, key_dim, 1),
        (num_heads, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
    )
    input_addresses = {tensor.buffer_address() for tensor in inputs}
    output_addresses = set()
    for index, (output, shape) in enumerate(zip(first, expected_shapes, strict=True)):
        expected_dtype = ttnn.bfloat16 if output_bf16_mask & (1 << index) else ttnn.float32
        assert output.dtype == expected_dtype
        assert output.layout == ttnn.TILE_LAYOUT
        assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
        assert tuple(output.shape) == shape
        assert output.buffer_address() not in input_addresses
        output_addresses.add(output.buffer_address())
    assert len(output_addresses) == 7

    cache_entries = device.num_program_cache_entries()
    repeated = _run(inputs, num_heads, output_bf16_mask=output_bf16_mask)
    ttnn.synchronize_device(device)
    assert device.num_program_cache_entries() == cache_entries

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = _run(inputs, num_heads, output_bf16_mask=output_bf16_mask)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for _ in range(2):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    for name, expected_output, first_tt, repeated_tt, traced_tt in zip(
        OUTPUT_NAMES, expected, first, repeated, traced, strict=True
    ):
        actual = ttnn.to_torch(first_tt)
        assert_accurate(expected_output, actual, name=name, pcc_threshold=0.999)
        assert_bit_identical(actual, ttnn.to_torch(repeated_tt), name=f"{name} eager repeat")
        assert_bit_identical(actual, ttnn.to_torch(traced_tt), name=f"{name} trace replay")
    for index, (snapshot, tensor) in enumerate(zip(snapshots, inputs, strict=True)):
        assert_bit_identical(snapshot, ttnn.to_torch(tensor), name=f"input {index} immutability")
    ttnn.release_trace(device, trace_id)


@pytest.mark.parametrize("host_index", range(9))
def test_prepare_chunk_recurrence_rejects_host_inputs(
    device: ttnn.Device,
    expect_error: Callable,
    host_index: int,
) -> None:
    host_inputs = _host_inputs(2, 2, 32, 32)
    inputs = list(_device_inputs(host_inputs, device))
    dtype = ttnn.bfloat16 if host_index < 4 else ttnn.float32
    inputs[host_index] = ttnn.from_torch(host_inputs[host_index], dtype=dtype, layout=ttnn.TILE_LAYOUT)
    with expect_error(
        RuntimeError,
        f"{('q', 'k', 'v', 'g', 'beta', 'eye', 'tril', 'ones', 'masks')[host_index]} must be an allocated device tensor",
    ):
        _run(tuple(inputs), 2)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("g_dtype", "g has wrong dtype"),
        ("layout", "k must use TILE layout"),
        ("rank", "rank 3 production-flat"),
        ("leading", "leading dimension 1"),
        ("qkg_shape", "q, k, and g must have matching shapes"),
        ("sequence", "matching sequence lengths"),
        ("sequence_chunk", "sequence length must be positive and divisible by 32"),
        ("head_divisibility", "flat widths must be divisible by num_heads"),
        ("key_alignment", "K and V must be positive and tile aligned"),
        ("value_alignment", "K and V must be positive and tile aligned"),
        ("beta", "beta shape must be"),
        ("eye", "eye shape must be"),
        ("masks", "masks shape must be"),
        ("sharded", "q must use interleaved memory"),
    ],
)
def test_prepare_chunk_recurrence_rejects_invalid_inputs(
    device: ttnn.Device,
    expect_error: Callable,
    case: str,
    message: str,
) -> None:
    host_inputs = list(_host_inputs(2, 2, 32, 32))
    inputs = list(_device_inputs(tuple(host_inputs), device))
    num_heads = 2
    if case == "g_dtype":
        inputs[3] = _to_device(host_inputs[3], device, ttnn.float32)
    elif case == "layout":
        inputs[1] = _to_device(host_inputs[1], device, ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    elif case == "rank":
        inputs[1] = _to_device(host_inputs[1].reshape(1, 2, 32, 64), device, ttnn.bfloat16)
    elif case == "leading":
        inputs[0] = _to_device(host_inputs[0].expand(2, -1, -1).clone(), device, ttnn.bfloat16)
    elif case == "qkg_shape":
        inputs[3] = _to_device(host_inputs[3][:, :, :32], device, ttnn.bfloat16)
    elif case == "sequence":
        inputs[2] = _to_device(host_inputs[2][:, :32], device, ttnn.bfloat16)
    elif case == "sequence_chunk":
        for index in range(4):
            inputs[index] = _to_device(host_inputs[index][:, :48], device, ttnn.bfloat16)
    elif case == "head_divisibility":
        num_heads = 3
    elif case == "key_alignment":
        num_heads = 2
        for index in (0, 1, 3):
            tensor = torch.randn(1, 64, 96).to(torch.bfloat16).float()
            inputs[index] = _to_device(tensor, device, ttnn.bfloat16)
    elif case == "value_alignment":
        inputs[2] = _to_device(torch.randn(1, 64, 96).to(torch.bfloat16).float(), device, ttnn.bfloat16)
    elif case == "beta":
        inputs[4] = _to_device(host_inputs[4][:, :1], device, ttnn.float32)
    elif case == "eye":
        inputs[5] = _to_device(torch.eye(64).reshape(1, 1, 64, 64), device, ttnn.float32)
    elif case == "masks":
        inputs[8] = _to_device(host_inputs[8][:, :, :, :64], device, ttnn.float32)
    elif case == "sharded":
        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
            [64, 64],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
        inputs[0] = _to_device(host_inputs[0], device, ttnn.bfloat16, memory_config=sharded)
    with expect_error(RuntimeError, message):
        _run(tuple(inputs), num_heads)


def test_prepare_chunk_recurrence_rejects_invalid_options(device: ttnn.Device, expect_error: Callable) -> None:
    host_inputs = _host_inputs(2, 2, 32, 32)
    inputs = _device_inputs(host_inputs, device)
    with expect_error(RuntimeError, "num_heads must be positive"):
        _run(inputs, 0)
    for mask in (0x08, 0x40):
        with expect_error(RuntimeError, "unsupported KDA prep BF16 mask"):
            _run(inputs, 2, output_bf16_mask=mask)

    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        [64, 64],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    with expect_error(RuntimeError, "output memory must be interleaved"):
        _run(inputs, 2, memory_config=sharded)


@pytest.mark.parametrize("removed_keyword", ["chunk_size", "v_flat", "normalize_qk", "scale"])
def test_prepare_chunk_recurrence_does_not_expose_prototype_modes(
    device: ttnn.Device,
    expect_error: Callable,
    removed_keyword: str,
) -> None:
    inputs = _device_inputs(_host_inputs(2, 2, 32, 32), device)
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.experimental.kda.prepare_chunk_recurrence(*inputs, 2, **{removed_keyword: True})
