# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent protocol generation and oracles for the KDA scan split."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence

import torch

import ttnn
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
)

CHUNK_SIZE = 32
PROTOCOL_NAMES = ("v_beta", "kd", "q_decay", "intra", "k_dec_t", "final_decay", "t_inv")
BF16_ALLOWED = frozenset({"v_beta", "kd", "q_decay", "k_dec_t", "final_decay"})


def host_protocol(
    batch_heads: int,
    num_chunks: int,
    key_dim: int,
    value_dim: int,
    *,
    bf16_names: Collection[str] = (),
    seed: int = 1731,
) -> tuple[torch.Tensor, ...]:
    """Generate a finite, stable seven-tensor protocol without invoking preparation."""
    unexpected = set(bf16_names) - BF16_ALLOWED
    if unexpected:
        raise ValueError(f"BF16 is not allowed for {sorted(unexpected)}")
    generator = torch.Generator().manual_seed(seed)
    base = (batch_heads, num_chunks)
    v_beta = 0.08 * torch.randn(*base, CHUNK_SIZE, value_dim, generator=generator)
    kd = 0.025 * torch.randn(*base, CHUNK_SIZE, key_dim, generator=generator)
    q_decay = 0.06 * torch.randn(*base, CHUNK_SIZE, key_dim, generator=generator)
    intra = torch.tril(0.025 * torch.randn(*base, CHUNK_SIZE, CHUNK_SIZE, generator=generator))
    k_dec_t = 0.025 * torch.randn(*base, key_dim, CHUNK_SIZE, generator=generator)
    final_decay = 0.86 + 0.08 * torch.rand(*base, key_dim, 1, generator=generator)
    strict_lower = torch.tril(0.015 * torch.randn(*base, CHUNK_SIZE, CHUNK_SIZE, generator=generator), diagonal=-1)
    identity = torch.eye(CHUNK_SIZE).reshape(1, 1, CHUNK_SIZE, CHUNK_SIZE)
    t_inv = torch.linalg.inv(identity + strict_lower)
    values = (v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv)
    return tuple(
        value.to(torch.bfloat16) if name in bf16_names else value.float()
        for name, value in zip(PROTOCOL_NAMES, values, strict=True)
    )


def initial_state(batch_heads: int, key_dim: int, value_dim: int, *, seed: int = 2718) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (0.04 * torch.randn(batch_heads, key_dim, value_dim, generator=generator)).float()


def _scan_state(protocol: Sequence[torch.Tensor], state: torch.Tensor) -> torch.Tensor:
    v_beta, kd, _, _, k_dec_t, final_decay, t_inv = (tensor.float() for tensor in protocol)
    state = state.float().clone()
    for chunk in range(v_beta.shape[1]):
        value_new = torch.matmul(t_inv[:, chunk], v_beta[:, chunk] - torch.matmul(kd[:, chunk], state))
        state = state * final_decay[:, chunk] + torch.matmul(k_dec_t[:, chunk], value_new)
    return state


def recurrent_oracle(protocol: Sequence[torch.Tensor], state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v_beta, kd, q_decay, intra, k_dec_t, final_decay, t_inv = (tensor.float() for tensor in protocol)
    state = state.float().clone()
    chunks = []
    for chunk in range(v_beta.shape[1]):
        value_new = torch.matmul(t_inv[:, chunk], v_beta[:, chunk] - torch.matmul(kd[:, chunk], state))
        output = torch.matmul(q_decay[:, chunk], state) + torch.matmul(intra[:, chunk], value_new)
        chunks.append(output)
        state = state * final_decay[:, chunk] + torch.matmul(k_dec_t[:, chunk], value_new)
    return torch.stack(chunks, dim=1).to(torch.bfloat16), state.float()


def summary_oracle(protocol: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    batch_heads, _, _, key_dim = protocol[1].shape
    value_dim = protocol[0].shape[-1]
    if key_dim != value_dim:
        raise ValueError("summary requires K == V")
    zero = torch.zeros(batch_heads, key_dim, value_dim)
    identity = torch.eye(key_dim).expand(batch_heads, -1, -1).clone()
    affine_b = _scan_state(protocol, zero)
    affine_a = _scan_state(protocol, identity) - affine_b
    return affine_a.float(), affine_b.float()


def assert_summary_reconstructs_state(
    protocol: Sequence[torch.Tensor], affine_a: torch.Tensor, affine_b: torch.Tensor
) -> None:
    batch_heads, key_dim, value_dim = affine_a.shape
    state = initial_state(batch_heads, key_dim, value_dim, seed=3141)
    expected = _scan_state(protocol, state)
    reconstructed = torch.matmul(affine_a.float(), state) + affine_b.float()
    assert_accurate(expected, reconstructed, name="semantic affine reconstruction", pcc_threshold=0.9999)


def to_device(
    tensor: torch.Tensor,
    device: ttnn.Device,
    *,
    dtype: ttnn.DataType | None = None,
    layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    resolved_dtype = dtype or (ttnn.bfloat16 if tensor.dtype == torch.bfloat16 else ttnn.float32)
    return ttnn.from_torch(tensor, dtype=resolved_dtype, layout=layout, device=device, memory_config=memory_config)


def device_protocol(protocol: Sequence[torch.Tensor], device: ttnn.Device) -> tuple[ttnn.Tensor, ...]:
    return tuple(to_device(tensor, device) for tensor in protocol)


def run_recurrent(
    protocol: Sequence[ttnn.Tensor],
    state: ttnn.Tensor,
    *,
    memory_config: ttnn.MemoryConfig | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
) -> list[ttnn.Tensor]:
    with ttnn.manage_config("throw_exception_on_fallback", True):
        return ttnn.experimental.kda.recurrent_chunk_scan(
            *protocol,
            state,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )


def run_summary(
    protocol: Sequence[ttnn.Tensor],
    *,
    memory_config: ttnn.MemoryConfig | None = None,
    compute_kernel_config: ttnn.DeviceComputeKernelConfig | None = None,
) -> list[ttnn.Tensor]:
    with ttnn.manage_config("throw_exception_on_fallback", True):
        return ttnn.experimental.kda.summarize_chunk_recurrence(
            *protocol,
            memory_config=memory_config,
            compute_kernel_config=compute_kernel_config,
        )


def assert_runtime_contract(
    device: ttnn.Device,
    inputs: Sequence[ttnn.Tensor],
    run: Callable[[], list[ttnn.Tensor]],
    expected: Sequence[torch.Tensor],
    *,
    names: Sequence[str],
    dtypes: Sequence[ttnn.DataType],
    shapes: Sequence[tuple[int, ...]],
    pcc_threshold: float = 0.999,
    expected_memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
) -> list[ttnn.Tensor]:
    snapshots = tuple(ttnn.to_torch(tensor).clone() for tensor in inputs)
    first = run()
    assert len(first) == len(expected)
    input_addresses = {tensor.buffer_address() for tensor in inputs}
    output_addresses = set()
    for output, dtype, shape in zip(first, dtypes, shapes, strict=True):
        assert output.dtype == dtype
        assert output.layout == ttnn.TILE_LAYOUT
        assert output.memory_config() == expected_memory_config
        assert tuple(output.shape) == shape
        assert output.buffer_address() not in input_addresses
        output_addresses.add(output.buffer_address())
    assert len(output_addresses) == len(first)

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    for _ in range(2):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    for name, golden, first_tt, traced_tt in zip(names, expected, first, traced, strict=True):
        actual = ttnn.to_torch(first_tt)
        assert_accurate(golden, actual, name=name, pcc_threshold=pcc_threshold)
        assert_bit_identical(actual, ttnn.to_torch(traced_tt), name=f"{name} trace replay")
    for index, (snapshot, tensor) in enumerate(zip(snapshots, inputs, strict=True)):
        assert_bit_identical(snapshot, ttnn.to_torch(tensor), name=f"input {index} immutability")
    ttnn.release_trace(device, trace_id)
    return first


def assert_outputs_accurate(
    expected: Sequence[torch.Tensor],
    actual: Sequence[ttnn.Tensor],
    *,
    names: Sequence[str],
    context: str,
    pcc_threshold: float = 0.999,
) -> None:
    for name, golden, actual_tt in zip(names, expected, actual, strict=True):
        assert_accurate(golden, ttnn.to_torch(actual_tt), name=f"{context} {name}", pcc_threshold=pcc_threshold)


def one_core_height_sharded(shape: tuple[int, int]) -> ttnn.MemoryConfig:
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        list(shape),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def group_summary_height_sharded(device: ttnn.Device, batch_heads: int, dim: int) -> ttnn.MemoryConfig:
    cores = ttnn.num_cores_to_corerangeset(
        batch_heads,
        device.compute_with_storage_grid_size(),
        row_wise=True,
    )
    return ttnn.create_sharded_memory_config(
        (batch_heads, dim, dim),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
