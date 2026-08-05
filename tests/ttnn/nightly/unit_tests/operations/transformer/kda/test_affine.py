# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Direct Blackhole coverage for KDA affine composition and prefix leaves."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.kda.utils import assert_accurate, assert_bit_identical

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 2_000_000}], indirect=True),
]


def _oracles(
    transform_a: torch.Tensor,
    transform_b: torch.Tensor,
    initial_state: torch.Tensor,
    batch_heads: int,
    groups_per_head: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key_dim, value_dim = transform_a.shape[-1], transform_b.shape[-1]
    transform_a = transform_a.reshape(batch_heads, groups_per_head, key_dim, key_dim).float()
    transform_b = transform_b.reshape(batch_heads, groups_per_head, key_dim, value_dim).float()
    entries = []
    composed_a = []
    composed_b = []
    for head in range(batch_heads):
        carry = initial_state[head].float()
        prefix_a = torch.eye(key_dim)
        prefix_b = torch.zeros(key_dim, value_dim)
        for group in range(groups_per_head):
            entries.append(carry)
            prefix_a = transform_a[head, group] @ prefix_a
            prefix_b = transform_a[head, group] @ prefix_b + transform_b[head, group]
            carry = transform_a[head, group] @ carry + transform_b[head, group]
        composed_a.append(prefix_a)
        composed_b.append(prefix_b)
    return torch.stack(composed_a), torch.stack(composed_b), torch.stack(entries)


def _to_device(tensor: torch.Tensor, device: ttnn.Device, dtype: ttnn.DataType = ttnn.float32) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _run_affine_leaves(
    transform_a: ttnn.Tensor,
    transform_b: ttnn.Tensor,
    initial_state: ttnn.Tensor,
    groups_per_head: int,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    composed_a, composed_b = ttnn.transformer.kda_affine_compose(
        transform_a,
        transform_b,
        groups_per_head,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    entries = ttnn.transformer.kda_affine_prefix(
        transform_a,
        transform_b,
        initial_state,
        groups_per_head,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return composed_a, composed_b, entries


@pytest.mark.parametrize("summary_dtype", [ttnn.float32, ttnn.bfloat16])
def test_kda_affine_leaves_correct_cache_trace_and_deterministic(
    device: ttnn.Device,
    summary_dtype: ttnn.DataType,
) -> None:
    batch_heads, groups_per_head, key_dim, value_dim = 2, 4, 32, 32
    generator = torch.Generator().manual_seed(914)
    eye = torch.eye(key_dim).reshape(1, key_dim, key_dim)
    transform_a = (0.96 * eye).expand(batch_heads * groups_per_head, -1, -1).clone()
    transform_a += 0.001 * torch.randn(transform_a.shape, generator=generator)
    transform_b = 0.01 * torch.randn(batch_heads * groups_per_head, key_dim, value_dim, generator=generator)
    initial_state = 0.01 * torch.randn(batch_heads, key_dim, value_dim, generator=generator)
    expected_a, expected_b, expected_entries = _oracles(
        transform_a, transform_b, initial_state, batch_heads, groups_per_head
    )

    a_tt = _to_device(transform_a, device, summary_dtype)
    b_tt = _to_device(transform_b, device, summary_dtype)
    state_tt = _to_device(initial_state, device)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        first = _run_affine_leaves(a_tt, b_tt, state_tt, groups_per_head)
    cache_entries = device.num_program_cache_entries()
    with ttnn.manage_config("throw_exception_on_fallback", True):
        repeated = _run_affine_leaves(a_tt, b_tt, state_tt, groups_per_head)
    ttnn.synchronize_device(device)
    assert device.num_program_cache_entries() == cache_entries

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        traced = _run_affine_leaves(a_tt, b_tt, state_tt, groups_per_head)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)

    expected = (expected_a, expected_b, expected_entries)
    first_host = tuple(ttnn.to_torch(tensor) for tensor in first)
    repeated_host = tuple(ttnn.to_torch(tensor) for tensor in repeated)
    traced_host = tuple(ttnn.to_torch(tensor) for tensor in traced)
    ttnn.release_trace(device, trace_id)
    for name, golden, actual, repeated_actual, traced_actual in zip(
        ("composed A", "composed B", "prefix entries"), expected, first_host, repeated_host, traced_host
    ):
        assert_accurate(golden, actual, name=f"{summary_dtype} {name}")
        assert_bit_identical(actual, repeated_actual, name=f"{summary_dtype} repeated {name}")
        assert_bit_identical(actual, traced_actual, name=f"{summary_dtype} traced {name}")


def test_kda_affine_prefix_rejects_bf16_state(device: ttnn.Device, expect_error) -> None:
    groups_per_head = 2
    transform = _to_device(torch.eye(32).repeat(groups_per_head, 1, 1), device)
    state = _to_device(torch.zeros(1, 32, 32), device, ttnn.bfloat16)
    with expect_error(RuntimeError, "initial_state must be device FLOAT32 TILE"):
        ttnn.transformer.kda_affine_prefix(transform, transform, state, groups_per_head)
