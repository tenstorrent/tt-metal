# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Direct device coverage for the KDA final chunk scan."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda.ops import (
    kda_recurrent_reference,
    l2_norm_reference,
)

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True),
]


def _to_device(tensor: torch.Tensor, device: ttnn.Device, dtype: ttnn.DataType) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _constants(device: ttnn.Device) -> tuple[ttnn.Tensor, ...]:
    size = 32
    eye = torch.eye(size).reshape(1, 1, size, size)
    tril = torch.tril(torch.ones(size, size)).reshape(1, 1, size, size)
    ones = torch.ones(1, 1, size, size)
    indices = torch.arange(size)
    rows, columns = indices[:, None], indices[None, :]
    lower_rows, lower_columns = rows < size // 2, columns < size // 2
    masks = torch.cat(
        [
            (lower_rows & lower_columns).float(),
            (~lower_rows & ~lower_columns).float(),
            (~lower_rows & lower_columns).float(),
        ],
        dim=1,
    ).reshape(1, 1, size, 3 * size)
    return tuple(_to_device(tensor, device, ttnn.float32) for tensor in (eye, tril, ones, masks))


def _case(device: ttnn.Device, *, state_dtype: ttnn.DataType = ttnn.float32):
    batch, sequence, heads, dim = 1, 64, 2, 32
    generator = torch.Generator().manual_seed(1731)
    shape = (batch, sequence, heads, dim)
    q = torch.randn(shape, generator=generator)
    k = torch.randn(shape, generator=generator)
    v = torch.randn(shape, generator=generator)
    gate = -0.02 * torch.rand(shape, generator=generator)
    beta = torch.sigmoid(torch.randn(batch, sequence, heads, generator=generator))
    state = 0.02 * torch.randn(batch, heads, dim, dim, generator=generator)
    expected_output, expected_state = kda_recurrent_reference(q, k, v, gate, beta, state)

    def chunk(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 2, 1, 3).reshape(batch * heads, sequence // 32, 32, dim)

    prep_inputs = (
        _to_device(chunk(l2_norm_reference(q) * dim**-0.5), device, ttnn.bfloat16),
        _to_device(chunk(l2_norm_reference(k)), device, ttnn.bfloat16),
        _to_device(chunk(v), device, ttnn.bfloat16),
        _to_device(chunk(gate), device, ttnn.float32),
        _to_device(
            beta.permute(0, 2, 1).reshape(batch * heads, sequence // 32, 32, 1),
            device,
            ttnn.float32,
        ),
        *_constants(device),
    )
    initial_state = _to_device(state.reshape(batch * heads, dim, dim), device, state_dtype)
    return prep_inputs, initial_state, expected_output, expected_state


def test_kda_final_chunk_scan_matches_reference_cache_trace_and_determinism(device: ttnn.Device) -> None:
    prep_inputs, initial_state, expected_output, expected_state = _case(device)
    prep = ttnn.transformer.kda_chunk_preparation(*prep_inputs)

    def run() -> list[ttnn.Tensor]:
        return ttnn.transformer.kda_final_chunk_scan(
            *prep,
            initial_state=initial_state,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    with ttnn.manage_config("throw_exception_on_fallback", True):
        outputs = run()
    ttnn.synchronize_device(device)
    first = [ttnn.to_torch(output) for output in outputs]
    actual_output = first[0].reshape(1, 2, 64, 32).permute(0, 2, 1, 3)
    actual_state = first[1].reshape(1, 2, 32, 32)
    for name, expected, actual in (
        ("output", expected_output, actual_output),
        ("state", expected_state, actual_state),
    ):
        assert torch.isfinite(actual).all(), f"{name} contains non-finite values"
        passing, message = comp_pcc(expected, actual, 0.999)
        assert passing, f"{name}: {message}"

    cache_entries = device.num_program_cache_entries()
    with ttnn.manage_config("throw_exception_on_fallback", True):
        repeated = run()
    ttnn.synchronize_device(device)
    assert device.num_program_cache_entries() == cache_entries
    for index, (expected, actual) in enumerate(zip(first, repeated)):
        assert torch.equal(expected, ttnn.to_torch(actual)), f"scan output {index} is not bit-identical"

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        traced = run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    for index, (expected, actual) in enumerate(zip(first, traced)):
        assert torch.equal(expected, ttnn.to_torch(actual)), f"traced scan output {index} is not bit-identical"
    ttnn.release_trace(device, trace_id)


def test_kda_final_chunk_scan_rejects_bf16_state(device: ttnn.Device, expect_error) -> None:
    prep_inputs, initial_state, _, _ = _case(device, state_dtype=ttnn.bfloat16)
    prep = ttnn.transformer.kda_chunk_preparation(*prep_inputs)
    with expect_error(RuntimeError, "initial_state must be FLOAT32"):
        ttnn.transformer.kda_final_chunk_scan(*prep, initial_state=initial_state)
