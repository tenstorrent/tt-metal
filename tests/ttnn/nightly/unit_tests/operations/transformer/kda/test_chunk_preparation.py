# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Direct device coverage for KDA chunk preparation."""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole

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


def _inputs(device: ttnn.Device, *, q_dtype: ttnn.DataType = ttnn.bfloat16) -> tuple[ttnn.Tensor, ...]:
    heads, chunks, chunk_size, dim = 2, 2, 32, 32
    generator = torch.Generator().manual_seed(731)
    q = torch.randn(heads, chunks, chunk_size, dim, generator=generator)
    k = torch.randn(heads, chunks, chunk_size, dim, generator=generator)
    v = torch.randn(heads, chunks, chunk_size, dim, generator=generator)
    gate = -0.02 * torch.rand(heads, chunks, chunk_size, dim, generator=generator)
    beta = torch.sigmoid(torch.randn(heads, chunks, chunk_size, 1, generator=generator))
    eye_host = torch.eye(chunk_size, dtype=torch.float32).reshape(1, 1, chunk_size, chunk_size)
    tril_host = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.float32)).reshape(
        1, 1, chunk_size, chunk_size
    )
    ones_host = torch.ones(1, 1, chunk_size, chunk_size, dtype=torch.float32)
    indices = torch.arange(chunk_size)
    rows, columns = indices[:, None], indices[None, :]
    lower_rows, lower_columns = rows < chunk_size // 2, columns < chunk_size // 2
    masks_host = torch.cat(
        [
            (lower_rows & lower_columns).float(),
            (~lower_rows & ~lower_columns).float(),
            (~lower_rows & lower_columns).float(),
        ],
        dim=1,
    ).reshape(1, 1, chunk_size, 3 * chunk_size)
    return (
        _to_device(q, device, q_dtype),
        _to_device(k, device, ttnn.bfloat16),
        _to_device(v, device, ttnn.bfloat16),
        _to_device(gate, device, ttnn.float32),
        _to_device(beta, device, ttnn.float32),
        _to_device(eye_host, device, ttnn.float32),
        _to_device(tril_host, device, ttnn.float32),
        _to_device(ones_host, device, ttnn.float32),
        _to_device(masks_host, device, ttnn.float32),
    )


def test_kda_chunk_preparation_numerics_cache_and_determinism(device: ttnn.Device) -> None:
    inputs = _inputs(device)
    expected_v_beta = ttnn.to_torch(inputs[2]).float() * ttnn.to_torch(inputs[4]).float()

    def run() -> list[ttnn.Tensor]:
        return ttnn.transformer.kda_chunk_preparation(
            *inputs,
            chunk_size=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            scale=32**-0.5,
        )

    with ttnn.manage_config("throw_exception_on_fallback", True):
        outputs = run()
    ttnn.synchronize_device(device)
    assert len(outputs) == 7
    expected_shapes = [
        (2, 2, 32, 32),
        (2, 2, 32, 32),
        (2, 2, 32, 32),
        (2, 2, 32, 32),
        (2, 2, 32, 32),
        (2, 2, 32, 1),
        (2, 2, 32, 32),
    ]
    first = []
    for index, (output, shape) in enumerate(zip(outputs, expected_shapes)):
        assert tuple(output.shape) == shape
        actual = ttnn.to_torch(output)
        assert torch.isfinite(actual).all(), f"prep output {index} contains non-finite values"
        first.append(actual)
    passing, message = comp_pcc(expected_v_beta, first[0], 0.999)
    assert passing, f"v_beta: {message}"

    cache_entries = device.num_program_cache_entries()
    with ttnn.manage_config("throw_exception_on_fallback", True):
        repeated = run()
    ttnn.synchronize_device(device)
    assert device.num_program_cache_entries() == cache_entries
    for index, (expected, actual_tt) in enumerate(zip(first, repeated)):
        assert torch.equal(expected, ttnn.to_torch(actual_tt)), f"prep output {index} is not bit-identical"

    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        traced = run()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    for index, (expected, actual_tt) in enumerate(zip(first, traced)):
        assert torch.equal(expected, ttnn.to_torch(actual_tt)), f"traced prep output {index} is not bit-identical"
    ttnn.release_trace(device, trace_id)


def test_kda_chunk_preparation_rejects_non_bf16_q(device: ttnn.Device, expect_error) -> None:
    inputs = _inputs(device, q_dtype=ttnn.float32)
    with expect_error(RuntimeError, "q has wrong dtype"):
        ttnn.transformer.kda_chunk_preparation(*inputs)
