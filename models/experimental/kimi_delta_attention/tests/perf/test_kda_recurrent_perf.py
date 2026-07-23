# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-profiler harness for the exact-shape fused KDA decode recurrence."""

import os

import pytest
import torch
from tracy import signpost

import ttnn
from models.common.utility_functions import run_for_blackhole

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True),
]


def _to_device(tensor: torch.Tensor, device: ttnn.Device) -> ttnn.Tensor:
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_kda_recurrent_device_perf(device: ttnn.Device) -> None:
    """Profile warm kernel duration; invoke through Tracy, not as a latency test."""
    heads, key_dim, value_dim = 32, 128, 128
    generator = torch.Generator().manual_seed(307)
    q = _to_device(torch.randn(heads, 1, key_dim, generator=generator), device)
    k = _to_device(torch.randn(heads, 1, key_dim, generator=generator), device)
    v = _to_device(torch.randn(heads, 1, value_dim, generator=generator), device)
    decay = _to_device(torch.rand(heads, key_dim, 1, generator=generator), device)
    beta = _to_device(torch.rand(heads, 1, 1, generator=generator), device)
    state = _to_device(0.05 * torch.randn(heads, key_dim, value_dim, generator=generator), device)

    def step() -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return ttnn.transformer.kda_recurrent_step(
            q,
            k,
            v,
            decay,
            beta,
            state,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    warm_output, warm_state = step()
    ttnn.synchronize_device(device)
    ttnn.deallocate(warm_output)
    ttnn.deallocate(warm_state)

    outputs: list[tuple[ttnn.Tensor, ttnn.Tensor]] = []
    repetitions = int(os.getenv("PERF_REPS", "20"))
    signpost(header="start")
    for _ in range(repetitions):
        outputs.append(step())
    ttnn.synchronize_device(device)
    signpost(header="stop")

    for output, final_state in outputs:
        ttnn.deallocate(output)
        ttnn.deallocate(final_state)
