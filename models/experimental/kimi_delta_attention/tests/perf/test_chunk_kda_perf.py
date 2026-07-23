# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-profiler harness for the exact-shape chunk-parallel KDA prefill."""

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


def test_chunk_kda_device_perf(device: ttnn.Device) -> None:
    """Profile warm T=640 chunk KDA; invoke through Tracy, not as a latency test."""
    batch, sequence, heads, key_dim, value_dim = 1, 640, 32, 128, 128
    generator = torch.Generator().manual_seed(503)
    shape = (batch, sequence, heads)

    def to_device(tensor: torch.Tensor, dtype: ttnn.DataType) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    q = to_device(torch.randn(*shape, key_dim, generator=generator), ttnn.bfloat16)
    k = to_device(torch.randn(*shape, key_dim, generator=generator), ttnn.bfloat16)
    v = to_device(torch.randn(*shape, value_dim, generator=generator), ttnn.bfloat16)
    gate = to_device(-0.02 * torch.rand(*shape, key_dim, generator=generator), ttnn.float32)
    beta = to_device(torch.rand(*shape, generator=generator), ttnn.float32)
    state = to_device(0.02 * torch.randn(batch, heads, key_dim, value_dim, generator=generator), ttnn.float32)

    def step() -> tuple[ttnn.Tensor, ttnn.Tensor]:
        output, final_state = ttnn.transformer.chunk_kda(
            q,
            k,
            v,
            gate,
            beta,
            initial_state=state,
            output_final_state=True,
            chunk_size=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        assert final_state is not None
        return output, final_state

    warm_output, warm_state = step()
    ttnn.synchronize_device(device)
    ttnn.deallocate(warm_output)
    ttnn.deallocate(warm_state)

    outputs: list[tuple[ttnn.Tensor, ttnn.Tensor]] = []
    repetitions = int(os.getenv("PERF_REPS", "10"))
    signpost(header="start")
    for _ in range(repetitions):
        outputs.append(step())
    ttnn.synchronize_device(device)
    signpost(header="stop")

    for output, final_state in outputs:
        ttnn.deallocate(output)
        ttnn.deallocate(final_state)
