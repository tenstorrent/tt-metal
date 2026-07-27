# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device tests for the fused KDA causal convolution."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True),
]


@pytest.mark.parametrize("stream_width", [512, 1024, 2048])
def test_kda_causal_convolution_channel_blocks(device: ttnn.Device, stream_width: int) -> None:
    """Match a serial four-tap reference across one and multiple channel blocks."""
    generator = torch.Generator().manual_seed(31)
    sequence = 64
    channels = 3 * stream_width
    inputs = torch.randn(1, sequence, channels, generator=generator, dtype=torch.bfloat16)
    state = torch.randn(1, 3, channels, generator=generator, dtype=torch.bfloat16)
    taps = [torch.randn(1, 1, channels, generator=generator, dtype=torch.bfloat16) for _ in range(4)]

    combined = torch.cat((state, inputs), dim=1)
    expected = F.silu(sum(combined[:, tap : tap + sequence] * taps[tap] for tap in range(4)))

    tt_inputs = ttnn.from_torch(
        inputs,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_state = ttnn.from_torch(
        state,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_taps = [
        ttnn.from_torch(
            tap,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for tap in taps
    ]

    outputs = ttnn.transformer.kda_causal_conv1d_split(
        tt_inputs,
        tt_state,
        *tt_taps,
        stream_width,
        stream_width,
        stream_width,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    actual = torch.cat([ttnn.to_torch(output) for output in outputs], dim=-1)
    passed, pcc = comp_pcc(expected, actual, pcc=0.999)
    max_abs = (expected.float() - actual.float()).abs().max().item()
    print(f"stream_width={stream_width}: PCC={pcc:.6f}, max_abs={max_abs:.6e}")
    assert passed, f"PCC {pcc:.6f} < 0.999"
