# SPDX-FileCopyrightText: 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf shapes for linear_gelu (production placement). Run under run_safe_pytest.sh --profile and read
DEVICE KERNEL DURATION [ns] from the profiler CSV; correctness is checked, perf is not asserted."""

import math

import pytest
import torch
import ttnn

from ttnn.operations.linear_gelu import linear_gelu

HIDDEN = 768
FFN = 3072

# (B, S): stock us, target us (eval/prompts/linear_gelu.txt).
PERF_SHAPES = [(8, 512), (16, 256), (32, 128), (8, 384), (4, 512), (8, 256), (2, 512), (4, 256)]


def _run(device, batch, seq, x_memory, out_memory):
    torch.manual_seed(42)
    x = torch.randn((batch, 1, seq, HIDDEN)).to(torch.bfloat16)
    w = (torch.randn((HIDDEN, FFN)) * (1.7 / math.sqrt(HIDDEN))).to(torch.bfloat16)
    b = (torch.randn((FFN,)) * 0.1 - 2.8).to(torch.bfloat16)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x_dev = ttnn.from_torch(x, memory_config=x_memory, **tile)
    w_dev = ttnn.from_torch(w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    b_dev = ttnn.from_torch(b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    out = linear_gelu(x_dev, w_dev, b_dev, memory_config=out_memory)
    ref = torch.nn.functional.gelu(x.double().reshape(batch, seq, HIDDEN) @ w.double() + b.double(), approximate="tanh")
    actual = ttnn.to_torch(out).reshape(batch, seq, FFN).double()
    pcc = torch.corrcoef(torch.stack([actual.flatten(), ref.flatten()]))[0, 1].item()
    assert pcc >= 0.9999, f"pcc {pcc}"


@pytest.mark.parametrize("batch, seq", PERF_SHAPES)
def test_linear_gelu_perf(device, batch, seq):
    _run(device, batch, seq, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)


# Guard rows for the perf refinements: DRAM placement and the non-aligned (W3 zero fill) path.
@pytest.mark.parametrize(
    "batch, seq, x_memory, out_memory",
    [
        (8, 512, ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
        (8, 512, ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
        (8, 500, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG),
    ],
    ids=["8x512-dram_x", "8x512-dram_out", "8x500-non_aligned"],
)
def test_linear_gelu_perf_guard(device, batch, seq, x_memory, out_memory):
    _run(device, batch, seq, x_memory, out_memory)
