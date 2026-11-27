# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1: ROW_MAJOR layout via the new_storage_rows regime.

Pins the regime (new buffer, input unchanged, one device program) and checks bit-exactness against
torch.nn.functional.pad in the input dtype. Covers the golden RM cases plus row-chunk edge cases:
a tail that ends mid-chunk, a tail starting exactly on a chunk boundary, odd bf16 widths (sub-4 B
tails), multi-chunk rows, L1 placement, and a non-default row_chunk_bytes knob.
"""

import sys

import pytest
import torch
import ttnn

from ttnn.operations.g6_pad import g6_pad

from .test_g6_pad import call_one_program, torch_g6_pad

g6_mod = sys.modules[g6_pad.__module__]

TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.int32: torch.int32,
    ttnn.uint32: torch.int32,
}


def make_input(shape, dtype):
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31 - 1, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31 - 1, shape, dtype=torch.int32)
    return (torch.randn(shape, dtype=torch.float32) * 100).to(TORCH_DTYPE[dtype])


def bits(t):
    return t.contiguous().view(torch.int16) if t.dtype == torch.bfloat16 else t.contiguous().view(torch.int32)


CASES = [
    # golden ROW_MAJOR cases
    ([1, 1, 1, 1], [[0, 0], [0, 0], [0, 0], [0, 31]], 0, ttnn.uint32, "dram"),
    ([1, 1, 1, 8], [[0, 0], [0, 0], [0, 0], [0, 24]], 0, ttnn.uint32, "dram"),
    ([5, 8192], [[0, 3], [0, 0]], 0, ttnn.int32, "dram"),
    ([5, 1], [[0, 3], [0, 0]], -1, ttnn.int32, "dram"),
    ([5, 1], [[0, 3], [0, 0]], 0, ttnn.uint32, "dram"),
    ([5, 3, 2560], [[0, 3], [0, 0], [0, 0]], 0, ttnn.bfloat16, "dram"),
    # extra coverage
    ([3, 7, 45], [[0, 2], [0, 5], [0, 19]], -3.5, ttnn.bfloat16, "dram"),  # odd bf16 widths, hw
    ([2, 3, 33, 1000], [[0, 0], [0, 1], [0, 0], [0, 77]], 7.0, ttnn.float32, "dram"),  # multi-chunk + tail
    ([4, 600], [[0, 0], [0, 424]], 2.0, ttnn.float32, "l1"),  # w only, L1
    ([1, 1, 2, 4097], [[0, 1], [0, 3]], -1, ttnn.int32, "dram"),  # few rows, wide, odd tail
    ([200, 64], [[0, 50], [0, 1]], 1.0, ttnn.bfloat16, "dram"),  # many rows (> grid)
]


@pytest.mark.parametrize("shape,padding,value,dtype,mem", CASES, ids=[f"case{i}" for i in range(len(CASES))])
def test_g6_pad_row_major(device, shape, padding, value, dtype, mem):
    x = make_input(shape, dtype)
    mc = ttnn.DRAM_MEMORY_CONFIG if mem == "dram" else ttnn.L1_MEMORY_CONFIG
    x_tt = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    out = call_one_program(x_tt, padding, value, positional=False)
    plan = g6_mod.last_plan
    assert plan["regime"] == "new_storage_rows"
    assert out.buffer_address() != x_tt.buffer_address()
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT
    expected = torch_g6_pad(x, padding, value)
    got = ttnn.to_torch(out)
    assert list(got.shape) == list(expected.shape)
    assert torch.equal(bits(got), bits(expected.to(got.dtype)))
    assert torch.equal(bits(ttnn.to_torch(x_tt)), bits(x)), "input modified"
    if plan["n_units"] > 1:
        assert plan["ncores"] > 1, "RM path must spread more than one output page across cores"


@pytest.mark.parametrize("chunk", [64, 192, 4096])
@pytest.mark.parametrize("block", [0, 1, 3])
def test_g6_pad_row_major_knobs(device, monkeypatch, chunk, block):
    monkeypatch.setitem(g6_mod.KNOBS, "row_chunk_bytes", chunk)
    monkeypatch.setitem(g6_mod.KNOBS, "block_tiles", block)
    monkeypatch.setitem(g6_mod.KNOBS, "min_units_per_core", 7)
    shape, padding, value = [3, 5, 301], [[0, 2], [0, 1], [0, 90]], -2.0
    x = make_input(shape, ttnn.float32)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    got = ttnn.to_torch(g6_pad(x_tt, padding, value))
    assert torch.equal(bits(got), bits(torch_g6_pad(x, padding, value)))


@pytest.mark.parametrize("split", [{"split_mode": 1}, {"split_mode": 2}, {"split_mode": 2, "fill_upfront": 1}])
def test_g6_pad_row_major_split(device, monkeypatch, split):
    """Refinement 3 split modes on the rows regime (dim-0 growth: two split regions, row tails)."""
    for k, v in split.items():
        monkeypatch.setitem(g6_mod.KNOBS, k, v)
    monkeypatch.setitem(g6_mod.KNOBS, "pipeline_min_units", 1)
    shape, padding, value = [3, 5, 301], [[0, 2], [0, 1], [0, 90]], -2.0
    x = make_input(shape, ttnn.float32)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    got = ttnn.to_torch(g6_pad(x_tt, padding, value))
    assert torch.equal(bits(got), bits(torch_g6_pad(x, padding, value)))
