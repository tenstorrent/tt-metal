# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests the design requires beyond the acceptance test (op_design.md, Work Distribution):

- every thread of the plan ran: two calls in one session on different inputs, each exact; the plan turned
  through the private `_slice(threads=...)` so the strided dealing is exercised;
- session order: a tt-metal program before slice, and slice before a tt-metal program (`to_layout`);
- R7: the output sits in the input's buffer, interleaved.
"""

import pytest
import torch

import ttnn
from ttnn.operations.slice.slice import _slice
from ttnn.operations.slice.slice_program_artifacts import uses_explicit_sync

# (shape, start, end): the YOLO calls and a ragged multi-page case per layout.
CASES = [
    ((1, 144, 400), (0, 64, 0), (1, 144, 400)),
    ((1, 144, 8400), (0, 0, 0), (1, 64, 8400)),
    ((4, 128, 47), (0, 64, 0), (4, 100, 47)),
    ((2, 512, 1024), (0, 256, 0), (2, 512, 1024)),
    # Row-major pages over one 8 KiB NoC packet take the ring's explicit protocol: several blocks per
    # cluster with a ragged last one (144 rows: 4 or 5 per cluster; 320 rows: 10 per cluster).
    ((1, 144, 8400), (0, 0, 0), (1, 144, 8400)),
    ((1, 352, 4128), (0, 32, 0), (1, 352, 4128)),
]
PLANS = [(1, 1), (2, 2), (2, 1), (3, 3), (4, 2)]  # W > R is refused (slice_program_artifacts.py)


def _to_device(device, t, layout, mem):
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mem)


def _expected(t, start, end):
    return t[tuple(slice(s, e) for s, e in zip(start, end))]


def _exact(out, expected):
    got = ttnn.to_torch(out)
    assert list(got.shape) == list(expected.shape), (got.shape, expected.shape)
    assert torch.equal(got, expected), (got.float() - expected.float()).abs().max()


@pytest.mark.parametrize("threads", PLANS, ids=lambda p: "R%dW%d" % p)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("case", CASES, ids=lambda c: "x".join(map(str, c[0])))
def test_every_thread_runs(device, case, layout, threads):
    """Two different inputs of the same shape into fresh outputs: a hart that did not run would leave its
    pages unwritten (or stale from the first call), and the second result would not be exact."""
    shape, start, end = case
    probe = _to_device(device, torch.zeros(shape, dtype=torch.bfloat16), layout, ttnn.DRAM_MEMORY_CONFIG)
    if uses_explicit_sync(probe.buffer_aligned_page_size()) and threads[0] != threads[1]:
        with pytest.raises(ValueError, match="R == W"):
            _slice(probe, start, end, threads=threads)
        return
    for seed in (0, 1):
        torch.manual_seed(seed)
        t = torch.randn(shape, dtype=torch.bfloat16)
        x = _to_device(device, t, layout, ttnn.DRAM_MEMORY_CONFIG)
        _exact(_slice(x, start, end, threads=threads), _expected(t, start, end))


def test_a_tt_metal_program_precedes(device):
    """tt-metal's own program (to_layout: tilize) runs first, then slice, in one session."""
    torch.manual_seed(2)
    t = torch.randn((4, 128, 512), dtype=torch.bfloat16)
    x = ttnn.to_layout(_to_device(device, t, ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG), ttnn.TILE_LAYOUT)
    _exact(ttnn.operations.slice.slice(x, [0, 32, 0], [4, 96, 512]), t[:, 32:96, :])


def test_a_tt_metal_program_follows(device):
    """slice (implicit-sync tagged transfers) then tt-metal's untilize, whose plain transfers would inherit a
    ring's tag if a hart did not hand it back."""
    torch.manual_seed(3)
    t = torch.randn((1, 144, 400), dtype=torch.bfloat16)
    x = _to_device(device, t, ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG)
    out = ttnn.operations.slice.slice(x, [0, 64, 0], [1, 144, 400])
    _exact(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT), t[:, 64:144, :])
    # And once more after that, for an RM input in DRAM.
    t2 = torch.randn((1, 144, 8400), dtype=torch.bfloat16)
    x2 = _to_device(device, t2, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)
    _exact(ttnn.operations.slice.slice(x2, (0, 0, 0), (1, 64, 8400)), t2[:, 0:64, :])


@pytest.mark.parametrize("mem", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram", "l1"])
def test_output_in_the_inputs_buffer(device, mem):
    """R7: the output is interleaved in the input's buffer type."""
    t = torch.randn((1, 96, 128), dtype=torch.bfloat16)
    x = _to_device(device, t, ttnn.TILE_LAYOUT, mem)
    out = ttnn.operations.slice.slice(x, [0, -64, 0], [1, -32, 128])
    assert out.memory_config().buffer_type == mem.buffer_type
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    _exact(out, t[:, 32:64, :])
