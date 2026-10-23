# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Regression test for the tt_memmove overlapping self-copy hazard
(ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp).

ttnn.experimental.slice_write on the RM-interleaved LAST_DIM path issues an in-place
overlapping self-copy: tt_memmove(dst = base + page_offset, src = base, stick_size),
with dst > src. On the NoC self-copy branch (taken when page_offset % 16 == 0, i.e.
DRAM input) this has no memmove overlap semantics, so an in-flight write can be read
back as a later source. The fix routes overlapping regions to the CPU memmove.

The stock slice_write test masks this two ways that this test avoids: it uses a
constant source (a shifted duplicate of equal values is invisible) and L1 input
(alignment 16 -> the racy NoC branch is never taken). Here we use DRAM input
(page_offset in {16,32,48}), a distinct per-element pattern, and >8192B (multi-packet)
rows, and assert the written region is bit-exact.

Note: on current silicon the buggy path is timing-masked (NoC reads outrun the
overlapping writes) so this passes even without the fix; it guards against a
regression that reroutes overlap onto the raw NoC self-copy. The mechanism is proven
in the PR via a forced write-before-read ordering (deterministic corruption).
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_equal


@pytest.mark.parametrize(
    "H, W_in, begin_last",
    [
        (4, 6000, 8),  # page_offset=16B, row=12000B (multi-packet)
        (4, 6000, 16),  # page_offset=32B
        (4, 6000, 24),  # page_offset=48B
        (2, 16000, 8),  # page_offset=16B, row=32000B (~4 packets)
    ],
)
def test_slice_write_last_dim_overlap(H, W_in, begin_last, device):
    torch.manual_seed(1234)
    W_out = begin_last + W_in + 64

    torch_src = torch.randn(H, W_in).bfloat16().float()
    begins, ends, strides = [0, begin_last], [H, begin_last + W_in], [1, 1]
    slices = tuple(slice(b, e, s) for b, e, s in zip(begins, ends, strides))

    torch_out_ref = torch.zeros(H, W_out, dtype=torch.bfloat16).float()
    torch_out_ref[slices] = torch_src

    # DRAM interleaved RM input (align 64) -> exercises the page_offset%16==0 NoC branch
    tt_in = ttnn.from_torch(torch_src.bfloat16(), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    tt_in = ttnn.to_memory_config(tt_in, ttnn.DRAM_MEMORY_CONFIG)
    tt_out = ttnn.from_torch(
        torch.zeros(H, W_out, dtype=torch.bfloat16), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    tt_out = ttnn.to_memory_config(tt_out, ttnn.DRAM_MEMORY_CONFIG)

    ttnn.experimental.slice_write(tt_in, tt_out, begins, ends, strides)

    out_host = ttnn.to_torch(tt_out).float()
    assert_equal(torch_src, out_host[slices])
