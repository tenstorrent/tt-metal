# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the bounded-ring sliding-tail rebuild plan (tt/attention/prefill.py
``_bounded_ring_tail_blocks``): a sliding-window continuation whose first chunk replayed the traced short bucket
(no stash) reads its prior window back from the ring, whose fill wrote position p to slot p % modulo. No device.

    pytest models/demos/gemma4/tests/unit/test_bounded_ring_tail_host.py --noconftest -p no:cacheprovider
"""

import torch

from models.demos.gemma4.tt.attention.prefill import _bounded_ring_tail_blocks

ROW = torch.arange(100, 116, dtype=torch.int32)  # a 16-block ring table row (1024 / 64), ring-local ids


def test_first_continuation_reads_the_leading_blocks():
    # first chunk of 100 tokens, continuation at 100: positions [0, 100) sit in blocks 0 and 1, rows 0..99
    blocks, first_row, take = _bounded_ring_tail_blocks(ROW, 100, 1024, 1024, 64)
    assert (blocks, first_row, take) == ([100, 101], 0, 100)
    # block-aligned offset
    assert _bounded_ring_tail_blocks(ROW, 128, 1024, 1024, 64) == ([100, 101], 0, 128)


def test_wrapped_window_walks_the_ring_in_position_order():
    # continuation at 1100 with a 1024 window: positions [76, 1100) -> slots 76..1023, then 0..75
    blocks, first_row, take = _bounded_ring_tail_blocks(ROW, 1100, 1024, 1024, 64)
    assert take == 1024 and first_row == 12  # 76 - 64
    assert blocks == [100 + b for b in range(1, 16)] + [100, 101]


def test_short_window_and_block_zero():
    row = torch.tensor([0, 1, 2, 3], dtype=torch.int32)  # the remapped sliding table may start at block 0
    assert _bounded_ring_tail_blocks(row, 64, 512, 256, 64) == ([0], 0, 64)
    # positions [172, 300): slots 172..255 (blocks 2, 3) then the wrap 256..299 (block 0); row 44 of the run
    assert _bounded_ring_tail_blocks(row, 300, 128, 256, 64) == ([2, 3, 0], 44, 128)


def test_rejects_what_the_ring_cannot_hold():
    assert _bounded_ring_tail_blocks(ROW, 0, 1024, 1024, 64) is None
    assert _bounded_ring_tail_blocks(ROW, 100, 1024, 1000, 64) is None  # modulo not block-aligned
    assert _bounded_ring_tail_blocks(torch.tensor([7], dtype=torch.int32), 100, 1024, 1024, 64) is None  # too short
