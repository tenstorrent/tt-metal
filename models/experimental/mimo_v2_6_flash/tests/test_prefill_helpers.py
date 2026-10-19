# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of prefill padding and the test PCC helper (no device needed)."""

import pytest
import torch

from models.experimental.mimo_v2_6_flash.tests.common import pcc
from models.experimental.mimo_v2_6_flash.tt.attention import TtMiMoAttention
from models.experimental.mimo_v2_6_flash.tt.model import prefill_padded_len

MAX_SEQ = 4096
BLOCK = 64


def chunks(n, chunk_size, start_pos=0):
    """(start, length) of each prefill chunk, as TtMiMoModel.prefill_enqueue splits them."""
    return [(start_pos + s, min(chunk_size, n - s)) for s in range(0, n, chunk_size)]


@pytest.mark.parametrize(
    "n, chunk_size, start_pos",
    [
        (4032, 4096, 64),  # prefix hit: a 4032-token suffix after a 64-token prefix fills the context exactly
        (4096, 704, 0),  # the last chunk (576 tokens at 3520) would pad to 640 and overrun
        (4096, 4096, 0),
        (1333, 512, 0),
    ],
)
def test_padding_stays_inside_the_context(n, chunk_size, start_pos):
    for start, length in chunks(n, chunk_size, start_pos):
        padded = prefill_padded_len(length, room=MAX_SEQ - start)
        assert length <= padded and padded % 32 == 0
        assert start + padded <= MAX_SEQ, f"chunk at {start} of {length} tokens pads to {padded}"
        # The SDPA program must still find a Q/K chunk size that divides the start and the padded length.
        program = TtMiMoAttention._prefill_sdpa_program(None, padded, start)
        assert start % program.q_chunk_size == 0 and padded % program.q_chunk_size == 0


def test_padding_unchanged_away_from_the_end():
    for n in (1, 31, 100, 129, 2047, 2049, 4000):
        assert prefill_padded_len(n, room=1 << 20) == prefill_padded_len(n)


def test_pcc_rejects_non_finite(expect_error):
    a = torch.randn(64)
    assert pcc(a, a) == pytest.approx(1.0)
    b = a.clone()
    b[3] = float("nan")
    with expect_error(AssertionError, "PCC is not finite"):
        pcc(a, b)
    with expect_error(AssertionError, "PCC is not finite"):
        pcc(torch.ones(64), a)  # a constant tensor has no correlation (NaN)
