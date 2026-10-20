# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of prefill padding and the test PCC helper (no device needed)."""

import pytest
import torch

from models.experimental.mimo_v2_6_flash.tests.common import pcc
from models.experimental.mimo_v2_6_flash.tt import kv
from models.experimental.mimo_v2_6_flash.tt.attention import TtMiMoAttention
from models.experimental.mimo_v2_6_flash.tt.common import pad_to
from models.experimental.mimo_v2_6_flash.tt.model import prefill_padded_len, prefill_warmup_chunks

BUCKETS = [32, 64, 96] + list(range(128, 2048, 128)) + list(range(2048, 4096 + 1, 256))

MAX_SEQ = 4096
BLOCK = 64
WINDOW = 128


def chunks(n, chunk_size, start_pos=0):
    """(start, length) of each prefill chunk, as TtMiMoModel.prefill_enqueue splits them."""
    return [(start_pos + s, min(chunk_size, n - s)) for s in range(0, n, chunk_size)]


def program_key(padded, start):
    """What a chunk's prefill programs key on besides the padded length: the SWA window blocks it reads and the SDPA
    chunk size of its global layers (attention.py)."""
    lead = start - max(0, start - WINDOW) // BLOCK * BLOCK
    return lead, TtMiMoAttention._prefill_sdpa_program(None, padded, start).q_chunk_size


@pytest.mark.parametrize(
    "n, chunk_size, start_pos",
    [
        (4032, 4096, 64),  # prefix hit: a 4032-token suffix after a 64-token prefix fills the context exactly
        (4096, 704, 0),  # the last chunk (576 tokens at 3520) pads to 640, past the context end
        (4096, 4096, 0),
        (1333, 512, 0),
    ],
)
def test_padding_is_a_bucket_and_tables_cover_it(n, chunk_size, start_pos):
    context_blocks = pad_to(-(-(MAX_SEQ + kv.PREFILL_MAX_PAD) // BLOCK), 8)
    table = torch.arange(MAX_SEQ // BLOCK, dtype=torch.int32).add(1).reshape(1, 1, -1)
    for start, length in chunks(n, chunk_size, start_pos):
        padded = prefill_padded_len(length)
        assert length <= padded <= length + kv.PREFILL_MAX_PAD and padded in BUCKETS
        # The SDPA program must still find a Q/K chunk size that divides the start and the padded length.
        program = TtMiMoAttention._prefill_sdpa_program(None, padded, start)
        assert start % program.q_chunk_size == 0 and padded % program.q_chunk_size == 0
        # Padding past the context lands in the scratch block; the context table reaches past the chunk's end.
        fill = kv.block_rows(table, 0, start // BLOCK, -(-padded // BLOCK), scratch=-7)
        real = min(-(-padded // BLOCK), MAX_SEQ // BLOCK - start // BLOCK)
        assert torch.equal(fill[0, 0, :real], table[0, 0, start // BLOCK : start // BLOCK + real])
        assert (fill[0, 0, real:] == -7).all()
        assert context_blocks * BLOCK >= start + padded


def test_warmup_covers_every_chunk():
    """Every (padded length, program key) a chunk of up to the warmup's tokens can run is in the warmup's list."""
    max_tokens, max_seq = 2048 + BLOCK - 1, 8192
    warm = {
        (prefill_padded_len(n), program_key(prefill_padded_len(n), s))
        for n, s in prefill_warmup_chunks(max_tokens, max_seq, BLOCK, WINDOW)
        if s
    }
    warm_first = {prefill_padded_len(n) for n, s in prefill_warmup_chunks(max_tokens, max_seq, BLOCK, WINDOW) if not s}
    leads, chunks_ = {}, {}
    for padded, (lead, chunk) in warm:
        leads.setdefault(padded, set()).add(lead)
        chunks_.setdefault(padded, set()).add(chunk)
    for n in range(1, max_tokens + 1):
        padded = prefill_padded_len(n)
        assert padded in warm_first
        for start in range(BLOCK, max_seq - n + 1, BLOCK):
            lead, chunk = program_key(padded, start)
            # SWA layers key on the lead, global layers on the chunk size: each must have been warmed for this length.
            assert lead in leads[padded] and chunk in chunks_[padded], (n, start)


def test_pcc_rejects_non_finite(expect_error):
    a = torch.randn(64)
    assert pcc(a, a) == pytest.approx(1.0)
    b = a.clone()
    b[3] = float("nan")
    with expect_error(AssertionError, "PCC is not finite"):
        pcc(a, b)
    with expect_error(AssertionError, "PCC is not finite"):
        pcc(torch.ones(64), a)  # a constant tensor has no correlation (NaN)
