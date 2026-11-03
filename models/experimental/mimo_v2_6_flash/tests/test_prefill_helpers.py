# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of prefill padding and the tests' PCC and greedy-agreement helpers (no device needed)."""

import pytest
import torch

from models.experimental.mimo_v2_6_flash.tests.common import first_divergence, pcc
from models.experimental.mimo_v2_6_flash.tt import kv
from models.experimental.mimo_v2_6_flash.tt.attention import GLOBAL_SDPA_MIN_START, TtMiMoAttention, later_chunk_split
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


@pytest.mark.parametrize("pair_split", [False, True], ids=["both_copies", "pair_split"])
def test_warmup_runs_the_grouped_program(pair_split):
    """From GLOBAL_SDPA_MIN_START a global layer's later chunk can take the grouped / K split program and its merge
    (attention.later_chunk_split); the warmup runs each later padded length that takes it once there, so serving
    compiles none of them between decode trace replays."""
    max_tokens, max_seq = 4096 + 2 * BLOCK - 1, 1 << 20
    listed = prefill_warmup_chunks(max_tokens, max_seq, BLOCK, WINDOW, pair_split=pair_split)
    align = 2 * BLOCK if pair_split else BLOCK
    warmed = {prefill_padded_len(n, s, pair_split) for n, s in listed if s >= GLOBAL_SDPA_MIN_START}
    takes_it = set()
    for n in range(1, max_tokens + 1):
        padded = prefill_padded_len(n, align, pair_split)
        start = pad_to(GLOBAL_SDPA_MIN_START, 256)
        if later_chunk_split(TtMiMoAttention._prefill_sdpa_program(None, padded, start).q_chunk_size, start):
            takes_it.add(padded)
    assert warmed == takes_it


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


def greedy_logits(*top):
    """Logits over a small vocabulary where token i scores top[i] and the rest score well below."""
    logits = torch.full((8,), 10.0)
    logits[: len(top)] = torch.tensor(top)
    return logits


@pytest.mark.parametrize(
    "gap, near_tie",
    [(0.0, True), (0.02, True), (2.0, True), (2.5, False), (5.0, False)],
    ids=["equal", "flip-0.02", "2-steps", "2.5-steps", "5-steps"],
)
def test_first_divergence_tolerates_only_bf16_near_ties(gap, near_tie):
    """test_model_chunked_tp8_dp4's greedy agreement: at the first different token each run's pick must score within
    TIE_STEPS bf16 steps (0.125 at logits in [16, 32)) of the other run's best; later tokens are not compared."""
    step = 0.125
    same = greedy_logits(22.8, 22.0)
    ref = [same, same, greedy_logits(22.84, 22.84 - 0.7 * step)]  # the reference picks token 0, token 1 0.7 steps back
    run = [same, same, greedy_logits(22.69 - gap * step, 22.69)]  # this run picks token 1, token 0 ``gap`` steps back
    # past the fork the histories differ: these tokens are not compared, however far apart
    assert first_divergence([0, 0, 1, 5], run, [0, 0, 0, 3], ref) == (2, near_tie)
    assert first_divergence([0, 0, 1], run, [0, 0, 0], ref)[1] == near_tie
    assert first_divergence([0, 0, 0, 4], ref, [0, 0, 0, 4], ref) is None
    # both directions count: this run's pick 5 steps behind in the reference's logits fails, whatever ``gap`` is
    far_ref = [same, same, greedy_logits(22.84, 22.84 - 5 * step)]
    assert first_divergence([0, 0, 1], run, [0, 0, 0], far_ref) == (2, False)
