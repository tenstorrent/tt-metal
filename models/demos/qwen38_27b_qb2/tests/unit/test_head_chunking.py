# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The DRAM head's chunks must fill their bank shards exactly on every bank count."""

import pytest

from models.demos.qwen38_27b_qb2.tt.model import head_decode_chunks

# Qwen3.8-27B's vocabulary, per device, on the two qualified platforms. QB2 has eight DRAM
# banks and a T3K twelve, and neither shard ends on its own bank boundary.
_VOCAB = 248320
_PLATFORMS = [(_VOCAB // 4, 8), (_VOCAB // 8, 12)]
_IDS = ["qb2_tp4", "t3k_tp8"]


@pytest.mark.parametrize("shard_vocab, banks", _PLATFORMS, ids=_IDS)
def test_every_chunk_fills_its_bank_shards_exactly(shard_vocab, banks):
    chunks, _ = head_decode_chunks(shard_vocab, banks)
    assert chunks
    for start, stop in chunks:
        # An over-allocated bank shard makes the matmul return wrong values without failing,
        # so this is the invariant the whole chunking exists to hold.
        assert (stop - start) % (banks * 64) == 0
        # The per-bank shard is also a whole number of tiles.
        assert ((stop - start) // banks) % 32 == 0


@pytest.mark.parametrize("shard_vocab, banks", _PLATFORMS, ids=_IDS)
def test_chunks_and_tail_tile_the_vocabulary_shard_without_gaps(shard_vocab, banks):
    chunks, tail = head_decode_chunks(shard_vocab, banks)
    pieces = chunks + ([tail] if tail is not None else [])
    assert pieces[0][0] == 0
    assert pieces[-1][1] == shard_vocab
    for (_, before), (after, _) in zip(pieces, pieces[1:]):
        assert before == after
    assert sum(stop - start for start, stop in pieces) == shard_vocab


@pytest.mark.parametrize("shard_vocab, banks", _PLATFORMS, ids=_IDS)
def test_only_the_unalignable_remainder_leaves_the_dram_path(shard_vocab, banks):
    chunks, tail = head_decode_chunks(shard_vocab, banks)
    assert tail is not None, "both qualified shards have a remainder; a change here needs a device run"
    # Whatever the interleaved head takes must be smaller than one bank-aligned step, or a
    # chunk was dropped from the fast path.
    assert 0 < tail[1] - tail[0] < banks * 64
    assert tail[0] == chunks[-1][1]


def test_a_shard_that_divides_needs_no_interleaved_tail():
    banks = 8
    chunks, tail = head_decode_chunks(banks * 64 * 3, banks)
    assert tail is None
    assert sum(stop - start for start, stop in chunks) == banks * 64 * 3


def test_banks_wider_than_the_target_still_make_progress():
    # A part with more banks than the target chunk width must not produce a zero stride.
    chunks, tail = head_decode_chunks(64 * 1024, banks=512)
    assert chunks and all(stop > start for start, stop in chunks)
