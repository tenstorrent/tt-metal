# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sequence packing (tt/packing.py): the planner, and embed() on real padded batches it packs: two chunks into one pass,
one pass's padding into fewer slots, and a last pass in its cheapest rows."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch

from models.common.utility_functions import run_for_blackhole
from models.experimental.nomic_embed_text_v2_moe.reference import embedding as reference_embedding
from models.experimental.nomic_embed_text_v2_moe.reference.preprocessing import NomicPromptPrefix
from models.experimental.nomic_embed_text_v2_moe.tt.chunking import plan_sorted_chunks
from models.experimental.nomic_embed_text_v2_moe.tt.model import TtNomicBertModel, encode
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS
from models.experimental.nomic_embed_text_v2_moe.tt.packing import (
    PACKED_ATTENTION_US,
    Segment,
    best_packing,
    cheaper_packing,
    chunked_cost_us,
    pack,
    packed_cost_us,
    packed_inputs,
    pooling_rows,
    row_segments,
)

COSINE_TOLERANCE = 0.01

# The real token counts of a 4096-token-budget batch of SciFact + XQuAD documents (nomic-opt c13): padded to its
# longest it fills 11 x 512 slots, two length-sorted chunks; no packing into rows of 512 fits 8 rows.
REAL_BATCH = [512, 400, 419, 260, 287, 398, 112, 246, 512, 352, 262]


@pytest.mark.parametrize("width", [512, 1024])
@pytest.mark.parametrize(
    "lengths",
    [REAL_BATCH, [512] * 9, [37, 5, 300, 300, 300, 1, 64, 480], [1] * 40],
    ids=["real", "full", "mixed", "tiny"],
)
def test_pack_places_every_text_once_within_its_row(lengths, width):
    passes = pack(lengths, width, MAX_CHUNK_TOKENS)
    seen = sorted(seg.index for p in passes for seg in p.segments)
    assert seen == list(range(len(lengths)))
    for p in passes:
        assert p.width == width and p.tokens <= max(MAX_CHUNK_TOKENS, width)
        for row in range(p.rows):
            spans = sorted((s.offset, s.offset + s.length) for s in p.segments if s.row == row)
            assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:])), "segments overlap"
            assert spans and spans[-1][1] <= width
        for seg in p.segments:
            assert seg.length == max(1, lengths[seg.index])


def test_the_real_batch_packs_into_one_pass_of_wide_rows():
    chunks = plan_sorted_chunks(sorted(REAL_BATCH, reverse=True), MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS)
    assert len(chunks) == 2
    assert len(pack(REAL_BATCH, 512, MAX_CHUNK_TOKENS)) == 2  # 9 rows of 512: no 8-row packing exists
    packed = best_packing(REAL_BATCH, MAX_CHUNK_TOKENS)
    assert [(p.rows, p.width) for p in packed] == [(4, 1024)]


def test_packing_is_chosen_only_where_its_modelled_time_is_lower():
    chunks = plan_sorted_chunks(sorted(REAL_BATCH, reverse=True), MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS)
    assert cheaper_packing(REAL_BATCH, MAX_CHUNK_TOKENS, chunks) is not None  # 2 chunked passes -> 1 packed
    full = [512] * 16  # full-length rows: packing saves nothing and pays the masked attention
    assert cheaper_packing(full, MAX_CHUNK_TOKENS, plan_sorted_chunks(full, MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS)) is None
    # 51 real documents (a 16384-token budget, nomic-opt): on stock SDPA, 4 unaligned packed passes lost to 5 chunked
    # ones (27.6 against 24.0 ms). Aligned to tiles the texts need 17 rows of 1024 (16,091 real tokens plus each
    # text's tile padding), so 5 passes either way, and packing wins only by computing fewer padded slots.
    docs = [512, 400, 419, 260, 287, 398, 112, 246, 352, 262, 512, 487, 123, 360, 223, 463, 164, 358, 390, 459, 512]
    docs += [284, 168, 340, 380, 185, 180, 409, 166, 128, 268, 207, 166, 512, 278, 308, 163, 257, 340, 232, 198]
    docs += [512, 187, 386, 380, 495, 263, 247, 441, 345, 367]
    chunks = plan_sorted_chunks(sorted(docs, reverse=True), MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS)
    packed = cheaper_packing(docs, MAX_CHUNK_TOKENS, chunks)
    assert packed is not None and len(packed) == len(chunks) == 5
    assert packed_cost_us(packed) < chunked_cost_us(chunks)


def test_the_last_pass_runs_in_its_cheapest_rows():
    # 8 full-length texts fill one pass of 4 x 1024; the 260-token text left over runs as 1 x 288, not a row of 1024
    passes = pack([512] * 8 + [260], 1024, MAX_CHUNK_TOKENS, PACKED_ATTENTION_US)
    assert [(p.rows, p.width) for p in passes] == [(4, 1024), (1, 288)]
    assert passes[1].segments == (Segment(8, 0, 0, 260),)
    assert [(p.rows, p.width) for p in pack([512] * 8 + [260], 1024, MAX_CHUNK_TOKENS)] == [(4, 1024), (1, 1024)]


def test_a_one_pass_batch_packs_its_padding_away():
    # The diary's 2,048-token batch of real passages: padded to the longest it computes 6 x 512 slots; packed in rows
    # of 512 it takes 5 x 512, and in its cheapest rows 3 x 736 (2,208 slots for 2,144 tiled tokens).
    lengths = [512, 400, 419, 260, 287, 170]
    chunks = plan_sorted_chunks(sorted(lengths, reverse=True), MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS)
    assert [(c.count, c.seqlen) for c in chunks] == [(6, 512)]
    packed = cheaper_packing(lengths, MAX_CHUNK_TOKENS, chunks)
    assert [(p.rows, p.width) for p in packed] == [(3, 736)]
    wide = cheaper_packing(lengths, MAX_CHUNK_TOKENS, chunks, narrow=False)
    assert [(p.rows, p.width) for p in wide] == [(5, 512)]


def test_a_pass_below_32_tile_rows_is_widened_where_that_is_faster():
    # Below 32 tile rows the generated ops refuse the pass and the stock ones run: one text of 512 tokens costs 2.51 ms
    # alone and 2.30 as a row of 1,024 (v15 sweep), one of 256 2.11 alone; so the first runs widened, the second not.
    for length, want in ((512, [(1, 1024)]), (256, None)):
        chunks = plan_sorted_chunks([length], MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS)
        packed = cheaper_packing([length], MAX_CHUNK_TOKENS, chunks)
        assert (None if packed is None else [(p.rows, p.width) for p in packed]) == want, length
    # 3 texts of about 300 tokens: padded as 3 x 320 they are 960 slots, below 32 tile rows; side by side they fill
    # one row of 1,024 (928 tiled tokens), exactly 32 tile rows
    chunks = plan_sorted_chunks([320, 300, 280], MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS)
    assert [(c.count, c.seqlen) for c in chunks] == [(3, 320)]
    assert [(p.rows, p.width) for p in cheaper_packing([320, 300, 280], MAX_CHUNK_TOKENS, chunks)] == [(1, 1024)]


def test_texts_start_on_tiles():
    for width in (512, 1024):
        for p in pack(REAL_BATCH + [1, 31, 33, 5], width, MAX_CHUNK_TOKENS):
            assert all(seg.offset % 32 == 0 for seg in p.segments)


def test_row_segments_reads_back_the_packing():
    (packed,) = best_packing(REAL_BATCH, MAX_CHUNK_TOKENS)
    _, segment_ids = packed_inputs(torch.zeros(len(REAL_BATCH), 512, dtype=torch.long), packed, pad_token_id=1)
    rows = row_segments(segment_ids)
    want = [[] for _ in range(packed.rows)]
    for seg in sorted(packed.segments, key=lambda s: (s.row, s.offset)):
        want[seg.row].append((seg.offset, seg.length))
    assert rows == want


def test_packed_inputs_and_pooling_rows_mark_each_text_slots():
    input_ids = torch.arange(1, 4 * 600 + 1).reshape(4, 600)
    lengths = [600 - 100 * i for i in range(4)]
    (packed,) = best_packing(lengths, MAX_CHUNK_TOKENS)
    ids, segments = packed_inputs(input_ids, packed, pad_token_id=0)
    weights = pooling_rows(packed)
    for t, seg in enumerate(packed.segments):
        row = ids[seg.row, seg.offset : seg.offset + seg.length]
        assert torch.equal(row, input_ids[seg.index, : seg.length])
        assert bool((segments[seg.row, seg.offset : seg.offset + seg.length] == seg.index).all())
        assert int(weights[t].sum()) == seg.length
    assert int((segments == -1).sum()) == packed.tokens - sum(lengths)
    assert bool((ids[segments == -1] == 0).all())


def _scifact_documents(count, start=0):
    """SciFact abstracts start .. start + count in dataset order, unfiltered (title + text), or skip without the
    cached dataset."""
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    try:
        from datasets import load_dataset

        corpus = load_dataset("mteb/scifact", "corpus", split="corpus")
    except Exception as exc:  # no network and no cache
        pytest.skip(f"mteb/scifact is not available: {type(exc).__name__}: {exc}")
    return [f"{r.get('title') or ''} {r['text']}".strip() for r in corpus.select(range(start, start + count))]


# SciFact abstracts in dataset order (start, count), packed by embed() (tt.packing.cheaper_packing): the first one (451
# tokens) runs widened to one row of 1,024, where the generated ops take the pass; the first 6 are one chunk of 6 x 512
# packed as 3 x 928; the first 10 are two chunks packed as one pass of 4 x 1024; the first 13 are 4 x 1024 plus a last
# pass of 2 x 736; and 12 from the tenth leave one text for a last pass of 1 x 192, below 32 tile rows (the stock ops
# with segmented attention).
PACKED_BATCHES = {
    (0, 1): [(1, 1024)],
    (0, 6): [(3, 928)],
    (0, 10): [(4, 1024)],
    (0, 13): [(4, 1024), (2, 736)],
    (9, 12): [(4, 1024), (1, 192)],
}


@run_for_blackhole()
@pytest.mark.parametrize("batch", sorted(PACKED_BATCHES), ids=lambda b: f"scifact{b[0]}-{b[0] + b[1]}")
def test_packed_embed_matches_the_reference_on_a_real_padded_batch(
    device, config, tt_config, state_dict, reference_model, tokenizer, batch
):
    """A real padded batch embed() packs (one pass's padding into fewer slots, two chunks into one pass, a last pass
    in its cheapest rows): every text's embedding stays within the model's cosine bar of the reference and of the
    chunked path."""
    texts = _scifact_documents(batch[1], batch[0])
    kwargs = dict(prompt_prefix=NomicPromptPrefix.PASSAGE)
    lengths = [min(512, len(ids)) for ids in tokenizer([f"search_document: {t}" for t in texts])["input_ids"]]
    chunks = plan_sorted_chunks(sorted(lengths, reverse=True), tt_config.max_chunk_tokens, tt_config.min_chunk_tokens)
    packed = cheaper_packing(lengths, tt_config.max_chunk_tokens, chunks)
    assert packed is not None and [(p.rows, p.width) for p in packed] == PACKED_BATCHES[batch], packed

    model = TtNomicBertModel(device, config, tt_config, state_dict)
    got = encode(model, tokenizer, texts, **kwargs)
    model.tt_config = replace(tt_config, packed_sequences=False)
    chunked = encode(model, tokenizer, texts, **kwargs)
    ref = reference_embedding.encode(reference_model, tokenizer, texts, **kwargs)

    cosine = (got * ref).sum(-1)
    assert bool(
        (cosine > 1.0 - COSINE_TOLERANCE).all()
    ), f"worst text cosine to the reference {float(cosine.min()):.6f}"
    agree = (got * chunked).sum(-1)
    assert bool(
        (agree > 1.0 - COSINE_TOLERANCE).all()
    ), f"worst text cosine to the chunked path {float(agree.min()):.6f}"


def test_texts_for_one_pass_takes_the_longest_prefix_of_one_pass():
    from models.experimental.nomic_embed_text_v2_moe.tt.packing import planned_passes, texts_for_one_pass

    queue = REAL_BATCH + [400, 300, 512]  # 11 real documents (3,904 tiled tokens) fill one pass of 4 x 1024
    take = texts_for_one_pass(queue, MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS)
    assert take == len(REAL_BATCH)
    assert planned_passes(queue[:take], MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS)[0] == 1
    assert planned_passes(queue[: take + 1], MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS)[0] == 2
    assert texts_for_one_pass([512] * 3, MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS) == 3
    assert texts_for_one_pass([512], MAX_CHUNK_TOKENS) == 1


def test_texts_for_full_passes_leaves_the_spill_for_the_next_batch():
    from models.experimental.nomic_embed_text_v2_moe.tt.packing import planned_passes, texts_for_full_passes

    queue = [512] * 10  # 10 full-length texts: 8 fill one pass, 2 would spill into a quarter-full second
    take = texts_for_full_passes(queue, MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS)
    assert take == 8
    assert planned_passes(queue[:take], MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS) == (1, 4096)
    assert texts_for_full_passes([512] * 3, MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS) == 0  # wait: 3 rows fill no pass
    assert texts_for_full_passes([512] * 3, MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS, flush=True) == 3
    passes, computed = planned_passes(REAL_BATCH, MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS)  # 11 real documents packed
    assert passes == 1 and computed <= 4096
