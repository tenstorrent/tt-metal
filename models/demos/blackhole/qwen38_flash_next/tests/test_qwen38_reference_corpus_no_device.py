# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The reference corpus tool without a model: packed arrays, the corpus files and their sha256s, the teacher-forced
driver on a synthetic runner (chunked prefill equals one pass, the argmax chain, windows), BF4 rounding, the device
column's conversion and the scorer's metrics on constructed rows.  The committed corpus is re-verified when present."""

from __future__ import annotations

import json
import math

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_reference_corpus as corpus

VOCAB = 40
HIDDEN = 8


class _SyntheticRunner(corpus._Runner):
    """A stateless-by-construction model: the hidden state of position p is a fixed function of tokens 0..p, so any
    chunking of the prefill yields the same rows; the state carries the consumed prefix."""

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(7)
        self.head_weight = torch.randn(VOCAB, HIDDEN, generator=generator)
        self.calls: list[int] = []

    def forward(self, ids, state):
        prefix = [] if state is None else list(state)
        rows = []
        for token in ids[0].tolist():
            prefix.append(token)
            generator = torch.Generator().manual_seed(sum((i + 1) * t for i, t in enumerate(prefix)) % 100_003)
            rows.append(torch.randn(HIDDEN, generator=generator))
        self.calls.append(len(prefix))
        return torch.stack(rows).view(1, len(rows), HIDDEN).to(torch.bfloat16), tuple(prefix)


def _text_item(item_id: str, tokens: int, windows=()) -> corpus.CorpusItem:
    return corpus.CorpusItem(item_id, "book", tuple((i * 7 + 3) % VOCAB for i in range(tokens)), 0, tuple(windows))


def _prompt_item(item_id: str, prompt: int, continuation: int) -> corpus.CorpusItem:
    return corpus.CorpusItem(item_id, "acceptance", tuple((i * 5 + 1) % VOCAB for i in range(prompt)), continuation, ())


# -- packed arrays and the corpus files --------------------------------------------------------------------------


def test_packed_arrays_round_trip_int32_and_float32(expect_error):
    ints = [0, 1, -1, 2**31 - 1, -(2**31)]
    floats = [0.0, 1.5, -2.25, 2.0**-100, float("nan")]  # float32-exact values, then a NaN
    assert corpus.unpack_array("i", corpus.pack_array("i", ints)) == ints
    back = corpus.unpack_array("f", corpus.pack_array("f", floats))
    assert back[:4] == floats[:4] and math.isnan(back[4])
    with expect_error(corpus.ReferenceCorpusError):
        corpus.pack_array("d", [1.0])


def test_corpus_files_round_trip_and_a_tampered_items_file_is_refused(expect_error, tmp_path):
    items = [
        _prompt_item("acceptance-a", 5, 4),
        _text_item("book-b", 12),
        _text_item("long-c", 40, [(8, 12), (36, 39)]),
    ]
    items[2] = corpus.CorpusItem("long-c", "long", items[2].token_ids, 0, items[2].windows)
    manifest = corpus.write_corpus(items, tmp_path, provenance={"note": "test"})
    assert manifest["positions"] == (5 - 1 + 4) + 11 + 39
    assert manifest["scored_positions"] == 8 + 11 + 7
    assert manifest["parts"] == {"acceptance": 1, "served": 0, "book": 1, "eval": 0, "long": 1}
    loaded, loaded_items = corpus.load_corpus(tmp_path)
    assert loaded["items_sha256"] == manifest["items_sha256"]
    assert [item.token_ids for item in loaded_items] == [item.token_ids for item in items]
    assert loaded_items[2].scored_positions() == [8, 9, 10, 11, 36, 37, 38]
    assert loaded_items[0].manifest_entry()["teacher"] == "hf-argmax"
    assert loaded_items[1].manifest_entry()["teacher"] == "text"

    path = tmp_path / corpus.ITEMS_NAME
    path.write_bytes(path.read_bytes().replace(b'"token_ids":[3,', b'"token_ids":[4,', 1))
    with expect_error(corpus.ReferenceCorpusError, match="sha256"):
        corpus.load_corpus(tmp_path)


def test_a_window_outside_the_positions_is_refused(expect_error):
    with expect_error(corpus.ReferenceCorpusError, match="outside"):
        _text_item("long-x", 10, [(5, 10)]).scored_positions()  # position 9 has no teacher


def test_select_items_by_part_and_id(expect_error):
    items = [_prompt_item("acceptance-a", 5, 4), _text_item("book-b", 12)]
    assert [i.item_id for i in corpus.select_items(items, ["book"], [])] == ["book-b"]
    assert [i.item_id for i in corpus.select_items(items, [], ["acceptance-a"])] == ["acceptance-a"]
    with expect_error(corpus.ReferenceCorpusError, match="unknown"):
        corpus.select_items(items, [], ["nope"])


def test_long_windows_sit_before_the_bridge_at_each_depth_and_at_the_end():
    assert corpus._long_windows(8_192) == ((1_984, 2_016), (8_128, 8_160))
    assert corpus._long_windows(32_704) == ((1_984, 2_016), (8_128, 8_160), (32_640, 32_672))


def test_messages_from_transcript_drops_the_empty_thinking_block_and_the_open_turn(expect_error):
    text = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nHello.<|im_end|>\n"
        "<|im_start|>user\nBye<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert corpus.messages_from_transcript(text) == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Bye"},
    ]
    with expect_error(corpus.ReferenceCorpusError):
        corpus.messages_from_transcript("no turns")


def test_moby_dick_body_starts_at_the_second_chapter_heading():
    raw = b"contents\r\nCHAPTER 1. Loomings.\r\nCHAPTER 2.\r\n\r\nCHAPTER 1. Loomings.\r\nCall me Ishmael.\r\n*** END OF THE PROJECT GUTENBERG EBOOK x"
    assert corpus.moby_dick_body(raw) == "CHAPTER 1. Loomings.\nCall me Ishmael.\n"


# -- the teacher-forced driver ----------------------------------------------------------------------------------


def test_topk_rows_are_descending_log_probs_with_the_teacher_log_prob():
    logits = torch.tensor([[0.0, 3.0, 1.0, 2.0] + [-5.0] * (VOCAB - 4)])
    ids, values, teacher = corpus.topk_rows(logits, [2], k=3)
    assert ids[0] == [1, 3, 2]
    logprobs = torch.log_softmax(logits, dim=-1)[0]
    assert values[0] == pytest.approx([float(logprobs[1]), float(logprobs[3]), float(logprobs[2])])
    assert teacher[0] == pytest.approx(float(logprobs[2]))
    assert corpus.topk_rows(logits, [None], k=3)[2] == [None]


def test_chunked_prefill_matches_one_pass_and_the_continuation_follows_the_argmax_chain():
    item = _prompt_item("acceptance-a", 10, 6)
    whole = corpus.reference_item(_SyntheticRunner(), item, chunk_tokens=64)
    chunked = corpus.reference_item(_SyntheticRunner(), item, chunk_tokens=3)
    assert whole.positions == list(range(15)) and chunked.positions == whole.positions
    assert chunked.top_ids == whole.top_ids
    assert [v for row in chunked.top_logprobs for v in row] == pytest.approx(
        [v for row in whole.top_logprobs for v in row]
    )
    # prompt positions teach the prompt; the continuation teaches its own argmax chain
    assert whole.teacher_ids[:9] == list(item.token_ids[1:])
    assert whole.teacher_ids[9:] == [row[0] for row in whole.top_ids[9:]]
    assert all(lp == pytest.approx(row[0]) for lp, row in zip(whole.teacher_logprobs[9:], whole.top_logprobs[9:]))
    assert all(len(row) == corpus.TOP_K for row in whole.top_ids)


def test_a_teacher_chain_is_fed_instead_of_the_runners_argmax():
    item = _prompt_item("acceptance-a", 6, 4)
    own = corpus.reference_item(_SyntheticRunner(), item, chunk_tokens=64)
    teacher_ids = [(t + 1) % VOCAB for t in own.teacher_ids[5:]]
    forced = corpus.reference_item(_SyntheticRunner(), item, chunk_tokens=64, teacher_ids=teacher_ids)
    assert forced.teacher_ids[5:] == teacher_ids
    assert forced.top_ids[:5] == own.top_ids[:5]  # the prompt positions do not depend on the chain
    assert forced.top_ids[6:] != own.top_ids[6:]  # positions after the first fed token do


def test_text_item_windows_keep_only_the_window_positions():
    item = corpus.CorpusItem("long-a", "long", tuple((i * 3) % VOCAB for i in range(30)), 0, ((4, 8), (26, 29)))
    result = corpus.reference_item(_SyntheticRunner(), item, chunk_tokens=7)
    assert result.positions == [4, 5, 6, 7, 26, 27, 28]
    assert result.teacher_ids == [item.token_ids[p + 1] for p in result.positions]
    full = corpus.reference_item(
        _SyntheticRunner(), corpus.CorpusItem("book-a", "book", item.token_ids, 0, ()), chunk_tokens=64
    )
    assert result.top_ids == [full.top_ids[p] for p in result.positions]


def test_full_logits_are_kept_per_item_when_asked(tmp_path):
    item = _text_item("book-a", 9)
    corpus.reference_item(_SyntheticRunner(), item, chunk_tokens=4, full_logits_dir=tmp_path)
    saved = torch.load(tmp_path / "book-a.pt")
    assert saved["positions"] == list(range(8)) and saved["logits_fp16"].shape == (8, VOCAB)
    assert saved["logits_fp16"].dtype == torch.float16


def test_produce_reference_resumes_and_refuses_another_producer(expect_error, tmp_path):
    items = [_prompt_item("acceptance-a", 5, 3), _text_item("book-b", 7)]
    manifest = corpus.write_corpus(items, tmp_path, provenance={})
    out = tmp_path / "ref.json"
    producer = {"kind": "synthetic"}
    corpus.produce_reference(
        _SyntheticRunner(),
        manifest,
        items[:1],
        out=out,
        chunk_tokens=8,
        teacher=None,
        full_logits_dir=None,
        producer=producer,
    )
    document, columns = corpus.load_reference(out, manifest=manifest)
    assert list(columns) == ["acceptance-a"] and document["producer"] == producer
    runner = _SyntheticRunner()
    corpus.produce_reference(
        runner, manifest, items, out=out, chunk_tokens=8, teacher=None, full_logits_dir=None, producer=producer
    )
    _, columns = corpus.load_reference(out, manifest=manifest)
    assert list(columns) == ["acceptance-a", "book-b"]
    assert runner.calls == [7]  # only the second item ran
    with expect_error(corpus.ReferenceCorpusError, match="producer"):
        corpus.produce_reference(
            runner,
            manifest,
            items,
            out=out,
            chunk_tokens=8,
            teacher=None,
            full_logits_dir=None,
            producer={"kind": "other"},
        )
    # the oracle column reads the HF column's chain
    _, teacher = corpus.load_reference(out, manifest=manifest)
    again = tmp_path / "again.json"
    corpus.produce_reference(
        _SyntheticRunner(),
        manifest,
        items,
        out=again,
        chunk_tokens=8,
        teacher=teacher,
        full_logits_dir=None,
        producer={"kind": "b"},
    )
    _, second = corpus.load_reference(again, manifest=manifest)
    assert second["acceptance-a"].teacher_ids == teacher["acceptance-a"].teacher_ids


def test_reference_item_documents_round_trip_including_missing_teacher_log_probs():
    item = corpus.ReferenceItem(
        "x", [0, 1], [3, 4], [[1, 2, 3], [4, 5, 6]], [[-0.1, -2.5, -3.0], [-0.2, -2.0, -4.0]], [None, -0.5]
    )
    back = corpus.ReferenceItem.from_document(json.loads(json.dumps(item.to_document())))
    assert back.positions == item.positions and back.top_ids == item.top_ids and back.teacher_ids == item.teacher_ids
    assert back.top_logprobs[1] == pytest.approx(item.top_logprobs[1])
    assert back.teacher_logprobs[0] is None and back.teacher_logprobs[1] == pytest.approx(-0.5)


def test_indexed_eager_experts_forward_is_bitwise_the_index_add_loop():
    from types import SimpleNamespace

    generator = torch.Generator().manual_seed(3)
    experts = SimpleNamespace(
        num_experts=6,
        gate_up_proj=torch.randn(6, 2 * 5, HIDDEN, dtype=torch.bfloat16, generator=generator),
        down_proj=torch.randn(6, HIDDEN, 5, dtype=torch.bfloat16, generator=generator),
        act_fn=torch.nn.functional.silu,
    )
    hidden = torch.randn(7, HIDDEN, dtype=torch.bfloat16, generator=generator)
    top_k_index = torch.stack([torch.randperm(6, generator=generator)[:3] for _ in range(7)])
    top_k_weights = torch.softmax(torch.randn(7, 3, generator=generator), dim=-1).to(torch.bfloat16)
    expected = torch.zeros_like(hidden)
    mask = torch.nn.functional.one_hot(top_k_index, num_classes=6).permute(2, 1, 0)
    for expert in range(6):
        position, token = torch.where(mask[expert])
        gate, up = torch.nn.functional.linear(hidden[token], experts.gate_up_proj[expert]).chunk(2, dim=-1)
        current = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, experts.down_proj[expert])
        expected.index_add_(0, token, (current * top_k_weights[token, position, None]).to(expected.dtype))
    actual = corpus.indexed_eager_experts_forward(experts, hidden, top_k_index, top_k_weights)
    assert torch.equal(actual, expected)


# -- BF4 rounding ------------------------------------------------------------------------------------------------


def test_bfp4_round_shares_the_block_exponent_and_keeps_three_mantissa_bits():
    column = [1.0, 0.3, -0.26, 0.0, 1.9, 0.124, 0.126, 0.5] + [0.0] * 8
    weight = torch.tensor(column, dtype=torch.bfloat16).view(16, 1)
    rounded = corpus.bfp4_round(weight).float().view(-1).tolist()
    # the block's largest exponent is 1.0's (2^0): the step is 2^-2 and the largest value 7 * 2^-2
    assert rounded[:8] == [1.0, 0.25, -0.25, 0.0, 1.75, 0.0, 0.25, 0.5]
    assert rounded[8:] == [0.0] * 8
    assert corpus.bfp4_round(corpus.bfp4_round(weight)).float().view(-1).tolist() == rounded


def test_bfp4_round_blocks_run_along_the_output_rows_of_a_linear_weight(expect_error):
    weight = torch.zeros(32, 3, dtype=torch.bfloat16)
    weight[0, 0] = 64.0  # block 0 column 0: scale 16
    weight[1, 0] = 1.0  # rounds to 0 under that exponent
    weight[17, 0] = 1.0  # block 1 column 0: its own scale 0.25
    weight[1, 1] = 1.0  # another column: its own block exponent
    rounded = corpus.bfp4_round(weight).float()
    assert rounded[0, 0] == 64.0 and rounded[1, 0] == 0.0 and rounded[17, 0] == 1.0 and rounded[1, 1] == 1.0
    assert rounded.shape == (32, 3) and corpus.bfp4_round(weight).dtype == torch.bfloat16
    with expect_error(corpus.ReferenceCorpusError):
        corpus.bfp4_round(torch.zeros(20, 2))


def test_bfp4_round_is_the_packers_round_to_nearest_even():
    # step 2^-2 under exponent 0: 0.625 = 2.5 steps -> 2 (even), 0.875 = 3.5 steps -> 4, 0.375 = 1.5 steps -> 2
    weight = torch.tensor([1.0, 0.625, 0.875, 0.375] + [0.0] * 12, dtype=torch.bfloat16).view(16, 1)
    assert corpus.bfp4_round(weight).float().view(-1).tolist()[:4] == [1.0, 0.5, 1.0, 0.5]


# -- device column and scorer -------------------------------------------------------------------------------------


def _reference_from_rows(item_id: str, rows: list[tuple[list[int], list[float]]], teacher=None) -> corpus.ReferenceItem:
    positions = list(range(len(rows)))
    ids = [row[0] for row in rows]
    return corpus.ReferenceItem(
        item_id,
        positions,
        [row[0][0] for row in rows] if teacher is None else teacher,
        ids,
        [row[1] for row in rows],
        [row[1][0] for row in rows],
    )


def test_device_records_become_a_column_normalised_over_the_candidate_row(expect_error, tmp_path):
    items = [_prompt_item("acceptance-a", 3, 2)]
    (tmp_path / "acceptance-a.json").write_text(
        json.dumps(
            {
                "schema": corpus.AGREEMENT_RECORDS_SCHEMA,
                "item_id": "acceptance-a",
                "positions": [
                    {
                        "position": 0,
                        "teacher_id": 6,
                        "argmax": 6,
                        "candidate_ids": [1, 6, 9],
                        "candidate_logits": [1.0, 3.0, 2.0],
                    },
                    {
                        "position": 1,
                        "teacher_id": 11,
                        "argmax": 9,
                        "candidate_ids": [9, 6],
                        "candidate_logits": [2.0, 2.0],
                    },
                ],
            }
        )
    )
    (columns,) = corpus.device_reference_items(tmp_path, items)
    assert columns.positions == [0, 1] and columns.teacher_ids == [6, 11]
    assert columns.top_ids == [[6, 9, 1], [9, 6]]  # a tie at the top keeps the device's own argmax first
    assert sum(math.exp(v) for v in columns.top_logprobs[0]) == pytest.approx(1.0)
    assert columns.top_logprobs[0][0] == pytest.approx(3.0 - math.log(math.exp(1) + math.exp(3) + math.exp(2)))
    assert columns.teacher_logprobs == [None, None]
    (tmp_path / "acceptance-a.json").write_text(
        json.dumps(
            {
                "schema": corpus.AGREEMENT_RECORDS_SCHEMA,
                "item_id": "acceptance-a",
                "positions": [
                    {
                        "position": 0,
                        "teacher_id": 6,
                        "argmax": 1,
                        "candidate_ids": [1, 6],
                        "candidate_logits": [1.0, 3.0],
                    }
                ],
            }
        )
    )
    with expect_error(corpus.ReferenceCorpusError, match="argmax"):
        corpus.device_reference_items(tmp_path, items)


def test_truncated_kl_is_zero_for_equal_rows_and_positive_for_a_shifted_row():
    a_ids, a_lp = [1, 2, 3], [math.log(0.5), math.log(0.3), math.log(0.2)]
    kl, mass = corpus._truncated_kl(a_ids, a_lp, a_ids, a_lp)
    assert kl == pytest.approx(0.0) and mass == pytest.approx(1.0)
    kl, mass = corpus._truncated_kl(a_ids, a_lp, [3, 2, 1], [math.log(0.5), math.log(0.3), math.log(0.2)])
    assert kl > 0.1
    kl, mass = corpus._truncated_kl(a_ids, a_lp, [1, 2, 7], [math.log(0.6), math.log(0.4), -5.0])
    assert mass == pytest.approx(0.8)  # only ids 1 and 2 are shared
    assert corpus._truncated_kl(a_ids, a_lp, [8, 9], [-0.1, -0.2]) is None


def test_scorer_rates_clear_margin_first_divergence_and_segments():
    lp = lambda *values: [math.log(v) for v in values]  # noqa: E731
    a = _reference_from_rows(
        "book-a",
        [
            ([1, 2, 3], lp(0.7, 0.2, 0.1)),
            ([4, 5, 6], lp(0.5, 0.45, 0.05)),
            ([7, 8, 9], lp(0.9, 0.05, 0.05)),
            ([1, 2, 3], lp(0.6, 0.3, 0.1)),
        ],
    )
    b = _reference_from_rows(
        "book-a",
        [
            ([1, 2, 3], lp(0.7, 0.2, 0.1)),
            ([5, 4, 6], lp(0.5, 0.45, 0.05)),
            ([8, 7, 12], lp(0.9, 0.05, 0.05)),
            ([1, 3, 2], lp(0.6, 0.3, 0.1)),
        ],
    )
    items = [_text_item("book-a", 5)]
    score = corpus.score_references({"book-a": a}, {"book-a": b}, items, clear_margin=0.125)
    row = score["items"][0]
    assert row["positions"] == 4 and row["top1"] == 0.5 and row["top5"] == 1.0 and row["a_in_b_top32"] == 1.0
    # position 1's margin log(0.5/0.45) = 0.105 is inside the clear margin: the other three are clear, one of them wrong
    assert row["clear_positions"] == 3 and row["clear_top1"] == pytest.approx(2 / 3, abs=1e-4)
    assert row["first_divergence"] == 1
    assert row["kl_mean"] > 0 and row["segments"] == {"0": 0.5}
    assert score["corpus"]["top1"] == 0.5 and score["parts"]["book"]["positions"] == 4
    text = corpus.format_score(score, a_name="a", b_name="b")
    assert "book-a" in text and "corpus" in text
    identical = corpus.score_references({"book-a": a}, {"book-a": a}, items, clear_margin=0.125)
    assert identical["corpus"]["top1"] == 1.0 and identical["corpus"]["kl_mean"] == 0.0
    assert identical["items"][0]["first_divergence"] is None
    assert identical["corpus"]["teacher_logprob_gap_mean"] == 0.0


def test_scorer_aligns_positions_and_needs_a_shared_item(expect_error):
    a = _reference_from_rows("book-a", [([1, 2], [-0.1, -2.4]), ([3, 4], [-0.1, -2.4]), ([5, 6], [-0.1, -2.4])])
    b = corpus.ReferenceItem("book-a", [1, 2], [4, 6], [[3, 4], [6, 5]], [[-0.1, -2.4], [-0.1, -2.4]], [None, None])
    score = corpus.score_references({"book-a": a}, {"book-a": b}, [_text_item("book-a", 4)], clear_margin=0.125)
    assert score["items"][0]["positions"] == 2 and score["items"][0]["top1"] == 0.5
    assert score["items"][0]["first_divergence"] == 2 and score["items"][0]["teacher_logprob_gap_mean"] is None
    with expect_error(corpus.ReferenceCorpusError, match="share no items"):
        corpus.score_references({"book-a": a}, {"other": b}, [], clear_margin=0.125)


def test_verify_writes_and_checks_a_reference_stamp(expect_error, tmp_path):
    items = [_text_item("book-b", 7)]
    manifest = corpus.write_corpus(items, tmp_path, provenance={})
    out = tmp_path / "ref.json"
    corpus.produce_reference(
        _SyntheticRunner(),
        manifest,
        items,
        out=out,
        chunk_tokens=8,
        teacher=None,
        full_logits_dir=None,
        producer={"kind": "s"},
    )
    stamp = tmp_path / "stamp.json"
    assert corpus.main(["verify", "--corpus", str(tmp_path), "--reference", str(out), "--stamp", str(stamp)]) == 0
    written = json.loads(stamp.read_text())
    assert written["items"] == ["book-b"] and written["positions"] == 6 and written["producer"] == {"kind": "s"}
    assert written["sha256"] == corpus._sha256_file(out) and written["bytes"] == out.stat().st_size
    assert (
        corpus.main(["verify", "--corpus", str(tmp_path), "--reference", str(out), "--expect-stamp", str(stamp)]) == 0
    )
    out.write_text(out.read_text().replace('"kind": "s"', '"kind": "t"'))
    with expect_error(corpus.ReferenceCorpusError, match="differs from the stamp"):
        corpus.main(["verify", "--corpus", str(tmp_path), "--reference", str(out), "--expect-stamp", str(stamp)])


# -- the committed corpus ----------------------------------------------------------------------------------------


@pytest.mark.skipif(not (corpus.REFERENCE_DIR / corpus.MANIFEST_NAME).exists(), reason="the corpus is not frozen yet")
def test_the_committed_corpus_verifies_and_has_the_planned_shape():
    manifest, items = corpus.load_corpus()
    assert manifest["parts"] == {"acceptance": 12, "served": 16, "book": 2, "eval": 4, "long": 2}
    assert len(items) == 36
    assert all(
        item.continuation_tokens == corpus.CONTINUATION_TOKENS
        for item in items
        if item.part in ("acceptance", "served", "eval")
    )
    assert all(item.continuation_tokens == 0 for item in items if item.part in ("book", "long"))
    assert {item.prompt_tokens for item in items if item.part == "book"} == {corpus.BOOK_TOKENS}
    assert sorted(item.prompt_tokens for item in items if item.part == "long") == sorted(corpus.LONG_TOKENS)
    assert manifest["scored_positions"] == sum(len(item.scored_positions()) for item in items)
    by_id = {item.item_id: item for item in items}
    assert by_id["book-moby-dick"].token_ids == by_id["long-moby-dick-8192"].token_ids[: corpus.BOOK_TOKENS]
    assert by_id["long-moby-dick-8192"].token_ids == by_id["long-moby-dick-32704"].token_ids[:8192]
