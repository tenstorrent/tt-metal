# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host tests for Gemma4's assistant (MTP) drafter on the plugin's speculative-decoding contract.

The runner supplies the candidate block and owns the accept walk; the adapter
answers verify steps from the posterior its own ``serving_step`` replay
produced. These tests drive ``spec_plan``, ``prefill_forward``,
``decode_forward``, ``propose_draft_tokens``, the warmups and the lifecycle
hooks in the runner's order over a recording target stub and a scripted
batched decoder (``mtp_contract_harness``).
"""

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tests.unit.conftest import import_adapter
from models.demos.gemma4.tests.unit.mtp_contract_harness import (
    DEFAULT_SHAPES,
    VOCAB,
    _block,
    _commands,
    _names,
    _ordinary,
    _positions,
    _prefill,
    _propose,
    _sampling,
    _start_solo,
    _tensor,
    _verify,
    build_mtp_model,
    install_decoder,
    logits_result,
    stub_drafter_modules,
)


@pytest.fixture
def mtp(adapter, monkeypatch):
    return build_mtp_model(adapter, monkeypatch)


@pytest.fixture
def assistant_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"backbone_hidden_size": 5376, "hidden_size": 1024}))
    return str(tmp_path)


def _steps(model):
    return _names(model, "step")


# -- declarations and the plan ---------------------------------------------------------


def test_declares_the_contract_rail(adapter):
    caps = adapter.Gemma4MTPContractForCausalLM.model_capabilities
    assert caps["output_tokens_per_step"] == 1
    assert caps["supports_spec_decode"] is True
    assert "tt_adaptive_block_output" not in caps and "tt_block_kv_extent_tokens" not in caps
    assert caps["spec_requirements"] == ("device_propose",)
    assert caps["spec_hidden_handoff"] == ("on_device",)
    assert caps["supports_sample_on_device"] is True
    assert caps["max_device_top_k"] == 32
    assert caps["supports_device_penalties"] is False
    # The resident decode is the base's env-driven value; the deferred verify is this class's own (below).
    assert caps["supports_async_decode"] is adapter.Gemma4ForCausalLM.model_capabilities["supports_async_decode"]
    assert caps["supports_async_decode"] is True and caps["supports_async_spec_decode"] is True
    # Prefix caching is the base's env-driven value: sessions key on slot + generation, not block ids.
    assert caps["supports_prefix_caching"] == adapter.Gemma4ForCausalLM.model_capabilities["supports_prefix_caching"]
    assert adapter.Gemma4MTPContractForCausalLM._SPEC_N == adapter.Gemma4MTPContractForCausalLM._SPEC_DRAFT_LEN + 1


def test_k_by_rows_table_parses_clamps_and_checks_the_tile(adapter, expect_error):
    assert adapter.spec_k_by_rows_table("5:5,8:3,16:1", 5) == ((5, 5), (8, 3), (16, 1))
    assert adapter.spec_k_by_rows_table(" 5:5 , 8:3 ", 3) == ((5, 3), (8, 3))
    with expect_error(ValueError, "strictly ascend"):
        adapter.spec_k_by_rows_table("8:3,5:5", 5)
    with expect_error(ValueError, "above the device argmax tile"):
        adapter.spec_k_by_rows_table("8:5", 5)
    with expect_error(ValueError, "not max_rows:k"):
        adapter.spec_k_by_rows_table("8", 5)
    with expect_error(ValueError, "is empty"):
        adapter.spec_k_by_rows_table(" , ", 5)


def test_plan_k_by_rows_appends_the_tail_only_when_the_table_stops_short(adapter):
    table = ((5, 5), (8, 3), (16, 1))
    assert adapter.spec_plan_k_by_rows(table, 16) == table
    assert adapter.spec_plan_k_by_rows(table, 8) == table
    assert adapter.spec_plan_k_by_rows(table, 32) == table + ((32, 1),)


def test_capture_shapes_add_the_solo_record_and_clamp_to_the_row_cap(adapter):
    table = ((5, 5), (8, 3), (16, 1))
    assert adapter.spec_capture_shapes(table, 5, 16) == DEFAULT_SHAPES
    assert adapter.spec_capture_shapes(table, 5, 6) == ((1, 5), (5, 5), (6, 3))
    assert adapter.spec_capture_shapes(table, 5, 1) == ((1, 5),)
    assert adapter.spec_capture_shapes(((4, 3),), 3, 16) == ((1, 3), (4, 3))


def test_spec_plan_is_positional_and_declares_the_assistant_drafter(adapter, monkeypatch, assistant_dir):
    monkeypatch.setenv("GEMMA4_ASSISTANT_MODEL", assistant_dir)
    monkeypatch.delenv("GEMMA4_SPEC_K_BY_ROWS", raising=False)
    plan = adapter.Gemma4MTPContractForCausalLM.spec_plan(None, 32, 5)
    assert plan.effective_k == 5 and plan.lanes_per_request == 1
    assert plan.accept_modes == ("argmax_ids", "logits")
    assert plan.drafter_state == "shared_with_target"
    assert plan.drafter_target_cache_requires == ("named_layer_caches", "absolute_positions")
    assert plan.supports_narrow_decode is True
    assert plan.k_by_rows == ((5, 5), (8, 3), (16, 1), (32, 1))
    assert plan.extra_bytes_per_seq == 5376 * 2 + 16 and plan.extra_bytes_per_token == 0


def test_spec_plan_reads_the_table_and_clamps_to_the_requested_k(adapter, monkeypatch, assistant_dir):
    monkeypatch.setenv("GEMMA4_ASSISTANT_MODEL", assistant_dir)
    monkeypatch.setenv("GEMMA4_SPEC_K_BY_ROWS", "4:7,8:3")
    plan = adapter.Gemma4MTPContractForCausalLM.spec_plan(None, 8, 3)
    assert plan.effective_k == 3
    assert plan.k_by_rows == ((4, 3), (8, 3))
    plan = adapter.Gemma4MTPContractForCausalLM.spec_plan(None, 4, 9)
    assert plan.effective_k == 5 and plan.k_by_rows == ((4, 5), (8, 3))


@pytest.mark.parametrize(
    ("target", "repo"),
    [
        ("google/gemma-4-31B-it", "google/gemma-4-31B-it-assistant"),
        ("google/gemma-4-12B-it", "google/gemma-4-12B-it-assistant"),
        ("google/gemma-4-26B-A4B-it", "google/gemma-4-26B-A4B-it-assistant"),
        ("/data/weights/gemma-4-26b-a4b-it/", "google/gemma-4-26B-A4B-it-assistant"),
        ("/data/weights/GEMMA-4-12B-IT", "google/gemma-4-12B-it-assistant"),
    ],
)
def test_the_assistant_follows_the_target_checkpoint_name(adapter, monkeypatch, target, repo):
    calls = []
    monkeypatch.setattr(
        adapter, "_hf_resolve_repo", lambda repo_id, cache_dir: calls.append((repo_id, cache_dir)) or "/snap"
    )
    assert adapter.assistant_repo_for(target) == repo
    assert adapter._assistant_default_snapshot(target) == "/snap"
    assert calls == [(repo, "models--" + repo.replace("/", "--"))]


def test_an_unknown_target_checkpoint_raises_instead_of_defaulting_to_31b(adapter, monkeypatch, expect_error):
    monkeypatch.setattr(adapter, "_hf_resolve_repo", lambda repo_id, cache_dir: "/snap")
    with expect_error(ValueError, "gemma-4-9B-it"):
        adapter.assistant_repo_for("google/gemma-4-9B-it")
    with expect_error(ValueError, "no assistant checkpoint is known"):
        adapter._assistant_default_snapshot("/weights/mystery-model")
    with expect_error(ValueError, "more than one checkpoint"):
        adapter.assistant_repo_for("/data/31b-box/gemma-4-12B-it")


def test_spec_plan_rejects_a_target_without_a_known_assistant(adapter, monkeypatch):
    monkeypatch.delenv("GEMMA4_ASSISTANT_MODEL", raising=False)
    monkeypatch.setenv("HF_MODEL", "google/gemma-4-9B-it")
    outcome = adapter.Gemma4MTPContractForCausalLM.spec_plan(None, 8, 5)
    assert type(outcome).__name__ == "SpecReject" and "gemma-4-9B-it" in outcome.reason


def test_spec_plan_rejects_without_an_assistant_snapshot(adapter, monkeypatch, tmp_path):
    monkeypatch.setenv("GEMMA4_ASSISTANT_MODEL", str(tmp_path / "missing"))
    outcome = adapter.Gemma4MTPContractForCausalLM.spec_plan(None, 8, 5)
    assert type(outcome).__name__ == "SpecReject"
    assert "GEMMA4_ASSISTANT_MODEL" in outcome.reason and outcome.supported_k == ()


def test_spec_plan_rejects_an_unreadable_assistant_config(adapter, monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 1024}))
    monkeypatch.setenv("GEMMA4_ASSISTANT_MODEL", str(tmp_path))
    outcome = adapter.Gemma4MTPContractForCausalLM.spec_plan(None, 8, 5)
    assert type(outcome).__name__ == "SpecReject" and "backbone_hidden_size" in outcome.reason


def test_spec_plan_rejects_a_table_the_argmax_tile_cannot_hold(adapter, monkeypatch, assistant_dir):
    monkeypatch.setenv("GEMMA4_ASSISTANT_MODEL", assistant_dir)
    monkeypatch.setenv("GEMMA4_SPEC_K_BY_ROWS", "8:5")
    outcome = adapter.Gemma4MTPContractForCausalLM.spec_plan(None, 8, 5)
    assert type(outcome).__name__ == "SpecReject" and "argmax tile" in outcome.reason
    assert outcome.supported_k == (5,)


# -- one request: narrow step, proposal, wide verify ------------------------------------


def test_solo_request_proposes_after_a_plain_first_step_and_answers_the_verify(mtp):
    proposal = _start_solo(mtp)
    # The first step was the plain decode; the proposal reseeded the row and replayed the solo record.
    assert _names(mtp, "decode") == [("decode", [3], [2])]
    assert _names(mtp, "reseed") == [("reseed", (1, 5), [0], [150], [3])]
    step = _steps(mtp)[0]
    assert step[1:7] == ((1, 5), [0], [150], [3], [None], 1024)
    assert step[7] == [[10, 11, 0, 0]]
    assert proposal.draft_token_ids.tolist() == [[101, 102, 103, 104, 105]]
    out = _verify(mtp, [_block(150, [101, 102, 103, 104, 105])], [_positions(3, 6)], [5], [10])
    assert out.spec_mode == "argmax_ids" and out.hidden is None
    assert out.argmax_ids.tolist() == [[150, 151, 152, 153, 154, 155]]
    assert out.argmax_ids.dtype == torch.int32
    assert len(_names(mtp, "decode")) == 1, "the verify was answered from the retained posterior"


def test_next_proposal_seeds_from_the_verified_column_without_a_reseed(mtp):
    _start_solo(mtp)
    _verify(mtp, [_block(150, [101, 102, 103, 104, 105])], [_positions(3, 6)], [5], [10])
    # The runner accepted 2 drafts and committed the correction: 3 tokens, anchor at column 2.
    proposal = _propose(mtp, [[150, 151, 999, -1, -1, -1]], [[4, 5, 6, -1, -1, -1]], counts=[3])
    assert proposal.num_valid.tolist() == [5] and proposal.draft_token_ids.tolist() == [[201, 202, 203, 204, 205]]
    assert len(_names(mtp, "reseed")) == 1
    assert _steps(mtp)[-1][1:6] == ((1, 5), [0], [999], [6], [2])


def test_verify_with_fewer_drafts_than_proposed_is_a_valid_prefix(mtp):
    _start_solo(mtp)
    out = _verify(mtp, [_block(150, [101, 102, 103])], [_positions(3, 4)], [3], [10])
    assert out.argmax_ids.tolist() == [[150, 151, 152, 153, 154, 155]]


def test_verify_rejects_drafts_the_model_did_not_propose(mtp, expect_error):
    _start_solo(mtp)
    with expect_error(ValueError, "did not propose"):
        _verify(mtp, [_block(150, [101, 102, 777, 104, 105])], [_positions(3, 6)], [5], [10])


def test_verify_rejects_a_block_that_does_not_continue_the_proposal(mtp, expect_error):
    _start_solo(mtp)
    with expect_error(RuntimeError, "does not continue"):
        _verify(mtp, [_block(151, [101, 102, 103, 104, 105])], [_positions(3, 6)], [5], [10])


def test_verify_rejects_drafts_for_a_row_without_a_session(mtp, expect_error):
    _prefill(mtp)
    with expect_error(RuntimeError, "owns no drafter session"):
        _verify(mtp, [_block(150, [101, 102, 103, 104, 105])], [_positions(3, 6)], [5], [10])


def test_a_draftless_verify_answers_column_zero_from_the_retained_proposal(mtp):
    _start_solo(mtp)
    out = _verify(mtp, [_block(150, [])], [_positions(3, 1)], [0], [10])
    assert out.argmax_ids[0, 0].item() == 150 and len(_names(mtp, "decode")) == 1


def test_an_ordinary_step_answers_from_the_retained_proposal_when_no_drafts_were_sent(adapter, monkeypatch):
    """Without the resident decode (the runner reloads every decode's inputs) a greedy row whose retained
    proposal continues its token is answered from that proposal's column 0."""
    mtp = build_mtp_model(adapter, monkeypatch, resident=False)
    _start_solo(mtp)
    out = _ordinary(mtp, [150], [3], [10])
    assert out.tolist() == [150] and len(_names(mtp, "decode")) == 1
    proposal = _propose(mtp, [[150, -1, -1, -1, -1, -1]], [[4, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [5]
    assert _steps(mtp)[-1][5] == [0], "the seed column is the verified anchor column"


def test_under_the_resident_decode_an_ordinary_step_runs_the_plain_decode_and_drops_the_retained_proposal(mtp):
    """``supports_async_decode`` (the class default) keeps the plain decode's device-resident token and
    position across ordinary steps, and the runner reloads them only after a verify, so no ordinary step is
    answered from the drafter: every live row runs the plain decode, the retained proposal is dropped, and the
    steady decode that follows (reload_inputs=False, host inputs stale) is accepted with its commands untouched."""
    assert mtp._spec_resident_decode_declared()
    _start_solo(mtp)
    out = _ordinary(mtp, [150], [3], [10], result=_tensor([150]))
    assert out.tolist() == [150] and mtp._mtp_sessions[0].retained is None
    assert _names(mtp, "decode")[-1] == ("decode", [150], [3])
    steady = dict(
        reload_inputs=False, reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False
    )
    out = _ordinary(mtp, [150], [4], [10], result="device", **steady)
    assert out == "device" and mtp.decode_commands[-1] == steady
    proposal = _propose(mtp, [[160] + [-1] * 5], [[5] + [-1] * 5])
    assert proposal.num_valid.tolist() == [5]
    assert _names(mtp, "reseed")[-1] == ("reseed", (1, 5), [0], [160], [5])


def test_a_plain_ordinary_step_drops_the_retained_proposal_and_the_next_proposal_reseeds(mtp):
    _start_solo(mtp)
    out = _ordinary(mtp, [150], [3], [10], result="device", sampling=False)
    assert out == "device", "host sampling reads logits, so the plain decode answers"
    proposal = _propose(mtp, [[160, -1, -1, -1, -1, -1]], [[4, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [5]
    assert _names(mtp, "reseed")[-1] == ("reseed", (1, 5), [0], [160], [4])


def test_accepted_counts_on_the_narrow_step_are_consumed_without_reaching_the_plain_decode(mtp):
    _prefill(mtp)
    out = _ordinary(mtp, [3], [2], [10], result=_tensor([150]), accepted_counts=_tensor([3]))
    assert out.tolist() == [150]
    assert "accepted_counts" not in mtp.decode_kwargs[-1] and "spec_mode" not in mtp.decode_kwargs[-1]


def test_a_plain_decode_after_a_drafter_served_step_refuses_stale_resident_inputs(mtp, expect_error):
    _start_solo(mtp)
    _verify(mtp, [_block(150, [101, 102, 103, 104, 105])], [_positions(3, 6)], [5], [10])
    with expect_error(RuntimeError, "reload_inputs=False"):
        _ordinary(mtp, [160], [4], [10], result="device", reload_inputs=False)
    assert _ordinary(mtp, [160], [4], [10], result="device") == "device"


# -- several rows ------------------------------------------------------------------------


def test_five_rows_share_one_replay_and_verify_with_per_row_counts(mtp):
    _prefill(mtp, rows=5)
    out = _ordinary(mtp, [3, 4, 5, 6, 7], [2] * 5, [10, 12, 14, 16, 18], result=_tensor([30, 31, 32, 33, 34]))
    assert out.tolist() == [30, 31, 32, 33, 34]
    committed = [[30 + r] + [-1] * 5 for r in range(5)]
    proposal = _propose(mtp, committed, [[3] + [-1] * 5] * 5)
    assert proposal.num_valid.tolist() == [5] * 5
    step = _steps(mtp)[0]
    assert step[1:4] == ((5, 5), [0, 1, 2, 3, 4], [30, 31, 32, 33, 34])
    assert step[7] == [[10, 11, 0, 0], [12, 13, 0, 0], [14, 15, 0, 0], [16, 17, 0, 0], [18, 19, 0, 0]]
    drafts = proposal.draft_token_ids.tolist()
    valid = [5, 3, 0, 5, 1]
    blocks = [_block(30 + r, drafts[r][: valid[r]]) for r in range(5)]
    positions = [_positions(3, 1 + valid[r]) for r in range(5)]
    out = _verify(mtp, blocks, positions, valid, [10, 12, 14, 16, 18])
    expected = [[150 + 10 * r + j for j in range(6)] for r in range(5)]
    assert out.argmax_ids.tolist() == expected
    assert len(_names(mtp, "decode")) == 1, "every row answered from the shared replay"
    # Each row commits a different count; the next replay seeds every row from its own verified column.
    counts = [6, 2, 1, 4, 2]
    committed = [[expected[r][j] if j < counts[r] else -1 for j in range(6)] for r in range(5)]
    positions = [_positions(4, counts[r]) for r in range(5)]
    proposal = _propose(mtp, committed, positions, counts=counts)
    assert proposal.num_valid.tolist() == [5] * 5
    step = _steps(mtp)[-1]
    assert step[3] == [expected[r][counts[r] - 1] for r in range(5)]
    assert step[4] == [3 + counts[r] for r in range(5)]
    assert step[5] == [c - 1 for c in counts]
    assert len(_names(mtp, "reseed")) == 1


def test_padded_wire_rows_are_ignored_and_keep_the_placeholder(mtp):
    _prefill(mtp)
    _ordinary(mtp, [3, 0], [2, -1], [10, 0], result=_tensor([150, 0]))
    proposal = _propose(mtp, [[150] + [-1] * 5, [-1] * 6], [[3] + [-1] * 5, [-1] * 6])
    assert proposal.num_valid.tolist() == [5, 0]
    out = _verify(
        mtp, [_block(150, [101, 102, 103, 104, 105]), [0] + [-1] * 5], [_positions(3, 6), [-1] * 6], [5, 0], [10, 0]
    )
    assert out.argmax_ids[0].tolist() == [150, 151, 152, 153, 154, 155]
    assert out.argmax_ids[1].tolist() == [-1] * 6


@pytest.mark.parametrize(
    ("live", "shape", "offer"), [(2, (5, 5), 5), (6, (8, 3), 3), (9, (16, 1), 1), (16, (16, 1), 1)]
)
def test_the_record_is_the_smallest_covering_the_live_rows(mtp, live, shape, offer):
    _prefill(mtp, rows=live)
    keys = [10 + 2 * r for r in range(live)]
    _ordinary(mtp, [3] * live, [2] * live, keys, result=_tensor([30] * live))
    proposal = _propose(mtp, [[30] + [-1] * 5] * live, [[3] + [-1] * 5] * live)
    assert proposal.num_valid.tolist() == [offer] * live
    step = _steps(mtp)[0]
    assert step[1] == shape and len(step[7]) == shape[0]
    assert all(row == [0, 0, 0, 0] for row in step[7][live:]), "padding rows address the null block"
    assert proposal.draft_token_ids[:, offer:].eq(0).all()


def test_a_shorter_record_offers_fewer_drafts_and_the_verify_truncates_to_them(mtp):
    _prefill(mtp, rows=6)
    keys = [10 + 2 * r for r in range(6)]
    _ordinary(mtp, [3] * 6, [2] * 6, keys, result=_tensor([30] * 6))
    proposal = _propose(mtp, [[30] + [-1] * 5] * 6, [[3] + [-1] * 5] * 6)
    drafts = proposal.draft_token_ids.tolist()
    assert drafts[0] == [101, 102, 103, 0, 0]
    blocks = [_block(30, drafts[r][:3]) for r in range(6)]
    out = _verify(mtp, blocks, [_positions(3, 4)] * 6, [3] * 6, keys)
    assert out.argmax_ids[0].tolist() == [150, 151, 152, 153, -1, -1]


# -- logits mode ----------------------------------------------------------------------------


def test_logits_verify_returns_the_host_block_with_zero_padding(mtp):
    _prefill(mtp, rows=2)
    _ordinary(mtp, [3, 4, 0], [2, 2, -1], [10, 12, 0], result=_tensor([30, 31, 0]))
    proposal = _propose(mtp, [[30] + [-1] * 5, [31] + [-1] * 5, [-1] * 6], [[3] + [-1] * 5, [3] + [-1] * 5, [-1] * 6])
    drafts = proposal.draft_token_ids.tolist()
    blocks = [_block(30, drafts[0]), _block(31, drafts[1][:2]), [0] + [-1] * 5]
    out = _verify(mtp, blocks, [_positions(3, 6), _positions(3, 3), [-1] * 6], [5, 2, 0], [10, 12, 0], mode="logits")
    assert out.spec_mode == "logits" and out.argmax_ids is None and out.hidden is None
    assert tuple(out.logits.shape) == (3, 6, VOCAB) and out.logits.dtype == torch.bfloat16
    assert out.logits[0].float().argmax(dim=-1).tolist() == [150, 151, 152, 153, 154, 155]
    assert out.logits[1].float().argmax(dim=-1).tolist() == [160, 161, 162, 163, 164, 165]
    assert out.logits[2].eq(0).all(), "a padding row is all zeros"
    assert mtp._mtp_decoder.results[-1].readbacks == 1, "one readback per replay, shared by its rows"
    assert len(_names(mtp, "decode")) == 1


def test_logits_verify_zeroes_the_columns_past_a_shorter_record(mtp):
    _prefill(mtp, rows=6)
    keys = [10 + 2 * r for r in range(6)]
    _ordinary(mtp, [3] * 6, [2] * 6, keys, result=_tensor([30] * 6))
    proposal = _propose(mtp, [[30] + [-1] * 5] * 6, [[3] + [-1] * 5] * 6)
    drafts = proposal.draft_token_ids.tolist()
    out = _verify(
        mtp, [_block(30, drafts[r][:3]) for r in range(6)], [_positions(3, 4)] * 6, [3] * 6, keys, mode="logits"
    )
    assert tuple(out.logits.shape) == (6, 6, VOCAB)
    assert out.logits[0, :4].float().argmax(dim=-1).tolist() == [150, 151, 152, 153]
    assert out.logits[:, 4:].eq(0).all()


def test_logits_verify_refuses_device_sampling_parameters(mtp, expect_error):
    _start_solo(mtp)
    with expect_error(ValueError, "logits verify carries no device sampling parameters"):
        _verify(
            mtp, [_block(150, [101, 102, 103, 104, 105])], [_positions(3, 6)], [5], [10], mode="logits", sampling=True
        )


def test_logits_verify_takes_straddle_rows_from_the_plain_decode_logits(mtp):
    _start_solo(mtp)
    _prefill(mtp, slot=1, key=20)
    blocks = [_block(150, [101, 102, 103, 104, 105]), _block(9, [])]
    out = _verify(
        mtp, blocks, [_positions(3, 6), _positions(2, 1)], [5, 0], [10, 20], mode="logits", result=logits_result(2, 40)
    )
    assert _names(mtp, "decode")[-1] == ("decode", [150, 9], [-1, 2])
    assert out.logits[0].float().argmax(dim=-1).tolist() == [150, 151, 152, 153, 154, 155]
    assert out.logits[1, 0].float().argmax().item() == 41 and out.logits[1, 1:].eq(0).all()


# -- straddle: a peer joins while a proposal is outstanding -----------------------------------


def test_a_peer_joining_takes_the_plain_decode_while_the_drafted_row_is_answered(mtp):
    _start_solo(mtp)
    _prefill(mtp, slot=1, key=20)
    blocks = [_block(150, [101, 102, 103, 104, 105]), _block(9, [])]
    out = _verify(mtp, blocks, [_positions(3, 6), _positions(2, 1)], [5, 0], [10, 20], result=_tensor([0, 42]))
    assert out.argmax_ids[0].tolist() == [150, 151, 152, 153, 154, 155]
    assert out.argmax_ids[1, 0].item() == 42
    assert _names(mtp, "decode")[-1] == ("decode", [150, 9], [-1, 2]), "the drafted row is excluded"
    # The next proposal moves to the two-row record: the drafted row still seeds from the last
    # replay under its row id (the decoder gathers across records); only the newcomer is reseeded.
    proposal = _propose(
        mtp, [[150, 151, 152, 153, 154, 155], [42] + [-1] * 5], [_positions(4, 6), [3] + [-1] * 5], counts=[6, 1]
    )
    assert proposal.num_valid.tolist() == [5, 5]
    assert _names(mtp, "reseed")[-1] == ("reseed", (5, 5), [1], [42], [3])
    assert _steps(mtp)[-1][1:6] == ((5, 5), [0, 1], [155, 42], [9, 3], [5, None])


def test_a_peer_leaving_moves_the_survivor_back_to_the_solo_record_seeded_from_the_last_replay(mtp):
    _prefill(mtp, rows=2)
    _ordinary(mtp, [3, 4], [2, 2], [10, 12], result=_tensor([30, 31]))
    _propose(mtp, [[30] + [-1] * 5, [31] + [-1] * 5], [[3] + [-1] * 5] * 2)
    mtp.release_request(1)
    drafts = mtp._mtp_decoder.results[-1].drafts[0].tolist()
    _verify(mtp, [_block(30, drafts), [0] + [-1] * 5], [_positions(3, 6), [-1] * 6], [5, 0], [10, 0])
    proposal = _propose(mtp, [[150, 151, 152, 153, 154, 155], [-1] * 6], [_positions(4, 6), [-1] * 6], counts=[6, 1])
    assert proposal.num_valid.tolist() == [5, 0]
    assert len(_names(mtp, "reseed")) == 1, "the survivor kept its row id: its seed is in the last replay"
    assert _steps(mtp)[-1][1:6] == ((1, 5), [0], [155], [9], [5])


def test_a_moved_slot_reseeds_under_its_new_row_id(mtp):
    _start_solo(mtp)
    # The runner moves the request to row 2 inside a step (row 2 reads slot 0) and reports the
    # move once the step is accepted; the proposal that follows uses the new row id.
    blocks = [[0] + [-1] * 5, [0] + [-1] * 5, _block(150, [101, 102, 103, 104, 105])]
    out = _verify(
        mtp,
        blocks,
        [[-1] * 6, [-1] * 6, _positions(3, 6)],
        [0, 0, 5],
        [0, 0, 10],
        slot_remap=[2, 1, 0],
        sampling_params=_sampling(3),
    )
    assert out.argmax_ids[2].tolist() == [150, 151, 152, 153, 154, 155]
    mtp.note_state_slots_moved({0: 2, 2: 0})
    committed = [[-1] * 6, [-1] * 6, [150, 151, 152, 153, 154, 155]]
    proposal = _propose(mtp, committed, [[-1] * 6, [-1] * 6, _positions(4, 6)], counts=[1, 1, 6])
    assert proposal.num_valid.tolist() == [0, 0, 5]
    assert _names(mtp, "reseed")[-1] == ("reseed", (1, 5), [2], [155], [9])
    assert _steps(mtp)[-1][1:6] == ((1, 5), [2], [155], [9], [None])


# -- declines and resumption -------------------------------------------------------------------


def test_above_the_row_cap_every_row_declines_once_logged_then_resumes_when_the_batch_shrinks(
    mtp, adapter, monkeypatch
):
    messages = []
    monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: messages.append(str(message)))
    rows = 17
    _prefill(mtp, rows=rows)
    keys = [10 + 2 * r for r in range(rows)]
    _ordinary(mtp, [3] * rows, [2] * rows, keys, result=_tensor([30] * rows))
    for _ in range(2):
        proposal = _propose(mtp, [[30] + [-1] * 5] * rows, [[3] + [-1] * 5] * rows)
        assert proposal.num_valid.tolist() == [0] * rows and proposal.draft_token_ids.eq(0).all()
    assert _steps(mtp) == []
    assert [m for m in messages if "proposing nothing" in m] == [
        "Gemma4MTP speculative contract: proposing nothing (17 proposing rows exceed the largest record (16, 1) "
        "(GEMMA4_SPEC_MAX_ROWS=16))"
    ]
    out = _ordinary(mtp, [30] * rows, [3] * rows, keys, result="device")
    assert out == "device", "the ordinary step after a decline is the plain decode"
    # Fewer rows: the next proposal reseeds the survivors and drafts again.
    live = 4
    proposal = _propose(mtp, [[31] + [-1] * 5] * live, [[4] + [-1] * 5] * live)
    assert proposal.num_valid.tolist() == [5] * live
    assert _names(mtp, "reseed")[-1] == ("reseed", (5, 5), [0, 1, 2, 3], [31] * 4, [4] * 4)


def test_a_row_past_the_captured_widths_is_left_out_while_the_others_draft(mtp):
    _prefill(mtp, rows=2)
    _ordinary(mtp, [3, 4], [2, 1019], [10, 12], result=_tensor([30, 31]))
    proposal = _propose(mtp, [[30] + [-1] * 5, [31] + [-1] * 5], [[3] + [-1] * 5, [1020] + [-1] * 5])
    assert proposal.num_valid.tolist() == [5, 0]
    assert _steps(mtp)[-1][1:4] == ((1, 5), [0], [30])


def test_a_row_past_max_seq_len_declines(mtp):
    mtp.model_args[0].max_seq_len = 10
    install_decoder(mtp)
    _start_solo(mtp)  # rows 3..8 fit; after six committed tokens rows 9..14 do not
    _verify(mtp, [_block(150, [101, 102, 103, 104, 105])], [_positions(3, 6)], [5], [10])
    proposal = _propose(mtp, [[150, 151, 152, 153, 154, 155]], [_positions(4, 6)], counts=[6])
    assert proposal.num_valid.tolist() == [0]


def test_an_exact_window_ring_declines_before_the_verify_would_wrap(adapter, monkeypatch):
    mtp = build_mtp_model(adapter, monkeypatch, ring=1024, max_seq_len=4096)
    _prefill(mtp)
    _ordinary(mtp, [3], [1018], [10], result=_tensor([150]))
    assert _propose(mtp, [[150] + [-1] * 5], [[1019] + [-1] * 5]).num_valid.tolist() == [0]
    _ordinary(mtp, [150], [1019], [10], result=_tensor([151]))
    assert _propose(mtp, [[151] + [-1] * 5], [[1020] + [-1] * 5]).num_valid.tolist() == [0]


def test_ring_advice_names_the_setting_and_the_resolved_numbers(adapter, monkeypatch):
    messages = []
    monkeypatch.setattr(adapter.logger, "warning", lambda message, *a, **k: messages.append(message))
    mtp = build_mtp_model(adapter, monkeypatch, ring=1024)
    mtp._spec_ring_advice(mtp._SPEC_N, "Gemma4MTP speculative contract")
    assert len(messages) == 1 and messages[0].startswith("Gemma4MTP speculative contract")
    assert (
        "GEMMA4_SPEC_RING_HEADROOM_BLOCKS=16" in messages[0] and "ring 2048" in messages[0] and "P_v=6" in messages[0]
    )
    mtp = build_mtp_model(adapter, monkeypatch, ring=2048)
    mtp._spec_ring_advice(mtp._SPEC_N, "Gemma4MTP speculative contract")
    assert len(messages) == 1


def test_proposing_without_captured_records_declines(adapter, monkeypatch):
    mtp = build_mtp_model(adapter, monkeypatch, warm=False)
    _prefill(mtp)
    _ordinary(mtp, [3], [2], [10], result=_tensor([150]))
    assert _propose(mtp, [[150] + [-1] * 5], [[3] + [-1] * 5]).num_valid.tolist() == [0]


# -- identity: slots, generations, release ----------------------------------------------------


def test_slot_moves_follow_the_session(mtp):
    _start_solo(mtp)  # the prefill opened generation 1 of slot 0
    mtp.note_state_slots_moved({0: 3, 3: 0})
    assert set(mtp._mtp_sessions) == {3} and mtp._mtp_sessions[3].owner == (3, 1)
    assert mtp._spec_slot_gen[3] == 1
    out = _verify(
        mtp,
        [[0] + [-1] * 5, [0] + [-1] * 5, [0] + [-1] * 5, _block(150, [101, 102, 103, 104, 105])],
        [[-1] * 6, [-1] * 6, [-1] * 6, _positions(3, 6)],
        [0, 0, 0, 5],
        [0, 0, 0, 10],
    )
    assert out.argmax_ids[3].tolist() == [150, 151, 152, 153, 154, 155]


def test_slot_remap_during_a_step_reads_the_moved_session(mtp):
    _start_solo(mtp)
    out = _verify(
        mtp,
        [[0] + [-1] * 5, _block(150, [101, 102, 103, 104, 105])],
        [[-1] * 6, _positions(3, 6)],
        [0, 5],
        [0, 10],
        slot_remap=[1, 0],
        sampling_params=_sampling(2),
    )
    assert out.argmax_ids[1].tolist() == [150, 151, 152, 153, 154, 155]


def test_release_drops_the_session_and_a_new_occupant_opens_a_new_generation(mtp, adapter, monkeypatch):
    messages = []
    monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: messages.append(str(message)))
    _start_solo(mtp)  # prefill: generation 1; the session is owned by (0, 1)
    assert mtp._mtp_sessions[0].owner == (0, 1)
    mtp.release_request(0)
    assert mtp._mtp_sessions == {} and mtp._spec_slot_gen[0] == 2
    assert [m for m in messages if "contract summary" in m] == [
        "Gemma4MTP contract summary: slot 0 (release) 1 steps, 1 tokens, 1.00 tokens/step"
    ]
    _prefill(mtp, key=20)
    assert mtp._spec_slot_gen[0] == 3
    _ordinary(mtp, [3], [2], [20], result=_tensor([160]))
    _propose(mtp, [[160] + [-1] * 5], [[3] + [-1] * 5])
    assert mtp._mtp_sessions[0].owner == (0, 3)
    assert _names(mtp, "reseed")[-1] == ("reseed", (1, 5), [0], [160], [3])


def test_release_of_another_slot_keeps_the_session(mtp):
    _start_solo(mtp)
    mtp.release_request(3)
    assert 0 in mtp._mtp_sessions and mtp._mtp_sessions[0].retained is not None


def test_release_of_every_request_drops_every_session(mtp):
    _prefill(mtp, rows=2)
    _ordinary(mtp, [3, 4], [2, 2], [10, 12], result=_tensor([30, 31]))
    _propose(mtp, [[30] + [-1] * 5, [31] + [-1] * 5], [[3] + [-1] * 5] * 2)
    mtp.release_request(None)
    assert mtp._mtp_sessions == {}


def test_a_prefill_into_a_slot_drops_its_stale_session(mtp):
    _start_solo(mtp)
    _prefill(mtp, key=20)
    assert mtp._mtp_sessions == {}
    _ordinary(mtp, [3], [2], [20], result=_tensor([160]))
    proposal = _propose(mtp, [[160] + [-1] * 5], [[3] + [-1] * 5])
    assert proposal.num_valid.tolist() == [5]


def test_a_session_of_a_closed_generation_is_refused(mtp, expect_error):
    _start_solo(mtp)
    mtp._spec_close_slot_generation(0)
    with expect_error(RuntimeError, "generation"):
        _ordinary(mtp, [150], [3], [10], result=_tensor([1]))


def test_release_persistent_capture_drops_sessions_releases_the_decoder_and_reaches_the_base(mtp):
    _start_solo(mtp)
    mtp.release_persistent_capture()
    assert mtp._mtp_sessions == {} and mtp._mtp_decoder is None and not mtp._mtp_captured
    assert mtp.events[-1] == ("release",)
    assert mtp.base_releases == [1]


# -- sampling state ------------------------------------------------------------------------------


@pytest.mark.parametrize("step", ["ordinary", "draftless-verify", "retained-verify"])
def test_retained_answers_apply_sampling_commands_and_remap_once_without_drawing(adapter, monkeypatch, step):
    # Without the resident decode, so that the ordinary step too is answered from the retained proposal.
    mtp = build_mtp_model(adapter, monkeypatch, resident=False)
    sampler = mtp.model[0].sampling
    _start_solo(mtp)
    sampler.seed_manager.events.clear()
    mtp.sampling_commands.clear()
    params = _sampling(2, seed=[None, 17])
    commands = _commands()
    kwargs = dict(sampling_params=params, slot_remap=[1, 0], **commands)
    if step == "ordinary":
        _ordinary(mtp, [0, 150], [-1, 3], [0, 10], **kwargs)
    elif step == "draftless-verify":
        _verify(mtp, [[0] + [-1] * 5, _block(150, [])], [[-1] * 6, _positions(3, 1)], [0, 0], [0, 10], **kwargs)
    else:
        _verify(
            mtp,
            [[0] + [-1] * 5, _block(150, [101, 102, 103, 104, 105])],
            [[-1] * 6, _positions(3, 6)],
            [0, 5],
            [0, 10],
            **kwargs,
        )
    assert len(_names(mtp, "decode")) == 1, "no device forward for the answered row"
    assert len(mtp.sampling_commands) == 1
    assert {name: mtp.sampling_commands[0][name] for name in commands if name != "reload_page_table"} == {
        name: value for name, value in commands.items() if name != "reload_page_table"
    }
    events = sampler.seed_manager.events
    assert len([event for event in events if event[0] == "remap"]) == 1
    assert not any(event[0] == "draw" for event in events)
    assert sampler.seed_manager.seeds[1] == 17 and sampler.seed_manager.positions[1] == 3


def test_a_logits_verify_answered_from_the_replay_follows_the_slot_remap_only(mtp):
    sampler = mtp.model[0].sampling
    _start_solo(mtp)
    sampler.seed_manager.events.clear()
    mtp.sampling_commands.clear()
    _verify(
        mtp,
        [[0] + [-1] * 5, _block(150, [101, 102, 103, 104, 105])],
        [[-1] * 6, _positions(3, 6)],
        [0, 5],
        [0, 10],
        mode="logits",
        slot_remap=[1, 0],
    )
    assert [event[0] for event in sampler.seed_manager.events] == ["remap"]
    assert mtp.sampling_commands == []


def test_a_sampled_row_is_not_answered_from_the_retained_argmax_on_an_ordinary_step(mtp):
    _start_solo(mtp)
    out = _ordinary(mtp, [150], [3], [10], result=_tensor([77]), sampling_params=_sampling(1, temperature=0.7))
    assert out.tolist() == [77] and mtp._mtp_sessions[0].retained is None


def test_argmax_verify_without_device_sampling_takes_straddle_rows_from_the_plain_logits(mtp):
    _start_solo(mtp)
    _prefill(mtp, slot=1, key=20)
    blocks = [_block(150, [101, 102, 103, 104, 105]), _block(9, [])]
    out = _verify(
        mtp, blocks, [_positions(3, 6), _positions(2, 1)], [5, 0], [10, 20], sampling=False, result=logits_result(2, 40)
    )
    assert out.argmax_ids[1, 0].item() == 41 and out.argmax_ids[0].tolist() == [150, 151, 152, 153, 154, 155]


# -- warmup: base buckets first, prepare on the eager pass, capture on the traced pass -----------


def _warmup(model, enable_trace):
    model.warmup_model_decode(
        kv_cache=model.kv_cache, enable_trace=enable_trace, max_batch_size=32, num_blocks=8, can_sample_on_device=True
    )


def test_warmup_runs_the_base_buckets_then_prepares_then_captures(adapter, monkeypatch, assistant_dir):
    mtp = build_mtp_model(adapter, monkeypatch, warm=False, env={"GEMMA4_ASSISTANT_MODEL": assistant_dir})
    built = stub_drafter_modules(monkeypatch, mtp.events)
    _warmup(mtp, False)
    _warmup(mtp, True)
    kinds = [event[0] for event in mtp.events]
    assert kinds == ["base_warmup", "prepare", "base_warmup", "capture"]
    assert mtp.events[0] == ("base_warmup", False) and mtp.events[2] == ("base_warmup", True)
    assert mtp.events[1] == ("prepare", list(DEFAULT_SHAPES), 16)
    assistant = dict(built[0][1])
    assert built[0][0] == "assistant" and assistant["max_local_batch_size"] == 16
    assert assistant["assistant_path"] == assistant_dir and assistant["max_seq_len"] == 1024 + 64
    decoder = dict(built[1][1])
    assert built[1][0] == "decoder" and decoder["draft_len"] == 5
    assert tuple(decoder["page_table_torch"].shape) == (16, 8) and decoder["page_table_torch"].eq(0).all()
    assert mtp._mtp_captured and mtp._mtp_max_rows == 16 and mtp._mtp_shapes == DEFAULT_SHAPES
    assert mtp._mtp_decoder.widths[(1, 5)] == adapter.mtp_pv_width_ladder(1024, 5)
    assert mtp._mtp_decoder.widths[(16, 1)] == adapter.mtp_pv_width_ladder(1024, 1)


def test_warmup_caps_the_records_at_the_engine_batch_and_the_row_cap(adapter, monkeypatch, assistant_dir):
    mtp = build_mtp_model(
        adapter,
        monkeypatch,
        warm=False,
        max_batch_size=6,
        env={"GEMMA4_ASSISTANT_MODEL": assistant_dir, "GEMMA4_SPEC_MAX_ROWS": "8", "GEMMA4_SPEC_MAX_ISL": "512"},
    )
    stub_drafter_modules(monkeypatch, mtp.events)
    _warmup(mtp, False)
    assert mtp._mtp_shapes == ((1, 5), (5, 5), (6, 3)) and mtp._mtp_max_rows == 6
    assert mtp._mtp_widths[(1, 5)] == adapter.mtp_pv_width_ladder(512, 5)


def test_a_traced_only_warmup_prepares_before_it_captures(adapter, monkeypatch, assistant_dir):
    mtp = build_mtp_model(adapter, monkeypatch, warm=False, env={"GEMMA4_ASSISTANT_MODEL": assistant_dir})
    stub_drafter_modules(monkeypatch, mtp.events)
    _warmup(mtp, True)
    assert [event[0] for event in mtp.events] == ["base_warmup", "prepare", "capture"]


def test_warmup_refuses_a_second_capture_and_a_missing_assistant(
    adapter, monkeypatch, assistant_dir, tmp_path, expect_error
):
    mtp = build_mtp_model(adapter, monkeypatch, warm=False, env={"GEMMA4_ASSISTANT_MODEL": assistant_dir})
    stub_drafter_modules(monkeypatch, mtp.events)
    _warmup(mtp, False)
    _warmup(mtp, True)
    with expect_error(RuntimeError, "captured twice"):
        _warmup(mtp, True)
    missing = build_mtp_model(adapter, monkeypatch, warm=False, env={"GEMMA4_ASSISTANT_MODEL": str(tmp_path / "none")})
    with expect_error(RuntimeError, "GEMMA4_ASSISTANT_MODEL"):
        _warmup(missing, False)


# -- page tables -------------------------------------------------------------------------------


def test_page_tables_are_installed_from_the_step_and_refreshed_only_on_change(mtp):
    _start_solo(mtp)
    assert _names(mtp, "refresh") == [("refresh", [[[10, 11, 0, 0]], [[10, 11, 0, 0]]])]
    _verify(mtp, [_block(150, [101, 102, 103, 104, 105])], [_positions(3, 6)], [5], [10])
    _propose(mtp, [[150, 151, 152, 153, 154, 155]], [_positions(4, 6)], counts=[6])
    assert len(_names(mtp, "refresh")) == 1, "an unchanged table is not re-installed"
    # The block table grew by one block: the full per-layer tables are re-installed and the
    # record's flat row follows.
    _verify(
        mtp,
        [_block(155, [201, 202, 203, 204, 205])],
        [_positions(9, 6)],
        [5],
        [10],
        page_table=_tensor([[10, 11, 12, 0]]),
    )
    _propose(mtp, [[250, 251, 252, 253, 254, 255]], [_positions(10, 6)], counts=[6])
    assert _names(mtp, "refresh")[-1] == ("refresh", [[[10, 11, 12, 0]], [[10, 11, 12, 0]]])
    assert _steps(mtp)[-1][7] == [[10, 11, 12, 0]]


def test_the_flat_record_table_selects_the_drafting_rows_from_the_wire_table(mtp):
    _prefill(mtp, rows=3)
    _ordinary(mtp, [3, 4, 5], [2, 1019, 2], [10, 12, 14], result=_tensor([30, 31, 32]))
    proposal = _propose(
        mtp, [[30] + [-1] * 5, [31] + [-1] * 5, [32] + [-1] * 5], [[3] + [-1] * 5, [1020] + [-1] * 5, [3] + [-1] * 5]
    )
    assert proposal.num_valid.tolist() == [5, 0, 5]
    step = _steps(mtp)[-1]
    assert step[2] == [0, 2] and step[7][:2] == [[10, 11, 0, 0], [14, 15, 0, 0]]
    assert _names(mtp, "refresh")[-1][1][0] == [[10, 11, 0, 0], [12, 13, 0, 0], [14, 15, 0, 0]]


def test_bounded_targets_leave_the_target_stash_alone(adapter, monkeypatch):
    """The decoder receives its per-layer tables explicitly; the plain decode installs the stash per call."""
    mtp = build_mtp_model(adapter, monkeypatch, ring=2048, max_seq_len=4096)
    _start_solo(mtp)
    assert not hasattr(mtp.model[0], "_active_page_tables_per_layer")
    assert len(_names(mtp, "refresh")) == 1


# -- the shared verify-logits buffer: read back at proposal time when a row will need it ----------


def test_a_greedy_only_replay_performs_no_readback(mtp):
    _start_solo(mtp)
    assert mtp._mtp_decoder.readbacks == 0 and mtp._mtp_sessions[0].retained.logits is None
    _verify(mtp, [_block(150, [101, 102, 103, 104, 105])], [_positions(3, 6)], [5], [10])
    assert mtp._mtp_decoder.readbacks == 0


def test_a_mixed_replay_reads_the_logits_back_exactly_once_at_proposal_time(mtp):
    _prefill(mtp, rows=3)
    mixed = _sampling(3, temperature=[0.7, 0.0, 0.0])
    _ordinary(mtp, [3, 4, 5], [2, 2, 2], [10, 12, 14], result=_tensor([30, 31, 32]), sampling_params=mixed)
    _propose(mtp, [[30 + r] + [-1] * 5 for r in range(3)], [[3] + [-1] * 5] * 3)
    decoder = mtp._mtp_decoder
    assert decoder.readbacks == 1 and decoder.results[-1].readbacks == 1
    for slot in range(3):
        retained = mtp._mtp_sessions[slot].retained
        assert tuple(retained.logits.shape) == (6, VOCAB) and retained.logits.dtype == torch.bfloat16
    drafts = [mtp._mtp_sessions[r].retained.drafts for r in range(3)]
    out = _verify(
        mtp, [_block(30 + r, drafts[r]) for r in range(3)], [_positions(3, 6)] * 3, [5] * 3, [10, 12, 14], mode="logits"
    )
    assert decoder.readbacks == 1, "the verify answers from the retained host logits"
    assert out.logits[0].float().argmax(dim=-1).tolist() == [150, 151, 152, 153, 154, 155]
    assert out.logits[2].float().argmax(dim=-1).tolist() == [170, 171, 172, 173, 174, 175]


def test_rows_of_a_logits_verify_are_read_back_at_the_next_proposal(mtp):
    _start_solo(mtp)
    _verify(mtp, [_block(150, [101, 102, 103, 104, 105])], [_positions(3, 6)], [5], [10], mode="logits")
    assert mtp._mtp_decoder.readbacks == 1, "a greedy-only replay is read back late, while its buffer is live"
    _propose(mtp, [[150, 151, 152, 153, 154, 155]], [_positions(4, 6)], counts=[6])
    assert mtp._mtp_decoder.readbacks == 2 and mtp._mtp_sessions[0].retained.logits is not None


def test_a_deferred_logits_verify_answers_from_the_logits_retained_at_proposal(mtp):
    # A (slot 0) samples, B (slot 1) is greedy: the shared replay is read back for A.
    _prefill(mtp, rows=2)
    _ordinary(
        mtp, [3, 4], [2, 2], [10, 12], result=_tensor([30, 31]), sampling_params=_sampling(2, temperature=[0.7, 0.0])
    )
    _propose(mtp, [[30] + [-1] * 5, [31] + [-1] * 5], [[3] + [-1] * 5] * 2)
    decoder = mtp._mtp_decoder
    assert decoder.readbacks == 1
    drafts_a = mtp._mtp_sessions[0].retained.drafts
    drafts_b = mtp._mtp_sessions[1].retained.drafts
    # The scheduler runs B alone: B's verify and B's next proposal rewrite the shared buffer.
    _verify(mtp, [[0] + [-1] * 5, _block(31, drafts_b)], [[-1] * 6, _positions(3, 6)], [0, 5], [0, 12])
    _propose(mtp, [[-1] * 6, [160, 161, 162, 163, 164, 165]], [[-1] * 6, _positions(4, 6)], counts=[1, 6])
    assert decoder.replays == 2 and decoder.readbacks == 1
    assert decoder.buffer[0, 0].argmax().item() != 150, "the buffer now holds B's replay"
    # A's deferred verify in logits mode answers from the logits retained with A's proposal. The
    # step itself is host-sampled, so B's greedy-only replay, still live in the buffer, is read
    # back for B's own future verify (one more readback, none of it for A).
    out = _verify(
        mtp, [_block(30, drafts_a), [0] + [-1] * 5], [_positions(3, 6), [-1] * 6], [5, 0], [10, 0], mode="logits"
    )
    assert out.logits[0].float().argmax(dim=-1).tolist() == [150, 151, 152, 153, 154, 155]
    assert decoder.readbacks == 2 and mtp._mtp_sessions[1].retained.logits is not None


def test_a_greedy_row_whose_buffer_was_overwritten_cannot_answer_a_logits_verify(mtp, expect_error):
    _start_solo(mtp)  # greedy A at slot 0: no readback
    _prefill(mtp, slot=1, key=20)
    _ordinary(mtp, [0, 9], [-1, 2], [0, 20], result=_tensor([0, 42]))
    _propose(mtp, [[-1] * 6, [42] + [-1] * 5], [[-1] * 6, [3] + [-1] * 5])
    assert mtp._mtp_decoder.replays == 2 and mtp._mtp_decoder.readbacks == 0
    with expect_error(RuntimeError, "overwritten the shared verify-logits buffer"):
        _verify(
            mtp,
            [_block(150, [101, 102, 103, 104, 105]), [0] + [-1] * 5],
            [_positions(3, 6), [-1] * 6],
            [5, 0],
            [10, 0],
            mode="logits",
        )


# -- multimodal composition ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "vision", "info"),
    [
        (
            "Gemma4MTPContractVisionForConditionalGeneration",
            "Gemma4VisionForConditionalGeneration",
            "Gemma4ProcessingInfo",
        ),
        (
            "Gemma4MTPContractUnifiedVisionForConditionalGeneration",
            "Gemma4UnifiedVisionForConditionalGeneration",
            "Gemma4UnifiedProcessingInfo",
        ),
    ],
)
def test_composed_vision_classes_keep_the_contract_above_the_multimodal_prefill(adapter, name, vision, info):
    cls = getattr(adapter, name)
    contract = adapter.Gemma4MTPContractForCausalLM
    vision_cls = getattr(adapter, vision)
    mro = cls.__mro__
    assert mro.index(contract) < mro.index(adapter.Gemma4SpecContractBase) < mro.index(vision_cls)
    assert mro.index(vision_cls) < mro.index(adapter.Gemma4ForCausalLM)
    assert adapter.SupportsMultiModal in mro
    # The contract's prefill_forward is the override; the next definition in the MRO is the
    # multimodal-staging base prefill, untouched by the vision classes.
    assert cls.prefill_forward is contract.prefill_forward
    after = [c for c in mro[mro.index(contract) + 1 :] if "prefill_forward" in c.__dict__]
    assert after[0] is adapter.Gemma4ForCausalLM
    assert cls.decode_forward is contract.decode_forward
    assert cls.propose_draft_tokens is contract.propose_draft_tokens
    assert cls.model_capabilities["supports_spec_decode"] is True
    registry = sys.modules["vllm.multimodal"].MULTIMODAL_REGISTRY
    if isinstance(registry, SimpleNamespace):
        pytest.skip("vLLM absent: the registry is the conftest stub; registration is checked on the vLLM runners")
    factory = cls.__dict__["_processor_factory"]
    assert factory.info is getattr(adapter, info)
    assert factory.processor is adapter.Gemma4MultiModalProcessor
    assert factory.dummy_inputs is adapter.Gemma4DummyInputsBuilder


def test_the_serving_classes_are_the_baseline_and_the_two_contract_rails(adapter):
    """The block-output rails are gone: every speculating class serves the plugin contract."""
    classes = sorted(name for name in dir(adapter) if name.startswith("Gemma4") and name.endswith("ForCausalLM"))
    assert classes == ["Gemma4DFlashContractForCausalLM", "Gemma4ForCausalLM", "Gemma4MTPContractForCausalLM"]
    for name in classes[::2]:
        cls = getattr(adapter, name)
        assert issubclass(cls, adapter.Gemma4SpecContractBase)
        assert cls.model_capabilities["output_tokens_per_step"] == 1
    assert issubclass(adapter.Gemma4DFlashContractForCausalLM, adapter.Gemma4DFlashBase)


def test_a_deferred_greedy_row_is_read_back_when_a_sampled_request_takes_its_first_step(mtp):
    """G replays greedy-only (no readback); G's verify is deferred; S prefills and takes its narrow
    step, which reads G's still-live replay back (exactly once); S's proposal overwrites the buffer;
    G's logits verify answers G's own columns with no further readback."""
    _start_solo(mtp)  # G at slot 0
    decoder = mtp._mtp_decoder
    assert decoder.readbacks == 0 and mtp._mtp_sessions[0].retained.logits is None
    _prefill(mtp, slot=1, key=20)  # S
    _ordinary(
        mtp, [0, 9], [-1, 2], [0, 20], result=_tensor([0, 42]), sampling_params=_sampling(2, temperature=[0.0, 0.7])
    )
    assert decoder.readbacks == 1 and mtp._mtp_sessions[0].retained.logits is not None
    _propose(mtp, [[-1] * 6, [42] + [-1] * 5], [[-1] * 6, [3] + [-1] * 5])
    assert decoder.replays == 2 and decoder.readbacks == 2, "S's own proposal reads its replay back"
    assert decoder.buffer[0, 0].argmax().item() != 150, "the buffer now holds S's replay"
    out = _verify(
        mtp,
        [_block(150, [101, 102, 103, 104, 105]), [0] + [-1] * 5],
        [_positions(3, 6), [-1] * 6],
        [5, 0],
        [10, 0],
        mode="logits",
    )
    assert out.logits[0].float().argmax(dim=-1).tolist() == [150, 151, 152, 153, 154, 155]
    assert decoder.readbacks == 2


# -- the record follows the proposing rows, never the wire batch ---------------------------------


def _wire(model, live_rows, anchors, positions, keys, total, result, **kwargs):
    """An ordinary step over ``total`` wire rows with the given live rows (-1 positions elsewhere)."""
    toks = [0] * total
    poss = [-1] * total
    ks = [0] * total
    for r, a, p, key in zip(live_rows, anchors, positions, keys):
        toks[r], poss[r], ks[r] = a, p, key
    return _ordinary(model, toks, poss, ks, result=result, **kwargs)


def _committed(total, live_rows, anchors, positions):
    committed = [[-1] * 6 for _ in range(total)]
    posits = [[-1] * 6 for _ in range(total)]
    for r, a, p in zip(live_rows, anchors, positions):
        committed[r] = [a] + [-1] * 5
        posits[r] = [p] + [-1] * 5
    return committed, posits


def test_one_proposing_row_in_a_wide_batch_takes_the_solo_record_without_padding(mtp):
    _prefill(mtp, slot=7, key=10)
    _wire(mtp, [7], [3], [2], [10], 20, _tensor([0] * 7 + [30] + [0] * 12))
    committed, posits = _committed(20, [7], [30], [3])
    proposal = _propose(mtp, committed, posits)
    assert proposal.num_valid.tolist() == [0] * 7 + [5] + [0] * 12
    step = _steps(mtp)[-1]
    assert step[1] == (1, 5) and step[2] == [7] and step[7] == [[10, 11, 0, 0]]
    assert _names(mtp, "reseed")[-1] == ("reseed", (1, 5), [7], [30], [3])
    assert mtp._mtp_decoder.reseed_tables[-1] == [[10, 11, 0, 0]]


def test_rows_no_record_can_hold_leave_before_the_record_is_chosen(mtp):
    """20 live rows, 19 of them at the end of max_seq_len: the one proposing row takes the solo record."""
    mtp.model_args[0].max_seq_len = 1024
    _prefill(mtp, rows=20)
    keys = [10 + 2 * r for r in range(20)]
    positions = [1023] * 20
    positions[5] = 2
    _ordinary(mtp, [3] * 20, positions, keys, result=_tensor([30] * 20))
    committed, posits = _committed(20, range(20), [30] * 20, [p + 1 for p in positions])
    proposal = _propose(mtp, committed, posits)
    assert proposal.num_valid.tolist() == [0] * 5 + [5] + [0] * 14
    step = _steps(mtp)[-1]
    assert step[1] == (1, 5) and step[2] == [5] and step[7] == [[20, 21, 0, 0]]


def test_three_proposing_rows_in_a_wide_batch_pad_the_record_with_zero_tables(mtp):
    live = [4, 9, 17]
    for slot, key in zip(live, (10, 20, 30)):
        _prefill(mtp, slot=slot, key=key)
    _wire(mtp, live, [3, 4, 5], [2, 2, 2], [10, 20, 30], 32, _tensor([0] * 32))
    committed, posits = _committed(32, live, [30, 31, 32], [3, 3, 3])
    proposal = _propose(mtp, committed, posits)
    assert [int(v) for v in proposal.num_valid[live]] == [5, 5, 5] and int(proposal.num_valid.sum()) == 15
    step = _steps(mtp)[-1]
    assert step[1] == (5, 5) and step[2] == live
    assert step[7] == [[10, 11, 0, 0], [20, 21, 0, 0], [30, 31, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_seventeen_proposing_rows_decline_above_the_largest_record(mtp):
    _prefill(mtp, rows=17)
    keys = [10 + 2 * r for r in range(17)]
    _ordinary(mtp, [3] * 17, [2] * 17, keys, result=_tensor([30] * 17))
    committed, posits = _committed(17, range(17), [30] * 17, [3] * 17)
    assert _propose(mtp, committed, posits).num_valid.tolist() == [0] * 17 and _steps(mtp) == []


def test_a_reseed_of_a_subset_of_the_record_rows_carries_only_their_tables(mtp):
    """Two rows in steady state and a newcomer: the reseed is the newcomer's own call (its table first,
    zeros after), the replay's table lists every drafting row; the decoder's zero-padding check holds."""
    _prefill(mtp, rows=2)
    _ordinary(mtp, [3, 4], [2, 2], [10, 12], result=_tensor([30, 31]))
    _propose(mtp, [[30] + [-1] * 5, [31] + [-1] * 5], [[3] + [-1] * 5] * 2)
    drafts = [mtp._mtp_sessions[r].retained.drafts for r in range(2)]
    _prefill(mtp, slot=2, key=40)  # C joins: its first decode is the straddle verify below
    blocks = [_block(30, drafts[0]), _block(31, drafts[1]), _block(9, [])]
    _verify(
        mtp,
        blocks,
        [_positions(3, 6), _positions(3, 6), _positions(2, 1)],
        [5, 5, 0],
        [10, 12, 40],
        result=_tensor([0, 0, 42]),
    )
    committed = [[150, 151, 152, 153, 154, 155], [160, 161, 162, 163, 164, 165], [42] + [-1] * 5]
    proposal = _propose(mtp, committed, [_positions(4, 6), _positions(4, 6), [3] + [-1] * 5], counts=[6, 6, 1])
    assert proposal.num_valid.tolist() == [5, 5, 5]
    assert _names(mtp, "reseed")[-1] == ("reseed", (5, 5), [2], [42], [3])
    assert mtp._mtp_decoder.reseed_tables[-1] == [[40, 41, 0, 0]] + [[0, 0, 0, 0]] * 4
    step = _steps(mtp)[-1]
    assert step[2:6] == ([0, 1, 2], [155, 165, 42], [9, 9, 3], [5, 5, None])
    assert step[7] == [[10, 11, 0, 0], [12, 13, 0, 0], [40, 41, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_three_non_contiguous_rows_over_two_rounds_keep_every_per_row_value_aligned(mtp):
    """Slots 4, 9, 17 of a 20-row wire batch through propose/verify/propose/verify with different accepted
    counts: the decoder receives each row's own anchor, position, seed column and table in record order,
    and the runner receives each wire row's own posterior, in both accept modes."""
    live = [4, 9, 17]
    keys = [10, 20, 30]
    for slot, key in zip(live, keys):
        _prefill(mtp, slot=slot, key=key)
    first = [30, 31, 32]
    _wire(
        mtp,
        live,
        [3, 4, 5],
        [2, 2, 2],
        keys,
        20,
        _tensor([first[live.index(r)] if r in live else 0 for r in range(20)]),
    )
    committed, posits = _committed(20, live, first, [3, 3, 3])
    proposal = _propose(mtp, committed, posits)
    decoder = mtp._mtp_decoder
    step = _steps(mtp)[-1]
    assert step[1:7] == ((5, 5), live, first, [3, 3, 3], [None, None, None], 1024)
    assert step[7][:3] == [[10, 11, 0, 0], [20, 21, 0, 0], [30, 31, 0, 0]] and step[7][3:] == [[0, 0, 0, 0]] * 2
    for i, r in enumerate(live):
        retained = mtp._mtp_sessions[r].retained
        assert retained.shape_row == i and retained.row_id == r
        assert retained.drafts == [100 + 10 * i + j for j in range(1, 6)]
        assert proposal.draft_token_ids[r].tolist() == retained.drafts
    # Round 1 verify (argmax mode): rows accept 5, 2 and 0 drafts; every wire row answers its own posterior.
    valid = [5, 2, 0]
    blocks = [[0] + [-1] * 5] * 20
    positions = [[-1] * 6] * 20
    for i, r in enumerate(live):
        blocks[r] = _block(first[i], mtp._mtp_sessions[r].retained.drafts[: valid[i]])
        positions[r] = _positions(3, 1 + valid[i])
    out = _verify(
        mtp,
        blocks,
        positions,
        [valid[live.index(r)] if r in live else 0 for r in range(20)],
        [keys[live.index(r)] if r in live else 0 for r in range(20)],
    )
    expected1 = [[150 + 10 * i + j for j in range(6)] for i in range(3)]
    for i, r in enumerate(live):
        assert out.argmax_ids[r].tolist() == expected1[i]
    assert len(_names(mtp, "decode")) == 1, "no device forward for a fully answered verify"
    # Round 2 propose: counts 6 / 3 / 1 commit the accepted prefix + bonus; each row seeds from ITS column.
    counts = [6, 3, 1]
    committed = [[-1] * 6 for _ in range(20)]
    posits = [[-1] * 6 for _ in range(20)]
    for i, r in enumerate(live):
        committed[r] = [expected1[i][j] if j < counts[i] else -1 for j in range(6)]
        posits[r] = _positions(4, counts[i])
    proposal = _propose(mtp, committed, posits, counts=[counts[live.index(r)] if r in live else 1 for r in range(20)])
    step = _steps(mtp)[-1]
    assert step[2] == live
    assert step[3] == [expected1[i][counts[i] - 1] for i in range(3)]
    assert step[4] == [3 + c for c in counts]
    assert step[5] == [c - 1 for c in counts]
    assert len(_names(mtp, "reseed")) == 1 and decoder.replays == 2
    assert step[7][:3] == [[10, 11, 0, 0], [20, 21, 0, 0], [30, 31, 0, 0]]
    for i, r in enumerate(live):
        retained = mtp._mtp_sessions[r].retained
        assert retained.shape_row == i and retained.row_id == r and retained.anchor_position == 3 + counts[i]
        assert proposal.draft_token_ids[r].tolist() == [200 + 10 * i + j for j in range(1, 6)]
    # Round 2 verify in logits mode (a sampled request in the batch): each wire row's columns are its own.
    valid = [4, 5, 1]
    blocks = [[0] + [-1] * 5] * 20
    positions = [[-1] * 6] * 20
    for i, r in enumerate(live):
        blocks[r] = _block(expected1[i][counts[i] - 1], mtp._mtp_sessions[r].retained.drafts[: valid[i]])
        positions[r] = _positions(3 + counts[i], 1 + valid[i])
    out = _verify(
        mtp,
        blocks,
        positions,
        [valid[live.index(r)] if r in live else 0 for r in range(20)],
        [keys[live.index(r)] if r in live else 0 for r in range(20)],
        mode="logits",
    )
    assert tuple(out.logits.shape) == (20, 6, VOCAB)
    for i, r in enumerate(live):
        assert out.logits[r].float().argmax(dim=-1).tolist() == [250 + 10 * i + j for j in range(6)]
    for r in range(20):
        if r not in live:
            assert out.logits[r].eq(0).all()


def test_a_sampled_request_is_verified_in_logits_mode_from_its_own_columns(mtp):
    """The runner picks the mode per step (logits once a live row samples); the adapter answers either mode,
    and in logits mode the sampled row's columns come from its own record row, not a neighbour's."""
    live = [2, 7]
    for slot, key in zip(live, (10, 20)):
        _prefill(mtp, slot=slot, key=key)
    sampling = _sampling(10, temperature=[0.7 if r == 7 else 0.0 for r in range(10)])
    _wire(mtp, live, [3, 4], [2, 2], [10, 20], 10, _tensor([0, 0, 30, 0, 0, 0, 0, 31, 0, 0]), sampling_params=sampling)
    assert mtp._mtp_row_needs_logits == {2: False, 7: True}
    committed, posits = _committed(10, live, [30, 31], [3, 3])
    _propose(mtp, committed, posits)
    decoder = mtp._mtp_decoder
    assert decoder.readbacks == 1, "a sampled row in the replay: the logits are read back at proposal time"
    assert mtp._mtp_sessions[7].retained.logits.float().argmax(dim=-1).tolist() == [160, 161, 162, 163, 164, 165]
    blocks = [[0] + [-1] * 5] * 10
    positions = [[-1] * 6] * 10
    for i, r in enumerate(live):
        blocks[r] = _block([30, 31][i], mtp._mtp_sessions[r].retained.drafts)
        positions[r] = _positions(3, 6)
    out = _verify(
        mtp,
        blocks,
        positions,
        [5 if r in live else 0 for r in range(10)],
        [10 if r == 2 else 20 if r == 7 else 0 for r in range(10)],
        mode="logits",
    )
    assert out.spec_mode == "logits"
    assert out.logits[2].float().argmax(dim=-1).tolist() == [150, 151, 152, 153, 154, 155]
    assert out.logits[7].float().argmax(dim=-1).tolist() == [160, 161, 162, 163, 164, 165]
    assert decoder.readbacks == 1


# -- asynchronous scheduling: the declarations, the deferred verify readback, overlapped ordinary steps ----


def test_the_deferred_verify_declaration_is_inherited_by_the_composed_classes_and_not_by_dflash(adapter):
    for name in (
        "Gemma4MTPContractVisionForConditionalGeneration",
        "Gemma4MTPContractUnifiedVisionForConditionalGeneration",
    ):
        caps = getattr(adapter, name).model_capabilities
        assert caps["supports_async_decode"] is True and caps["supports_async_spec_decode"] is True
    dflash = adapter.Gemma4DFlashContractForCausalLM.model_capabilities
    assert dflash["supports_async_decode"] is False and dflash["supports_async_spec_decode"] is False


@pytest.mark.parametrize(
    ("env", "async_decode", "async_spec"),
    [
        ({"GEMMA4_SUPPORTS_ASYNC_SPEC_DECODE": "0"}, True, False),
        ({"GEMMA4_SUPPORTS_ASYNC_DECODE": "0"}, False, True),
        ({"GEMMA4_SUPPORTS_ASYNC_DECODE": "false", "GEMMA4_SUPPORTS_ASYNC_SPEC_DECODE": "no"}, False, False),
    ],
)
def test_the_async_declarations_follow_their_environment_gates(env, async_decode, async_spec):
    with pytest.MonkeyPatch.context() as patch:
        module = import_adapter(patch, env)
        caps = module.Gemma4MTPContractForCausalLM.model_capabilities
        assert caps["supports_async_decode"] is async_decode and caps["supports_async_spec_decode"] is async_spec
        assert module.Gemma4MTPContractVisionForConditionalGeneration.model_capabilities is caps
        dflash = module.Gemma4DFlashContractForCausalLM.model_capabilities
        assert dflash["supports_async_decode"] is False and dflash["supports_async_spec_decode"] is False


def test_read_decode_output_passes_a_host_tensor_through_and_routes_device_output_to_the_base_reader(mtp):
    """The runner's async read (``submit_decode`` with ``async_read=True``): a verify's host tensor comes back
    as ``(tensor, [])``, the plain decode's device output goes to the base reader with the async flag."""
    ids = torch.tensor([[150, 151]], dtype=torch.int32)
    host, events = mtp.read_decode_output(ids, async_read=True)
    assert host is ids and events == []
    assert mtp.read_decode_output(ids) is ids
    _prefill(mtp)
    out = _ordinary(mtp, [3], [2], [10], result="device")
    assert mtp.read_decode_output(out, async_read=True) == ["device"]
    assert _names(mtp, "read")[-1] == ("read", True)


def _read(model, out):
    """The deferred read of a verify: the mode's tensor, unwrapped from the VerifyOutput as the runner does,
    handed to ``read_decode_output`` and waited on later (the host tensor carries no events)."""
    tensor = out.argmax_ids if out.spec_mode == "argmax_ids" else out.logits
    host, events = model.read_decode_output(tensor, async_read=True)
    assert events == []
    return host


def _greedy_walk(block, valid, ids):
    """The plugin's greedy walk (``accept_greedy_drafts``): every draft the target argmax agrees with, then the
    argmax at the first disagreeing column (the correction, or the bonus past the last draft)."""
    accepted = 0
    while accepted < valid and int(ids[accepted]) == int(block[1 + accepted]):
        accepted += 1
    return [int(ids[j]) for j in range(accepted + 1)], accepted + 1


def _script_two_replays(model):
    """Replay 1 agrees with three drafts, replay 2 with two (every id below the stub vocabulary)."""
    model._mtp_decoder.script = [
        [([101, 102, 103, 104, 105], [101, 102, 103, 477, 478, 479])],
        [([201, 202, 203, 204, 205], [201, 202, 488, 489, 490, 491])],
    ]


def _verify_walk_propose(model, anchor, position, drafts, mode, deferred):
    """One verify of ``drafts`` after ``anchor`` at ``position``, read now (synchronous order) or through the
    deferred read, walked on the host, committed and proposed from."""
    block = _block(anchor, drafts)
    out = _verify(model, [block], [_positions(position, 6)], [5], [10], mode=mode)
    host = _read(model, out) if deferred else (out.argmax_ids if mode == "argmax_ids" else out.logits)
    ids = host[0] if mode == "argmax_ids" else host[0].float().argmax(dim=-1)
    committed, count = _greedy_walk(block, 5, ids)
    proposal = _propose(model, [committed + [-1] * (6 - count)], [_positions(position + 1, count)], counts=[count])
    return committed, count, proposal.draft_token_ids[0].tolist()


@pytest.mark.parametrize("mode", ["argmax_ids", "logits"])
def test_a_deferred_verify_readback_commits_what_the_synchronous_order_commits(adapter, monkeypatch, mode):
    """Two rounds of verify / read / walk / propose: reading the verify's tensor later through
    ``read_decode_output(async_read=True)`` commits the same tokens, seeds the same columns and drives the
    decoder with the same calls as reading it with the submission."""
    runs = {}
    for deferred in (False, True):
        model = build_mtp_model(adapter, monkeypatch)
        _script_two_replays(model)
        drafts = _start_solo(model).draft_token_ids[0].tolist()
        committed1, count1, drafts2 = _verify_walk_propose(model, 150, 3, drafts, mode, deferred)
        committed2, count2, drafts3 = _verify_walk_propose(model, committed1[-1], 3 + count1, drafts2, mode, deferred)
        runs[deferred] = (
            committed1,
            count1,
            drafts2,
            committed2,
            count2,
            drafts3,
            _steps(model),
            _names(model, "reseed"),
            _names(model, "decode"),
            model._mtp_decoder.readbacks,
        )
    assert runs[True] == runs[False]
    committed1, count1, drafts2, committed2, count2, drafts3, steps, reseeds, decodes, _ = runs[True]
    assert (committed1, count1) == ([101, 102, 103, 477], 4) and (committed2, count2) == ([201, 202, 488], 3)
    assert drafts2 == [201, 202, 203, 204, 205] and drafts3 == [301, 302, 303, 304, 305]
    assert [step[5] for step in steps] == [[None], [3], [2]], "each proposal seeds from the committed column"
    assert len(reseeds) == 1 and len(decodes) == 1, "the first plain step only; every verify was answered"


def test_the_retained_replay_survives_a_straddle_plain_decode_until_the_deferred_read_and_the_next_proposal(mtp):
    """A logits verify whose peer row takes the plain decode (a synchronous device read inside the step)
    answers the drafted row from the retained replay; the deferred read after that straddle returns that same
    block, and the proposal that follows still seeds the drafted row from the replay's verify hidden."""
    _start_solo(mtp)
    _prefill(mtp, slot=1, key=20)
    decoder = mtp._mtp_decoder
    replay = mtp._mtp_sessions[0].retained.replay
    blocks = [_block(150, [101, 102, 103, 104, 105]), _block(9, [])]
    out = _verify(
        mtp, blocks, [_positions(3, 6), _positions(2, 1)], [5, 0], [10, 20], mode="logits", result=logits_result(2, 40)
    )
    assert _names(mtp, "decode")[-1] == ("decode", [150, 9], [-1, 2]) and _names(mtp, "read")[-1] == ("read", False)
    assert decoder.readbacks == 1, "the greedy-only replay was read back at the step, before the straddle"
    host = _read(mtp, out)
    assert host is out.logits
    assert host[0].float().argmax(dim=-1).tolist() == [150, 151, 152, 153, 154, 155]
    assert host[1, 0].float().argmax().item() == 41 and host[1, 1:].eq(0).all()
    assert mtp._mtp_sessions[0].retained.replay is replay and mtp._mtp_last_replay is replay
    committed = [[150, 151, 152, 153, 154, 155], [41] + [-1] * 5]
    proposal = _propose(mtp, committed, [_positions(4, 6), [3] + [-1] * 5], counts=[6, 1])
    assert proposal.num_valid.tolist() == [5, 5]
    assert _names(mtp, "reseed")[-1] == ("reseed", (5, 5), [1], [41], [3])
    assert _steps(mtp)[-1][1:6] == ((5, 5), [0, 1], [155, 41], [9, 3], [5, None])


def test_an_ordinary_step_submitted_before_the_proposal_drops_it_and_the_next_proposal_reseeds(mtp, expect_error):
    """Overlap under asynchronous scheduling: the runner submits ordinary step N+1 before step N's proposal is
    published, then publishes N's and N+1's proposals at their drains. The adapter sees propose, propose with no
    verify between: the first proposal is dropped rather than verified (contract s.4d), the row reseeds at the
    token the overlapped step committed, and the dropped drafts are never accepted by a verify."""
    _prefill(mtp)
    _ordinary(mtp, [3], [2], [10], result=_tensor([150]))
    steady = dict(
        reload_inputs=False, reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False
    )
    _ordinary(mtp, [150], [3], [10], result=_tensor([151]), **steady)
    first = _propose(mtp, [[150] + [-1] * 5], [[3] + [-1] * 5])
    assert first.draft_token_ids[0].tolist() == [101, 102, 103, 104, 105]
    assert mtp._mtp_sessions[0].retained is not None and not mtp._mtp_sessions[0].retained.consumed
    second = _propose(mtp, [[151] + [-1] * 5], [[4] + [-1] * 5])
    assert second.draft_token_ids[0].tolist() == [201, 202, 203, 204, 205] and mtp._mtp_dropped_proposals == 1
    assert _names(mtp, "reseed")[-1] == ("reseed", (1, 5), [0], [151], [4])
    assert _steps(mtp)[-1][1:6] == ((1, 5), [0], [151], [4], [None])
    with expect_error(ValueError, "did not propose"):
        _verify(mtp, [_block(151, [101, 102, 103, 104, 105])], [_positions(4, 6)], [5], [10])
    out = _verify(mtp, [_block(151, [201, 202, 203, 204, 205])], [_positions(4, 6)], [5], [10])
    assert out.argmax_ids[0].tolist() == [250, 251, 252, 253, 254, 255]
