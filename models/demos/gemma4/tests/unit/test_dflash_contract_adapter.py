# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host tests for Gemma4 dFlash on the plugin's speculative-decoding contract.

The runner supplies the candidate block and owns the accept walk; this class
answers verify steps from the posterior its own propose replay produced. These
tests drive ``prefill_forward``, ``decode_forward`` and ``propose_draft_tokens``
in the runner's order over a recording fused-decoder stub. What a replay
computes on the device is not something a host test can attest.
"""

import json

import pytest
import torch

from models.demos.gemma4.tests.unit.conftest import build_model, import_adapter
from models.demos.gemma4.tests.unit.dflash_contract_harness import (
    DeviceResult,
    _ordinary,
    _prefill,
    _propose,
    _start_solo,
    _table,
    _tensor,
    _verify,
    make_expect_error,
)


@pytest.fixture
def expect_error():
    return make_expect_error()


def _names(model, kind):
    return [event for event in model.events if event[0] == kind]


def _logits(rows, hit):
    """Host-sampling logits: row r's argmax is ``hit + r``."""
    logits = torch.full((1, 1, rows, 600), -1.0)
    for r in range(rows):
        logits[0, 0, r, hit + r] = 1.0
    return logits


# -- declarations and admission ------------------------------------------------


def test_declares_the_contract_rail(adapter):
    caps = adapter.Gemma4DFlashContractForCausalLM.model_capabilities
    assert caps["output_tokens_per_step"] == 1
    assert caps["supports_spec_decode"] is True
    assert "tt_adaptive_block_output" not in caps
    assert caps["spec_requirements"] == ("device_propose",)
    assert caps["spec_hidden_handoff"] == ("on_device",)
    assert caps["supports_sample_on_device"] is True
    assert caps["supports_chunked_prefill"] is True
    assert caps["supports_async_decode"] is False
    assert caps["supports_async_spec_decode"] is False


@pytest.mark.parametrize("value", ["1", "true"])
def test_async_gate_parses_like_the_other_gates(value):
    with pytest.MonkeyPatch.context() as patch:
        module = import_adapter(patch, {"GEMMA4_CONTRACT_ASYNC": value})
        caps = module.Gemma4DFlashContractForCausalLM.model_capabilities
        assert caps["supports_async_decode"] is True
        assert caps["supports_async_spec_decode"] is True


def test_verify_count_pins_the_inherited_width_arithmetic(adapter):
    cls = adapter.Gemma4DFlashContractForCausalLM
    assert cls._SPEC_CONTRACT_K == 5
    assert cls._SPEC_V == 5 and cls._SPEC_N == 6
    assert cls._SPEC_BLOCK >= 2


@pytest.fixture
def drafter_snapshot(tmp_path, monkeypatch):
    cfg = {
        "num_hidden_layers": 5,
        "hidden_size": 5376,
        "head_dim": 128,
        "num_key_value_heads": 8,
        "block_size": 16,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("GEMMA4_DFLASH_DRAFTER", str(tmp_path))
    monkeypatch.setenv("MESH_DEVICE", "P150x8")
    return tmp_path


def test_spec_plan_admits_concurrency_and_declares_narrow_decode(adapter, drafter_snapshot):
    from vllm_tt_plugin.spec_decode import SpecPlan

    plan = adapter.Gemma4DFlashContractForCausalLM.spec_plan(None, 32, 5)
    assert isinstance(plan, SpecPlan)
    assert plan.effective_k == 5
    assert plan.supports_narrow_decode is True
    assert plan.accept_modes == ("argmax_ids",)


@pytest.mark.parametrize(
    "env, needle",
    [
        ({"GEMMA4_DFLASH_PACKED": "0"}, "packed"),
        ({"GEMMA4_DFLASH_WIDTH_SET": "0"}, "GEMMA4_DFLASH_WIDTH_SET"),
        ({"GEMMA4_DFLASH_WARMUP_DECODE": "0"}, "GEMMA4_DFLASH_WARMUP_DECODE"),
    ],
)
def test_spec_plan_rejects_configurations_that_capture_during_serving(
    adapter, drafter_snapshot, monkeypatch, env, needle
):
    from vllm_tt_plugin.spec_decode import SpecReject

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    out = adapter.Gemma4DFlashContractForCausalLM.spec_plan(None, 1, 5)
    assert isinstance(out, SpecReject)
    assert needle in out.reason


def test_spec_plan_rejects_a_verify_count_outside_the_drafter_block(adapter, drafter_snapshot, monkeypatch):
    from vllm_tt_plugin.spec_decode import SpecReject

    monkeypatch.setenv("GEMMA4_DFLASH_BLOCK", "4")  # block_size 4 admits verify counts 1..3
    out = adapter.Gemma4DFlashContractForCausalLM.spec_plan(None, 1, 5)
    assert isinstance(out, SpecReject)
    assert "outside [1, 3]" in out.reason


# -- 3.1 the first solo ordinary step -------------------------------------------


def test_solo_prefill_then_ordinary_step_bootstraps_and_answers_from_the_replay(model):
    _prefill(model, prompt_len=2, key=10)
    assert model._spec_pending is not None
    assert model._spec_pending[1] == 2
    assert model.model[0].keep_last == 12
    out = _ordinary(model, [3], [2], [10])
    assert out.dtype == torch.int32 and out.tolist() == [150]
    assert len(_names(model, "bootstrap")) == 1
    assert _names(model, "replay") == [("replay", True, 2)]
    assert _names(model, "decode") == []  # the plain decode never ran
    assert model._dflash_retained is None
    assert model._dflash_owner_tables[0].tolist() == _table([10]).tolist()
    assert model._slots_prefilled_since_decode == {0}
    assert model._spec_owner_slot == 0


def test_padded_solo_ordinary_step_answers_at_the_live_row(model):
    _prefill(model, prompt_len=2, key=10)
    out = _ordinary(model, [3, 0, 0, 0], [2, -1, -1, -1], [10, 0, 0, 0])
    assert out.tolist() == [150, 0, 0, 0]


def test_host_sampled_ordinary_step_runs_the_plain_decode_and_drops_the_taps(model):
    _prefill(model, prompt_len=2, key=10)
    taps = list(model._spec_pending[0])
    out = _ordinary(model, [3], [2], [10], result="device", sampling=False)
    assert out == "device"
    assert model._spec_pending is None
    assert all(tap.releases == 1 for tap in taps)
    assert _names(model, "bootstrap") == []


def test_two_live_rows_on_the_first_step_run_the_plain_decode_and_release(model):
    _prefill(model, prompt_len=2, key=10)
    out = _ordinary(model, [3, 4], [2, 5], [10, 20], result="device")
    assert out == "device"
    assert model._spec_pending is None and not model._spec_active
    # The next solo step for the same request has nothing to start from.
    out = _ordinary(model, [3], [3], [10], result="device")
    assert out == "device"
    assert _names(model, "bootstrap") == []


def test_a_request_prefilled_with_another_never_proposes(model):
    _prefill(model, prompt_len=2, key=10, rows=2)
    assert model._spec_pending is None
    assert model.model[0].tap_layers is None
    _ordinary(model, [3], [2], [10], result="device")
    proposal = _propose(model, [[7, -1, -1, -1, -1, -1]], [[3, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [0]
    assert model._spec_decoder.replays == 0


def test_a_request_at_a_row_other_than_zero_is_not_bootstrapped(model):
    _prefill(model, prompt_len=2, key=10, slot=1)
    out = _ordinary(model, [0, 3], [-1, 2], [0, 10], result="device")
    assert out == "device"
    assert model._spec_pending is None and _names(model, "bootstrap") == []


def test_bootstrap_refuses_a_position_no_captured_width_covers(model):
    _prefill(model, prompt_len=2, key=10)
    model._spec_decoder._pv_widths[1024]["trace"] = None  # prepared, never captured
    out = _ordinary(model, [3], [2], [10], result="device")
    assert out == "device"
    assert model._spec_pending is None and _names(model, "bootstrap") == []


# -- 3.2 propose: commit, refresh, declines --------------------------------------


def test_propose_commits_the_runners_count_and_anchor_before_the_replay(model):
    _start_solo(model)
    commits = _names(model, "commit")
    replays = _names(model, "replay")
    assert commits == [("commit", 1, 150)]
    assert model.events.index(commits[0]) < model.events.index(replays[1])
    retained = model._dflash_retained
    assert retained.anchor_token == 150 and retained.anchor_position == 3
    assert retained.drafts == [201, 202, 203, 204, 205]
    assert retained.posterior == [250, 251, 252, 253, 254, 255]
    assert retained.consumed is False


def test_propose_refreshes_the_verify_page_tables_when_the_block_table_changes(model):
    _start_solo(model)
    assert len(_names(model, "refresh")) == 1  # the bootstrap's own refresh
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    # The runner allocated a block: the owner's table grew.
    model._dflash_owner_tables = (_tensor([[10, 11, 12, 0]]), None, model.kv_cache)
    _propose(model, [[201, 202, 251, -1, -1, -1]], [[4, 5, 6, -1, -1, -1]], counts=[3])
    refreshes = _names(model, "refresh")
    assert refreshes[-1] == ("refresh", [10, 11, 12, 0])
    assert model.events.index(refreshes[-1]) < model.events.index(_names(model, "replay")[-1])
    assert _names(model, "commit")[-1] == ("commit", 3, 251)


def test_propose_without_a_table_change_does_not_refresh(model):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    _propose(model, [[201, 202, 251, -1, -1, -1]], [[4, 5, 6, -1, -1, -1]], counts=[3])
    assert len(_names(model, "refresh")) == 1


def test_propose_installs_per_layer_tables_under_bounded_sliding(adapter, monkeypatch):
    model = build_model(adapter, monkeypatch, ring=2048)  # headroom: ring > window + P_v
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    model._dflash_owner_tables = (_tensor([[10, 11, 12, 0]]), None, model.kv_cache)
    _propose(model, [[201, 202, 251, -1, -1, -1]], [[4, 5, 6, -1, -1, -1]], counts=[3])
    installed = model.model[0]._active_page_tables_per_layer
    assert [t.tolist() for t in installed] == [[[10, 11, 12, 0]], [[10, 11, 12, 0]]]


def test_propose_declines_past_the_widest_captured_width(model):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    last_covered = 1024 - 6 - 64
    model._spec_decoder.start = last_covered  # committing one token moves past it
    proposal = _propose(model, [[201, 0, 0, -1, -1, -1]], [[last_covered + 1, 0, 0, -1, -1, -1]], counts=[1])
    assert proposal.num_valid.tolist() == [0]
    assert not model._spec_active
    assert model.events[-1] == ("release_decoder", False)


def test_propose_declines_when_the_verify_rows_exceed_max_seq_len(model):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    model.model_args[0].max_seq_len = 11  # rows 6..11 need 12 positions
    proposal = _propose(model, [[201, 202, 251, -1, -1, -1]], [[4, 5, 6, -1, -1, -1]], counts=[3])
    assert proposal.num_valid.tolist() == [0]
    assert not model._spec_active


def test_propose_declines_before_wrapping_an_exact_window_ring(adapter, monkeypatch):
    model = build_model(adapter, monkeypatch, ring=1024, widths=(4096,))  # ring == window
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    model._spec_decoder.start = 1024 - 6 - 1
    proposal = _propose(model, [[201, 0, 0, -1, -1, -1]], [[1024 - 6, 0, 0, -1, -1, -1]], counts=[1])
    assert proposal.num_valid.tolist() == [5]  # rows 1018..1023 fit
    _verify(model, [[201, 301, 302, 303, 304, 305]], [list(range(1024 - 6, 1024))], [5], keys=[10])
    proposal = _propose(model, [[301, 0, 0, -1, -1, -1]], [[1024 - 5, 0, 0, -1, -1, -1]], counts=[1])
    assert proposal.num_valid.tolist() == [0]  # rows 1019..1024 wrap
    assert not model._spec_active


def test_a_ring_with_headroom_does_not_decline(adapter, monkeypatch):
    model = build_model(adapter, monkeypatch, ring=2048, widths=(4096,))
    model.model_args[0].max_seq_len = 8192
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    model._spec_decoder.start = 2048 - 6
    proposal = _propose(model, [[201, 0, 0, -1, -1, -1]], [[2048 - 5, 0, 0, -1, -1, -1]], counts=[1])
    assert proposal.num_valid.tolist() == [5]


def test_after_a_decline_the_next_ordinary_step_is_plain(model):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    model.model_args[0].max_seq_len = 11
    _propose(model, [[201, 202, 251, -1, -1, -1]], [[4, 5, 6, -1, -1, -1]], counts=[3])
    out = _ordinary(model, [251], [6], [10], result="device")
    assert out == "device"
    proposal = _propose(model, [[9, -1, -1, -1, -1, -1]], [[7, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [0]


def test_propose_declines_when_the_runners_anchor_position_disagrees_with_the_decoder(model):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    proposal = _propose(model, [[201, 202, 251, -1, -1, -1]], [[40, 41, 42, -1, -1, -1]], counts=[3])
    assert proposal.num_valid.tolist() == [0]
    assert not model._spec_active


def test_propose_shape_follows_the_block_rows_and_declines_padding_rows(model):
    _prefill(model, prompt_len=2, key=10)
    _ordinary(model, [3, 0, 0, 0], [2, -1, -1, -1], [10, 0, 0, 0])
    committed = torch.zeros((4, 6), dtype=torch.int32)
    committed[0, 0] = 150
    positions = torch.full((4, 6), -1, dtype=torch.int32)
    positions[0] = torch.arange(3, 9)
    out = model.propose_draft_tokens(5, committed, positions, torch.ones(4, dtype=torch.int32))
    assert tuple(out.draft_token_ids.shape) == (4, 5)
    assert out.draft_token_ids.dtype == torch.int32
    assert out.draft_token_ids[0].tolist() == [201, 202, 203, 204, 205]
    assert out.num_valid.tolist() == [5, 0, 0, 0]
    assert out.draft_scores is None


def test_a_batched_proposal_declines_every_row_and_ends_the_session(model):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    committed = torch.zeros((2, 6), dtype=torch.int32)
    positions = torch.tensor([[4, 5, 6, -1, -1, -1], [9, -1, -1, -1, -1, -1]], dtype=torch.int32)
    out = model.propose_draft_tokens(5, committed, positions, torch.tensor([3, 1], dtype=torch.int32))
    assert out.num_valid.tolist() == [0, 0]
    assert not model._spec_active
    assert model._spec_decoder.replays == 2


# -- 3.3 verify -------------------------------------------------------------------


def test_verify_answers_from_the_retained_posterior_without_the_device(model):
    _start_solo(model)
    replays = model._spec_decoder.replays
    out = _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    assert out.spec_mode == "argmax_ids"
    assert out.argmax_ids.dtype == torch.int32
    assert out.argmax_ids.tolist() == [[250, 251, 252, 253, 254, 255]]
    assert out.hidden is None
    assert model._spec_decoder.replays == replays
    assert _names(model, "decode") == []
    assert model._dflash_retained.consumed is True


def test_verify_tolerates_a_block_truncated_to_num_valid_drafts(model):
    _start_solo(model)
    out = _verify(model, [[150, 201, 202, -1, -1, -1]], [list(range(3, 9))], [2], keys=[10])
    assert out.argmax_ids[0].tolist()[:3] == [250, 251, 252]


def test_verify_refuses_draft_ids_this_model_did_not_propose(model, expect_error):
    _start_solo(model)
    with expect_error(ValueError, "did not propose"):
        _verify(model, [[150, 201, 777, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])


def test_verify_refuses_a_block_that_does_not_continue_the_proposal(model, expect_error):
    _start_solo(model)
    with expect_error(RuntimeError, "does not continue"):
        _verify(model, [[151, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])


def test_verify_with_drafts_and_no_retained_proposal_raises(model, expect_error):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    with expect_error(RuntimeError, "no proposal is retained"):
        _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])


def test_verify_with_drafts_for_a_row_without_a_session_raises(model, expect_error):
    with expect_error(RuntimeError, "owns no drafter session"):
        _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])


def test_verify_for_a_padded_solo_block_answers_at_the_live_row(model):
    _prefill(model, prompt_len=2, key=10)
    _ordinary(model, [3, 0, 0, 0], [2, -1, -1, -1], [10, 0, 0, 0])
    committed = torch.zeros((4, 6), dtype=torch.int32)
    committed[0, 0] = 150
    positions = torch.full((4, 6), -1, dtype=torch.int32)
    positions[0] = torch.arange(3, 9)
    model.propose_draft_tokens(5, committed, positions, torch.ones(4, dtype=torch.int32))
    blocks = torch.zeros((4, 6), dtype=torch.int32)
    blocks[0] = torch.tensor([150, 201, 202, 203, 204, 205])
    out = _verify(model, blocks.tolist(), positions.tolist(), [5, 0, 0, 0], keys=[10, 0, 0, 0])
    assert tuple(out.argmax_ids.shape) == (4, 6)
    assert out.argmax_ids[0].tolist() == [250, 251, 252, 253, 254, 255]
    assert bool(out.argmax_ids[1:].eq(-1).all())


def test_straddle_verify_answers_the_drafted_row_and_decodes_the_peer(model):
    """A peer joined after the proposal. The drafted row is answered from the
    device's own evaluation of its drafts; the peer takes the plain decode,
    which runs with the drafted row at position -1; speculation ends."""
    _start_solo(model)
    out = _verify(
        model,
        [[150, 201, 202, 203, 204, 205], [40, 0, 0, 0, 0, 0]],
        [list(range(3, 9)), [9, 10, 11, 12, 13, 14]],
        [5, 0],
        keys=[10, 20],
        result=DeviceResult(_tensor([0, 600])),
    )
    assert out.argmax_ids[0].tolist() == [250, 251, 252, 253, 254, 255]
    assert int(out.argmax_ids[1, 0]) == 600
    assert _names(model, "decode") == [("decode", [150, 40], [-1, 9])]
    assert not model._spec_active and model._dflash_retained is None
    assert model._slots_prefilled_since_decode == {0}
    proposal = _propose(
        model,
        [[201, 202, 251, -1, -1, -1], [600, -1, -1, -1, -1, -1]],
        [[4, 5, 6, -1, -1, -1], [10, -1, -1, -1, -1, -1]],
        counts=[3, 1],
    )
    assert proposal.num_valid.tolist() == [0, 0]


def test_straddle_verify_finds_the_owner_through_slot_remap(model):
    """The peer took row 0 and the owner moved to row 1 while its state still
    sits in slot 0; the runner says so with slot_remap and reports the move
    only after the step."""
    _start_solo(model)
    out = _verify(
        model,
        [[40, 0, 0, 0, 0, 0], [150, 201, 202, 203, 204, 205]],
        [[9, 10, 11, 12, 13, 14], list(range(3, 9))],
        [0, 5],
        keys=[20, 10],
        result=DeviceResult(_tensor([600, 0])),
        slot_remap=[1, 0],
    )
    assert out.argmax_ids[1].tolist() == [250, 251, 252, 253, 254, 255]
    assert int(out.argmax_ids[0, 0]) == 600
    assert _names(model, "decode") == [("decode", [40, 150], [9, -1])]
    assert not model._spec_active
    model.note_state_slots_moved({0: 1, 1: 0})  # the runner settles the permutation afterwards


def test_ordinary_step_finds_the_owner_through_slot_remap(model):
    _start_solo(model)
    out = _ordinary(model, [0, 150], [-1, 3], [0, 10], slot_remap=[1, 0])
    assert out.tolist() == [0, 250]


def test_verify_for_a_row_whose_slot_owns_nothing_raises_even_at_the_owners_row(model, expect_error):
    _start_solo(model)
    with expect_error(RuntimeError, "owns no drafter session"):
        _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10], slot_remap=[3])


def test_straddle_verify_converts_host_logits_at_the_steps_row_count(model):
    _start_solo(model)
    out = _verify(
        model,
        [[150, 201, 202, 203, 204, 205], [40, 0, 0, 0, 0, 0]],
        [list(range(3, 9)), [9, 10, 11, 12, 13, 14]],
        [5, 0],
        keys=[10, 20],
        result=DeviceResult(_logits(2, 500)),
        sampling=False,
    )
    assert int(out.argmax_ids[1, 0]) == 501
    assert model.model[0].convert_calls == [(2, False)]


def test_batched_draftless_verify_decodes_column_zero_only(model):
    rows = 3
    blocks = torch.arange(rows * 6, dtype=torch.int32).reshape(rows, 6)
    positions = torch.tensor([[100 + 6 * r + c for c in range(6)] for r in range(rows)], dtype=torch.int32)
    out = _verify(
        model,
        blocks.tolist(),
        positions.tolist(),
        [0] * rows,
        keys=[10, 20, 30],
        result=DeviceResult(_tensor([7, 8, 9])),
    )
    assert _names(model, "decode") == [("decode", [0, 6, 12], [100, 106, 112])]
    assert out.argmax_ids[:, 0].tolist() == [7, 8, 9]
    assert bool(out.argmax_ids[:, 1:].eq(-1).all())


def test_padded_rows_keep_the_sentinel_in_the_plain_decode(model):
    blocks = torch.zeros((4, 6), dtype=torch.int32)
    positions = torch.full((4, 6), -1, dtype=torch.int32)
    positions[0] = torch.arange(100, 106)
    positions[1] = torch.arange(200, 206)
    _verify(
        model,
        blocks.tolist(),
        positions.tolist(),
        [0, 0, 0, 0],
        keys=[10, 20, 0, 0],
        result=DeviceResult(_tensor([1, 2, 0, 0])),
    )
    assert _names(model, "decode")[0][2] == [100, 200, -1, -1]


def test_an_ordinary_step_without_drafts_for_an_outstanding_proposal_answers_its_first_id(model):
    """The scheduler dropped the drafts (it does near max_model_len). The
    retained posterior's first id is still the target's choice after the
    anchor, and the next propose commits one token."""
    _start_solo(model)
    out = _ordinary(model, [150], [3], [10])
    assert out.tolist() == [250]
    assert model._dflash_retained.consumed
    proposal = _propose(model, [[250, -1, -1, -1, -1, -1]], [[4, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [5]
    assert _names(model, "commit")[-1] == ("commit", 1, 250)


def test_a_peer_joining_on_an_ordinary_step_ends_speculation_for_good(model):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    out = _ordinary(model, [251, 40], [6, 9], [10, 20], result="device")
    assert out == "device" and not model._spec_active
    _ordinary(model, [7], [7], [10], result="device")
    proposal = _propose(model, [[8, -1, -1, -1, -1, -1]], [[8, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [0]


# -- device-resident decode inputs after drafter-served steps ---------------------


def test_the_first_plain_decode_after_a_straddle_reloads_host_inputs(model):
    """The traced decode keeps tokens and positions on the device and reloads them
    only on a layout change. The owner's rows were answered from the drafter, so
    the peers' decode in the straddle and the owner's first plain decode after
    it both carry reset_batch."""
    _start_solo(model)
    _verify(
        model,
        [[150, 201, 202, 203, 204, 205], [40, 0, 0, 0, 0, 0]],
        [list(range(3, 9)), [9, 10, 11, 12, 13, 14]],
        [5, 0],
        keys=[10, 20],
        result=DeviceResult(_tensor([0, 600])),
    )
    assert model.reloads == [True]
    _ordinary(model, [251, 600], [6, 10], [10, 20], result="device")
    assert model.reloads == [True, True]
    _ordinary(model, [7, 8], [7, 11], [10, 20], result="device")
    assert model.reloads == [True, True, False]


def test_the_first_plain_decode_after_a_decline_reloads_host_inputs(model):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    model.model_args[0].max_seq_len = 11
    _propose(model, [[201, 202, 251, -1, -1, -1]], [[4, 5, 6, -1, -1, -1]], counts=[3])
    _ordinary(model, [251], [6], [10], result="device")
    _ordinary(model, [9], [7], [10], result="device")
    assert model.reloads == [True, False]


def test_a_plain_decode_with_no_drafter_history_does_not_force_a_reload(model):
    _prefill(model, prompt_len=2, key=10, rows=2)
    _ordinary(model, [3, 4], [2, 2], [10, 12], result="device")
    assert model.reloads == [False]


def test_release_logs_committed_tokens_per_step_for_the_session(model, adapter, monkeypatch):
    """The block rail logs tokens per iteration on release; the contract rail
    logs the same per session, counting the first ordinary step as a step."""
    messages = []
    monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: messages.append(str(message)))
    _start_solo(model)  # first ordinary step (1 token) then the first proposal
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    _propose(model, [[201, 202, 251, -1, -1, -1]], [[4, 5, 6, -1, -1, -1]], counts=[3])
    model.release_request(0)
    summary = [m for m in messages if "contract summary" in m]
    assert summary == ["Gemma4DFlash contract summary: 2 steps, 4 tokens, 2.00 tokens/step"]
    assert model._dflash_steps == 0 and model._dflash_committed == 0


# -- identity and release --------------------------------------------------------


def test_release_of_the_owner_clears_the_retained_proposal(model):
    _start_solo(model)
    model.release_request(0)
    assert model._dflash_retained is None and not model._spec_active
    assert model._dflash_live_owner is None and model._spec_owner_slot is None


def test_release_of_another_slot_keeps_the_session(model):
    _start_solo(model)
    model.release_request(3)
    assert model._spec_active and model._dflash_retained is not None


def test_release_of_the_pending_owner_frees_its_taps(model):
    _prefill(model, prompt_len=2, key=10, slot=2)
    taps = list(model._spec_pending[0])
    model.release_request(1)
    assert model._spec_pending is not None
    model.release_request(2)
    assert model._spec_pending is None and all(tap.releases == 1 for tap in taps)


def test_a_reused_slot_does_not_match_a_stale_owner(model):
    _prefill(model, prompt_len=2, key=10)
    model.release_request(0)  # the owner finished before decoding
    _prefill(model, prompt_len=2, key=20, rows=2)  # slots 0 and 1, never speculable
    out = _ordinary(model, [3], [2], [20], result="device")
    assert out == "device" and _names(model, "bootstrap") == []


def test_slot_moves_follow_the_owner(model):
    _start_solo(model)
    model.note_state_slots_moved({0: 3, 3: 0})
    assert model._dflash_live_owner[0] == 3 and model._spec_owner_slot == 3
    model.release_request(0)
    assert model._spec_active
    model.release_request(3)
    assert not model._spec_active


def test_release_persistent_capture_clears_the_rail_and_reaches_the_base(model):
    _start_solo(model)
    model.release_persistent_capture()
    assert model._dflash_retained is None and model._spec_pending is None
    assert ("release_decoder", True) in model.events
    assert model.base_releases == [1]


# -- 3.6 scheduler-chunked prefill ------------------------------------------------


def test_chunked_prefill_accumulates_taps_across_scheduler_chunks(adapter, monkeypatch):
    model = build_model(adapter, monkeypatch, widths=(4096,))
    model.model_args[0].max_seq_len = 8192
    _prefill(model, prompt_len=1024, key=10)
    _prefill(model, prompt_len=2048, key=10, start=1024)
    _prefill(model, prompt_len=2100, key=10, start=2048)
    taps, n = model._spec_pending
    assert n == 2100
    assert [tap.tag for tap in taps] == [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    _ordinary(model, [3], [2100], [10])
    assert _names(model, "ingest")[0][1] == 2100


def test_chunk_groups_below_the_context_window_are_freed(model):
    model._spec_decoder.cap = 1024
    _prefill(model, prompt_len=1024, key=10)
    first = list(model._spec_pending[0])
    _prefill(model, prompt_len=2048, key=10, start=1024)
    _prefill(model, prompt_len=2100, key=10, start=2048)
    taps, _ = model._spec_pending
    assert all(tap.releases == 1 for tap in first)
    assert [tap.tag for tap in taps] == [(2, 0), (2, 1), (3, 0), (3, 1)]


def test_a_solo_prefill_of_another_request_replaces_the_pending_taps(model):
    _prefill(model, prompt_len=1024, key=10, slot=0)
    first = list(model._spec_pending[0])
    _prefill(model, prompt_len=64, key=20, slot=1)
    assert all(tap.releases == 1 for tap in first)
    assert model._spec_pending[1] == 64 and model._dflash_pending_owner[0] == 1
    # The first request's next chunk starts over from its own rows.
    _prefill(model, prompt_len=2048, key=10, slot=0, start=1024)
    assert model._dflash_pending_owner[0] == 0
    assert [tap.tag for tap in model._spec_pending[0]] == [(3, 0), (3, 1)]


def test_an_unaligned_chunk_start_stores_nothing(model):
    _prefill(model, prompt_len=1000, key=10)
    _prefill(model, prompt_len=2000, key=10, start=1000)  # 1000 is not a multiple of 128
    assert model._spec_pending is None


def test_a_prompt_over_the_spec_ceiling_stores_nothing(model, monkeypatch):
    monkeypatch.setenv("GEMMA4_DFLASH_MAX_SPEC_ISL", "100")
    _prefill(model, prompt_len=200, key=10)
    assert model._spec_pending is None and model.model[0].tap_layers is None


def test_warmup_prefill_leaves_the_pending_taps_alone(model):
    _prefill(model, prompt_len=2, key=10)
    model.prefill_forward(
        tokens=torch.zeros((1, 128), dtype=torch.int32),
        prompt_lens=[128],
        page_table=_table([0]),
        kv_cache=model.kv_cache,
        warmup_prefill=True,
    )
    assert model._spec_pending is not None


def test_every_prefill_on_this_rail_is_eager(model):
    _prefill(model, prompt_len=2, key=10)
    _prefill(model, prompt_len=2, key=20, rows=2)
    assert all(event[3] is False for event in _names(model, "prefill"))


def test_a_solo_prefill_keeps_a_live_session_for_its_straddle_verify(model):
    _start_solo(model)
    _prefill(model, prompt_len=2, key=20, slot=1)
    assert model._spec_active and model._dflash_retained is not None
    out = _verify(
        model,
        [[150, 201, 202, 203, 204, 205], [40, 0, 0, 0, 0, 0]],
        [list(range(3, 9)), [2, 3, 4, 5, 6, 7]],
        [5, 0],
        keys=[10, 20],
        result=DeviceResult(_tensor([0, 600])),
    )
    assert out.argmax_ids[0].tolist() == [250, 251, 252, 253, 254, 255]
    assert model._spec_pending is None and not model._spec_active


# -- 3.7 block-table hygiene ---------------------------------------------------------


def test_prefill_zeroes_block_table_columns_past_the_prompt(model):
    stale = _tensor([[10, 11, 99, 98]])
    model.prefill_forward(
        tokens=torch.full((1, 70), 7, dtype=torch.int32),
        prompt_lens=[70],
        empty_slots=[0],
        page_table=stale,
        kv_cache=model.kv_cache,
        page_tables_per_layer=[stale, stale],
        warmup_prefill=False,
    )
    page_table, per_layer = model.last_prefill_tables
    assert page_table.tolist() == [[10, 11, 0, 0]]
    assert [t.tolist() for t in per_layer] == [[[10, 11, 0, 0]], [[10, 11, 0, 0]]]
    assert stale.tolist() == [[10, 11, 99, 98]]  # the runner's tensor is untouched


def test_prefill_masks_per_row_and_leaves_ring_layers_alone(adapter, monkeypatch):
    model = build_model(adapter, monkeypatch, ring=2048)
    tables = _tensor([[10, 11, 99, 98], [20, 21, 22, 97]])
    model.prefill_forward(
        tokens=torch.full((2, 130), 7, dtype=torch.int32),
        prompt_lens=[70, 130],
        empty_slots=[0, 1],
        page_table=tables,
        kv_cache=model.kv_cache,
        page_tables_per_layer=[tables, tables],
        warmup_prefill=False,
    )
    page_table, per_layer = model.last_prefill_tables
    assert page_table.tolist() == [[10, 11, 0, 0], [20, 21, 22, 0]]
    assert per_layer[0] is tables  # layer 0 is the bounded sliding layer
    assert per_layer[1].tolist() == [[10, 11, 0, 0], [20, 21, 22, 0]]


# -- helpers --------------------------------------------------------------------------


def test_live_row_detection_uses_the_padding_sentinel(adapter):
    cls = adapter.Gemma4DFlashContractForCausalLM
    positions = torch.full((4, 6), -1, dtype=torch.int32)
    positions[1] = torch.arange(6)
    assert cls._dflash_live_rows(positions, 4) == [1]
    assert cls._dflash_live_rows(torch.tensor([5, -1, 7]), 3) == [0, 2]
    assert cls._dflash_live_rows(None, 2) == [0, 1]


def test_column_zero_narrowing_shapes_match_the_plain_decode(adapter):
    cls = adapter.Gemma4DFlashContractForCausalLM
    block = torch.arange(12, dtype=torch.int32).reshape(2, 6)
    positions = torch.tensor([[3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14]], dtype=torch.int32)
    args, kwargs = cls._dflash_column_zero((), {"tokens": block, "start_pos": positions}, 2, exclude_row=1)
    assert kwargs["tokens"].tolist() == [[0], [6]]
    assert kwargs["start_pos"].tolist() == [3, -1]
    args, kwargs = cls._dflash_column_zero(
        (), {"tokens": torch.tensor([[7], [8]]), "start_pos": torch.tensor([1, 2])}, 2
    )
    assert kwargs["tokens"].tolist() == [[7], [8]] and kwargs["start_pos"].tolist() == [1, 2]


def test_ring_advice_names_the_setting_and_the_resolved_numbers(adapter, monkeypatch):
    model = build_model(adapter, monkeypatch, ring=1024)
    messages = []
    monkeypatch.setattr(adapter.logger, "warning", lambda message, *a, **k: messages.append(message))
    model._dflash_ring_advice()
    assert len(messages) == 1
    assert "GEMMA4_SPEC_RING_HEADROOM_BLOCKS=16" in messages[0]
    assert "ring 2048" in messages[0] and "P_v=6" in messages[0]
    model = build_model(adapter, monkeypatch, ring=2048)
    model._dflash_ring_advice()
    assert len(messages) == 1


def test_retained_proposal_is_a_plain_record(adapter):
    proposal = adapter.RetainedProposal(anchor_token=1, anchor_position=2, drafts=[3], posterior=[4, 5])
    assert proposal.consumed is False
    assert (proposal.anchor_token, proposal.anchor_position, proposal.drafts, proposal.posterior) == (1, 2, [3], [4, 5])
