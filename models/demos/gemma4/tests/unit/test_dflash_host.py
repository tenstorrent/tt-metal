# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host tests for Gemma4 dFlash on the plugin's speculative-decoding contract.

The runner supplies the candidate block and owns the accept walk; this class
answers verify steps from the posterior its own propose replay produced. These
tests drive ``prefill_forward``, ``decode_forward`` and ``propose_draft_tokens``
in the runner's order over a recording fused-decoder stub. What a replay
computes on the device is not something a host test can attest.
"""

import ast
import dataclasses
import inspect
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tests.unit import mtp_contract_harness as mh
from models.demos.gemma4.tests.unit.conftest import build_model, import_adapter
from models.demos.gemma4.tests.unit.dflash_contract_harness import (
    DeviceResult,
    _ordinary,
    _prefill,
    _propose,
    _sampling,
    _start_solo,
    _table,
    _tensor,
    _verify,
    make_expect_error,
)
from models.demos.gemma4.tests.unit.dflash_sampling_harness import attach_sampling_state
from models.demos.gemma4.tt.spec_policy import CostLine


@pytest.fixture
def expect_error_base():
    return make_expect_error()


@pytest.fixture
def mtp(adapter, monkeypatch):
    """The MTP (assistant drafter) contract rail over its recording stubs, warm: the rail's twin of ``model``."""
    return mh.build_mtp_model(adapter, monkeypatch)


@pytest.fixture
def assistant_dir(tmp_path):
    """An assistant checkpoint dir (only ``config.json`` is read), beside the dFlash ``drafter_snapshot``."""
    path = tmp_path / "assistant"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"backbone_hidden_size": 5376, "hidden_size": 1024}))
    return str(path)


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
    assert caps["max_device_top_k"] == 32
    assert caps["supports_device_penalties"] is False
    assert caps["supports_chunked_prefill"] is True
    assert caps["supports_async_decode"] is False
    assert caps["supports_async_spec_decode"] is False

    # -- the MTP rail: declares the contract rail
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
    # -- the MTP rail: the deferred-verify declaration is inherited by the composed classes and not by dflash
    for name in (
        "Gemma4MTPContractVisionForConditionalGeneration",
        "Gemma4MTPContractUnifiedVisionForConditionalGeneration",
    ):
        caps = getattr(adapter, name).model_capabilities
        assert caps["supports_async_decode"] is True and caps["supports_async_spec_decode"] is True
    dflash = adapter.Gemma4DFlashContractForCausalLM.model_capabilities
    assert dflash["supports_async_decode"] is False and dflash["supports_async_spec_decode"] is False
    # -- the MTP rail: the async declarations follow their environment gates
    for env, async_decode, async_spec in [
        ({"GEMMA4_SUPPORTS_ASYNC_SPEC_DECODE": "0"}, True, False),
        ({"GEMMA4_SUPPORTS_ASYNC_DECODE": "0"}, False, True),
        ({"GEMMA4_SUPPORTS_ASYNC_DECODE": "false", "GEMMA4_SUPPORTS_ASYNC_SPEC_DECODE": "no"}, False, False),
    ]:
        with pytest.MonkeyPatch.context() as patch:
            module = import_adapter(patch, env)
            caps = module.Gemma4MTPContractForCausalLM.model_capabilities
            assert caps["supports_async_decode"] is async_decode and caps["supports_async_spec_decode"] is async_spec
            assert module.Gemma4MTPContractVisionForConditionalGeneration.model_capabilities is caps
            dflash = module.Gemma4DFlashContractForCausalLM.model_capabilities
            assert dflash["supports_async_decode"] is False and dflash["supports_async_spec_decode"] is False


def _plugin_sampling_router():
    """The plugin's ``SamplingInputBatch`` and ``TTModelRunner``, or a skip.

    The routing under test is plugin code. The tt-metal unit-test image has
    only the conftest's ``vllm_tt_plugin`` stub, so these tests run where the
    plugin is installed or on PYTHONPATH.
    """
    input_batch = pytest.importorskip("vllm_tt_plugin.input_batch")
    model_runner = pytest.importorskip("vllm_tt_plugin.model_runner")
    return input_batch.SamplingInputBatch, model_runner.TTModelRunner


@pytest.mark.parametrize(
    ("temperature", "top_k", "device_sampling"),
    [
        (0.0, 262144, True),
        (0.7, 1, True),
        (0.7, 32, True),
        (0.7, 33, False),
        (0.7, 262144, False),
    ],
)
def test_contract_sampled_top_k_uses_host_when_the_device_cannot_honor_it(adapter, temperature, top_k, device_sampling):
    SamplingInputBatch, TTModelRunner = _plugin_sampling_router()

    sampling = SamplingInputBatch(1)
    sampling.temperature[0] = temperature
    sampling.top_k[0] = top_k
    runner = SimpleNamespace(
        # The plugin runner reads its seeded-sampling policy before the top-k rule (None = device default).
        seeded_sampling_policy=None,
        sample_on_device_mode="decode_only",
        num_devices=8,
        tt_data_parallel_size=1,
        model=SimpleNamespace(model_capabilities=adapter.Gemma4DFlashContractForCausalLM.model_capabilities),
        model_config=SimpleNamespace(logits_processors=[]),
        input_batch=SimpleNamespace(
            sampling=sampling,
            req_id_to_index={"request": 0},
            no_penalties=True,
            no_allowed_token_ids=True,
            max_num_logprobs=None,
        ),
    )

    assert TTModelRunner.check_perform_device_sampling(runner, True, False) is device_sampling


def test_contract_penalty_request_uses_host_sampler(adapter):
    SamplingInputBatch, TTModelRunner = _plugin_sampling_router()

    sampling = SamplingInputBatch(1)
    sampling.temperature[0] = 0.7
    sampling.top_k[0] = 2
    runner = SimpleNamespace(
        # The plugin runner reads its seeded-sampling policy before the top-k rule (None = device default).
        seeded_sampling_policy=None,
        sample_on_device_mode="decode_only",
        num_devices=8,
        tt_data_parallel_size=1,
        model=SimpleNamespace(model_capabilities=adapter.Gemma4DFlashContractForCausalLM.model_capabilities),
        model_config=SimpleNamespace(logits_processors=[]),
        input_batch=SimpleNamespace(
            sampling=sampling,
            req_id_to_index={"request": 0},
            no_penalties=False,
            no_allowed_token_ids=True,
            max_num_logprobs=None,
        ),
    )

    assert TTModelRunner.check_perform_device_sampling(runner, True, False) is False


@pytest.mark.parametrize("value", ["1", "true"])
def test_dflash_cannot_enable_resident_decode_with_an_environment_override(value):
    with pytest.MonkeyPatch.context() as patch:
        module = import_adapter(patch, {"GEMMA4_CONTRACT_ASYNC": value})
        caps = module.Gemma4DFlashContractForCausalLM.model_capabilities
        assert caps["supports_async_decode"] is False
        assert caps["supports_async_spec_decode"] is False


def test_verify_count_pins_the_inherited_width_arithmetic(adapter, expect_error):
    cls = adapter.Gemma4DFlashContractForCausalLM
    assert cls._SPEC_CONTRACT_K == 5
    assert cls._SPEC_V == 5 and cls._SPEC_N == 6

    # -- the MTP rail: k by rows table parses clamps and checks the tile
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
    # -- the MTP rail: plan k by rows appends the tail only when the table stops short
    table = ((5, 5), (8, 3), (16, 1))
    assert adapter.spec_plan_k_by_rows(table, 16) == table
    assert adapter.spec_plan_k_by_rows(table, 8) == table
    assert adapter.spec_plan_k_by_rows(table, 32) == table + ((32, 1),)
    # -- the MTP rail: capture shapes add the solo record and clamp to the row cap
    table = ((5, 5), (8, 3), (16, 1))
    assert adapter.spec_capture_shapes(table, 5, 16) == mh.DEFAULT_SHAPES
    assert adapter.spec_capture_shapes(table, 5, 6) == ((1, 5), (5, 5), (6, 3))
    assert adapter.spec_capture_shapes(table, 5, 1) == ((1, 5),)
    assert adapter.spec_capture_shapes(((4, 3),), 3, 16) == ((1, 3), (4, 3))
    # -- the MTP rail: extra records join the table's as a band's second draft length, never its default
    assert adapter.spec_extra_shapes("5:3, 1:2", 5) == ((5, 3), (1, 2)) and adapter.spec_extra_shapes("", 5) == ()
    assert adapter.spec_capture_shapes(table, 5, 16, extra=((5, 3), (1, 2))) == (
        (1, 2),
        (1, 5),
        (5, 3),
        (5, 5),
        (8, 3),
        (16, 1),
    )
    assert adapter.spec_capture_shapes(table, 5, 4, extra=((5, 3), (4, 5))) == ((1, 5), (4, 3), (4, 5))
    with expect_error(ValueError, "drafts above"):
        adapter.spec_extra_shapes("5:6", 5)
    with expect_error(ValueError, "above the device argmax tile"):
        adapter.spec_extra_shapes("8:5", 5)
    with expect_error(ValueError, "is not rows:k"):
        adapter.spec_extra_shapes("x", 5)


def test_contract_rail_shares_the_dflash_base_and_the_contract_base(adapter, monkeypatch):
    cls = adapter.Gemma4DFlashContractForCausalLM
    assert issubclass(cls, adapter.Gemma4DFlashBase)
    assert issubclass(cls, adapter.Gemma4SpecContractBase)
    assert cls.model_capabilities["output_tokens_per_step"] == 1

    # -- the MTP rail: the serving classes are the baseline and the two contract rails
    # The block-output rails are gone: every speculating class serves the plugin contract.
    classes = sorted(name for name in dir(adapter) if name.startswith("Gemma4") and name.endswith("ForCausalLM"))
    assert classes == ["Gemma4DFlashContractForCausalLM", "Gemma4ForCausalLM", "Gemma4MTPContractForCausalLM"]
    for name in classes[::2]:
        cls = getattr(adapter, name)
        assert issubclass(cls, adapter.Gemma4SpecContractBase)
        assert cls.model_capabilities["output_tokens_per_step"] == 1
    assert issubclass(adapter.Gemma4DFlashContractForCausalLM, adapter.Gemma4DFlashBase)

    # -- the acceptance-adaptive policy: the composed vision rail shares it
    _dflash_the_composed_vision_rail_shares_the_policy(adapter, monkeypatch)


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


def test_spec_plan_admits_concurrency_and_declares_narrow_decode(adapter, drafter_snapshot, monkeypatch, assistant_dir):
    from vllm_tt_plugin.spec_decode import SpecPlan

    plan = adapter.Gemma4DFlashContractForCausalLM.spec_plan(None, 32, 5)
    assert isinstance(plan, SpecPlan)
    assert plan.effective_k == 5
    assert plan.supports_narrow_decode is True
    assert plan.accept_modes == ("argmax_ids",)

    # -- the MTP rail: spec plan is positional and declares the assistant drafter
    monkeypatch.setenv("GEMMA4_ASSISTANT_MODEL", assistant_dir)
    monkeypatch.delenv("GEMMA4_SPEC_K_BY_ROWS", raising=False)
    plan = adapter.Gemma4MTPContractForCausalLM.spec_plan(None, 32, 5)
    assert plan.effective_k == 5 and plan.lanes_per_request == 1
    assert plan.accept_modes == ("argmax_ids", "logits")
    assert plan.drafter_state == "shared_with_target"
    assert plan.drafter_target_cache_requires == ("named_layer_caches", "absolute_positions")
    assert plan.supports_narrow_decode is True
    assert plan.k_by_rows == ((5, 5), (32, 1))  # the default table plus the plan's admission tail
    assert plan.extra_bytes_per_seq == 5376 * 2 + 16 and plan.extra_bytes_per_token == 0
    # -- the MTP rail: spec plan reads the table and clamps to the requested k
    monkeypatch.setenv("GEMMA4_ASSISTANT_MODEL", assistant_dir)
    monkeypatch.setenv("GEMMA4_SPEC_K_BY_ROWS", "4:7,8:3")
    plan = adapter.Gemma4MTPContractForCausalLM.spec_plan(None, 8, 3)
    assert plan.effective_k == 3
    assert plan.k_by_rows == ((4, 3), (8, 3))
    plan = adapter.Gemma4MTPContractForCausalLM.spec_plan(None, 4, 9)
    assert plan.effective_k == 5 and plan.k_by_rows == ((4, 5), (8, 3))


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


def test_spec_plan_rejects_a_verify_count_outside_the_drafter_block(
    adapter, drafter_snapshot, monkeypatch, expect_error, assistant_dir, tmp_path
):
    from vllm_tt_plugin.spec_decode import SpecReject

    monkeypatch.setenv("GEMMA4_DFLASH_BLOCK", "4")  # block_size 4 admits verify counts 1..3
    out = adapter.Gemma4DFlashContractForCausalLM.spec_plan(None, 1, 5)
    assert isinstance(out, SpecReject)
    assert "outside [1, 3]" in out.reason

    # -- the MTP rail: the assistant follows the target checkpoint name
    for target, repo in [
        ("google/gemma-4-31B-it", "google/gemma-4-31B-it-assistant"),
        ("google/gemma-4-12B-it", "google/gemma-4-12B-it-assistant"),
        ("google/gemma-4-26B-A4B-it", "google/gemma-4-26B-A4B-it-assistant"),
        ("/data/weights/gemma-4-26b-a4b-it/", "google/gemma-4-26B-A4B-it-assistant"),
        ("/data/weights/GEMMA-4-12B-IT", "google/gemma-4-12B-it-assistant"),
    ]:
        calls = []
        monkeypatch.setattr(
            adapter, "_hf_resolve_repo", lambda repo_id, cache_dir: calls.append((repo_id, cache_dir)) or "/snap"
        )
        assert adapter.assistant_repo_for(target) == repo, target
        assert adapter._assistant_default_snapshot(target) == "/snap"
        assert calls == [(repo, "models--" + repo.replace("/", "--"))], target
    # -- the MTP rail: an unknown target checkpoint raises instead of defaulting to 31b
    monkeypatch.setattr(adapter, "_hf_resolve_repo", lambda repo_id, cache_dir: "/snap")
    with expect_error(ValueError, "gemma-4-9B-it"):
        adapter.assistant_repo_for("google/gemma-4-9B-it")
    with expect_error(ValueError, "no assistant checkpoint is known"):
        adapter._assistant_default_snapshot("/weights/mystery-model")
    with expect_error(ValueError, "more than one checkpoint"):
        adapter.assistant_repo_for("/data/31b-box/gemma-4-12B-it")
    # -- the MTP rail: spec plan rejects a target without a known assistant
    monkeypatch.delenv("GEMMA4_ASSISTANT_MODEL", raising=False)
    monkeypatch.setenv("HF_MODEL", "google/gemma-4-9B-it")
    outcome = adapter.Gemma4MTPContractForCausalLM.spec_plan(None, 8, 5)
    assert type(outcome).__name__ == "SpecReject" and "gemma-4-9B-it" in outcome.reason
    monkeypatch.delenv("HF_MODEL", raising=False)
    # -- the MTP rail: spec plan rejects without an assistant snapshot
    monkeypatch.setenv("GEMMA4_ASSISTANT_MODEL", str(tmp_path / "missing"))
    outcome = adapter.Gemma4MTPContractForCausalLM.spec_plan(None, 8, 5)
    assert type(outcome).__name__ == "SpecReject"
    assert "GEMMA4_ASSISTANT_MODEL" in outcome.reason and outcome.supported_k == ()
    # -- the MTP rail: spec plan rejects an unreadable assistant config
    (tmp_path / "unreadable").mkdir()
    (tmp_path / "unreadable" / "config.json").write_text(json.dumps({"hidden_size": 1024}))
    monkeypatch.setenv("GEMMA4_ASSISTANT_MODEL", str(tmp_path / "unreadable"))
    outcome = adapter.Gemma4MTPContractForCausalLM.spec_plan(None, 8, 5)
    assert type(outcome).__name__ == "SpecReject" and "backbone_hidden_size" in outcome.reason
    # -- the MTP rail: spec plan rejects a table the argmax tile cannot hold
    monkeypatch.setenv("GEMMA4_ASSISTANT_MODEL", assistant_dir)
    monkeypatch.setenv("GEMMA4_SPEC_K_BY_ROWS", "8:5")
    outcome = adapter.Gemma4MTPContractForCausalLM.spec_plan(None, 8, 5)
    assert type(outcome).__name__ == "SpecReject" and "argmax tile" in outcome.reason
    assert outcome.supported_k == (5,)


# -- 3.1 the first solo ordinary step -------------------------------------------


def test_solo_prefill_then_ordinary_step_bootstraps_and_answers_from_the_replay(model, mtp, adapter, monkeypatch):
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
    assert model._spec_owner_slot == 0

    # -- the MTP rail: solo request proposes after a plain first step and answers the verify
    proposal = mh._start_solo(mtp)
    # The first step was the plain decode; the proposal reseeded the row and replayed the solo record.
    assert _names(mtp, "decode") == [("decode", [3], [2])]
    assert _names(mtp, "reseed") == [("reseed", (1, 5), [0], [150], [3])]
    step = mh._steps(mtp)[0]
    assert step[1:7] == ((1, 5), [0], [150], [3], [None], 1024)
    assert step[7] == [[10, 11, 0, 0]]
    assert proposal.draft_token_ids.tolist() == [[101, 102, 103, 104, 105]]
    out = mh._verify(mtp, [mh._block(150, [101, 102, 103, 104, 105])], [mh._positions(3, 6)], [5], [10])
    assert out.spec_mode == "argmax_ids" and out.hidden is None
    assert out.argmax_ids.tolist() == [[150, 151, 152, 153, 154, 155]]
    assert out.argmax_ids.dtype == torch.int32
    assert len(_names(mtp, "decode")) == 1, "the verify was answered from the retained posterior"

    # -- the acceptance-adaptive policy: the first request speculates from the optimistic prior and the
    # accepted counts feed the row's EWMA and the batch prior (both rails)
    _dflash_the_first_request_speculates_from_the_optimistic_prior_and_the_counts_feed_its_ewma(adapter, monkeypatch)
    _mtp_the_first_proposal_speculates_from_the_optimistic_prior_and_high_acceptance_keeps_k5(adapter, monkeypatch)
    _mtp_acceptance_statistics_come_from_the_accepted_counts_of_the_verified_drafts(adapter, monkeypatch)
    _mtp_a_new_request_under_a_pessimistic_prior_gets_one_trial_whose_verify_counts_at_once(adapter, monkeypatch)
    _mtp_a_draftless_verify_and_an_ordinary_step_observe_nothing(adapter, monkeypatch)


def test_padded_solo_ordinary_step_answers_at_the_live_row(model):
    _prefill(model, prompt_len=2, key=10)
    out = _ordinary(model, [3, 0, 0, 0], [2, -1, -1, -1], [10, 0, 0, 0])
    assert out.tolist() == [150, 0, 0, 0]


def test_host_sampled_ordinary_step_runs_the_plain_decode_and_drops_the_taps(
    model, mtp, adapter, monkeypatch, expect_error
):
    _prefill(model, prompt_len=2, key=10)
    taps = list(model._spec_pending[0])
    out = _ordinary(model, [3], [2], [10], result="device", sampling=False)
    assert out == "device"
    assert model._spec_pending is None
    assert all(tap.releases == 1 for tap in taps)
    assert _names(model, "bootstrap") == []

    # -- the MTP rail: a plain ordinary step drops the retained proposal and the next proposal reseeds
    mh._start_solo(mtp)
    out = mh._ordinary(mtp, [150], [3], [10], result="device", sampling=False)
    assert out == "device", "host sampling reads logits, so the plain decode answers"
    proposal = mh._propose(mtp, [[160, -1, -1, -1, -1, -1]], [[4, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [5]
    assert _names(mtp, "reseed")[-1] == ("reseed", (1, 5), [0], [160], [4])
    # -- the MTP rail: read_decode_output passes a host tensor through and routes device output to the base reader
    # The runner's async read (``submit_decode`` with ``async_read=True``): a verify's host tensor comes back
    # as ``(tensor, [])``, the plain decode's device output goes to the base reader with the async flag.
    mtp = mh.build_mtp_model(adapter, monkeypatch)
    ids = torch.tensor([[150, 151]], dtype=torch.int32)
    host, events = mtp.read_decode_output(ids, async_read=True)
    assert host is ids and events == []
    assert mtp.read_decode_output(ids) is ids
    mh._prefill(mtp)
    out = mh._ordinary(mtp, [3], [2], [10], result="device")
    assert mtp.read_decode_output(out, async_read=True) == ["device"]
    assert _names(mtp, "read")[-1] == ("read", True)

    # -- the acceptance-adaptive policy: without a cost table or switched off the rails run as before, and a
    # bad knob is refused at construction (both rails)
    _dflash_without_a_cost_table_every_solo_greedy_request_speculates_and_a_warning_says_so_once(adapter, monkeypatch)
    _mtp_without_a_cost_table_the_rows_based_record_serves_and_a_warning_says_so_once(adapter, monkeypatch)
    _dflash_adaptive_off_keeps_the_rail_as_before_whatever_the_table_says(adapter, monkeypatch)
    _mtp_adaptive_off_keeps_the_rows_based_record_whatever_the_table_says(adapter, monkeypatch)
    _mtp_a_bad_policy_knob_is_refused_when_the_adapter_is_built(adapter, monkeypatch, expect_error=expect_error)


def test_two_live_rows_on_the_first_step_run_the_plain_decode_and_release(model, mtp):
    _prefill(model, prompt_len=2, key=10)
    out = _ordinary(model, [3, 4], [2, 5], [10, 20], result="device")
    assert out == "device"
    assert model._spec_pending is None and not model._spec_active
    # The next solo step for the same request has nothing to start from.
    out = _ordinary(model, [3], [3], [10], result="device")
    assert out == "device"
    assert _names(model, "bootstrap") == []

    # -- the MTP rail: rows no record can hold leave before the record is chosen
    # 20 live rows, 19 of them at the end of max_seq_len: the one proposing row takes the solo record.
    mtp.model_args[0].max_seq_len = 1024
    mh._prefill(mtp, rows=20)
    keys = [10 + 2 * r for r in range(20)]
    positions = [1023] * 20
    positions[5] = 2
    mh._ordinary(mtp, [3] * 20, positions, keys, result=_tensor([30] * 20))
    committed, posits = mh._committed(20, range(20), [30] * 20, [p + 1 for p in positions])
    proposal = mh._propose(mtp, committed, posits)
    assert proposal.num_valid.tolist() == [0] * 5 + [5] + [0] * 14
    step = mh._steps(mtp)[-1]
    assert step[1] == (1, 5) and step[2] == [5] and step[7] == [[20, 21, 0, 0]]


def test_a_request_prefilled_with_another_never_proposes(model, mtp):
    _prefill(model, prompt_len=2, key=10, rows=2)
    assert model._spec_pending is None
    assert model.model[0].tap_layers is None
    _ordinary(model, [3], [2], [10], result="device")
    proposal = _propose(model, [[7, -1, -1, -1, -1, -1]], [[3, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [0]
    assert model._spec_decoder.replays == 0

    # -- the MTP rail: three proposing rows in a wide batch pad the record with zero tables
    live = [4, 9, 17]
    for slot, key in zip(live, (10, 20, 30)):
        mh._prefill(mtp, slot=slot, key=key)
    mh._wire(mtp, live, [3, 4, 5], [2, 2, 2], [10, 20, 30], 32, _tensor([0] * 32))
    committed, posits = mh._committed(32, live, [30, 31, 32], [3, 3, 3])
    proposal = mh._propose(mtp, committed, posits)
    assert [int(v) for v in proposal.num_valid[live]] == [5, 5, 5] and int(proposal.num_valid.sum()) == 15
    step = mh._steps(mtp)[-1]
    assert step[1] == (5, 5) and step[2] == live
    assert step[7] == [[10, 11, 0, 0], [20, 21, 0, 0], [30, 31, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_a_request_at_a_row_other_than_zero_is_not_bootstrapped(model, mtp):
    _prefill(model, prompt_len=2, key=10, slot=1)
    out = _ordinary(model, [0, 3], [-1, 2], [0, 10], result="device")
    assert out == "device"
    assert model._spec_pending is None and _names(model, "bootstrap") == []

    # -- the MTP rail: one proposing row in a wide batch takes the solo record without padding
    mh._prefill(mtp, slot=7, key=10)
    mh._wire(mtp, [7], [3], [2], [10], 20, _tensor([0] * 7 + [30] + [0] * 12))
    committed, posits = mh._committed(20, [7], [30], [3])
    proposal = mh._propose(mtp, committed, posits)
    assert proposal.num_valid.tolist() == [0] * 7 + [5] + [0] * 12
    step = mh._steps(mtp)[-1]
    assert step[1] == (1, 5) and step[2] == [7] and step[7] == [[10, 11, 0, 0]]
    assert _names(mtp, "reseed")[-1] == ("reseed", (1, 5), [7], [30], [3])
    assert mtp._mtp_decoder.reseed_tables[-1] == [[10, 11, 0, 0]]


def test_bootstrap_refuses_a_position_no_captured_width_covers(model, adapter, monkeypatch):
    _prefill(model, prompt_len=2, key=10)
    model._spec_decoder._pv_widths[1024]["trace"] = None  # prepared, never captured
    out = _ordinary(model, [3], [2], [10], result="device")
    assert out == "device"
    assert model._spec_pending is None and _names(model, "bootstrap") == []

    # -- the MTP rail: proposing without captured records declines
    mtp = mh.build_mtp_model(adapter, monkeypatch, warm=False)
    mh._prefill(mtp)
    mh._ordinary(mtp, [3], [2], [10], result=_tensor([150]))
    assert mh._propose(mtp, [[150] + [-1] * 5], [[3] + [-1] * 5]).num_valid.tolist() == [0]


# -- 3.2 propose: commit, refresh, declines --------------------------------------


def test_propose_commits_the_runners_count_and_anchor_before_the_replay(model, mtp, adapter, monkeypatch):
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

    # -- the MTP rail: next proposal seeds from the verified column without a reseed
    mh._start_solo(mtp)
    mh._verify(mtp, [mh._block(150, [101, 102, 103, 104, 105])], [mh._positions(3, 6)], [5], [10])
    # The runner accepted 2 drafts and committed the correction: 3 tokens, anchor at column 2.
    proposal = mh._propose(mtp, [[150, 151, 999, -1, -1, -1]], [[4, 5, 6, -1, -1, -1]], counts=[3])
    assert proposal.num_valid.tolist() == [5] and proposal.draft_token_ids.tolist() == [[201, 202, 203, 204, 205]]
    assert len(_names(mtp, "reseed")) == 1
    assert mh._steps(mtp)[-1][1:6] == ((1, 5), [0], [999], [6], [2])

    # -- the acceptance-adaptive policy: the record a proposal replays, the reseed as the entry cost, the
    # prior-gated bootstrap and the measured table (both rails)
    _mtp_a_batch_may_replay_a_smaller_record_than_its_rows_select(adapter, monkeypatch)
    _mtp_the_reseed_cost_of_entering_speculation_is_part_of_the_candidates_cost(adapter, monkeypatch)
    _mtp_a_row_continuing_in_the_last_replay_pays_no_reseed(adapter, monkeypatch)
    _dflash_a_new_request_skips_the_bootstrap_when_the_prior_says_the_replay_would_not_pay(adapter, monkeypatch)
    _dflash_the_cost_table_is_consumed_as_measured(adapter, monkeypatch)


def test_propose_refreshes_the_verify_page_tables_when_the_block_table_changes(model, mtp):
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

    # -- the MTP rail: page tables are installed from the step and refreshed only on change
    mh._start_solo(mtp)
    assert _names(mtp, "refresh") == [("refresh", [[[10, 11, 0, 0]], [[10, 11, 0, 0]]])]
    mh._verify(mtp, [mh._block(150, [101, 102, 103, 104, 105])], [mh._positions(3, 6)], [5], [10])
    # the change check reads no table whose storage and version the last install already saw, and compares a
    # new object of the same content once per distinct object (both layers share the step's one table): an
    # unchanged step costs a key check per layer, not a compare per layer
    equal_calls = []
    real_equal = torch.equal

    def counted_equal(a, b):
        equal_calls.append((tuple(a.shape), tuple(b.shape)))
        return real_equal(a, b)

    torch.equal = counted_equal
    try:
        mh._propose(mtp, [[150, 151, 152, 153, 154, 155]], [mh._positions(4, 6)], counts=[6])
        assert len(equal_calls) <= 1, equal_calls  # the harness hands a new tensor object each step
        _full, layers = mtp._mtp_step_page_tables()
        assert not mtp._mtp_tables_changed(layers) and len(equal_calls) <= 1, "the same objects again: a key check"
        layers[0][0, 2] = 12  # a block appended in place: the version moves, the content is read and differs
        assert mtp._mtp_tables_changed(layers) and len(equal_calls) == 2
        layers[0][0, 2] = 0
    finally:
        torch.equal = real_equal
    assert len(_names(mtp, "refresh")) == 1, "an unchanged table is not re-installed"
    assert len({id(t) for t in mtp._mtp_installed_tables}) == 1 and len(mtp._mtp_installed_tables) == 2
    assert mtp._mtp_installed_keys[0] == mtp._mtp_installed_keys[1] and mtp._mtp_installed_keys[0][2] == (1, 4)
    # The block table grew by one block: the full per-layer tables are re-installed and the
    # record's flat row follows.
    mh._verify(
        mtp,
        [mh._block(155, [201, 202, 203, 204, 205])],
        [mh._positions(9, 6)],
        [5],
        [10],
        page_table=_tensor([[10, 11, 12, 0]]),
    )
    mh._propose(mtp, [[250, 251, 252, 253, 254, 255]], [mh._positions(10, 6)], counts=[6])
    assert _names(mtp, "refresh")[-1] == ("refresh", [[[10, 11, 12, 0]], [[10, 11, 12, 0]]])
    assert mh._steps(mtp)[-1][7] == [[10, 11, 12, 0]]


def test_propose_without_a_table_change_does_not_refresh(model, mtp):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    _propose(model, [[201, 202, 251, -1, -1, -1]], [[4, 5, 6, -1, -1, -1]], counts=[3])
    assert len(_names(model, "refresh")) == 1

    # -- the MTP rail: the flat record table selects the drafting rows from the wire table
    mh._prefill(mtp, rows=3)
    mh._ordinary(mtp, [3, 4, 5], [2, 1019, 2], [10, 12, 14], result=_tensor([30, 31, 32]))
    proposal = mh._propose(
        mtp, [[30] + [-1] * 5, [31] + [-1] * 5, [32] + [-1] * 5], [[3] + [-1] * 5, [1020] + [-1] * 5, [3] + [-1] * 5]
    )
    assert proposal.num_valid.tolist() == [5, 0, 5]
    step = mh._steps(mtp)[-1]
    assert step[2] == [0, 2] and step[7][:2] == [[10, 11, 0, 0], [14, 15, 0, 0]]
    assert _names(mtp, "refresh")[-1][1][0] == [[10, 11, 0, 0], [12, 13, 0, 0], [14, 15, 0, 0]]


def test_propose_installs_per_layer_tables_under_bounded_sliding(adapter, monkeypatch):
    model = build_model(adapter, monkeypatch, ring=2048)  # headroom: ring > window + P_v
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    model._dflash_owner_tables = (_tensor([[10, 11, 12, 0]]), None, model.kv_cache)
    _propose(model, [[201, 202, 251, -1, -1, -1]], [[4, 5, 6, -1, -1, -1]], counts=[3])
    installed = model.model[0]._active_page_tables_per_layer
    assert [t.tolist() for t in installed] == [[[10, 11, 12, 0]], [[10, 11, 12, 0]]]

    # -- the MTP rail: bounded targets leave the target stash alone
    # The decoder receives its per-layer tables explicitly; the plain decode installs the stash per call.
    mtp = mh.build_mtp_model(adapter, monkeypatch, ring=2048, max_seq_len=4096)
    mh._start_solo(mtp)
    assert not hasattr(mtp.model[0], "_active_page_tables_per_layer")
    assert len(_names(mtp, "refresh")) == 1


def test_propose_declines_past_the_widest_captured_width(model, mtp):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    last_covered = 1024 - 6 - 64
    model._spec_decoder.start = last_covered  # committing one token moves past it
    proposal = _propose(model, [[201, 0, 0, -1, -1, -1]], [[last_covered + 1, 0, 0, -1, -1, -1]], counts=[1])
    assert proposal.num_valid.tolist() == [0]
    assert not model._spec_active
    assert model.events[-1] == ("release_decoder", False)

    # -- the MTP rail: a row past the captured widths is left out while the others draft
    mh._prefill(mtp, rows=2)
    mh._ordinary(mtp, [3, 4], [2, 1019], [10, 12], result=_tensor([30, 31]))
    proposal = mh._propose(mtp, [[30] + [-1] * 5, [31] + [-1] * 5], [[3] + [-1] * 5, [1020] + [-1] * 5])
    assert proposal.num_valid.tolist() == [5, 0]
    assert mh._steps(mtp)[-1][1:4] == ((1, 5), [0], [30])


def test_propose_declines_when_the_verify_rows_exceed_max_seq_len(model, mtp):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    model.model_args[0].max_seq_len = 11  # rows 6..11 need 12 positions
    proposal = _propose(model, [[201, 202, 251, -1, -1, -1]], [[4, 5, 6, -1, -1, -1]], counts=[3])
    assert proposal.num_valid.tolist() == [0]
    assert not model._spec_active

    # -- the MTP rail: a row past max seq len declines
    mtp.model_args[0].max_seq_len = 10
    mh.install_decoder(mtp)
    mh._start_solo(mtp)  # rows 3..8 fit; after six committed tokens rows 9..14 do not
    mh._verify(mtp, [mh._block(150, [101, 102, 103, 104, 105])], [mh._positions(3, 6)], [5], [10])
    proposal = mh._propose(mtp, [[150, 151, 152, 153, 154, 155]], [mh._positions(4, 6)], counts=[6])
    assert proposal.num_valid.tolist() == [0]


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

    # -- the MTP rail: an exact window ring declines before the verify would wrap
    mtp = mh.build_mtp_model(adapter, monkeypatch, ring=1024, max_seq_len=4096)
    mh._prefill(mtp)
    mh._ordinary(mtp, [3], [1018], [10], result=_tensor([150]))
    assert mh._propose(mtp, [[150] + [-1] * 5], [[1019] + [-1] * 5]).num_valid.tolist() == [0]
    mh._ordinary(mtp, [150], [1019], [10], result=_tensor([151]))
    assert mh._propose(mtp, [[151] + [-1] * 5], [[1020] + [-1] * 5]).num_valid.tolist() == [0]


def test_a_ring_with_headroom_does_not_decline(adapter, monkeypatch):
    model = build_model(adapter, monkeypatch, ring=2048, widths=(4096,))
    model.model_args[0].max_seq_len = 8192
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    model._spec_decoder.start = 2048 - 6
    proposal = _propose(model, [[201, 0, 0, -1, -1, -1]], [[2048 - 5, 0, 0, -1, -1, -1]], counts=[1])
    assert proposal.num_valid.tolist() == [5]


def test_after_a_decline_the_next_ordinary_step_is_plain(model, mtp, adapter, monkeypatch):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    model.model_args[0].max_seq_len = 11
    _propose(model, [[201, 202, 251, -1, -1, -1]], [[4, 5, 6, -1, -1, -1]], counts=[3])
    out = _ordinary(model, [251], [6], [10], result="device")
    assert out == "device"
    proposal = _propose(model, [[9, -1, -1, -1, -1, -1]], [[7, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [0]

    # -- the MTP rail: above the row cap every row declines once logged then resumes when the batch shrinks
    messages = []
    monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: messages.append(str(message)))
    rows = 17
    mh._prefill(mtp, rows=rows)
    keys = [10 + 2 * r for r in range(rows)]
    # the all-declined step's per-row python budget: the blocks are read to the host once per step and the
    # ordinary step walks no session, so no row is read tensor by tensor and no row's sampling is inspected
    # (a replay's readback is the only reader of that); the c128 Galaxy point paid ~2 ms a step in both
    budget = {"row_reads": 0, "sampling_checks": 0}
    base = adapter.Gemma4SpecContractBase
    orig_row, orig_can = base._spec_row, base._spec_row_can_use_argmax

    def counted_row(values, rows_, row):
        budget["row_reads"] += 1
        return orig_row(values, rows_, row)

    def counted_can(cls, sampling_params, row, grammar_bitmask=None):
        budget["sampling_checks"] += 1
        return orig_can.__func__(cls, sampling_params, row, grammar_bitmask)

    monkeypatch.setattr(base, "_spec_row", staticmethod(counted_row))
    monkeypatch.setattr(base, "_spec_row_can_use_argmax", classmethod(counted_can))
    mh._ordinary(mtp, [3] * rows, [2] * rows, keys, result=_tensor([30] * rows))
    for _ in range(2):
        proposal = mh._propose(mtp, [[30] + [-1] * 5] * rows, [[3] + [-1] * 5] * rows)
        assert proposal.num_valid.tolist() == [0] * rows and proposal.draft_token_ids.eq(0).all()
    assert mh._steps(mtp) == []
    assert [m for m in messages if "proposing nothing" in m] == [
        "Gemma4MTP speculative contract: proposing nothing (17 proposing rows exceed the largest record (16, 1) "
        "(GEMMA4_SPEC_MAX_ROWS=16))"
    ]
    out = mh._ordinary(mtp, [30] * rows, [3] * rows, keys, result="device")
    assert out == "device", "the ordinary step after a decline is the plain decode"
    assert budget == {"row_reads": 0, "sampling_checks": 0}, budget
    assert mtp._mtp_row_needs_logits.get(0) is False and budget["sampling_checks"] == 1, "evaluated on first use"
    assert mtp._mtp_row_needs_logits.get(0) is False and budget["sampling_checks"] == 1, "and cached"
    assert not mtp._mtp_row_needs_logits.any() and budget["sampling_checks"] == rows, "a greedy batch walks all"
    assert mtp._mtp_row_needs_logits.get(99, "off") == "off" and mtp._mtp_row_needs_logits.values() == [False] * rows
    monkeypatch.setattr(base, "_spec_row", staticmethod(orig_row))
    monkeypatch.setattr(base, "_spec_row_can_use_argmax", orig_can)
    # Fewer rows: the next proposal reseeds the survivors and drafts again.
    live = 4
    proposal = mh._propose(mtp, [[31] + [-1] * 5] * live, [[4] + [-1] * 5] * live)
    assert proposal.num_valid.tolist() == [5] * live
    assert _names(mtp, "reseed")[-1] == ("reseed", (5, 5), [0, 1, 2, 3], [31] * 4, [4] * 4)
    # -- the MTP rail: seventeen proposing rows decline above the largest record
    mtp = mh.build_mtp_model(adapter, monkeypatch)
    mh._prefill(mtp, rows=17)
    keys = [10 + 2 * r for r in range(17)]
    mh._ordinary(mtp, [3] * 17, [2] * 17, keys, result=_tensor([30] * 17))
    committed, posits = mh._committed(17, range(17), [30] * 17, [3] * 17)
    assert mh._propose(mtp, committed, posits).num_valid.tolist() == [0] * 17 and mh._steps(mtp) == []

    # -- the acceptance-adaptive policy: falling acceptance leaves for the plain step, the amortized retry,
    # the rate-limited switch log (both rails)
    _dflash_rejections_end_the_session_and_the_request_runs_plain(adapter, monkeypatch)
    _dflash_after_n_plain_steps_the_next_request_bootstraps_and_its_first_proposal_is_the_retry(adapter, monkeypatch)
    _mtp_falling_acceptance_drops_k5_to_k3_then_plain_and_retries_at_k1_after_n_plain_steps(adapter, monkeypatch)
    _mtp_record_switches_are_rate_limited_to_one_line_per_interval(adapter, monkeypatch)
    _mtp_every_record_switch_is_logged_with_the_ewma_the_cost_ratio_and_the_gain(adapter, monkeypatch)


def test_propose_declines_when_the_runners_anchor_position_disagrees_with_the_decoder(model):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    proposal = _propose(model, [[201, 202, 251, -1, -1, -1]], [[40, 41, 42, -1, -1, -1]], counts=[3])
    assert proposal.num_valid.tolist() == [0]
    assert not model._spec_active


def test_propose_shape_follows_the_block_rows_and_declines_padding_rows(model, mtp, adapter, monkeypatch):
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

    # -- the MTP rail: padded wire rows are ignored and keep the placeholder
    mh._prefill(mtp)
    mh._ordinary(mtp, [3, 0], [2, -1], [10, 0], result=_tensor([150, 0]))
    proposal = mh._propose(mtp, [[150] + [-1] * 5, [-1] * 6], [[3] + [-1] * 5, [-1] * 6])
    assert proposal.num_valid.tolist() == [5, 0]
    out = mh._verify(
        mtp,
        [mh._block(150, [101, 102, 103, 104, 105]), [0] + [-1] * 5],
        [mh._positions(3, 6), [-1] * 6],
        [5, 0],
        [10, 0],
    )
    assert out.argmax_ids[0].tolist() == [150, 151, 152, 153, 154, 155]
    assert out.argmax_ids[1].tolist() == [-1] * 6
    # -- the MTP rail: the record is the smallest covering the live rows
    for live, shape, offer in [(2, (5, 5), 5), (6, (8, 3), 3), (9, (16, 1), 1), (16, (16, 1), 1)]:
        mtp = mh.build_mtp_model(adapter, monkeypatch)
        mh._prefill(mtp, rows=live)
        keys = [10 + 2 * r for r in range(live)]
        mh._ordinary(mtp, [3] * live, [2] * live, keys, result=_tensor([30] * live))
        proposal = mh._propose(mtp, [[30] + [-1] * 5] * live, [[3] + [-1] * 5] * live)
        assert proposal.num_valid.tolist() == [offer] * live, live
        step = mh._steps(mtp)[0]
        assert step[1] == shape and len(step[7]) == shape[0], live
        assert all(row == [0, 0, 0, 0] for row in step[7][live:]), "padding rows address the null block"
        assert proposal.draft_token_ids[:, offer:].eq(0).all()
    # -- the MTP rail: a shorter record offers fewer drafts and the verify truncates to them
    mtp = mh.build_mtp_model(adapter, monkeypatch)
    mh._prefill(mtp, rows=6)
    keys = [10 + 2 * r for r in range(6)]
    mh._ordinary(mtp, [3] * 6, [2] * 6, keys, result=_tensor([30] * 6))
    proposal = mh._propose(mtp, [[30] + [-1] * 5] * 6, [[3] + [-1] * 5] * 6)
    drafts = proposal.draft_token_ids.tolist()
    assert drafts[0] == [101, 102, 103, 0, 0]
    blocks = [mh._block(30, drafts[r][:3]) for r in range(6)]
    out = mh._verify(mtp, blocks, [mh._positions(3, 4)] * 6, [3] * 6, keys)
    assert out.argmax_ids[0].tolist() == [150, 151, 152, 153, -1, -1]


def test_a_batched_proposal_declines_every_row_and_ends_the_session(model, mtp):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    committed = torch.zeros((2, 6), dtype=torch.int32)
    positions = torch.tensor([[4, 5, 6, -1, -1, -1], [9, -1, -1, -1, -1, -1]], dtype=torch.int32)
    out = model.propose_draft_tokens(5, committed, positions, torch.tensor([3, 1], dtype=torch.int32))
    assert out.num_valid.tolist() == [0, 0]
    assert not model._spec_active
    assert model._spec_decoder.replays == 2

    # -- the MTP rail: five rows share one replay and verify with per row counts
    mh._prefill(mtp, rows=5)
    out = mh._ordinary(mtp, [3, 4, 5, 6, 7], [2] * 5, [10, 12, 14, 16, 18], result=_tensor([30, 31, 32, 33, 34]))
    assert out.tolist() == [30, 31, 32, 33, 34]
    committed = [[30 + r] + [-1] * 5 for r in range(5)]
    proposal = mh._propose(mtp, committed, [[3] + [-1] * 5] * 5)
    assert proposal.num_valid.tolist() == [5] * 5
    step = mh._steps(mtp)[0]
    assert step[1:4] == ((5, 5), [0, 1, 2, 3, 4], [30, 31, 32, 33, 34])
    assert step[7] == [[10, 11, 0, 0], [12, 13, 0, 0], [14, 15, 0, 0], [16, 17, 0, 0], [18, 19, 0, 0]]
    drafts = proposal.draft_token_ids.tolist()
    valid = [5, 3, 0, 5, 1]
    blocks = [mh._block(30 + r, drafts[r][: valid[r]]) for r in range(5)]
    positions = [mh._positions(3, 1 + valid[r]) for r in range(5)]
    out = mh._verify(mtp, blocks, positions, valid, [10, 12, 14, 16, 18])
    expected = [[150 + 10 * r + j for j in range(6)] for r in range(5)]
    assert out.argmax_ids.tolist() == expected
    assert len(_names(mtp, "decode")) == 1, "every row answered from the shared replay"
    # Each row commits a different count; the next replay seeds every row from its own verified column.
    counts = [6, 2, 1, 4, 2]
    committed = [[expected[r][j] if j < counts[r] else -1 for j in range(6)] for r in range(5)]
    positions = [mh._positions(4, counts[r]) for r in range(5)]
    proposal = mh._propose(mtp, committed, positions, counts=counts)
    assert proposal.num_valid.tolist() == [5] * 5
    step = mh._steps(mtp)[-1]
    assert step[3] == [expected[r][counts[r] - 1] for r in range(5)]
    assert step[4] == [3 + counts[r] for r in range(5)]
    assert step[5] == [c - 1 for c in counts]
    assert len(_names(mtp, "reseed")) == 1


# -- 3.3 verify -------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["argmax_ids", "logits"])
def test_verify_answers_from_the_retained_posterior_without_the_device(model, mode, adapter, monkeypatch):
    """dFlash answers in argmax_ids mode only; the MTP rail answers either mode, read with the submission or
    through the deferred read of asynchronous scheduling."""
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

    # -- the MTP rail: a deferred verify readback commits what the synchronous order commits
    # Two rounds of verify / read / walk / propose: reading the verify's tensor later through
    # ``read_decode_output(async_read=True)`` commits the same tokens, seeds the same columns and drives the
    # decoder with the same calls as reading it with the submission.
    runs = {}
    for deferred in (False, True):
        mtp = mh.build_mtp_model(adapter, monkeypatch)
        mh._script_two_replays(mtp)
        drafts = mh._start_solo(mtp).draft_token_ids[0].tolist()
        committed1, count1, drafts2 = mh._verify_walk_propose(mtp, 150, 3, drafts, mode, deferred)
        committed2, count2, drafts3 = mh._verify_walk_propose(mtp, committed1[-1], 3 + count1, drafts2, mode, deferred)
        runs[deferred] = (
            committed1,
            count1,
            drafts2,
            committed2,
            count2,
            drafts3,
            mh._steps(mtp),
            _names(mtp, "reseed"),
            _names(mtp, "decode"),
            mtp._mtp_decoder.readbacks,
        )
    assert runs[True] == runs[False]
    committed1, count1, drafts2, committed2, count2, drafts3, steps, reseeds, decodes, _ = runs[True]
    assert (committed1, count1) == ([101, 102, 103, 477], 4) and (committed2, count2) == ([201, 202, 488], 3)
    assert drafts2 == [201, 202, 203, 204, 205] and drafts3 == [301, 302, 303, 304, 305]
    assert [step[5] for step in steps] == [[None], [3], [2]], "each proposal seeds from the committed column"
    assert len(reseeds) == 1 and len(decodes) == 1, "the first plain step only; every verify was answered"


def test_verify_tolerates_a_block_truncated_to_num_valid_drafts(model, mtp):
    _start_solo(model)
    out = _verify(model, [[150, 201, 202, -1, -1, -1]], [list(range(3, 9))], [2], keys=[10])
    assert out.argmax_ids[0].tolist()[:3] == [250, 251, 252]

    # -- the MTP rail: verify with fewer drafts than proposed is a valid prefix
    mh._start_solo(mtp)
    out = mh._verify(mtp, [mh._block(150, [101, 102, 103])], [mh._positions(3, 4)], [3], [10])
    assert out.argmax_ids.tolist() == [[150, 151, 152, 153, 154, 155]]


def test_verify_refuses_draft_ids_this_model_did_not_propose(model, expect_error_base, mtp, expect_error):
    _start_solo(model)
    with expect_error_base(ValueError, "did not propose"):
        _verify(model, [[150, 201, 777, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])

    # -- the MTP rail: verify rejects drafts the model did not propose
    mh._start_solo(mtp)
    with expect_error(ValueError, "did not propose"):
        mh._verify(mtp, [mh._block(150, [101, 102, 777, 104, 105])], [mh._positions(3, 6)], [5], [10])


def test_verify_refuses_a_block_that_does_not_continue_the_proposal(model, expect_error_base, mtp, expect_error):
    _start_solo(model)
    with expect_error_base(RuntimeError, "does not continue"):
        _verify(model, [[151, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])

    # -- the MTP rail: verify rejects a block that does not continue the proposal
    mh._start_solo(mtp)
    with expect_error(RuntimeError, "does not continue"):
        mh._verify(mtp, [mh._block(151, [101, 102, 103, 104, 105])], [mh._positions(3, 6)], [5], [10])


def test_verify_with_drafts_and_no_retained_proposal_raises(model, expect_error_base):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    with expect_error_base(RuntimeError, "no proposal is retained"):
        _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])


def test_verify_with_drafts_for_a_row_without_a_session_raises(model, expect_error_base, mtp, expect_error):
    with expect_error_base(RuntimeError, "owns no drafter session"):
        _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])

    # -- the MTP rail: verify rejects drafts for a row without a session
    mh._prefill(mtp)
    with expect_error(RuntimeError, "owns no drafter session"):
        mh._verify(mtp, [mh._block(150, [101, 102, 103, 104, 105])], [mh._positions(3, 6)], [5], [10])


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


def test_straddle_verify_answers_the_drafted_row_and_decodes_the_peer(model, mtp, adapter, monkeypatch):
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
    proposal = _propose(
        model,
        [[201, 202, 251, -1, -1, -1], [600, -1, -1, -1, -1, -1]],
        [[4, 5, 6, -1, -1, -1], [10, -1, -1, -1, -1, -1]],
        counts=[3, 1],
    )
    assert proposal.num_valid.tolist() == [0, 0]

    # -- the MTP rail: a peer joining takes the plain decode while the drafted row is answered
    mh._start_solo(mtp)
    mh._prefill(mtp, slot=1, key=20)
    blocks = [mh._block(150, [101, 102, 103, 104, 105]), mh._block(9, [])]
    out = mh._verify(mtp, blocks, [mh._positions(3, 6), mh._positions(2, 1)], [5, 0], [10, 20], result=_tensor([0, 42]))
    assert out.argmax_ids[0].tolist() == [150, 151, 152, 153, 154, 155]
    assert out.argmax_ids[1, 0].item() == 42
    assert _names(mtp, "decode")[-1] == ("decode", [150, 9], [-1, 2]), "the drafted row is excluded"
    # The next proposal moves to the two-row record: the drafted row still seeds from the last
    # replay under its row id (the decoder gathers across records); only the newcomer is reseeded.
    proposal = mh._propose(
        mtp, [[150, 151, 152, 153, 154, 155], [42] + [-1] * 5], [mh._positions(4, 6), [3] + [-1] * 5], counts=[6, 1]
    )
    assert proposal.num_valid.tolist() == [5, 5]
    assert _names(mtp, "reseed")[-1] == ("reseed", (5, 5), [1], [42], [3])
    assert mh._steps(mtp)[-1][1:6] == ((5, 5), [0, 1], [155, 42], [9, 3], [5, None])
    # -- the MTP rail: the retained replay survives a straddle plain decode until the deferred read
    # A logits verify whose peer row takes the plain decode (a synchronous device read inside the step)
    # answers the drafted row from the retained replay; the deferred read after that straddle returns that
    # same block, and the proposal that follows still seeds the drafted row from the replay's verify hidden.
    mtp = mh.build_mtp_model(adapter, monkeypatch)
    mh._start_solo(mtp)
    mh._prefill(mtp, slot=1, key=20)
    decoder = mtp._mtp_decoder
    replay = mtp._mtp_sessions[0].retained.replay
    blocks = [mh._block(150, [101, 102, 103, 104, 105]), mh._block(9, [])]
    out = mh._verify(
        mtp,
        blocks,
        [mh._positions(3, 6), mh._positions(2, 1)],
        [5, 0],
        [10, 20],
        mode="logits",
        result=mh.logits_result(2, 40),
    )
    assert _names(mtp, "decode")[-1] == ("decode", [150, 9], [-1, 2]) and _names(mtp, "read")[-1] == ("read", False)
    assert decoder.readbacks == 1, "the greedy-only replay was read back at the step, before the straddle"
    host = mh._read(mtp, out)
    assert host is out.logits
    assert host[0].float().argmax(dim=-1).tolist() == [150, 151, 152, 153, 154, 155]
    assert host[1, 0].float().argmax().item() == 41 and host[1, 1:].eq(0).all()
    assert mtp._mtp_sessions[0].retained.replay is replay and mtp._mtp_last_replay is replay
    committed = [[150, 151, 152, 153, 154, 155], [41] + [-1] * 5]
    proposal = mh._propose(mtp, committed, [mh._positions(4, 6), [3] + [-1] * 5], counts=[6, 1])
    assert proposal.num_valid.tolist() == [5, 5]
    assert _names(mtp, "reseed")[-1] == ("reseed", (5, 5), [1], [41], [3])
    assert mh._steps(mtp)[-1][1:6] == ((5, 5), [0, 1], [155, 41], [9, 3], [5, None])


def test_straddle_verify_finds_the_owner_through_slot_remap(model, mtp):
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

    # -- the MTP rail: a moved slot reseeds under its new row id
    mh._start_solo(mtp)
    # The runner moves the request to row 2 inside a step (row 2 reads slot 0) and reports the
    # move once the step is accepted; the proposal that follows uses the new row id.
    blocks = [[0] + [-1] * 5, [0] + [-1] * 5, mh._block(150, [101, 102, 103, 104, 105])]
    out = mh._verify(
        mtp,
        blocks,
        [[-1] * 6, [-1] * 6, mh._positions(3, 6)],
        [0, 0, 5],
        [0, 0, 10],
        slot_remap=[2, 1, 0],
        sampling_params=_sampling(3),
    )
    assert out.argmax_ids[2].tolist() == [150, 151, 152, 153, 154, 155]
    mtp.note_state_slots_moved({0: 2, 2: 0})
    committed = [[-1] * 6, [-1] * 6, [150, 151, 152, 153, 154, 155]]
    proposal = mh._propose(mtp, committed, [[-1] * 6, [-1] * 6, mh._positions(4, 6)], counts=[1, 1, 6])
    assert proposal.num_valid.tolist() == [0, 0, 5]
    assert _names(mtp, "reseed")[-1] == ("reseed", (1, 5), [2], [155], [9])
    assert mh._steps(mtp)[-1][1:6] == ((1, 5), [2], [155], [9], [None])


def test_ordinary_step_finds_the_owner_through_slot_remap(model, mtp):
    _start_solo(model)
    out = _ordinary(model, [0, 150], [-1, 3], [0, 10], slot_remap=[1, 0])
    assert out.tolist() == [0, 250]

    # -- the MTP rail: slot remap during a step reads the moved session
    mh._start_solo(mtp)
    out = mh._verify(
        mtp,
        [[0] + [-1] * 5, mh._block(150, [101, 102, 103, 104, 105])],
        [[-1] * 6, mh._positions(3, 6)],
        [0, 5],
        [0, 10],
        slot_remap=[1, 0],
        sampling_params=_sampling(2),
    )
    assert out.argmax_ids[1].tolist() == [150, 151, 152, 153, 154, 155]


def test_verify_for_a_row_whose_slot_owns_nothing_raises_even_at_the_owners_row(
    model, expect_error_base, mtp, expect_error
):
    _start_solo(model)
    with expect_error_base(RuntimeError, "owns no drafter session"):
        _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10], slot_remap=[3])

    # -- the MTP rail: a session of a closed generation is refused
    mh._start_solo(mtp)
    mtp._spec_close_slot_generation(0)
    with expect_error(RuntimeError, "generation"):
        mh._ordinary(mtp, [150], [3], [10], result=_tensor([1]))


def test_straddle_verify_rejects_host_sampling_before_consuming_a_proposal(
    model, expect_error_base, mtp, adapter, monkeypatch, expect_error
):
    _start_solo(model)
    ordinary_result = DeviceResult(_logits(2, 500))
    model.results.append(ordinary_result)
    retained = model._dflash_retained
    events = list(model.events)
    with expect_error_base(ValueError, "cannot apply host sampling controls"):
        _verify(
            model,
            [[150, 201, 202, 203, 204, 205], [40, 0, 0, 0, 0, 0]],
            [list(range(3, 9)), [9, 10, 11, 12, 13, 14]],
            [5, 0],
            keys=[10, 20],
            sampling=False,
        )
    assert model.events == events
    assert model.results == [ordinary_result]
    assert model._dflash_retained is retained and not retained.consumed
    assert model._spec_active

    # -- the MTP rail: logits verify returns the host block with zero padding
    mh._prefill(mtp, rows=2)
    mh._ordinary(mtp, [3, 4, 0], [2, 2, -1], [10, 12, 0], result=_tensor([30, 31, 0]))
    proposal = mh._propose(
        mtp, [[30] + [-1] * 5, [31] + [-1] * 5, [-1] * 6], [[3] + [-1] * 5, [3] + [-1] * 5, [-1] * 6]
    )
    drafts = proposal.draft_token_ids.tolist()
    blocks = [mh._block(30, drafts[0]), mh._block(31, drafts[1][:2]), [0] + [-1] * 5]
    out = mh._verify(
        mtp, blocks, [mh._positions(3, 6), mh._positions(3, 3), [-1] * 6], [5, 2, 0], [10, 12, 0], mode="logits"
    )
    assert out.spec_mode == "logits" and out.argmax_ids is None and out.hidden is None
    assert tuple(out.logits.shape) == (3, 6, mh.VOCAB) and out.logits.dtype == torch.bfloat16
    assert out.logits[0].float().argmax(dim=-1).tolist() == [150, 151, 152, 153, 154, 155]
    assert out.logits[1].float().argmax(dim=-1).tolist() == [160, 161, 162, 163, 164, 165]
    assert out.logits[2].eq(0).all(), "a padding row is all zeros"
    assert mtp._mtp_decoder.results[-1].readbacks == 1, "one readback per replay, shared by its rows"
    assert len(_names(mtp, "decode")) == 1
    # -- the MTP rail: logits verify zeroes the columns past a shorter record
    mtp = mh.build_mtp_model(adapter, monkeypatch)
    mh._prefill(mtp, rows=6)
    keys = [10 + 2 * r for r in range(6)]
    mh._ordinary(mtp, [3] * 6, [2] * 6, keys, result=_tensor([30] * 6))
    proposal = mh._propose(mtp, [[30] + [-1] * 5] * 6, [[3] + [-1] * 5] * 6)
    drafts = proposal.draft_token_ids.tolist()
    out = mh._verify(
        mtp, [mh._block(30, drafts[r][:3]) for r in range(6)], [mh._positions(3, 4)] * 6, [3] * 6, keys, mode="logits"
    )
    assert tuple(out.logits.shape) == (6, 6, mh.VOCAB)
    assert out.logits[0, :4].float().argmax(dim=-1).tolist() == [150, 151, 152, 153]
    assert out.logits[:, 4:].eq(0).all()
    # -- the MTP rail: logits verify refuses device sampling parameters
    mtp = mh.build_mtp_model(adapter, monkeypatch)
    mh._start_solo(mtp)
    with expect_error(ValueError, "logits verify carries no device sampling parameters"):
        mh._verify(
            mtp,
            [mh._block(150, [101, 102, 103, 104, 105])],
            [mh._positions(3, 6)],
            [5],
            [10],
            mode="logits",
            sampling=True,
        )
    # -- the MTP rail: logits verify takes straddle rows from the plain decode logits
    mtp = mh.build_mtp_model(adapter, monkeypatch)
    mh._start_solo(mtp)
    mh._prefill(mtp, slot=1, key=20)
    blocks = [mh._block(150, [101, 102, 103, 104, 105]), mh._block(9, [])]
    out = mh._verify(
        mtp,
        blocks,
        [mh._positions(3, 6), mh._positions(2, 1)],
        [5, 0],
        [10, 20],
        mode="logits",
        result=mh.logits_result(2, 40),
    )
    assert _names(mtp, "decode")[-1] == ("decode", [150, 9], [-1, 2])
    assert out.logits[0].float().argmax(dim=-1).tolist() == [150, 151, 152, 153, 154, 155]
    assert out.logits[1, 0].float().argmax().item() == 41 and out.logits[1, 1:].eq(0).all()
    # -- the MTP rail: argmax verify without device sampling takes straddle rows from the plain logits
    mtp = mh.build_mtp_model(adapter, monkeypatch)
    mh._start_solo(mtp)
    mh._prefill(mtp, slot=1, key=20)
    blocks = [mh._block(150, [101, 102, 103, 104, 105]), mh._block(9, [])]
    out = mh._verify(
        mtp,
        blocks,
        [mh._positions(3, 6), mh._positions(2, 1)],
        [5, 0],
        [10, 20],
        sampling=False,
        result=mh.logits_result(2, 40),
    )
    assert out.argmax_ids[1, 0].item() == 41 and out.argmax_ids[0].tolist() == [150, 151, 152, 153, 154, 155]


def test_draftless_host_verify_does_not_bootstrap_an_argmax_session(model, expect_error_base):
    _prefill(model)
    pending = model._spec_pending
    events = list(model.events)
    with expect_error_base(ValueError, "cannot apply host sampling controls"):
        _verify(model, [[3, 0, 0, 0, 0, 0]], [list(range(2, 8))], [0], keys=[10], sampling=False)
    assert model.events == events
    assert model._spec_pending is pending and not model._spec_active


@pytest.mark.parametrize(
    "message, step_kwargs",
    [
        (
            "cannot return requested logprobs",
            lambda: dict(sampling_params=_sampling(1, enable_log_probs=True, num_logprobs=0)),
        ),
        ("cannot apply a grammar bitmask", lambda: dict(grammar_bitmask=torch.full((1, 19), -1, dtype=torch.int32))),
    ],
    ids=["logprobs", "grammar"],
)
@pytest.mark.parametrize("retained", [False, True])
def test_verify_rejects_unsupported_outputs_before_mutating_the_session(
    model, expect_error_base, retained, message, step_kwargs
):
    if retained:
        _start_solo(model)
        tokens, positions, valid = [150, 201, 202, 203, 204, 205], list(range(3, 9)), 5
    else:
        _prefill(model)
        tokens, positions, valid = [3, 0, 0, 0, 0, 0], list(range(2, 8)), 0
    proposal = model._dflash_retained
    pending = model._spec_pending
    events = list(model.events)
    with expect_error_base(ValueError, message):
        _verify(model, [tokens], [positions], [valid], keys=[10], **step_kwargs())
    assert model.events == events
    assert model._spec_pending is pending and model._dflash_retained is proposal
    assert proposal is None or not proposal.consumed


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


def test_padded_rows_keep_the_sentinel_in_the_plain_decode(model, mtp):
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

    # -- the MTP rail: three non contiguous rows over two rounds keep every per row value aligned
    # Slots 4, 9, 17 of a 20-row wire batch through propose/verify/propose/verify with different accepted
    #     counts: the decoder receives each row's own anchor, position, seed column and table in record order,
    #     and the runner receives each wire row's own posterior, in both accept modes.
    live = [4, 9, 17]
    keys = [10, 20, 30]
    for slot, key in zip(live, keys):
        mh._prefill(mtp, slot=slot, key=key)
    first = [30, 31, 32]
    mh._wire(
        mtp,
        live,
        [3, 4, 5],
        [2, 2, 2],
        keys,
        20,
        _tensor([first[live.index(r)] if r in live else 0 for r in range(20)]),
    )
    committed, posits = mh._committed(20, live, first, [3, 3, 3])
    proposal = mh._propose(mtp, committed, posits)
    decoder = mtp._mtp_decoder
    step = mh._steps(mtp)[-1]
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
        blocks[r] = mh._block(first[i], mtp._mtp_sessions[r].retained.drafts[: valid[i]])
        positions[r] = mh._positions(3, 1 + valid[i])
    out = mh._verify(
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
        posits[r] = mh._positions(4, counts[i])
    proposal = mh._propose(
        mtp, committed, posits, counts=[counts[live.index(r)] if r in live else 1 for r in range(20)]
    )
    step = mh._steps(mtp)[-1]
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
        blocks[r] = mh._block(expected1[i][counts[i] - 1], mtp._mtp_sessions[r].retained.drafts[: valid[i]])
        positions[r] = mh._positions(3 + counts[i], 1 + valid[i])
    out = mh._verify(
        mtp,
        blocks,
        positions,
        [valid[live.index(r)] if r in live else 0 for r in range(20)],
        [keys[live.index(r)] if r in live else 0 for r in range(20)],
        mode="logits",
    )
    assert tuple(out.logits.shape) == (20, 6, mh.VOCAB)
    for i, r in enumerate(live):
        assert out.logits[r].float().argmax(dim=-1).tolist() == [250 + 10 * i + j for j in range(6)]
    for r in range(20):
        if r not in live:
            assert out.logits[r].eq(0).all()


def test_an_ordinary_step_without_drafts_for_an_outstanding_proposal_answers_its_first_id(
    model, mtp, adapter, monkeypatch, expect_error
):
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

    # -- the MTP rail: an ordinary step answers from the retained proposal when no drafts were sent
    # Without the resident decode (the runner reloads every decode's inputs) a greedy row whose retained
    # proposal continues its token is answered from that proposal's column 0.
    plain = mh.build_mtp_model(adapter, monkeypatch, resident=False)
    mh._start_solo(plain)
    out = mh._ordinary(plain, [150], [3], [10])
    assert out.tolist() == [150] and len(_names(plain, "decode")) == 1
    proposal = mh._propose(plain, [[150, -1, -1, -1, -1, -1]], [[4, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [5]
    assert mh._steps(plain)[-1][5] == [0], "the seed column is the verified anchor column"
    # -- the MTP rail: under the resident decode an ordinary step runs the plain decode and drops the proposal
    # ``supports_async_decode`` (the class default) keeps the plain decode's device-resident token and
    # position across ordinary steps, and the runner reloads them only after a verify, so no ordinary step is
    # answered from the drafter: every live row runs the plain decode, the retained proposal is dropped, and
    # the steady decode that follows (reload_inputs=False, host inputs stale) is accepted with its commands
    # untouched.
    assert mtp._spec_resident_decode_declared()
    mh._start_solo(mtp)
    out = mh._ordinary(mtp, [150], [3], [10], result=_tensor([150]))
    assert out.tolist() == [150] and mtp._mtp_sessions[0].retained is None
    assert _names(mtp, "decode")[-1] == ("decode", [150], [3])
    steady = dict(
        reload_inputs=False, reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False
    )
    out = mh._ordinary(mtp, [150], [4], [10], result="device", **steady)
    assert out == "device" and mtp.decode_commands[-1] == steady
    proposal = mh._propose(mtp, [[160] + [-1] * 5], [[5] + [-1] * 5])
    assert proposal.num_valid.tolist() == [5]
    assert _names(mtp, "reseed")[-1] == ("reseed", (1, 5), [0], [160], [5])
    # -- the MTP rail: an ordinary step submitted before the proposal drops it and the next proposal reseeds
    # Overlap under asynchronous scheduling: the runner submits ordinary step N+1 before step N's proposal is
    # published, then publishes N's and N+1's proposals at their drains. The adapter sees propose, propose with
    # no verify between: the first proposal is dropped rather than verified (contract s.4d), the row reseeds
    # at the token the overlapped step committed, and the dropped drafts are never accepted by a verify.
    mtp = mh.build_mtp_model(adapter, monkeypatch)
    messages = []
    monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: messages.append(str(message)))
    mh._prefill(mtp)
    mh._ordinary(mtp, [3], [2], [10], result=_tensor([150]))
    mh._ordinary(mtp, [150], [3], [10], result=_tensor([151]), **steady)
    first = mh._propose(mtp, [[150] + [-1] * 5], [[3] + [-1] * 5])
    assert first.draft_token_ids[0].tolist() == [101, 102, 103, 104, 105]
    assert mtp._mtp_sessions[0].retained is not None and not mtp._mtp_sessions[0].retained.consumed
    second = mh._propose(mtp, [[151] + [-1] * 5], [[4] + [-1] * 5])
    assert second.draft_token_ids[0].tolist() == [201, 202, 203, 204, 205] and mtp._mtp_dropped_proposals == 1
    assert mtp._mtp_sessions[0].dropped == 1
    assert _names(mtp, "reseed")[-1] == ("reseed", (1, 5), [0], [151], [4])
    assert mh._steps(mtp)[-1][1:6] == ((1, 5), [0], [151], [4], [None])
    with expect_error(ValueError, "did not propose"):
        mh._verify(mtp, [mh._block(151, [101, 102, 103, 104, 105])], [mh._positions(4, 6)], [5], [10])
    out = mh._verify(mtp, [mh._block(151, [201, 202, 203, 204, 205])], [mh._positions(4, 6)], [5], [10])
    assert out.argmax_ids[0].tolist() == [250, 251, 252, 253, 254, 255]
    # The session's summary line at release carries the dropped count; the engine-life total is logged and
    # reset by release_persistent_capture.
    mtp.release_request(0)
    summary = [m for m in messages if m.startswith("Gemma4MTP contract summary: slot 0 (release)")]
    assert len(summary) == 1 and summary[0].endswith(", 1 proposals dropped unverified"), summary
    mtp.release_persistent_capture()
    assert "Gemma4MTP speculative contract: 1 proposals dropped unverified over the engine's life" in messages
    assert mtp._mtp_dropped_proposals == 0
    # -- the MTP rail: an overtaken proposal whose first draft the overlapped step committed seeds the next
    # The overlapped step's token against the dropped proposal's first draft is one position of acceptance
    # evidence (observed on the row), and when they agree the replay's verify hidden at column 1 is the seed of
    # the next proposal (no reseed): the row continues exactly as after a verify that accepted one draft.
    mtp = mh.build_mtp_model(adapter, monkeypatch)
    mh._prefill(mtp)
    mh._ordinary(mtp, [3], [2], [10], result=_tensor([150]))
    mh._ordinary(mtp, [150], [3], [10], result=_tensor([101]), **steady)
    first = mh._propose(mtp, [[150] + [-1] * 5], [[3] + [-1] * 5])
    assert first.draft_token_ids[0].tolist() == [101, 102, 103, 104, 105]
    stats = mtp._mtp_sessions[0].stats
    assert stats.ewma_accepted is None
    second = mh._propose(mtp, [[101] + [-1] * 5], [[4] + [-1] * 5])
    assert second.draft_token_ids[0].tolist() == [201, 202, 203, 204, 205] and mtp._mtp_dropped_proposals == 1
    assert mtp._mtp_sessions[0].dropped == 1 and stats.ewma_accepted is not None
    assert _names(mtp, "reseed") == [("reseed", (1, 5), [0], [150], [3])], "the first proposal reseeded; column 1 seeds"
    assert mh._steps(mtp)[-1][1:6] == ((1, 5), [0], [101], [4], [1])
    out = mh._verify(mtp, [mh._block(101, [201, 202, 203, 204, 205])], [mh._positions(4, 6)], [5], [10])
    assert out.argmax_ids[0].tolist() == [250, 251, 252, 253, 254, 255]
    # a rule-2 ordinary decode (a non-speculable peer) drops the proposal the same way, counted
    mh._prefill(mtp, slot=1, key=20)
    third = mh._propose(mtp, [[250] + [-1] * 5], [[5] + [-1] * 5], counts=[1])
    assert third.num_valid.tolist() == [5] and mtp._mtp_sessions[0].retained is not None
    mh._ordinary(mtp, [250, 3], [5, 2], [10, 20], result=_tensor([260, 40]))  # reloaded: first plain after a verify
    assert mtp._mtp_sessions[0].retained is None and mtp._mtp_sessions[0].dropped == 2
    assert mtp._mtp_dropped_proposals == 2


def test_a_peer_joining_on_an_ordinary_step_ends_speculation_for_good(model, mtp):
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    out = _ordinary(model, [251, 40], [6, 9], [10, 20], result="device")
    assert out == "device" and not model._spec_active
    _ordinary(model, [7], [7], [10], result="device")
    proposal = _propose(model, [[8, -1, -1, -1, -1, -1]], [[8, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [0]

    # -- the MTP rail: a peer leaving moves the survivor back to the solo record seeded from the last replay
    mh._prefill(mtp, rows=2)
    mh._ordinary(mtp, [3, 4], [2, 2], [10, 12], result=_tensor([30, 31]))
    mh._propose(mtp, [[30] + [-1] * 5, [31] + [-1] * 5], [[3] + [-1] * 5] * 2)
    mtp.release_request(1)
    drafts = mtp._mtp_decoder.results[-1].drafts[0].tolist()
    mh._verify(mtp, [mh._block(30, drafts), [0] + [-1] * 5], [mh._positions(3, 6), [-1] * 6], [5, 0], [10, 0])
    proposal = mh._propose(
        mtp, [[150, 151, 152, 153, 154, 155], [-1] * 6], [mh._positions(4, 6), [-1] * 6], counts=[6, 1]
    )
    assert proposal.num_valid.tolist() == [5, 0]
    assert len(_names(mtp, "reseed")) == 1, "the survivor kept its row id: its seed is in the last replay"
    assert mh._steps(mtp)[-1][1:6] == ((1, 5), [0], [155], [9], [5])


_PAD = [-1] * 6


@pytest.mark.parametrize(
    "anchors, positions, keys",
    [
        ([0, 40], [-1, 9], [0, 20]),
        ([0, 40, 41], [-1, 9, 12], [0, 20, 30]),
    ],
    ids=["one-other-request", "two-other-requests"],
)
def test_a_step_without_the_owner_keeps_its_outstanding_proposal(model, anchors, positions, keys):
    """The scheduler keeps the drafts of a request it did not schedule and sends
    them on its next step, so that step must still find the proposal."""
    _start_solo(model)
    out = _ordinary(model, anchors, positions, keys, result="device")
    assert out == "device"
    committed = [_PAD] + [[a] + [-1] * 5 for a in anchors[1:]]
    proposal = _propose(model, committed, [_PAD] + [[p + 1] + [-1] * 5 for p in positions[1:]])
    assert proposal.num_valid.tolist() == [0] * len(anchors)
    assert model._spec_active and not model._dflash_retained.consumed
    out = _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    assert out.argmax_ids.tolist() == [[250, 251, 252, 253, 254, 255]]


def test_a_draftless_verify_without_the_owner_keeps_its_outstanding_proposal(model, mtp):
    _start_solo(model)
    out = _verify(
        model,
        [_PAD, [40, 0, 0, 0, 0, 0]],
        [_PAD, [9, 10, 11, 12, 13, 14]],
        [0, 0],
        keys=[0, 20],
        result=DeviceResult(_tensor([0, 600])),
    )
    assert int(out.argmax_ids[1, 0]) == 600
    assert model._spec_active and not model._dflash_retained.consumed
    out = _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    assert out.argmax_ids.tolist() == [[250, 251, 252, 253, 254, 255]]

    # -- the MTP rail: a sampled request is verified in logits mode from its own columns
    # The runner picks the mode per step (logits once a live row samples); the adapter answers either mode,
    #     and in logits mode the sampled row's columns come from its own record row, not a neighbour's.
    live = [2, 7]
    for slot, key in zip(live, (10, 20)):
        mh._prefill(mtp, slot=slot, key=key)
    sampling = _sampling(10, temperature=[0.7 if r == 7 else 0.0 for r in range(10)])
    mh._wire(
        mtp, live, [3, 4], [2, 2], [10, 20], 10, _tensor([0, 0, 30, 0, 0, 0, 0, 31, 0, 0]), sampling_params=sampling
    )
    assert mtp._mtp_row_needs_logits == {2: False, 7: True}
    committed, posits = mh._committed(10, live, [30, 31], [3, 3])
    mh._propose(mtp, committed, posits)
    decoder = mtp._mtp_decoder
    assert decoder.readbacks == 1, "a sampled row in the replay: the logits are read back at proposal time"
    assert mtp._mtp_sessions[7].retained.logits.float().argmax(dim=-1).tolist() == [160, 161, 162, 163, 164, 165]
    blocks = [[0] + [-1] * 5] * 10
    positions = [[-1] * 6] * 10
    for i, r in enumerate(live):
        blocks[r] = mh._block([30, 31][i], mtp._mtp_sessions[r].retained.drafts)
        positions[r] = mh._positions(3, 6)
    out = mh._verify(
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


# -- caller-owned decode updates after drafter-served steps ----------------------


def test_drafter_history_does_not_override_the_reload_commands(model, mtp):
    commands = dict(
        reload_inputs=True, reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False
    )
    _start_solo(model)
    _verify(
        model,
        [[150, 201, 202, 203, 204, 205], [40, 0, 0, 0, 0, 0]],
        [list(range(3, 9)), [9, 10, 11, 12, 13, 14]],
        [5, 0],
        keys=[10, 20],
        result=DeviceResult(_tensor([0, 600])),
        **commands,
    )
    assert model.decode_commands == [commands]
    _ordinary(model, [251, 600], [6, 10], [10, 20], result="device", **commands)
    assert model.decode_commands == [commands, commands]
    _ordinary(model, [7, 8], [7, 11], [10, 20], result="device", reload_inputs=False)
    assert model.reloads == [True, True, False]

    # -- the MTP rail: accepted counts on the narrow step are consumed without reaching the plain decode
    mh._prefill(mtp)
    out = mh._ordinary(mtp, [3], [2], [10], result=_tensor([150]), accepted_counts=_tensor([3]))
    assert out.tolist() == [150]
    assert "accepted_counts" not in mtp.decode_kwargs[-1] and "spec_mode" not in mtp.decode_kwargs[-1]


def _decline_after_a_verify(model):
    """The drafter answers the first ordinary step and a verify, then the next proposal declines."""
    _start_solo(model)
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    model.model_args[0].max_seq_len = 11
    _propose(model, [[201, 202, 251, -1, -1, -1]], [[4, 5, 6, -1, -1, -1]], counts=[3])
    return [251], [6], [10]


def _peer_joins_an_outstanding_proposal(model):
    """The drafted row is answered from its proposal and runs the peer's plain decode at position -1."""
    _start_solo(model)
    _verify(
        model,
        [[150, 201, 202, 203, 204, 205], [40, 0, 0, 0, 0, 0]],
        [list(range(3, 9)), [9, 10, 11, 12, 13, 14]],
        [5, 0],
        keys=[10, 20],
        result=DeviceResult(_tensor([0, 600])),
    )
    return [251, 600], [6, 10], [10, 20]


@pytest.mark.parametrize(
    "history", [_decline_after_a_verify, _peer_joins_an_outstanding_proposal], ids=["decline", "peer-joined"]
)
def test_a_plain_decode_after_a_drafter_served_step_refuses_resident_inputs(
    model, expect_error_base, history, mtp, expect_error
):
    """A drafter-served row never updated the plain decode's resident token and
    position. Without reload_inputs the plain decode would start from the stale
    token, so the step raises before the device runs; a reloading decode clears it."""
    anchors, positions, keys = history(model)
    decodes = _names(model, "decode")
    with expect_error_base(RuntimeError, "reload_inputs=False"):
        _ordinary(model, anchors, positions, keys, result="device", reload_inputs=False)
    assert _names(model, "decode") == decodes
    out = _ordinary(model, anchors, positions, keys)
    assert out == "device"
    _ordinary(model, [a + 1 for a in anchors], [p + 1 for p in positions], keys, result="device", reload_inputs=False)
    assert model.reloads[-2:] == [True, False]

    # -- the MTP rail: a plain decode after a drafter served step refuses stale resident inputs
    mh._start_solo(mtp)
    mh._verify(mtp, [mh._block(150, [101, 102, 103, 104, 105])], [mh._positions(3, 6)], [5], [10])
    with expect_error(RuntimeError, "reload_inputs=False"):
        mh._ordinary(mtp, [160], [4], [10], result="device", reload_inputs=False)
    assert mh._ordinary(mtp, [160], [4], [10], result="device") == "device"


@pytest.mark.parametrize("reload_inputs", [True, False])
def test_plain_decode_forwards_the_reload_command(model, reload_inputs):
    _prefill(model, prompt_len=2, key=10, rows=2)
    _ordinary(model, [3, 4], [2, 2], [10, 12], result="device", reload_inputs=reload_inputs)
    assert model.reloads == [reload_inputs]


def test_release_logs_committed_tokens_per_step_for_the_session(model, adapter, monkeypatch, mtp):
    """The block rail logs tokens per iteration on release; the contract rail
    logs the same per session, counting the first ordinary step as a step."""
    messages = []
    monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: messages.append(str(message)))
    _start_solo(model)  # first ordinary step (1 token) then the first proposal
    _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    _propose(model, [[201, 202, 251, -1, -1, -1]], [[4, 5, 6, -1, -1, -1]], counts=[3])
    model.release_request(0)
    summary = [m for m in messages if "contract summary" in m]
    assert summary == [
        "Gemma4DFlash contract summary: 2 steps, 4 tokens, 2.00 tokens/step, 2.00 accepted drafts/verify (EWMA over 1)"
    ]
    assert model._dflash_steps == 0 and model._dflash_committed == 0

    # -- the MTP rail: release drops the session and a new occupant opens a new generation
    messages = []
    monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: messages.append(str(message)))
    mh._start_solo(mtp)  # prefill: generation 1; the session is owned by (0, 1)
    assert mtp._mtp_sessions[0].owner == (0, 1)
    mtp.release_request(0)
    assert mtp._mtp_sessions == {} and mtp._spec_slot_gen[0] == 2
    assert [m for m in messages if "contract summary" in m] == [
        "Gemma4MTP contract summary: slot 0 (release) 1 steps, 1 tokens, 1.00 tokens/step, no verified drafts"
    ]
    mh._prefill(mtp, key=20)
    assert mtp._spec_slot_gen[0] == 3
    mh._ordinary(mtp, [3], [2], [20], result=_tensor([160]))
    mh._propose(mtp, [[160] + [-1] * 5], [[3] + [-1] * 5])
    assert mtp._mtp_sessions[0].owner == (0, 3)
    assert _names(mtp, "reseed")[-1] == ("reseed", (1, 5), [0], [160], [3])

    # -- the acceptance-adaptive policy: the session summary carries the accepted-drafts EWMA (both rails)
    _dflash_the_session_summary_reports_the_accepted_drafts_ewma(adapter, monkeypatch)
    _mtp_the_session_summary_reports_the_accepted_drafts_ewma(adapter, monkeypatch)


# -- rows that cannot use the fused argmax output --------------------------------

_NOT_GREEDY = [
    {"temperature": 1.0},
    {"presence_penalty": 0.5},
    {"frequency_penalty": 0.5},
    {"repetition_penalty": 1.2},
]
_NOT_GREEDY_IDS = ["temperature", "presence-penalty", "frequency-penalty", "repetition-penalty"]


@pytest.mark.parametrize("values", _NOT_GREEDY, ids=_NOT_GREEDY_IDS)
def test_a_solo_row_that_does_not_sample_greedily_runs_the_plain_decode(model, values):
    """The session answers with the target's argmax ids. The plain decode draws
    with the row's own parameters, so the row takes it and its taps are freed."""
    _prefill(model, prompt_len=2, key=10)
    taps = list(model._spec_pending[0])
    out = _ordinary(model, [3], [2], [10], result="device", sampling_params=_sampling(1, **values))
    assert out == "device"
    assert _names(model, "decode") == [("decode", [3], [2])]
    assert _names(model, "bootstrap") == [] and model._spec_decoder.replays == 0
    assert model._spec_pending is None and all(tap.releases == 1 for tap in taps)


@pytest.mark.parametrize("active", [False, True], ids=["pending-prefill", "active-session"])
def test_a_greedy_solo_row_under_a_grammar_bitmask_runs_the_plain_decode(model, active):
    """The fused replay's argmax never sees the bitmask, so a greedy row under
    one takes the plain decode and leaves speculation."""
    if active:
        _start_solo(model)
        token, position = 150, 3
    else:
        _prefill(model, prompt_len=2, key=10)
        token, position = 3, 2
    replays = model._spec_decoder.replays
    bitmask = torch.full((1, 19), -1, dtype=torch.int32)
    out = _ordinary(model, [token], [position], [10], result="device", grammar_bitmask=bitmask)
    assert out == "device"
    assert _names(model, "decode") == [("decode", [token], [position])]
    assert model._spec_decoder.replays == replays
    assert model._spec_pending is None and not model._spec_active


def test_a_sampled_solo_row_is_never_proposed_for(model, mtp):
    sampled = _sampling(1, temperature=1.0)
    _prefill(model, prompt_len=2, key=10)
    _ordinary(model, [3], [2], [10], result="device", sampling_params=sampled)
    proposal = _propose(model, [[600, -1, -1, -1, -1, -1]], [[3, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [0]
    out = _ordinary(model, [600], [3], [10], result="device", sampling_params=sampled)
    assert out == "device"
    assert model._spec_decoder.replays == 0 and not model._spec_active

    # -- the MTP rail: a deferred logits verify answers from the logits retained at proposal
    mh._prefill(mtp, rows=2)
    mh._ordinary(
        mtp, [3, 4], [2, 2], [10, 12], result=_tensor([30, 31]), sampling_params=_sampling(2, temperature=[0.7, 0.0])
    )
    mh._propose(mtp, [[30] + [-1] * 5, [31] + [-1] * 5], [[3] + [-1] * 5] * 2)
    decoder = mtp._mtp_decoder
    assert decoder.readbacks == 1
    drafts_a = mtp._mtp_sessions[0].retained.drafts
    drafts_b = mtp._mtp_sessions[1].retained.drafts
    # The scheduler runs B alone: B's verify and B's next proposal rewrite the shared buffer.
    mh._verify(mtp, [[0] + [-1] * 5, mh._block(31, drafts_b)], [[-1] * 6, mh._positions(3, 6)], [0, 5], [0, 12])
    mh._propose(mtp, [[-1] * 6, [160, 161, 162, 163, 164, 165]], [[-1] * 6, mh._positions(4, 6)], counts=[1, 6])
    assert decoder.replays == 2 and decoder.readbacks == 1
    assert decoder.buffer[0, 0].argmax().item() != 150, "the buffer now holds B's replay"
    # A's deferred verify in logits mode answers from the logits retained with A's proposal. The
    # step itself is host-sampled, so B's greedy-only replay, still live in the buffer, is read
    # back for B's own future verify (one more readback, none of it for A).
    out = mh._verify(
        mtp, [mh._block(30, drafts_a), [0] + [-1] * 5], [mh._positions(3, 6), [-1] * 6], [5, 0], [10, 0], mode="logits"
    )
    assert out.logits[0].float().argmax(dim=-1).tolist() == [150, 151, 152, 153, 154, 155]
    assert decoder.readbacks == 2 and mtp._mtp_sessions[1].retained.logits is not None


@pytest.mark.parametrize(
    "enable_log_probs",
    [True, [True], torch.tensor(True), torch.tensor([True])],
    ids=["scalar", "list", "scalar-tensor", "row-tensor"],
)
@pytest.mark.parametrize("active", [False, True], ids=["pending-prefill", "active-session"])
def test_a_row_requesting_logprobs_keeps_the_ordinary_result_and_releases_its_session(model, enable_log_probs, active):
    if active:
        _start_solo(model)
        token, position = 150, 3
        taps = []
    else:
        _prefill(model, prompt_len=2, key=10)
        token, position = 3, 2
        taps = list(model._spec_pending[0])
    replays = model._spec_decoder.replays
    sampling = dataclasses.replace(_sampling(1, num_logprobs=0), enable_log_probs=enable_log_probs)
    result = (_tensor([600]), torch.tensor([-0.5]))

    out = _ordinary(model, [token], [position], [10], result=result, sampling_params=sampling)

    assert out is result
    assert _names(model, "decode") == [("decode", [token], [position])]
    assert model._spec_decoder.replays == replays
    assert model._spec_pending is None and not model._spec_active
    assert model._dflash_retained is None and all(tap.releases == 1 for tap in taps)
    proposal = _propose(model, [[600, -1, -1, -1, -1, -1]], [[position + 1, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [0]


def test_logprobs_for_another_row_leave_the_session_owners_proposal_alone(model, mtp):
    _start_solo(model)
    sampling = _sampling(2, enable_log_probs=[False, True], num_logprobs=0)
    result = (_tensor([0, 600]), torch.tensor([0.0, -0.5]))

    out = _ordinary(model, [0, 40], [-1, 9], [0, 20], result=result, sampling_params=sampling)

    assert out is result
    assert model._spec_active and not model._dflash_retained.consumed
    out = _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    assert out.argmax_ids.tolist() == [[250, 251, 252, 253, 254, 255]]

    # -- the MTP rail: a mixed replay reads the logits back exactly once at proposal time
    mh._prefill(mtp, rows=3)
    mixed = _sampling(3, temperature=[0.7, 0.0, 0.0])
    mh._ordinary(mtp, [3, 4, 5], [2, 2, 2], [10, 12, 14], result=_tensor([30, 31, 32]), sampling_params=mixed)
    mh._propose(mtp, [[30 + r] + [-1] * 5 for r in range(3)], [[3] + [-1] * 5] * 3)
    decoder = mtp._mtp_decoder
    assert decoder.readbacks == 1 and decoder.results[-1].readbacks == 1
    for slot in range(3):
        retained = mtp._mtp_sessions[slot].retained
        assert tuple(retained.logits.shape) == (6, mh.VOCAB) and retained.logits.dtype == torch.bfloat16
    drafts = [mtp._mtp_sessions[r].retained.drafts for r in range(3)]
    out = mh._verify(
        mtp,
        [mh._block(30 + r, drafts[r]) for r in range(3)],
        [mh._positions(3, 6)] * 3,
        [5] * 3,
        [10, 12, 14],
        mode="logits",
    )
    assert decoder.readbacks == 1, "the verify answers from the retained host logits"
    assert out.logits[0].float().argmax(dim=-1).tolist() == [150, 151, 152, 153, 154, 155]
    assert out.logits[2].float().argmax(dim=-1).tolist() == [170, 171, 172, 173, 174, 175]


def test_logprobs_on_padding_do_not_disqualify_the_live_row(model, mtp):
    _prefill(model, prompt_len=2, key=10)
    out = _ordinary(
        model,
        [3, 0, 0, 0],
        [2, -1, -1, -1],
        [10, 0, 0, 0],
        sampling_params=_sampling(4, enable_log_probs=[False, True, True, True]),
    )
    assert out.tolist() == [150, 0, 0, 0]
    assert len(_names(model, "bootstrap")) == 1 and _names(model, "decode") == []

    # -- the MTP rail: rows of a logits verify are read back at the next proposal
    mh._start_solo(mtp)
    mh._verify(mtp, [mh._block(150, [101, 102, 103, 104, 105])], [mh._positions(3, 6)], [5], [10], mode="logits")
    assert mtp._mtp_decoder.readbacks == 1, "a greedy-only replay is read back late, while its buffer is live"
    mh._propose(mtp, [[150, 151, 152, 153, 154, 155]], [mh._positions(4, 6)], counts=[6])
    assert mtp._mtp_decoder.readbacks == 2 and mtp._mtp_sessions[0].retained.logits is not None


def test_a_greedy_row_with_top_k_and_top_p_still_speculates(model, mtp):
    _prefill(model, prompt_len=2, key=10)
    out = _ordinary(model, [3], [2], [10], sampling_params=_sampling(1, top_k=20, top_p=0.9))
    assert out.tolist() == [150]
    assert len(_names(model, "bootstrap")) == 1

    # -- the MTP rail: a greedy only replay performs no readback
    mh._start_solo(mtp)
    assert mtp._mtp_decoder.readbacks == 0 and mtp._mtp_sessions[0].retained.logits is None
    mh._verify(mtp, [mh._block(150, [101, 102, 103, 104, 105])], [mh._positions(3, 6)], [5], [10])
    assert mtp._mtp_decoder.readbacks == 0


@pytest.mark.parametrize(
    "temperature, bootstraps",
    [([0.0, 1.0, 1.0, 1.0], True), ([1.0, 0.0, 0.0, 0.0], False)],
    ids=["greedy-live-row", "sampled-live-row"],
)
def test_a_padded_step_reads_only_the_live_rows_sampling(model, temperature, bootstraps):
    _prefill(model, prompt_len=2, key=10)
    out = _ordinary(
        model,
        [3, 0, 0, 0],
        [2, -1, -1, -1],
        [10, 0, 0, 0],
        result="device",
        sampling_params=_sampling(4, temperature=temperature),
    )
    assert (len(_names(model, "bootstrap")) == 1) is bootstraps
    if bootstraps:
        assert out.tolist() == [150, 0, 0, 0]
        assert model.results == ["device"]  # the plain decode never ran
    else:
        assert out == "device"


def test_a_draftless_verify_of_a_greedy_solo_row_starts_the_session(model, mtp):
    _prefill(model, prompt_len=2, key=10)
    out = _verify(model, [[3, 0, 0, 0, 0, 0]], [list(range(2, 8))], [0], keys=[10])
    assert out.argmax_ids[0].tolist() == [150, -1, -1, -1, -1, -1]
    assert len(_names(model, "bootstrap")) == 1 and _names(model, "decode") == []

    # -- the MTP rail: a draftless verify answers column zero from the retained proposal
    mh._start_solo(mtp)
    out = mh._verify(mtp, [mh._block(150, [])], [mh._positions(3, 1)], [0], [10])
    assert out.argmax_ids[0, 0].item() == 150 and len(_names(mtp, "decode")) == 1


@pytest.mark.parametrize("values", _NOT_GREEDY, ids=_NOT_GREEDY_IDS)
def test_a_draftless_verify_of_a_solo_row_that_does_not_sample_greedily_takes_the_plain_decode(model, values):
    """Column 0 is the plain decode's device draw for the row, not the session's argmax."""
    _prefill(model, prompt_len=2, key=10)
    taps = list(model._spec_pending[0])
    out = _verify(
        model,
        [[3, 0, 0, 0, 0, 0]],
        [list(range(2, 8))],
        [0],
        keys=[10],
        result=DeviceResult(_tensor([600])),
        sampling_params=_sampling(1, **values),
    )
    assert out.argmax_ids[0].tolist() == [600, -1, -1, -1, -1, -1]
    assert _names(model, "decode") == [("decode", [3], [2])]
    assert _names(model, "bootstrap") == [] and model._spec_decoder.replays == 0
    assert model._spec_pending is None and all(tap.releases == 1 for tap in taps)


@pytest.mark.parametrize("step", ["ordinary", "verify"])
def test_a_sampled_row_leaves_another_requests_outstanding_proposal_alone(model, step):
    """The owner is not in this step and its drafts come on its next step."""
    _start_solo(model)
    sampled = _sampling(2, temperature=[0.0, 1.0])
    if step == "ordinary":
        out = _ordinary(model, [0, 40], [-1, 9], [0, 20], result="device", sampling_params=sampled)
        assert out == "device"
    else:
        out = _verify(
            model,
            [_PAD, [40, 0, 0, 0, 0, 0]],
            [_PAD, [9, 10, 11, 12, 13, 14]],
            [0, 0],
            keys=[0, 20],
            result=DeviceResult(_tensor([0, 600])),
            sampling_params=sampled,
        )
        assert int(out.argmax_ids[1, 0]) == 600
    assert model._spec_active and not model._dflash_retained.consumed
    out = _verify(model, [[150, 201, 202, 203, 204, 205]], [list(range(3, 9))], [5], keys=[10])
    assert out.argmax_ids.tolist() == [[250, 251, 252, 253, 254, 255]]


def test_a_session_owner_seen_sampling_on_a_later_step_ends_its_session(model, mtp):
    """A request that changes from greedy to sampled must leave its argmax session."""
    _prefill(model, prompt_len=2, key=10)
    first = _verify(model, [[3, 0, 0, 0, 0, 0]], [[2, 3, 4, 5, 6, 7]], [0], keys=[10])
    assert first.argmax_ids[0, 0] == 150
    assert model._spec_active
    out = _ordinary(model, [150], [3], [10], result="device", sampling_params=_sampling(1, temperature=1.0))
    assert out == "device"
    assert not model._spec_active and model._dflash_retained is None
    assert ("release_decoder", False) in model.events
    proposal = _propose(model, [[600, -1, -1, -1, -1, -1]], [[4, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [0]

    # -- the MTP rail: a sampled row is not answered from the retained argmax on an ordinary step
    mh._start_solo(mtp)
    out = mh._ordinary(mtp, [150], [3], [10], result=_tensor([77]), sampling_params=_sampling(1, temperature=0.7))
    assert out.tolist() == [77] and mtp._mtp_sessions[0].retained is None


@pytest.mark.parametrize("verify", [False, True], ids=["ordinary", "draftless-verify"])
@pytest.mark.parametrize("sampling_values", [{"temperature": 1.0}, {"presence_penalty": 0.5}])
def test_sampled_request_after_greedy_completion_uses_the_real_decode_contract(
    model, adapter, monkeypatch, verify, sampling_values
):
    """Run the real decode interface over recording trace and sampling operations."""
    _start_solo(model)
    model.release_request(0)
    _prefill(model, prompt_len=2, key=20)

    source_path = Path(__file__).resolve().parents[2] / "tt" / "generator.py"
    tree = ast.parse(source_path.read_text())
    mixin = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ChunkedPrefillPageTableGuardMixin"
    )
    decode = next(node for node in mixin.body if isinstance(node, ast.FunctionDef) and node.name == "decode_forward")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), decode],
        type_ignores=[],
    )
    namespace = {"torch": torch, "Mode": SimpleNamespace(DECODE="decode")}
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)

    def plain(self, *args, page_tables_per_layer=None, **kwargs):
        return namespace["decode_forward"](self, *args, **kwargs)

    trace_calls, sample_calls = [], []

    def trace(**kwargs):
        trace_calls.append(kwargs)
        return "logits"

    def sample(logits, **kwargs):
        assert logits == "logits"
        sample_calls.append(kwargs)
        return _tensor([600])

    monkeypatch.setattr(adapter.Gemma4ForCausalLM, "decode_forward", plain)
    monkeypatch.setattr(model, "_clear_sequential_batch_page_tables", lambda: None, raising=False)
    monkeypatch.setattr(model, "mode", "prefill", raising=False)
    monkeypatch.setattr(model, "data_parallel", 1, raising=False)
    monkeypatch.setattr(model.model[0], "switch_mode", lambda mode: None, raising=False)
    monkeypatch.setattr(model, "_decode_forward_trace_text", trace, raising=False)
    monkeypatch.setattr(model, "sample_decode_on_device", sample, raising=False)
    sampling_params = _sampling(1, **sampling_values)
    kwargs = dict(sampling_params=sampling_params, read_from_device=False)
    if verify:
        out = _verify(model, [[40, 0, 0, 0, 0, 0]], [[2, 3, 4, 5, 6, 7]], [0], keys=[20], **kwargs)
        assert out.argmax_ids[0].tolist() == [600, -1, -1, -1, -1, -1]
    else:
        out = _ordinary(model, [40], [2], [20], **kwargs)
        assert out.tolist() == [600]
    assert not model._spec_active and model._spec_pending is None
    assert len(trace_calls) == len(sample_calls) == 1
    assert trace_calls[0]["tokens"][0].tolist() == [[40]]
    assert trace_calls[0]["current_pos"][0].tolist() == [2]
    assert trace_calls[0]["reload_inputs"] is True
    assert trace_calls[0]["reload_page_table"] is False
    assert sample_calls[0]["sampling_params"] is sampling_params
    assert sample_calls[0]["reload_sampling_params"] is True
    assert sample_calls[0]["reset_sampling_state"] is True


def test_argmax_eligibility_reads_per_row_values_and_shared_scalars(adapter, mtp, expect_error):
    can_use_argmax = adapter.Gemma4DFlashContractForCausalLM._dflash_row_can_use_argmax
    lists = _sampling(2, temperature=[0.0, 0.7])
    assert can_use_argmax(lists, 0) and not can_use_argmax(lists, 1)
    tensors = dataclasses.replace(lists, temperature=torch.zeros(2), repetition_penalty=torch.tensor([1.0, 1.1]))
    assert can_use_argmax(tensors, 0) and not can_use_argmax(tensors, 1)
    shared = dataclasses.replace(_sampling(1), presence_penalty=0.0, frequency_penalty=0.2)
    assert not can_use_argmax(shared, 0)
    assert can_use_argmax(dataclasses.replace(shared, frequency_penalty=0.0), 0)
    log_probs = dataclasses.replace(_sampling(2), enable_log_probs=torch.tensor([False, True]))
    assert can_use_argmax(log_probs, 0) and not can_use_argmax(log_probs, 1)

    # -- the MTP rail: a greedy row whose buffer was overwritten cannot answer a logits verify
    mh._start_solo(mtp)  # greedy A at slot 0: no readback
    mh._prefill(mtp, slot=1, key=20)
    mh._ordinary(mtp, [0, 9], [-1, 2], [0, 20], result=_tensor([0, 42]))
    mh._propose(mtp, [[-1] * 6, [42] + [-1] * 5], [[-1] * 6, [3] + [-1] * 5])
    assert mtp._mtp_decoder.replays == 2 and mtp._mtp_decoder.readbacks == 0
    with expect_error(RuntimeError, "overwritten the shared verify-logits buffer"):
        mh._verify(
            mtp,
            [mh._block(150, [101, 102, 103, 104, 105]), [0] + [-1] * 5],
            [mh._positions(3, 6), [-1] * 6],
            [5, 0],
            [10, 0],
            mode="logits",
        )


# -- identity and release --------------------------------------------------------


def test_release_of_the_owner_clears_the_retained_proposal(model, mtp):
    _start_solo(model)
    model.release_request(0)
    assert model._dflash_retained is None and not model._spec_active
    assert model._dflash_live_owner is None and model._spec_owner_slot is None

    # -- the MTP rail: release of every request drops every session
    mh._prefill(mtp, rows=2)
    mh._ordinary(mtp, [3, 4], [2, 2], [10, 12], result=_tensor([30, 31]))
    mh._propose(mtp, [[30] + [-1] * 5, [31] + [-1] * 5], [[3] + [-1] * 5] * 2)
    mtp.release_request(None)
    assert mtp._mtp_sessions == {}


def test_release_of_another_slot_keeps_the_session(model, mtp):
    _start_solo(model)
    model.release_request(3)
    assert model._spec_active and model._dflash_retained is not None

    # -- the MTP rail: release of another slot keeps the session
    mh._start_solo(mtp)
    mtp.release_request(3)
    assert 0 in mtp._mtp_sessions and mtp._mtp_sessions[0].retained is not None


def test_release_of_the_pending_owner_frees_its_taps(model):
    _prefill(model, prompt_len=2, key=10, slot=2)
    taps = list(model._spec_pending[0])
    model.release_request(1)
    assert model._spec_pending is not None
    model.release_request(2)
    assert model._spec_pending is None and all(tap.releases == 1 for tap in taps)


def test_a_reused_slot_does_not_match_a_stale_owner(model, mtp):
    _prefill(model, prompt_len=2, key=10)
    model.release_request(0)  # the owner finished before decoding
    _prefill(model, prompt_len=2, key=20, rows=2)  # slots 0 and 1, never speculable
    out = _ordinary(model, [3], [2], [20], result="device")
    assert out == "device" and _names(model, "bootstrap") == []

    # -- the MTP rail: a prefill into a slot drops its stale session
    mh._start_solo(mtp)
    mh._prefill(mtp, key=20)
    assert mtp._mtp_sessions == {}
    mh._ordinary(mtp, [3], [2], [20], result=_tensor([160]))
    proposal = mh._propose(mtp, [[160] + [-1] * 5], [[3] + [-1] * 5])
    assert proposal.num_valid.tolist() == [5]


def test_slot_moves_follow_the_owner(model, mtp):
    _start_solo(model)
    model.note_state_slots_moved({0: 3, 3: 0})
    assert model._dflash_live_owner[0] == 3 and model._spec_owner_slot == 3
    model.release_request(0)
    assert model._spec_active
    model.release_request(3)
    assert not model._spec_active

    # -- the MTP rail: slot moves follow the session
    mh._start_solo(mtp)  # the prefill opened generation 1 of slot 0
    mtp.note_state_slots_moved({0: 3, 3: 0})
    assert set(mtp._mtp_sessions) == {3} and mtp._mtp_sessions[3].owner == (3, 1)
    assert mtp._spec_slot_gen[3] == 1
    out = mh._verify(
        mtp,
        [[0] + [-1] * 5, [0] + [-1] * 5, [0] + [-1] * 5, mh._block(150, [101, 102, 103, 104, 105])],
        [[-1] * 6, [-1] * 6, [-1] * 6, mh._positions(3, 6)],
        [0, 0, 0, 5],
        [0, 0, 0, 10],
    )
    assert out.argmax_ids[3].tolist() == [150, 151, 152, 153, 154, 155]


def test_release_persistent_capture_clears_the_rail_and_reaches_the_base(model, mtp):
    _start_solo(model)
    model.release_persistent_capture()
    assert model._dflash_retained is None and model._spec_pending is None
    assert ("release_decoder", True) in model.events
    assert model.base_releases == [1]

    # -- the MTP rail: release persistent capture drops sessions releases the decoder and reaches the base
    mh._start_solo(mtp)
    mtp.release_persistent_capture()
    assert mtp._mtp_sessions == {} and mtp._mtp_decoder is None and not mtp._mtp_captured
    assert mtp.events[-1] == ("release",)
    assert mtp.base_releases == [1]


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


def test_a_solo_prefill_of_another_request_drops_the_pending_taps_and_takes_none(model, monkeypatch):
    """Two rows prefilled before a proposal share their first step, so neither can bootstrap: the first's taps
    are dropped and the second captures none (the hook stays disarmed); the base prefill still runs, eager."""
    _prefill(model, prompt_len=1024, key=10, slot=0)
    first = list(model._spec_pending[0])
    _prefill(model, prompt_len=64, key=20, slot=1)
    assert all(tap.releases == 1 for tap in first)
    assert model._spec_pending is None and model._dflash_pending_owner is None
    assert model.model[0].tap_layers is None and _names(model, "prefill")[-1] == ("prefill", 1, [64], False)
    # released before any step, the two rows are gone: the next solo prefill takes taps and bootstraps
    model.release_request(0)
    model.release_request(1)
    _start_solo(model, prompt_len=2, key=30)
    assert model._spec_active
    # a session live on another row: a peer's solo prefill takes no taps, and the session stays
    _prefill(model, prompt_len=2, key=40, slot=1)
    assert model._spec_pending is None and model._spec_active and model.model[0].tap_layers is None
    # rows the last step served and still live: the same (two live rows ended the session; both still
    # decode when slot 2 prefills)
    _ordinary(model, [3, 5], [3, 2], [30, 40], result="device")
    assert not model._spec_active and model._dflash_live_slots == {0, 1}
    _prefill(model, prompt_len=2, key=50, slot=2)
    assert model._spec_pending is None and model.model[0].tap_layers is None
    # the batch drained (every served and prefilled row released): the taps are taken again
    for slot in (0, 1, 2):
        model.release_request(slot)
    assert model._dflash_live_slots == set() and model._dflash_prefilled_slots == set()
    _prefill(model, prompt_len=2, key=60, slot=0)
    assert model._spec_pending is not None and model._spec_pending[1] == 2 and model._dflash_pending_owner[0] == 0
    # GEMMA4_DFLASH_TRACE_PLAIN_PREFILL=1: a prefill that takes no taps keeps the caller's trace setting; one
    # that takes them stays eager
    monkeypatch.setenv("GEMMA4_DFLASH_TRACE_PLAIN_PREFILL", "1")
    _prefill(model, prompt_len=2, key=70, slot=1)  # slot 0 prefilled since the last step: this one takes none
    assert _names(model, "prefill")[-1] == ("prefill", 1, [2], True) and model._spec_pending is None
    model.release_request(1)
    _prefill(model, prompt_len=2, key=80, slot=0)  # a new request in slot 0, alone again: taps, eager
    assert _names(model, "prefill")[-1][3] is False and model._spec_pending is not None
    # slot moves follow the sets
    model.note_state_slots_moved({0: 3, 3: 0})
    assert model._dflash_prefilled_slots == {3}


def test_a_chunk_whose_earlier_taps_were_replaced_stores_nothing(model):
    """Its own taps cover only the chunk and would seed the drafter as if they
    were the whole prompt, so the request decodes plain."""
    _prefill(model, prompt_len=1024, key=10, slot=0)
    _prefill(model, prompt_len=64, key=20, slot=1)
    _prefill(model, prompt_len=2048, key=10, slot=0, start=1024)
    assert model._spec_pending is None and model._dflash_pending_owner is None
    assert model.model[0].tap_layers is None
    out = _ordinary(model, [3], [2048], [10], result="device")
    assert out == "device" and _names(model, "bootstrap") == []


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


def test_a_solo_prefill_keeps_a_live_session_for_its_straddle_verify(model, mtp):
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

    # -- the MTP rail: a deferred greedy row is read back when a sampled request takes its first step
    # G replays greedy-only (no readback); G's verify is deferred; S prefills and takes its narrow
    #     step, which reads G's still-live replay back (exactly once); S's proposal overwrites the buffer;
    #     G's logits verify answers G's own columns with no further readback.
    mh._start_solo(mtp)  # G at slot 0
    decoder = mtp._mtp_decoder
    assert decoder.readbacks == 0 and mtp._mtp_sessions[0].retained.logits is None
    mh._prefill(mtp, slot=1, key=20)  # S
    mh._ordinary(
        mtp, [0, 9], [-1, 2], [0, 20], result=_tensor([0, 42]), sampling_params=_sampling(2, temperature=[0.0, 0.7])
    )
    assert decoder.readbacks == 1 and mtp._mtp_sessions[0].retained.logits is not None
    mh._propose(mtp, [[-1] * 6, [42] + [-1] * 5], [[-1] * 6, [3] + [-1] * 5])
    assert decoder.replays == 2 and decoder.readbacks == 2, "S's own proposal reads its replay back"
    assert decoder.buffer[0, 0].argmax().item() != 150, "the buffer now holds S's replay"
    out = mh._verify(
        mtp,
        [mh._block(150, [101, 102, 103, 104, 105]), [0] + [-1] * 5],
        [mh._positions(3, 6), [-1] * 6],
        [5, 0],
        [10, 0],
        mode="logits",
    )
    assert out.logits[0].float().argmax(dim=-1).tolist() == [150, 151, 152, 153, 154, 155]
    assert decoder.readbacks == 2


# -- 3.7 block-table hygiene ---------------------------------------------------------


def test_prefill_zeroes_block_table_columns_past_the_prompt(model, mtp):
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

    # -- the MTP rail: a reseed of a subset of the record rows carries only their tables
    # Two rows in steady state and a newcomer: the reseed is the newcomer's own call (its table first,
    #     zeros after), the replay's table lists every drafting row; the decoder's zero-padding check holds.
    mh._prefill(mtp, rows=2)
    mh._ordinary(mtp, [3, 4], [2, 2], [10, 12], result=_tensor([30, 31]))
    mh._propose(mtp, [[30] + [-1] * 5, [31] + [-1] * 5], [[3] + [-1] * 5] * 2)
    drafts = [mtp._mtp_sessions[r].retained.drafts for r in range(2)]
    mh._prefill(mtp, slot=2, key=40)  # C joins: its first decode is the straddle verify below
    blocks = [mh._block(30, drafts[0]), mh._block(31, drafts[1]), mh._block(9, [])]
    mh._verify(
        mtp,
        blocks,
        [mh._positions(3, 6), mh._positions(3, 6), mh._positions(2, 1)],
        [5, 5, 0],
        [10, 12, 40],
        result=_tensor([0, 0, 42]),
    )
    committed = [[150, 151, 152, 153, 154, 155], [160, 161, 162, 163, 164, 165], [42] + [-1] * 5]
    proposal = mh._propose(mtp, committed, [mh._positions(4, 6), mh._positions(4, 6), [3] + [-1] * 5], counts=[6, 6, 1])
    assert proposal.num_valid.tolist() == [5, 5, 5]
    assert _names(mtp, "reseed")[-1] == ("reseed", (5, 5), [2], [42], [3])
    assert mtp._mtp_decoder.reseed_tables[-1] == [[40, 41, 0, 0]] + [[0, 0, 0, 0]] * 4
    step = mh._steps(mtp)[-1]
    assert step[2:6] == ([0, 1, 2], [155, 165, 42], [9, 9, 3], [5, 5, None])
    assert step[7] == [[10, 11, 0, 0], [12, 13, 0, 0], [40, 41, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


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
    # the retained proposal is a plain record
    proposal = adapter.RetainedProposal(anchor_token=1, anchor_position=2, drafts=[3], posterior=[4, 5])
    assert proposal.consumed is False
    assert (proposal.anchor_token, proposal.anchor_position, proposal.drafts, proposal.posterior) == (1, 2, [3], [4, 5])


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

    # -- the MTP rail: ring advice names the setting and the resolved numbers
    messages = []
    monkeypatch.setattr(adapter.logger, "warning", lambda message, *a, **k: messages.append(message))
    mtp = mh.build_mtp_model(adapter, monkeypatch, ring=1024)
    mtp._spec_ring_advice(mtp._SPEC_N, "Gemma4MTP speculative contract")
    assert len(messages) == 1 and messages[0].startswith("Gemma4MTP speculative contract")
    assert (
        "GEMMA4_SPEC_RING_HEADROOM_BLOCKS=16" in messages[0] and "ring 2048" in messages[0] and "P_v=6" in messages[0]
    )
    mtp = mh.build_mtp_model(adapter, monkeypatch, ring=2048)
    mtp._spec_ring_advice(mtp._SPEC_N, "Gemma4MTP speculative contract")
    assert len(messages) == 1


# ===== merged from tests/unit/test_dflash_capture_release.py: Tests for the ``release_persistent_capture`` chain.
@pytest.fixture
def expect_error_dflash_capture_release():
    return make_expect_error()


def test_contract_rail_releases_its_session_then_the_width_set_then_the_base(model):
    _start_solo(model)
    model.release_persistent_capture()
    assert model._dflash_retained is None and model._dflash_live_owner is None
    assert model._spec_pending is None
    assert ("release_decoder", True) in model.events
    assert model.base_releases == [1]


def test_contract_rail_reaches_the_base_when_its_own_release_fails(model, expect_error_dflash_capture_release):
    _start_solo(model)

    def failing(drop_page_tables=True, teardown=False):
        raise RuntimeError("trace release failed")

    model._spec_release_decoder = failing
    with expect_error_dflash_capture_release(RuntimeError, "trace release failed"):
        model.release_persistent_capture()
    assert model.base_releases == [1]


# ===== merged from tests/unit/test_dflash_sampling_state.py: Host coverage for sampling state when dFlash returns a fused argmax token.
@pytest.fixture
def expect_error_dflash_sampling_state():
    return make_expect_error()


def test_fused_bootstrap_resets_sampling_before_the_first_proposal_declines(model):
    sampler = model.model[0].sampling
    _prefill(model, key=10)
    _ordinary(
        model,
        [3],
        [2],
        [10],
        result="sampled",
        sampling_params=_sampling(1, temperature=1.0, top_k=20, presence_penalty=0.5, seed=7),
    )
    assert sampler.uploads[-1]["k"][0] == 20 and sampler._penalties_active
    model.release_request(0)
    _prefill(model, key=20)
    model.model_args[0].max_seq_len = 8
    greedy = _sampling(1, seed=19)
    out = _ordinary(model, [3], [2], [20], sampling_params=greedy)
    assert out.tolist() == [150]
    assert len(sampler.uploads) == 2 and sampler.uploads[-1]["k"][0] == 1
    assert not sampler._penalties_active
    assert sampler.seed_manager.seeds[0] == 19 and sampler.seed_manager.positions[0] == 2
    assert len([event for event in sampler.seed_manager.events if event[0] == "draw"]) == 1

    proposal = _propose(model, [[150, -1, -1, -1, -1, -1]], [[3, -1, -1, -1, -1, -1]])
    assert proposal.num_valid.tolist() == [0]
    _ordinary(
        model,
        [150],
        [3],
        [20],
        result="plain",
        sampling_params=greedy,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    assert len(sampler.uploads) == 2 and sampler.uploads[-1]["k"][0] == 1
    assert not sampler._penalties_active
    assert sampler.seed_manager.seeds[0] == 19 and sampler.seed_manager.positions[0] == 3
    assert len([event for event in sampler.seed_manager.events if event[0] == "draw"]) == 2
    assert model.decode_commands[-1] == dict(
        reload_inputs=True, reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False
    )


@pytest.mark.parametrize("step", ["ordinary", "draftless-verify", "retained-verify"])
def test_fused_outputs_apply_sampling_commands_and_remap_once_without_drawing(model, step, adapter, monkeypatch):
    sampler = model.model[0].sampling
    if step == "retained-verify":
        _start_solo(model)
    else:
        _prefill(model, slot=1)
    sampler.seed_manager.events.clear()
    model.sampling_commands.clear()
    params = _sampling(2, seed=[17, None] if step != "retained-verify" else [None, 17])
    commands = dict(reload_inputs=True, reload_page_table=False, reload_sampling_params=True, reset_sampling_state=True)
    kwargs = dict(sampling_params=params, slot_remap=[1, 0], **commands)
    if step == "ordinary":
        _ordinary(model, [3, 0], [2, -1], [10, 0], **kwargs)
        row, position = 0, 2
    elif step == "draftless-verify":
        _verify(model, [[3, 0, 0, 0, 0, 0], [-1] * 6], [list(range(2, 8)), [-1] * 6], [0, 0], [10, 0], **kwargs)
        row, position = 0, 2
    else:
        _verify(
            model,
            [[-1] * 6, [150, 201, 202, 203, 204, 205]],
            [[-1] * 6, list(range(3, 9))],
            [0, 5],
            [0, 10],
            **kwargs,
        )
        row, position = 1, 3
    assert len(model.sampling_commands) == 1
    assert {name: model.sampling_commands[0][name] for name in commands if name != "reload_page_table"} == {
        name: value for name, value in commands.items() if name != "reload_page_table"
    }
    events = sampler.seed_manager.events
    assert len([event for event in events if event[0] == "remap"]) == 1
    assert not any(event[0] == "draw" for event in events)
    assert sampler.seed_manager.seeds[row] == 17
    assert sampler.seed_manager.positions[row] == position

    # -- the MTP rail: retained answers apply sampling commands and remap once without drawing
    # Without the resident decode, so that the ordinary step too is answered from the retained proposal.
    mtp = mh.build_mtp_model(adapter, monkeypatch, resident=False)
    sampler = mtp.model[0].sampling
    mh._start_solo(mtp)
    sampler.seed_manager.events.clear()
    mtp.sampling_commands.clear()
    params = _sampling(2, seed=[None, 17])
    commands = mh._commands()
    kwargs = dict(sampling_params=params, slot_remap=[1, 0], **commands)
    if step == "ordinary":
        mh._ordinary(mtp, [0, 150], [-1, 3], [0, 10], **kwargs)
    elif step == "draftless-verify":
        mh._verify(
            mtp, [[0] + [-1] * 5, mh._block(150, [])], [[-1] * 6, mh._positions(3, 1)], [0, 0], [0, 10], **kwargs
        )
    else:
        mh._verify(
            mtp,
            [[0] + [-1] * 5, mh._block(150, [101, 102, 103, 104, 105])],
            [[-1] * 6, mh._positions(3, 6)],
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


def test_retained_fused_output_preserves_false_sampling_commands(model, mtp):
    _start_solo(model)
    sampler = model.model[0].sampling
    before = len(sampler.uploads)
    sampler.seed_manager.events.clear()
    _verify(
        model,
        [[150, 201, 202, 203, 204, 205]],
        [list(range(3, 9))],
        [5],
        [10],
        reload_inputs=False,
        reload_page_table=True,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    assert len(sampler.uploads) == before
    assert model.sampling_commands[-1]["reload_inputs"] is False
    assert model.sampling_commands[-1]["reload_sampling_params"] is False
    assert model.sampling_commands[-1]["reset_sampling_state"] is False
    assert not any(event[0] in {"reset", "align", "draw"} for event in sampler.seed_manager.events)

    # -- the MTP rail: a logits verify answered from the replay follows the slot remap only
    sampler = mtp.model[0].sampling
    mh._start_solo(mtp)
    sampler.seed_manager.events.clear()
    mtp.sampling_commands.clear()
    mh._verify(
        mtp,
        [[0] + [-1] * 5, mh._block(150, [101, 102, 103, 104, 105])],
        [[-1] * 6, mh._positions(3, 6)],
        [0, 5],
        [0, 10],
        mode="logits",
        slot_remap=[1, 0],
    )
    assert [event[0] for event in sampler.seed_manager.events] == ["remap"]
    assert mtp.sampling_commands == []


def test_host_verify_rejection_preserves_seed_state_before_remap(model, expect_error_dflash_sampling_state):
    _start_solo(model)
    sampler = model.model[0].sampling
    sampler.seed_manager.seeds[0] = 23
    sampler.seed_manager.events.clear()
    model.sampling_commands.clear()
    with expect_error_dflash_sampling_state(ValueError, "cannot apply host sampling controls"):
        _verify(
            model,
            [[-1] * 6, [150, 201, 202, 203, 204, 205]],
            [[-1] * 6, list(range(3, 9))],
            [0, 5],
            [0, 10],
            sampling=False,
            slot_remap=[1, 0],
        )
    assert sampler.seed_manager.seeds[0] == 23
    assert sampler.seed_manager.seeds[1] is None
    assert sampler.seed_manager.events == []
    assert model.sampling_commands == []


@pytest.mark.parametrize("data_parallel,sampling_dp", [(1, 1), (2, 1), (1, 2)])
def test_shared_preparation_keeps_parameters_history_and_seeds_with_their_sampling_groups(
    adapter, data_parallel, sampling_dp
):
    model = SimpleNamespace(model=[SimpleNamespace(sampling_dp=sampling_dp) for _ in range(data_parallel)])
    attach_sampling_state(model)
    rows = 32 * max(data_parallel, sampling_dp)
    top_k = [1 + row % 20 for row in range(rows)]
    params = _sampling(rows, temperature=1.0, top_k=top_k, presence_penalty=0.5, seed=list(range(100, 100 + rows)))
    positions = torch.arange(rows, dtype=torch.int32)
    positions[-1] = -1
    prompts = torch.arange(2 * rows, dtype=torch.int32).reshape(rows, 2)
    outputs = torch.arange(rows, dtype=torch.int32).reshape(rows, 1)
    active = model._prepare_decode_sampling_state(
        params,
        start_pos=torch.chunk(positions, data_parallel),
        prompt_tokens=prompts,
        output_tokens=outputs,
        slot_remap=list(range(rows)),
        reload_inputs=True,
        reload_sampling_params=True,
        reset_sampling_state=True,
    )
    per_model = rows // data_parallel
    for index, target in enumerate(model.model):
        sampler = target.sampling
        start, end = index * per_model, (index + 1) * per_model
        assert sampler.uploads[-1]["k"] == top_k[start:end]
        assert torch.equal(sampler.prompt_resets[-1], prompts[start:end])
        assert torch.equal(sampler.output_resets[-1], outputs[start:end])
        expected_active = [row for row in range(per_model) if int(positions[start + row]) >= 0]
        assert active[index] == expected_active
        assert [sampler.seed_manager.seeds[row] for row in expected_active] == [
            100 + start + row for row in expected_active
        ]
        assert not any(event[0] == "draw" for event in sampler.seed_manager.events)


@pytest.mark.parametrize(
    "missing", ["reload_inputs", "reload_page_table", "reload_sampling_params", "reset_sampling_state"]
)
def test_fused_decode_requires_each_command_before_mutating_state(model, expect_error_dflash_sampling_state, missing):
    _prefill(model)
    commands = dict(reload_inputs=True, reload_page_table=False, reload_sampling_params=True, reset_sampling_state=True)
    del commands[missing]
    events = list(model.events)
    with expect_error_dflash_sampling_state(TypeError, missing):
        model.decode_forward(
            tokens=_tensor([[3]]),
            start_pos=_tensor([2]),
            page_table=_table([10]),
            kv_cache=model.kv_cache,
            sampling_params=_sampling(1),
            **commands,
        )
    assert model.events == events
    assert model.sampling_commands == []


# ===== merged from tests/unit/test_dflash_width_prepare.py: Host tests for the dFlash verify-width preparation and capture order.
@pytest.fixture
def expect_error_dflash_width_prepare():
    return make_expect_error()


@pytest.fixture
def decoder(drafter_module, monkeypatch):
    cls = drafter_module.DFlashFusedDecoder
    dec = cls.__new__(cls)
    dec.use_packed = True
    dec._pv_widths = {}
    dec._prepared_widths = set()
    dec.pv_sk = None
    dec.trace = None
    dec._out = None
    dec._replay_on_first = False
    dec.events = []
    dec.target = SimpleNamespace(
        dflash_capture_taps=lambda layers, buffers=None, keep_last=None: dec.events.append(
            ("taps", None if layers is None else tuple(layers))
        )
    )
    dec.drafter = SimpleNamespace(target_layer_ids=[1, 2])
    dec.tap_bufs = "bufs"
    dec.mesh_device = "mesh"

    def install(width):
        dec._pv_widths.setdefault(int(width), {"pv_sk": int(width), "trace": None})
        dec.events.append(("install", int(width)))

    def activate(record):
        dec.pv_sk = record["pv_sk"]
        dec.events.append(("activate", record["pv_sk"]))

    def body():
        dec.events.append(("body", dec.pv_sk))
        return ("ids", "fc")

    monkeypatch.setattr(dec, "_pv_ring_meta", lambda: dec.events.append(("ring_meta",)))
    monkeypatch.setattr(dec, "_pv_width_install", install)
    monkeypatch.setattr(dec, "_seed_ctx_cache", lambda: dec.events.append(("seed",)))
    monkeypatch.setattr(dec, "_pv_width_activate", activate)
    monkeypatch.setattr(dec, "_pv_upload", lambda start: dec.events.append(("pv_upload", int(start))))
    monkeypatch.setattr(
        dec, "_upload_iter_inputs", lambda anchor, start: dec.events.append(("iter_inputs", anchor, start))
    )
    monkeypatch.setattr(dec, "_body", body)
    monkeypatch.setattr(dec, "restore_model_logits_mode", lambda: dec.events.append(("restore",)))
    runtime = sys.modules["ttnn"]
    counter = [0]

    def begin(mesh, cq_id=0):
        counter[0] += 1
        dec.events.append(("begin_trace", counter[0]))
        return counter[0]

    monkeypatch.setattr(runtime, "begin_trace_capture", begin)
    monkeypatch.setattr(runtime, "end_trace_capture", lambda mesh, tid, cq_id=0: dec.events.append(("end_trace", tid)))
    monkeypatch.setattr(runtime, "release_trace", lambda mesh, tid: dec.events.append(("release_trace", tid)))
    monkeypatch.setattr(runtime, "synchronize_device", lambda mesh: None)
    return dec


def _kinds(dec, *kinds):
    return [event for event in dec.events if event[0] in kinds]


def test_prepare_installs_every_width_before_seeding_and_running_any_body(decoder):
    decoder.prepare_widths([4096, 2048])
    kinds = [event[0] for event in decoder.events]
    assert kinds[:4] == ["ring_meta", "install", "install", "seed"]
    assert _kinds(decoder, "install") == [("install", 2048), ("install", 4096)]
    assert _kinds(decoder, "body") == [("body", 2048), ("body", 4096)]
    assert _kinds(decoder, "begin_trace") == []
    assert decoder._prepared_widths == {2048, 4096}
    assert decoder.events[-2:] == [("taps", None), ("restore",)]
    assert all(record["trace"] is None for record in decoder._pv_widths.values())


def test_prepare_is_idempotent(decoder):
    decoder.prepare_widths([2048])
    seen = list(decoder.events)
    decoder.prepare_widths([2048])
    assert decoder.events == seen


def test_capture_prepares_first_and_records_one_trace_per_width(decoder):
    cost = decoder.capture_widths([2048, 4096])
    bodies = _kinds(decoder, "body")
    begins = _kinds(decoder, "begin_trace")
    assert [b[1] for b in bodies] == [2048, 4096, 2048, 4096]
    assert decoder.events.index(begins[0]) > decoder.events.index(bodies[1])
    assert sorted(cost) == [2048, 4096]
    assert decoder._pv_widths[2048]["trace"] == 1 and decoder._pv_widths[4096]["trace"] == 2
    assert decoder.trace == 2
    assert decoder._replay_on_first is True
    assert decoder._out == ("ids", "fc")


def test_captured_widths_are_not_captured_again(decoder):
    decoder.capture_widths([2048])
    decoder.capture_widths([2048])
    assert len(_kinds(decoder, "begin_trace")) == 1


def test_a_new_width_after_capture_is_refused(decoder, expect_error_dflash_width_prepare):
    decoder.capture_widths([2048])
    with expect_error_dflash_width_prepare(RuntimeError, "after trace capture"):
        decoder.prepare_widths([4096])
    with expect_error_dflash_width_prepare(RuntimeError, "after trace capture"):
        decoder.capture_widths([2048, 4096])


def test_a_failed_capture_releases_its_trace_and_leaves_the_width_uncaptured(
    decoder, monkeypatch, expect_error_dflash_width_prepare
):
    calls = [0]

    def body():
        calls[0] += 1
        if calls[0] == 2:  # the capture pass, after the prepare pass
            raise RuntimeError("boom")
        return ("ids", "fc")

    monkeypatch.setattr(decoder, "_body", body)
    with expect_error_dflash_width_prepare(RuntimeError, "boom"):
        decoder.capture_widths([2048])
    assert _kinds(decoder, "release_trace") == [("release_trace", 1)]
    assert decoder._pv_widths[2048]["trace"] is None
    assert decoder.events[-2:] == [("taps", None), ("restore",)]


def test_unpacked_verification_cannot_use_the_width_set(decoder, expect_error_dflash_width_prepare):
    decoder.use_packed = False
    with expect_error_dflash_width_prepare(NotImplementedError, "packed-verify"):
        decoder.prepare_widths([2048])


def test_verify_count_from_the_adapter_overrides_the_environment(drafter_module, monkeypatch):
    resolve = drafter_module.DFlashFusedDecoder._resolve_verify_count
    monkeypatch.delenv("GEMMA4_DFLASH_VERIFY", raising=False)
    assert resolve(15, True, 5) == 5
    monkeypatch.setenv("GEMMA4_DFLASH_VERIFY", "7")
    assert resolve(15, True, 5) == 5
    assert resolve(15, True, None) == 7
    monkeypatch.delenv("GEMMA4_DFLASH_VERIFY", raising=False)
    assert resolve(15, True, None) == 15
    assert resolve(15, False, None) == 15


@pytest.mark.parametrize("bad", [0, 16, -1])
def test_verify_count_outside_the_drafter_block_is_refused(drafter_module, bad, expect_error_dflash_width_prepare):
    with expect_error_dflash_width_prepare(ValueError, r"\[1, 15\]"):
        drafter_module.DFlashFusedDecoder._resolve_verify_count(15, True, bad)


def test_verify_count_below_the_block_needs_packed_verification(drafter_module, expect_error_dflash_width_prepare):
    with expect_error_dflash_width_prepare(ValueError, "packed"):
        drafter_module.DFlashFusedDecoder._resolve_verify_count(15, False, 5)
    assert drafter_module.DFlashFusedDecoder._resolve_verify_count(15, False, 15) == 15


class _RecordingDecoder:
    instances = []

    def __init__(self, target, drafter, kv_layers, page_table, ctx_cap=2048, *, verify_count=None):
        self.verify_count = verify_count
        self.page_table = page_table
        self.events = []
        self._pv_widths = {}
        _RecordingDecoder.instances.append(self)

    def prepare_widths(self, widths):
        self.events.append(("prepare", tuple(widths)))
        for width in widths:
            self._pv_widths.setdefault(int(width), {"trace": None})

    def capture_widths(self, widths):
        self.events.append(("capture", tuple(widths)))
        for width in widths:
            self._pv_widths[int(width)]["trace"] = object()
        return {int(width): 0.1 for width in widths}

    def measure_widths(self, top_position):
        captured = sorted(w for w, r in self._pv_widths.items() if r["trace"] is not None)
        self.events.append(("measure", int(top_position), tuple(captured)))
        return {w: (0, 40.0 + i, int(top_position), 60.0 + i) for i, w in enumerate(captured)}


def _warm(adapter, monkeypatch, plain_ms=33.0, env=None):
    """A contract model whose fused decoder is the recording stub and whose base warmup is a no-op."""
    _RecordingDecoder.instances = []
    monkeypatch.setattr(
        sys.modules["models.demos.gemma4.tt.dflash_drafter"], "DFlashFusedDecoder", _RecordingDecoder, raising=False
    )
    model = build_model(adapter, monkeypatch, env=env)
    model._spec_decoder = None
    model._spec_width_ladder = None
    model.model_args[0].max_seq_len = 4096
    model._harness_warmup_model_decode = lambda self, *a, **k: "warm"

    def time_plain_step(self, warmup_kwargs, page_table, position=0, repeats=2, batch=None):
        batch = int(batch or warmup_kwargs.get("max_batch_size") or model.model_args[0].max_batch_size)
        model.events.append(("time_plain", list(page_table[:batch].shape), int(position)))
        return plain_ms + position / 100.0 - (0.0 if batch == 32 else float(batch))

    monkeypatch.setattr(adapter.Gemma4DFlashContractForCausalLM, "_spec_time_plain_step", time_plain_step)
    return model


@pytest.fixture
def warm_model(adapter, monkeypatch):
    return _warm(adapter, monkeypatch)


def test_eager_warmup_prepares_and_traced_warmup_captures_on_the_same_decoder(
    warm_model, adapter, monkeypatch, assistant_dir
):
    model = warm_model
    assert model.warmup_model_decode(enable_trace=False, kv_cache=model.kv_cache, num_blocks=64) == "warm"
    assert len(_RecordingDecoder.instances) == 1
    dec = _RecordingDecoder.instances[0]
    assert dec.verify_count == 5
    assert [e[0] for e in dec.events] == ["prepare"]
    assert model._spec_decoder is dec
    assert model.warmup_model_decode(enable_trace=True, kv_cache=model.kv_cache, num_blocks=64) == "warm"
    assert len(_RecordingDecoder.instances) == 1
    assert [e[0] for e in dec.events] == ["prepare", "capture", "measure"]
    assert dec.events[0][1] == dec.events[1][1] == tuple(model._spec_width_ladder)

    # -- the MTP rail: warmup runs the base buckets then prepares then captures
    mtp = mh.build_mtp_model(adapter, monkeypatch, warm=False, env={"GEMMA4_ASSISTANT_MODEL": assistant_dir})
    built = mh.stub_drafter_modules(monkeypatch, mtp.events)
    mh._warmup(mtp, False)
    mh._warmup(mtp, True)
    kinds = [event[0] for event in mtp.events]
    # after the capture the policy measures the records (the per-layer tables installed first) and times the
    # plain step at position 0 and at the top
    assert kinds == [
        "base_warmup",
        "prepare",
        "base_warmup",
        "capture",
        "refresh",
        "measure",
        "time_plain",
        "time_plain",
    ]
    assert mtp.events[0] == ("base_warmup", False) and mtp.events[2] == ("base_warmup", True)
    assert mtp.events[1] == ("prepare", [(1, 5), (5, 5)], 5)  # the default table: K=5 up to 5 live rows
    assistant = dict(built[0][1])
    assert built[0][0] == "assistant" and assistant["max_local_batch_size"] == 5  # the table's last max_rows
    assert assistant["assistant_path"] == assistant_dir and assistant["max_seq_len"] == 1024 + 64
    decoder = dict(built[1][1])
    assert built[1][0] == "decoder" and decoder["draft_len"] == 5
    assert tuple(decoder["page_table_torch"].shape) == (5, 8) and decoder["page_table_torch"].eq(0).all()
    assert mtp._mtp_captured and mtp._mtp_max_rows == 5 and mtp._mtp_shapes == ((1, 5), (5, 5))
    assert mtp._mtp_decoder.widths[(1, 5)] == adapter.mtp_pv_width_ladder(1024, 5)
    assert mtp._mtp_decoder.widths[(5, 5)] == adapter.mtp_pv_width_ladder(1024, 5)
    # -- the MTP rail: warmup caps the records at the engine batch and the row cap
    mtp = mh.build_mtp_model(
        adapter,
        monkeypatch,
        warm=False,
        max_batch_size=6,
        env={
            "GEMMA4_ASSISTANT_MODEL": assistant_dir,
            "GEMMA4_SPEC_K_BY_ROWS": "5:5,8:3,16:1",  # a table past the engine batch: capped at (6, 3)
            "GEMMA4_SPEC_MAX_ROWS": "8",
            "GEMMA4_SPEC_MAX_ISL": "512",
            "GEMMA4_SPEC_EXTRA_SHAPES": "5:3",  # the 1-5 row band's second draft length: a candidate, not the default
        },
    )
    mh.stub_drafter_modules(monkeypatch, mtp.events)
    mh._warmup(mtp, False)
    assert mtp._mtp_shapes == ((1, 5), (5, 3), (5, 5), (6, 3)) and mtp._mtp_max_rows == 6
    assert mtp._mtp_table_shapes == ((1, 5), (5, 5), (6, 3))
    assert mtp._mtp_shape_for(3) == (5, 5) and mtp._mtp_shape_for(6) == (6, 3) and mtp._mtp_shape_for(7) is None
    assert (5, 3) in mtp._mtp_widths
    assert mtp._mtp_widths[(1, 5)] == adapter.mtp_pv_width_ladder(512, 5)
    # -- the MTP rail: a traced only warmup prepares before it captures
    mtp = mh.build_mtp_model(adapter, monkeypatch, warm=False, env={"GEMMA4_ASSISTANT_MODEL": assistant_dir})
    mh.stub_drafter_modules(monkeypatch, mtp.events)
    mh._warmup(mtp, True)
    assert [event[0] for event in mtp.events] == [
        "base_warmup",
        "prepare",
        "capture",
        "refresh",
        "measure",
        "time_plain",
        "time_plain",
    ]

    # -- the acceptance-adaptive policy: the traced warmup measures the cost lines and times the plain step,
    # the env plain cost skips the timing, and the policy switched off measures nothing (both rails)
    _dflash_the_traced_warmup_measures_every_captured_width_and_times_the_plain_step(adapter, monkeypatch)
    _dflash_the_plain_step_cost_from_the_env_skips_the_timed_replay(adapter, monkeypatch)
    _dflash_adaptive_off_measures_nothing_at_warmup(adapter, monkeypatch)
    _mtp_warmup_measures_the_records_and_times_the_plain_step_after_the_capture(
        adapter, monkeypatch, assistant_dir=assistant_dir
    )
    _mtp_the_plain_step_cost_from_the_env_skips_the_timed_replay(adapter, monkeypatch, assistant_dir=assistant_dir)
    _mtp_adaptive_off_measures_nothing_at_warmup(adapter, monkeypatch, assistant_dir=assistant_dir)


def test_a_changed_ladder_between_preparation_and_capture_is_refused(
    warm_model, expect_error_dflash_width_prepare, adapter, monkeypatch, expect_error, assistant_dir, tmp_path
):
    model = warm_model
    model.warmup_model_decode(enable_trace=False, kv_cache=model.kv_cache, num_blocks=64)
    model._spec_width_ladder = [1234]
    with expect_error_dflash_width_prepare(RuntimeError, "widths changed"):
        model.warmup_model_decode(enable_trace=True, kv_cache=model.kv_cache, num_blocks=64)

    # -- the MTP rail: warmup refuses a second capture and a missing assistant
    mtp = mh.build_mtp_model(adapter, monkeypatch, warm=False, env={"GEMMA4_ASSISTANT_MODEL": assistant_dir})
    mh.stub_drafter_modules(monkeypatch, mtp.events)
    mh._warmup(mtp, False)
    mh._warmup(mtp, True)
    with expect_error(RuntimeError, "captured twice"):
        mh._warmup(mtp, True)
    missing = mh.build_mtp_model(
        adapter, monkeypatch, warm=False, env={"GEMMA4_ASSISTANT_MODEL": str(tmp_path / "none")}
    )
    with expect_error(RuntimeError, "GEMMA4_ASSISTANT_MODEL"):
        mh._warmup(missing, False)


# -- the composed vision class: the taps capture around the multimodal prefill ----------------------


def _mm_prefill(model, tokens, prompt_len, *, start=None, key=10, offsets=None, lengths=None, pixel_values=None):
    """One multimodal prefill call as the plugin makes it: payload plus placeholder metadata per user."""
    kwargs = dict(
        tokens=tokens,
        prompt_lens=[prompt_len],
        empty_slots=[0],
        page_table=_table([key]),
        kv_cache=model.kv_cache,
        warmup_prefill=False,
        mm_placeholder_offsets=[offsets],
        mm_placeholder_lengths=[lengths],
    )
    if pixel_values is not None:
        kwargs["pixel_values"] = [pixel_values]
    if start is not None:
        kwargs["start_pos"] = [start]
    return model.prefill_forward(**kwargs)


def _image_prompt(prompt_len, offset, length, image_id=5):
    tokens = torch.full((1, prompt_len), 7, dtype=torch.int32)
    tokens[0, offset : offset + length] = image_id
    return tokens


def test_composed_vision_class_keeps_the_dflash_contract_above_the_multimodal_prefill(adapter):
    cls = adapter.Gemma4DFlashContractVisionForConditionalGeneration
    contract = adapter.Gemma4DFlashContractForCausalLM
    mro = cls.__mro__
    assert mro.index(contract) < mro.index(adapter.Gemma4SpecContractBase) < mro.index(adapter.Gemma4DFlashBase)
    assert mro.index(adapter.Gemma4DFlashBase) < mro.index(adapter.Gemma4VisionForConditionalGeneration)
    assert mro.index(adapter.Gemma4VisionForConditionalGeneration) < mro.index(adapter.Gemma4ForCausalLM)
    assert adapter.SupportsMultiModal in mro
    # The taps override is the class's prefill, and it reaches the multimodal staging by calling the
    # base prefill by name, not through super(), so the vision class adds nothing on that path.
    assert cls.prefill_forward is contract.prefill_forward
    source = inspect.getsource(contract.prefill_forward)
    assert "Gemma4ForCausalLM.prefill_forward(self" in source and "super().prefill_forward" not in source
    assert cls.decode_forward is contract.decode_forward and cls.propose_draft_tokens is contract.propose_draft_tokens
    assert cls.model_capabilities["supports_spec_decode"] is True
    registry = sys.modules["vllm.multimodal"].MULTIMODAL_REGISTRY
    # -- the MTP rail: the composed vision classes keep the contract above the multimodal prefill
    for name, vision, info in [
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
    ]:
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
        if not isinstance(registry, SimpleNamespace):
            factory = cls.__dict__["_processor_factory"]
            assert factory.info is getattr(adapter, info)
            assert factory.processor is adapter.Gemma4MultiModalProcessor
            assert factory.dummy_inputs is adapter.Gemma4DummyInputsBuilder
    cls, contract = adapter.Gemma4DFlashContractVisionForConditionalGeneration, adapter.Gemma4DFlashContractForCausalLM
    if isinstance(registry, SimpleNamespace):
        pytest.skip("vLLM absent: the registry is the conftest stub; registration is checked on the vLLM runners")
    factory = cls.__dict__["_processor_factory"]
    assert factory.info is adapter.Gemma4ProcessingInfo and factory.processor is adapter.Gemma4MultiModalProcessor


def test_an_image_prefill_captures_taps_over_the_prefilled_rows_like_a_text_prefill(adapter, monkeypatch):
    text = build_model(adapter, monkeypatch)
    _prefill(text, prompt_len=8, key=10)
    vision = build_model(adapter, monkeypatch, cls=adapter.Gemma4DFlashContractVisionForConditionalGeneration)
    vision.is_multimodal = True
    vision.image_token_id = 5
    _mm_prefill(vision, _image_prompt(8, 1, 4), 8, offsets=[1], lengths=[4], pixel_values=[torch.zeros(4, 8)])
    for model in (text, vision):
        taps, end = model._spec_pending
        assert end == 8 and [tap.shape for tap in taps] == [(1, 1, 8, 8), (1, 1, 8, 8)]
        assert _names(model, "capture")[-2:] == [("capture", [1, 2]), ("capture", None)]
    # The multimodal payload and placeholder metadata reached the base prefill untouched, eagerly.
    assert {"pixel_values", "mm_placeholder_offsets", "mm_placeholder_lengths"} <= set(vision.prefill_kwargs[-1])
    assert _names(vision, "prefill") == [("prefill", 1, [8], False)]


def test_a_continuation_whose_restart_the_base_moves_into_a_vision_item_is_not_speculable(adapter, monkeypatch):
    model = build_model(adapter, monkeypatch, cls=adapter.Gemma4DFlashContractVisionForConditionalGeneration)
    model.is_multimodal = True
    model.image_token_id = 5
    _mm_prefill(model, _image_prompt(256, 10, 20), 256, offsets=[10], lengths=[20])
    assert model._spec_pending[1] == 256
    # A continuation at the SDPA-aligned 256 that cuts no item keeps the taps and extends them.
    _mm_prefill(model, _image_prompt(400, 10, 20), 400, start=256, offsets=[10], lengths=[20])
    taps, end = model._spec_pending
    assert end == 400 and [tap.shape[2] for tap in taps] == [256, 256, 144, 144]
    # A continuation at 256 cutting the image run [200, 300) is restarted at 128 by the base, which
    # re-prefills rows 128..255: their taps would repeat, so the chunk runs without speculation.
    model = build_model(adapter, monkeypatch, cls=adapter.Gemma4DFlashContractVisionForConditionalGeneration)
    model.is_multimodal = True
    model.image_token_id = 5
    _mm_prefill(model, _image_prompt(256, 10, 20), 256, offsets=[10], lengths=[20])
    _mm_prefill(model, _image_prompt(400, 200, 100), 400, start=256, offsets=[200], lengths=[100])
    assert model._spec_pending is None
    assert _names(model, "capture")[-1] == ("capture", None)
    assert _names(model, "prefill")[-1] == ("prefill", 1, [400], False), "the base prefill still runs"


# ===== ctx-cache commit: every local KV head is merged (TP8 one head, TP4 two, replicated eight)
def test_commit_ctx_caches_gathers_every_local_kv_head(drafter_module, monkeypatch):
    """TP8 keeps the one-gather graph; TP4 (two local KV heads) and the replicated
    drafter gather each head through the shared row map, stack the heads on dim 1
    and write the whole cache (the old graph refused more than one head)."""
    for local_kv in (1, 2, 8):
        _commit_ctx_caches_case(drafter_module, monkeypatch, local_kv)


def _commit_ctx_caches_case(drafter_module, monkeypatch, local_kv):
    runtime = sys.modules["ttnn"]
    calls = []

    class T:
        def __init__(self, shape, tag):
            self.shape = tuple(shape)
            self.tag = tag
            self.freed = False

        def __getitem__(self, idx):
            head = idx[1]
            calls.append(("slice", self.tag, head.start))
            return T((self.shape[0], 1) + self.shape[2:], f"{self.tag}[{head.start}]")

        def deallocate(self, force=False):
            self.freed = True

    def concat(ts, dim):
        calls.append(("concat", dim, tuple(t.tag for t in ts)))
        shape = list(ts[0].shape)
        shape[dim] = sum(t.shape[dim] for t in ts)
        return T(shape, "cat")

    def reshape(t, shape):
        calls.append(("reshape", tuple(shape)))
        return T(shape, "view")

    def embedding(idx, table, layout=None):
        calls.append(("embedding", table.shape))
        return T((1, idx.shape[1], table.shape[1]), "gathered")

    def assign(src, dst):
        calls.append(("assign", src.shape, dst.tag))

    for name, fn in (("concat", concat), ("reshape", reshape), ("embedding", embedding), ("assign", assign)):
        monkeypatch.setattr(runtime, name, fn, raising=False)

    cls = drafter_module.DFlashFusedDecoder
    dec = cls.__new__(cls)
    cap, P_v, hd, layers = 2048, 6, 128, 2
    dec.cap, dec.P_v = cap, P_v
    dec.drafter = SimpleNamespace(local_kv=local_kv, head_dim=hd)
    dec.merge_idx = T((1, cap), "merge_idx")
    dec.ctx_k = [T((1, local_kv, cap, hd), f"k{li}") for li in range(layers)]
    dec.ctx_v = [T((1, local_kv, cap, hd), f"v{li}") for li in range(layers)]
    kv_new = [(T((1, local_kv, P_v, hd), f"kn{li}"), T((1, local_kv, P_v, hd), f"vn{li}")) for li in range(layers)]

    dec._commit_ctx_caches(kv_new)

    assigns = [c for c in calls if c[0] == "assign"]
    assert [c[2] for c in assigns] == ["k0", "v0", "k1", "v1"]
    assert all(c[1] == (1, local_kv, cap, hd) for c in assigns), "the whole cache is written"
    gathers = [c for c in calls if c[0] == "embedding"]
    assert len(gathers) == 2 * layers * local_kv
    assert all(c[1] == (cap + P_v, hd) for c in gathers), "one head's [cache | new] rows per gather"
    sliced_heads = sorted({c[2] for c in calls if c[0] == "slice"})
    assert sliced_heads == list(range(local_kv))
    head_stacks = [c for c in calls if c[0] == "concat" and c[1] == 1]
    assert len(head_stacks) == (0 if local_kv == 1 else 2 * layers)
    assert all(len(c[2]) == local_kv for c in head_stacks)
    assert all(t.freed for layer in kv_new for t in layer)


# ===== packed verify page tables follow vLLM's KV-cache group slots (not the layer type)
def test_packed_verify_slot_sources_hand_every_group_its_own_table(drafter_module):
    """vLLM-grouped per-layer tables with distinct block ids per group: each slot
    reads its own group's row; with nothing installed every slot reads the flat
    table; a slot whose layer has no installed table falls back to the flat row.
    Then the slot-row padding rule and the K-cache source rule of the same page-table path."""
    from models.demos.gemma4.tt.spec_decode import serving_kv_group_slots

    types = (["sliding_attention"] * 5 + ["full_attention"]) * 8
    slots, n_slots = serving_kv_group_slots(types)
    assert n_slots == 6
    blocks = 16
    installed = [torch.arange(blocks, dtype=torch.int64).reshape(1, blocks) + 1000 * slots[i] for i in range(48)]
    flat = torch.full((1, blocks), 7, dtype=torch.int64)

    sources = drafter_module.packed_verify_slot_sources(installed, flat, slots)
    assert len(sources) == n_slots
    for slot, src in enumerate(sources):
        assert torch.equal(src, installed[slots.index(slot)][0]), f"slot {slot}"
    # every layer resolves to its own group's ids through its slot
    for i in range(48):
        assert int(sources[slots[i]][0]) == 1000 * slots[i], f"layer {i}"
    # the old per-type rule would have read layer 0's row for layer 1
    assert not torch.equal(sources[slots[1]], installed[0][0])

    assert all(torch.equal(src, flat[0]) for src in drafter_module.packed_verify_slot_sources(None, flat, slots))
    partial = list(installed)
    partial[1] = None  # slot 1's first layer has no table -> flat
    assert torch.equal(drafter_module.packed_verify_slot_sources(partial, flat, slots)[1], flat[0])

    # the slot row pads and never clamps
    src = torch.arange(5, dtype=torch.int64) + 1
    padded = drafter_module.packed_verify_slot_row(src, 8)
    assert padded.shape == (1, 8) and padded.dtype == torch.int32
    assert padded[0].tolist() == [1, 2, 3, 4, 5, 0, 0, 0]
    clipped = drafter_module.packed_verify_slot_row(src, 3)
    assert clipped[0].tolist() == [1, 2, 3]
    # The K cache prefers the served caches and tolerates None: under vLLM the attention modules' own
    # kv_cache is None and the decoder verifies through the served per-layer [k, v] list; the demo paths
    # hand caches per forward and get None (block 64).
    served = [["k0", "v0"], ["k1", "v1"]]
    layer_with_own = SimpleNamespace(self_attn=SimpleNamespace(kv_cache=["own_k", "own_v"]))
    layer_without = SimpleNamespace(self_attn=SimpleNamespace(kv_cache=None))
    f = drafter_module.packed_verify_k_cache
    assert f(served, layer_without, 1) == "k1"
    assert f(served, layer_with_own, 0) == "k0"
    assert f(None, layer_with_own, 0) == "own_k"
    assert f(None, layer_without, 0) is None
    assert f([[None, None]], layer_without, 0) is None
    assert f(served, layer_without, 5) is None


# -- the acceptance-adaptive policy (tt/spec_policy.py) on this rail: helpers the tests above call as blocks


# -- the acceptance-adaptive policy on this rail (tt/spec_policy.py) ------------------------------

# The 31B CI leg: 46.9 ms per fused replay against the ~34 ms plain step (1.38x), one width rung.
DFLASH_STEP_MS = 46.9
DFLASH_PLAIN_MS = 34.0


def _with_costs(model, step_ms=DFLASH_STEP_MS, plain_ms=DFLASH_PLAIN_MS):
    """Install the cost table the traced warmup would have measured: one record, ``(1, 5)``, per width (flat
    in the position)."""
    model._spec_decoder.costs = {1024: float(step_ms)}
    model._spec_costs = {("step", (1, 5), 1024): CostLine.flat(step_ms)}
    model._spec_plain_cost = {32: CostLine.flat(plain_ms)}
    return model


def _costed(adapter, monkeypatch):
    """A contract model with the CI leg's costs, a three-step retry at a 0.6 overhead and no probe."""
    env = {
        "GEMMA4_SPEC_RETRY_STEPS": "3",
        "GEMMA4_SPEC_RETRY_OVERHEAD": "0.6",
        "GEMMA4_SPEC_PROBE_STEPS": "0",
        "GEMMA4_SPEC_CALIB_MIN": "1000",  # the pinned sequences read the scripted table
        "GEMMA4_SPEC_MIN_GAIN": "1.10",  # the band mechanics below are written at the 1.10 bar (the default is 1.25)
        "GEMMA4_SPEC_TRIAL_STEPS": "1",  # the scenarios describe one-step trials and retries
    }
    return _with_costs(build_model(adapter, monkeypatch, env=env))


def _inform_prior(model, cond):
    """Give the batch prior the conditionals ``cond`` as if every position had been observed."""
    prior = model._spec_policy.prior
    prior.cond = list(cond)
    prior.informed = [True] * len(cond)


def _dflash_rounds(model, accepted_per_round, key=10):
    """Drive one solo request in the runner's order: per round the drafts its verify accepts (unused once
    the session has ended and the row runs plain). Returns the draft length every proposal offered, the
    first one right after the bootstrap; 0 is a plain step."""
    proposal = _start_solo(model, key=key)
    anchor, pos = 150, 3
    offered = [int(proposal.num_valid[0])]
    for accepted in accepted_per_round:
        k = offered[-1]
        if k:
            drafts = proposal.draft_token_ids[0, :k].tolist()
            out = _verify(model, [[anchor] + drafts], [list(range(pos, pos + 1 + k))], [k], keys=[key])
            ids = out.argmax_ids[0].tolist()
            n = min(int(accepted), k) + 1
            committed = ids[:n] + [-1] * (6 - n)
            positions = list(range(pos + 1, pos + 1 + n)) + [-1] * (6 - n)
            proposal = _propose(model, [committed], [positions], counts=[n])
            anchor, pos = ids[n - 1], pos + n
        else:
            assert _ordinary(model, [anchor], [pos], [key], result="device") == "device"
            anchor, pos = anchor + 1, pos + 1
            proposal = _propose(model, [[anchor] + [-1] * 5], [[pos] + [-1] * 5])
        offered.append(int(proposal.num_valid[0]))
    return offered


def _dflash_without_a_cost_table_every_solo_greedy_request_speculates_and_a_warning_says_so_once(
    adapter, monkeypatch, expect_error=None
):
    model = build_model(adapter, monkeypatch)
    warned = []
    monkeypatch.setattr(adapter.logger, "warning", lambda message, *a, **k: warned.append(str(message)))
    _inform_prior(model, [0.0] * 5)  # with a table the policy would never start a session
    assert _dflash_rounds(model, [0, 0, 0]) == [5, 5, 5, 5]
    assert warned == [
        "Gemma4DFlash adaptive K: no cost table (the warmup measured none and GEMMA4_SPEC_PLAIN_STEP_MS is "
        "unset); speculation runs as without the policy"
    ]


def _dflash_the_first_request_speculates_from_the_optimistic_prior_and_the_counts_feed_its_ewma(
    adapter, monkeypatch, expect_error=None
):
    costed = _costed(adapter, monkeypatch)
    assert _dflash_rounds(costed, [2]) == [5, 5]  # the verify carried 5 drafts and committed 3 tokens
    stats = costed._dflash_stats
    assert stats.cond == pytest.approx([1.0, 1.0, 0.9, 1.0, 1.0]) and stats.observed == [1, 1, 1, 0, 0]
    assert stats.ewma_accepted == 2.0 and stats.spec_steps == 1
    assert costed._spec_policy.prior.cond == pytest.approx([1.0, 1.0, 0.9, 1.0, 1.0]), "the batch prior learns too"
    assert costed._dflash_retained.verified == 0, "the new proposal is not verified yet"


def _dflash_rejections_end_the_session_and_the_request_runs_plain(adapter, monkeypatch, expect_error=None):
    """From a 0.35 prior (1.53 tokens at 1.38x = 1.11x) the session starts; EWMA 0.1 on the first draft takes
    the gain under 1.10x at once, the record is held through the hysteresis band down to 0.95x and for one
    step below it, and the eighth proposal is the plain step: the session ends and every later step of the
    request is the base decode."""
    costed = _costed(adapter, monkeypatch)
    _inform_prior(costed, [0.35] * 5)
    assert _dflash_rounds(costed, [0] * 9) == [5] * 7 + [0] * 3
    assert len(_names(costed, "replay")) == 8 and len(_names(costed, "release_decoder")) == 1
    assert _names(costed, "decode") == [("decode", [850], [10]), ("decode", [851], [11])]
    assert not costed._spec_active and costed._dflash_stats is None
    prior = costed._spec_policy.prior
    assert prior.effective_cond(5) == pytest.approx([0.35 * 0.9**7] + [0.35] * 4)
    assert costed._spec_policy.declined == 3, "the decline and the two plain steps count toward the retry"
    # the instrumentation: every closed interval is one step of the mode the proposal decided
    calib = costed._spec_policy.calibration
    assert calib.modes[("step", (1, 5), 1024)].steps == 7 and calib.modes[("plain", 32)].steps == 2
    assert calib.reseeds[(1, 5)][0] == 1, "the bootstrap is the entry, measured once"
    assert costed._spec_policy.request_steps == 8.0, "the ended session's steps feed the entry horizon"
    assert calib.phases["replay"][0] == 7 and calib.phases["refresh"][0] == 8 and calib.phases["host"][0] == 10
    assert all(stats.wall_sum >= 0.0 for stats in calib.modes.values())


def _dflash_after_n_plain_steps_the_next_request_bootstraps_and_its_first_proposal_is_the_retry(
    adapter, monkeypatch, expect_error=None
):
    costed = _costed(adapter, monkeypatch)
    messages = []
    monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: messages.append(str(message)))
    monkeypatch.setattr(adapter, "SPEC_POLICY_LOG_INTERVAL_S", 0.0)
    _inform_prior(costed, [0.35] * 5)
    _dflash_rounds(costed, [0] * 9)
    costed.release_request(0)
    _prefill(costed, key=20)
    # three plain steps amortize the bootstrap replay plus one step at the fixture's 0.6 overhead:
    # (46.9 + 46.9 - 34) / (3 x 34) = 0.59
    assert _ordinary(costed, [3], [2], [20]).tolist() == [950], "the retry is due: the session starts"
    proposal = _propose(costed, [[950] + [-1] * 5], [[3] + [-1] * 5])
    assert proposal.num_valid.tolist() == [5] and costed._spec_policy.declined == 0 and costed._spec_policy.trial
    lines = [m for m in messages if m.startswith("Gemma4DFlash adaptive K:")]
    assert lines[0] == (
        "Gemma4DFlash adaptive K: 1 rows, cap (1, 5) -> record (1, 5) S_k=1024 (gain; was none); EWMA accepted "
        "prior, per-position 0.35/0.35/0.35/0.35/0.35, expected 1.54 tokens/row at 46.9 ms (1.38x plain 34.0, "
        "pool none), gain 1.11x"
    )
    assert lines[1].startswith(
        "Gemma4DFlash adaptive K: 1 rows, cap (1, 5) -> plain step (exit; was K=5); EWMA accepted 0."
    )
    assert "(retry; was K=0)" in lines[2] and "gain 0.91x" in lines[2]
    # another rejection: the trial observed, the policy declines at once and the request runs plain
    out = _verify(costed, [[950, 1001, 1002, 1003, 1004, 1005]], [list(range(3, 9))], [5], keys=[20])
    ids = out.argmax_ids[0].tolist()
    proposal = _propose(costed, [[ids[0]] + [-1] * 5], [[4] + [-1] * 5], counts=[1])
    assert proposal.num_valid.tolist() == [0] and not costed._spec_active


def _dflash_a_new_request_skips_the_bootstrap_when_the_prior_says_the_replay_would_not_pay(
    adapter, monkeypatch, expect_error=None
):
    # 1.43 expected tokens at 1.38x the plain step: 1.03x, below the entry margin but above the trial gain: the
    # bootstrap is the request's one trial, logged as such, and one rejection ends the session
    costed = _costed(adapter, monkeypatch)
    messages = []
    monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: messages.append(str(message)))
    monkeypatch.setattr(adapter, "SPEC_POLICY_LOG_INTERVAL_S", 0.0)
    _inform_prior(costed, [0.3] * 5)
    _prefill(costed, prompt_len=2, key=10)
    assert _ordinary(costed, [3], [2], [10], result="device") != "device"
    assert len(_names(costed, "bootstrap")) == 1
    proposal = _propose(costed, [[9] + [-1] * 5], [[3] + [-1] * 5])
    assert proposal.num_valid.tolist() == [5] and costed._spec_policy.trial
    lines = [m for m in messages if m.startswith("Gemma4DFlash adaptive K:")]
    assert "(trial; was none)" in lines[0] and "gain 1.03x" in lines[0]
    drafts = proposal.draft_token_ids[0, :5].tolist()
    out = _verify(costed, [[9] + drafts], [list(range(3, 9))], [5], keys=[10])
    ids = out.argmax_ids[0].tolist()
    proposal = _propose(costed, [[ids[0]] + [-1] * 5], [[4] + [-1] * 5], counts=[1])
    assert proposal.num_valid.tolist() == [0] and not costed._spec_active and costed._spec_policy.declined == 1
    # below the trial gain (1.11 expected tokens at 2.06x: 0.54x) the bootstrap is skipped: the taps are
    # dropped and the request runs plain from its first step
    costed = _with_costs(_costed(adapter, monkeypatch), step_ms=70.0)
    _inform_prior(costed, [0.1] * 5)
    _prefill(costed, prompt_len=2, key=10)
    assert costed._spec_pending is None and costed.model[0].tap_layers is None, "no taps: the prior declines"
    assert _ordinary(costed, [3], [2], [10], result="device") == "device"
    assert _names(costed, "bootstrap") == [] and _names(costed, "replay") == []
    assert costed._spec_pending is None
    proposal = _propose(costed, [[9] + [-1] * 5], [[3] + [-1] * 5])
    assert proposal.num_valid.tolist() == [0] and costed._spec_policy.declined == 1


def _dflash_the_cost_table_is_consumed_as_measured(adapter, monkeypatch, expect_error=None):
    """The same prior starts a session at the CI leg's replay cost (1.43x: the entry band), still starts one at a
    dearer one as the request's trial (0.96x: above the trial gain) and not at all below the trial gain."""
    for step_ms, cond, bootstraps in ((46.9, 0.5, True), (70.0, 0.5, True), (70.0, 0.1, False)):
        model = _with_costs(build_model(adapter, monkeypatch), step_ms=step_ms)
        _inform_prior(model, [cond] * 5)  # 1.97 expected tokens: 1.43x at 46.9 ms, 0.96x at 70 ms; 1.11: 0.54x
        _prefill(model, prompt_len=2, key=10)
        out = _ordinary(model, [3], [2], [10], result="device")
        assert (len(_names(model, "bootstrap")) == 1) is bootstraps, (step_ms, cond)
        assert (out == "device") is not bootstraps


def _dflash_adaptive_off_keeps_the_rail_as_before_whatever_the_table_says(adapter, monkeypatch, expect_error=None):
    model = _with_costs(build_model(adapter, monkeypatch, env={"GEMMA4_SPEC_ADAPTIVE": "0"}))
    _inform_prior(model, [0.0] * 5)
    assert _dflash_rounds(model, [0] * 7) == [5] * 8


def _dflash_the_session_summary_reports_the_accepted_drafts_ewma(adapter, monkeypatch, expect_error=None):
    costed = _costed(adapter, monkeypatch)
    messages = []
    monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: messages.append(str(message)))
    _dflash_rounds(costed, [5, 3])
    costed.release_request(0)
    assert [m for m in messages if "contract summary" in m] == [
        "Gemma4DFlash contract summary: 3 steps, 11 tokens, 3.67 tokens/step, 4.80 accepted drafts/verify (EWMA over 2)"
    ]


def _dflash_the_composed_vision_rail_shares_the_policy(adapter, monkeypatch, expect_error=None):
    vision = build_model(adapter, monkeypatch, cls=adapter.Gemma4DFlashContractVisionForConditionalGeneration)
    _with_costs(vision, step_ms=70.0)
    _inform_prior(vision, [0.1] * 5)  # 1.11 expected tokens at 2.06x: 0.54x, below the trial gain
    _prefill(vision, prompt_len=2, key=10)
    assert _ordinary(vision, [3], [2], [10], result="device") == "device" and _names(vision, "bootstrap") == []


def _dflash_the_traced_warmup_measures_every_captured_width_and_times_the_plain_step(
    adapter, monkeypatch, expect_error=None
):
    warm_model = _warm(adapter, monkeypatch)
    model = warm_model
    model.warmup_model_decode(enable_trace=False, kv_cache=model.kv_cache, num_blocks=64)
    assert model._spec_costs is None and _names(model, "time_plain") == []
    model.warmup_model_decode(enable_trace=True, kv_cache=model.kv_cache, num_blocks=64)
    ladder = sorted(model._spec_width_ladder)
    top = 4096 - 6  # the context minus one verify block: every cost is a line from position 0 to here
    assert model._spec_costs == {
        ("step", (1, 5), w): CostLine(0, 40.0 + i, top, 60.0 + i) for i, w in enumerate(ladder)
    }
    assert model._spec_plain_cost == {32: CostLine(0, 33.0, top, 33.0 + top / 100.0)}
    assert _names(_RecordingDecoder.instances[0], "measure") == [("measure", top, tuple(ladder))]
    assert _names(model, "time_plain") == [
        ("time_plain", [32, 64], 0),
        ("time_plain", [32, 64], top),
    ], "the decode warmup's zero table (batch x blocks) at position 0 and at the top"
    model.release_persistent_capture()
    assert model._spec_costs is None and model._spec_plain_cost is None
    # every warmed decode bucket gets its own plain line (the bucket of a step's live rows is its prior)
    model = _warm(adapter, monkeypatch)
    model.tt_supported_decode_batch_sizes = (1, 8, 32)
    model.warmup_model_decode(enable_trace=False, kv_cache=model.kv_cache, num_blocks=64)
    model.warmup_model_decode(enable_trace=True, kv_cache=model.kv_cache, num_blocks=64)
    assert [e[1] for e in _names(model, "time_plain")] == [[1, 64], [1, 64], [8, 64], [8, 64], [32, 64], [32, 64]]
    assert model._spec_plain_cost == {
        1: CostLine(0, 32.0, top, 32.0 + top / 100.0),
        8: CostLine(0, 25.0, top, 25.0 + top / 100.0),
        32: CostLine(0, 33.0, top, 33.0 + top / 100.0),
    }
    assert [model._spec_plain_bucket(n) for n in (1, 2, 8, 9, 32, 40)] == [1, 8, 8, 32, 32, 32]
    assert model._spec_plain_ms_for(2, 0) == (25.0, 8)


def _dflash_the_plain_step_cost_from_the_env_skips_the_timed_replay(adapter, monkeypatch, expect_error=None):
    model = _warm(adapter, monkeypatch, env={"GEMMA4_SPEC_PLAIN_STEP_MS": "35.5"})
    model.warmup_model_decode(enable_trace=False, kv_cache=model.kv_cache, num_blocks=64)
    model.warmup_model_decode(enable_trace=True, kv_cache=model.kv_cache, num_blocks=64)
    assert model._spec_plain_cost == {32: CostLine(0, 35.5, 4096 - 6, 35.5)} and _names(model, "time_plain") == []


def _dflash_adaptive_off_measures_nothing_at_warmup(adapter, monkeypatch, expect_error=None):
    model = _warm(adapter, monkeypatch, env={"GEMMA4_SPEC_ADAPTIVE": "0"})
    model.warmup_model_decode(enable_trace=False, kv_cache=model.kv_cache, num_blocks=64)
    model.warmup_model_decode(enable_trace=True, kv_cache=model.kv_cache, num_blocks=64)
    assert [e[0] for e in _RecordingDecoder.instances[0].events] == ["prepare", "capture"]
    assert model._spec_costs is None and model._spec_plain_cost is None


# -- the acceptance-adaptive policy on the MTP rail (through ``mh``): helpers the tests above call as blocks


# -- the acceptance-adaptive record choice (tt/spec_policy.py through the adapter) ---------------

# Scripted step costs per record; with the plain step at mh.PLAIN_MS = 20 they are 2.5x / 2.5x / 1.75x / 1.5x.
MTP_STEP_MS = {(1, 5): 50.0, (5, 5): 50.0, (8, 3): 35.0, (16, 1): 30.0}


def _mtp_adaptive(adapter, monkeypatch, env=None):
    """A model with the scripted cost table, a three-step retry at a 0.2 overhead and no probe."""
    # the pinned sequences read the scripted table: the host walls never calibrate a mode
    env = {
        "GEMMA4_SPEC_RETRY_STEPS": "3",
        "GEMMA4_SPEC_RETRY_OVERHEAD": "0.2",
        "GEMMA4_SPEC_PROBE_STEPS": "0",
        "GEMMA4_SPEC_CALIB_MIN": "1000",
        "GEMMA4_SPEC_TRIAL_STEPS": "1",  # the scenarios describe one-step trials and retries
        **(env or {}),
    }
    model = mh.build_mtp_model(adapter, monkeypatch, env=env)
    mh.install_decoder(model, step_ms=MTP_STEP_MS, plain_ms=mh.PLAIN_MS)
    return model


def _mtp_rounds(mtp, accepted_per_round):
    """Drive one request in the runner's order: per round the drafts its verify accepts (unused on a plain
    step). Returns the draft length every proposal offered, the first one from the prior; 0 is a plain step."""
    proposal = mh._start_solo(mtp)
    anchor, pos = 150, 3
    offered = [int(proposal.num_valid[0])]
    for accepted in accepted_per_round:
        k = offered[-1]
        if k:
            drafts = proposal.draft_token_ids[0, :k].tolist()
            out = mh._verify(mtp, [mh._block(anchor, drafts)], [mh._positions(pos, 1 + k)], [k], [10])
            ids = out.argmax_ids[0].tolist()
            n = min(int(accepted), k) + 1
            proposal = mh._propose(mtp, [ids[:n] + [-1] * (6 - n)], [mh._positions(pos + 1, n)], counts=[n])
            anchor, pos = ids[n - 1], pos + n
        else:
            out = mh._ordinary(mtp, [anchor], [pos], [10], result=_tensor([anchor + 1]))
            anchor, pos = int(out[0]), pos + 1
            proposal = mh._propose(mtp, [[anchor] + [-1] * 5], [[pos] + [-1] * 5])
        offered.append(int(proposal.num_valid[0]))
    return offered


def _mtp_the_first_proposal_speculates_from_the_optimistic_prior_and_high_acceptance_keeps_k5(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    adaptive = _mtp_adaptive(adapter, monkeypatch)
    assert _mtp_rounds(adaptive, [5, 5]) == [5, 5, 5]
    assert [s[1] for s in mh._steps(adaptive)] == [(1, 5)] * 3
    stats = adaptive._mtp_sessions[0].stats
    assert stats.cond == [1.0] * 5 and stats.ewma_accepted == 5.0 and stats.spec_steps == 2


def _mtp_a_new_request_under_a_pessimistic_prior_gets_one_trial_whose_verify_counts_at_once(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    """Prior 0.3 at every position, the records costed as a one-row step on the device ((1,5) at 34 ms against a
    20 ms plain step: 1.43 expected tokens = 0.84x, below the entry band, above the trial gain), so the first
    proposal replays the cheapest record (entry plus step: (1,5) at 170 + 34) as the trial. Five accepted drafts
    blend 0.25 x 1.0 + 0.75 x 0.37 = 0.53 into every position (2.07 tokens, 1.22x: the entry band) and the
    request speculates on; a rejection blends to 0.20 (1.29 tokens, 0.76x) and the request runs plain, no retry
    until the cadence allows."""
    reseeds = {(1, 5): 170.0, (5, 5): 200.0, (8, 3): 250.0, (16, 1): 300.0}
    steps = {(1, 5): 34.0, (5, 5): 40.0, (8, 3): 45.0, (16, 1): 50.0}
    outcomes = {}
    for name, rounds in (("accepted", [5, 5]), ("rejected", [0, 0, 0])):
        model = _mtp_adaptive(adapter, monkeypatch, env={"GEMMA4_SPEC_MIN_GAIN": "1.10"})  # the 1.22x entry band
        mh.install_decoder(model, step_ms=steps, plain_ms=mh.PLAIN_MS)
        model._spec_costs.update({("reseed", shape): CostLine.flat(ms) for shape, ms in reseeds.items()})
        messages = []
        monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: messages.append(str(message)))
        monkeypatch.setattr(adapter, "SPEC_POLICY_LOG_INTERVAL_S", 0.0)
        _inform_prior(model, [0.3] * 5)
        offered = _mtp_rounds(model, rounds)
        policy = model._spec_policy
        outcomes[name] = (
            offered,
            [m for m in messages if m.startswith("Gemma4MTP adaptive K:")],
            policy.last_reason,
            policy.trial,
            model._mtp_sessions[0].stats.effective_cond(5),
        )
        assert model._mtp_sessions[0].trialed
    offered, lines, reason, trial, cond = outcomes["accepted"]
    assert offered == [5, 5, 5] and "(trial; was none)" in lines[0] and "gain 0.84x" in lines[0]
    assert len(lines) == 1 and reason == "gain" and not trial, "the draft length did not change: no second line"
    assert cond == pytest.approx([0.4 * 1.0 + 0.6 * 0.433] * 5, abs=1e-3), "two own observations: 0.4 own, 0.6 prior"
    offered, lines, reason, trial, cond = outcomes["rejected"]
    assert offered == [5, 0, 0, 0] and "(trial; was none)" in lines[0] and reason == "declined" and not trial
    assert "-> plain step (declined; was K=5)" in lines[1] and "per-position 0.20/0.30/0.30/0.30/0.30" in lines[1]


def _mtp_acceptance_statistics_come_from_the_accepted_counts_of_the_verified_drafts(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    adaptive = _mtp_adaptive(adapter, monkeypatch)
    _mtp_rounds(adaptive, [2])  # the verify carried 5 drafts and committed 3 tokens: 2 accepted, the third rejected
    stats = adaptive._mtp_sessions[0].stats
    assert stats.cond == pytest.approx([1.0, 1.0, 0.9, 1.0, 1.0]) and stats.observed == [1, 1, 1, 0, 0]
    assert stats.ewma_accepted == 2.0
    assert adaptive._spec_policy.prior.cond == pytest.approx([1.0, 1.0, 0.9, 1.0, 1.0]), "the batch prior learns too"
    # one own observation at draft 3 (a rejection) counts 1 / (1 + 3) against the prior's 0.9: 0.675, and the
    # unobserved tail is capped by it
    assert stats.effective_cond(5) == pytest.approx([1.0, 1.0, 0.675, 0.675, 0.675])
    assert adaptive._mtp_sessions[0].retained.verified == 0, "the new proposal is not verified yet"


def _mtp_a_draftless_verify_and_an_ordinary_step_observe_nothing(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    adaptive = _mtp_adaptive(adapter, monkeypatch)
    mh._start_solo(adaptive)
    mh._verify(adaptive, [mh._block(150, [])], [mh._positions(3, 1)], [0], [10])  # answered from the retained column 0
    mh._propose(adaptive, [[150, 151, -1, -1, -1, -1]], [mh._positions(4, 2)], counts=[2])
    stats = adaptive._mtp_sessions[0].stats
    assert stats.spec_steps == 0 and stats.ewma_accepted is None and stats.cond == [1.0] * 5


def _mtp_falling_acceptance_drops_k5_to_k3_then_plain_and_retries_at_k1_after_n_plain_steps(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    """EWMA 0.1 on the first draft only: K=5 keeps its 2.5x record above 1.10x for five rejections, then (8,3)
    (1.75x) is the better gain for seven more, is held through the hysteresis band (gain between 0.95x and
    1.10x, then one step below) for four, and the plain step follows. With the fixture's 0.2 retry overhead,
    three plain steps amortize the cheapest record, (16, 1): its retry is a trial that observes one more
    rejection and leaves at once, and the plain steps resume."""
    adaptive = _mtp_adaptive(adapter, monkeypatch)
    assert _mtp_rounds(adaptive, [5] + [0] * 21) == [5] * 6 + [3] * 11 + [0] * 3 + [1] + [0] * 2
    assert [s[1] for s in mh._steps(adaptive)] == [(1, 5)] * 6 + [(8, 3)] * 11 + [(16, 1)]
    assert len(_names(adaptive, "decode")) == 1 + 4, "the first step and the four plain steps ran the base decode"
    assert _names(adaptive, "reseed")[-1][1:3] == ((16, 1), [0]), "the retry reseeds after the plain steps"
    assert adaptive._mtp_sessions[0].stats.cond[0] == pytest.approx(0.9**17)
    assert adaptive._spec_policy.declined == 2 and not adaptive._spec_policy.trial
    # the instrumentation: the closed intervals by mode, clean only without a switch, a reseed or a prefill
    calib = adaptive._spec_policy.calibration
    # an outlier (a host step over 3x the median) is a clean interval the window reports apart
    modes = {mode: (stats.steps, stats.clean + stats.outliers[0]) for mode, stats in calib.modes.items()}
    assert modes[("step", (1, 5), 1024)] == (6, 5), "the first interval starts with a reseed"
    assert modes[("step", (8, 3), 1024)] == (11, 10), "the first (8,3) interval starts at a switch"
    assert modes[("plain", 32)] == (4, 2), "the last decision's interval never closes; the switches are not clean"
    assert modes[("step", (16, 1), 1024)] == (1, 0), "the retry reseeds"
    assert calib.reseeds[(1, 5)][0] == 1 and calib.reseeds[(16, 1)][0] == 1
    assert {name: calls for name, (calls, _ms) in calib.phases.items()} == {
        "tables": 18,
        "refresh": 1,
        "reseed": 2,
        "replay": 18,
        "host": 23,
        # the runner's call boundaries (8b3f194aca3): a gap per proposal-to-step and step-to-proposal, by the step's kind
        "gap_pv_v": 18,
        "gap_vp_v": 18,
        "gap_pv_p": 4,
        "gap_vp_p": 5,
        "verify": 18,
    }
    assert calib.prefills == 1 and calib.window_prefills == 1
    assert calib.total[0] == 22 and calib.total[2] == 6 + 16 + 5, "every closed interval, with its tokens"
    assert (
        sum(entry[0] for entry in calib.dirty.values()) + sum(s.clean + s.outliers[0] for s in calib.modes.values())
        == 22
    )


def _mtp_a_batch_may_replay_a_smaller_record_than_its_rows_select(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    """Five rows select (5, 5) by rows; at 0.7 conditional acceptance (8, 3) pays better (2.53 tokens at 1.75x
    beats 2.94 at 2.5x), and its three padding rows carry the null block."""
    mtp = mh.build_mtp_model(adapter, monkeypatch)
    mh.install_decoder(mtp, step_ms=MTP_STEP_MS)
    mtp._spec_policy.prior.cond = [0.7] * 5
    mh._prefill(mtp, rows=5)
    keys = [10 + 2 * r for r in range(5)]
    mh._ordinary(mtp, [3] * 5, [2] * 5, keys, result=_tensor([30] * 5))
    proposal = mh._propose(mtp, [[30] + [-1] * 5] * 5, [[3] + [-1] * 5] * 5)
    assert proposal.num_valid.tolist() == [3] * 5
    step = mh._steps(mtp)[-1]
    assert step[1] == (8, 3) and step[2] == [0, 1, 2, 3, 4] and step[7][5:] == [[0, 0, 0, 0]] * 3
    # a row joining a speculating batch stays out of the replay (its token from the plain decode the verify
    # runs beside it) while its reseed does not pay over the entry horizon
    messages = []
    monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: messages.append(str(message)))
    monkeypatch.setattr(adapter, "SPEC_POLICY_LOG_INTERVAL_S", 0.0)
    mtp = mh.build_mtp_model(
        adapter, monkeypatch, env={"GEMMA4_SPEC_PROBE_STEPS": "0", "GEMMA4_SPEC_CALIB_MIN": "1000"}
    )
    mh.install_decoder(mtp, step_ms=MTP_STEP_MS)
    mh._prefill(mtp, rows=5)
    keys = [10 + 2 * r for r in range(5)]
    mh._ordinary(mtp, [3] * 5, [2] * 5, keys, result=_tensor([30] * 5))
    proposal = mh._propose(mtp, [[30] + [-1] * 5] * 5, [[3] + [-1] * 5] * 5)  # the first proposal: every row reseeds
    assert proposal.num_valid.tolist() == [5] * 5 and _names(mtp, "reseed")[-1][1:3] == ((5, 5), [0, 1, 2, 3, 4])
    drafts = proposal.draft_token_ids.tolist()
    blocks = [mh._block(30, drafts[r]) for r in range(5)]
    out = mh._verify(mtp, blocks, [mh._positions(3, 6)] * 5, [5] * 5, keys)
    ids = out.argmax_ids.tolist()
    mtp._spec_costs.update({("reseed", shape): 5000.0 for shape in mh.DEFAULT_SHAPES})
    # a sixth request joins: its first step is a plain decode beside the others' verify, then it proposes too
    mh._prefill(mtp, slot=5, key=20)
    committed = [ids[r] for r in range(5)] + [[40] + [-1] * 5]
    posits = [mh._positions(4, 6)] * 5 + [[3] + [-1] * 5]
    proposal = mh._propose(mtp, committed, posits, counts=[6] * 5 + [1])
    # 64 x (4 x 20 - 35) = 2880 ms of saving over the horizon does not cover a 5000 ms reseed of (8,3): the
    # five continuing rows replay (8,3) alone, the joiner commits its one token from the plain decode
    assert proposal.num_valid.tolist() == [3] * 5 + [0]
    step = _names(mtp, "step")[-1]
    assert step[1] == (8, 3) and step[2] == [0, 1, 2, 3, 4] and len(_names(mtp, "reseed")) == 1
    assert mtp._mtp_sessions[5].retained is None
    line = [m for m in messages if "for 5/6 rows" in m]
    assert len(line) == 1 and "-> record (8, 3) S_k=1024 for 5/6 rows (gain; was K=5)" in line[0]
    assert "at 55.0 ms (2.75x plain 20.0, pool " in line[0] and "), gain 1.27x" in line[0]


def _mtp_the_reseed_cost_of_entering_speculation_is_part_of_the_candidates_cost(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    """The reseed is the entry cost: the saving over the entry horizon must cover it before a row enters. The
    new request's trial is off here (GEMMA4_SPEC_TRIAL_GAIN above any gain): with it, a fresh request replays the
    cheapest record once whatever the horizon says, the reseed charged all the same."""
    mtp = mh.build_mtp_model(adapter, monkeypatch, env={"GEMMA4_SPEC_RETRY_STEPS": "0", "GEMMA4_SPEC_TRIAL_GAIN": "9"})
    mh.install_decoder(mtp, step_ms=MTP_STEP_MS)
    mtp._spec_costs.update({("reseed", shape): 2000.0 for shape in mh.DEFAULT_SHAPES})
    # At 0.8 every record clears 1.10x ((8,3): 2.95 tokens at 1.75x = 1.69x), but the saving over the 64-step
    # entry horizon, 64 x (2.95 x 20 - 35) = 1536 ms, does not cover a 2000 ms reseed: the row stays plain.
    mtp._spec_policy.prior.cond = [0.8] * 5
    mtp._spec_policy.prior.informed = [True] * 5
    mh._prefill(mtp)
    mh._ordinary(mtp, [3], [2], [10], result=_tensor([150]))
    proposal = mh._propose(mtp, [[150] + [-1] * 5], [[3] + [-1] * 5])
    assert proposal.num_valid.tolist() == [0] and mh._steps(mtp) == []
    assert mtp._mtp_sessions[0].trialed is False and mtp._spec_policy.last_reason == "declined"
    for shape in mh.DEFAULT_SHAPES:
        mtp._spec_costs[("reseed", shape)] = 0.0
    mh._ordinary(mtp, [150], [3], [10], result=_tensor([151]))
    proposal = mh._propose(mtp, [[151] + [-1] * 5], [[4] + [-1] * 5])
    assert proposal.num_valid.tolist() == [3] and _names(mtp, "reseed")[-1][1:4] == ((8, 3), [0], [151])


def _mtp_a_row_continuing_in_the_last_replay_pays_no_reseed(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    adaptive = _mtp_adaptive(adapter, monkeypatch)
    cls = type(adaptive)
    mh._start_solo(adaptive)
    session = adaptive._mtp_sessions[0]
    assert not cls._mtp_seed_continues(session, 0, adaptive._mtp_last_replay), "not verified yet"
    mh._verify(adaptive, [mh._block(150, [101, 102, 103, 104, 105])], [mh._positions(3, 6)], [5], [10])
    assert cls._mtp_seed_continues(session, 0, adaptive._mtp_last_replay)
    assert not cls._mtp_seed_continues(session, 1, adaptive._mtp_last_replay), "a moved slot reseeds"
    mh._ordinary(adaptive, [150], [3], [10], result=_tensor([151]))
    assert not cls._mtp_seed_continues(session, 0, adaptive._mtp_last_replay), "a plain step drops the proposal"


def _mtp_without_a_cost_table_the_rows_based_record_serves_and_a_warning_says_so_once(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    mtp = mh.build_mtp_model(adapter, monkeypatch)
    warned = []
    monkeypatch.setattr(adapter.logger, "warning", lambda message, *a, **k: warned.append(str(message)))
    mtp._spec_policy.prior.cond = [0.0] * 5  # with a table the policy would decline every record
    assert _mtp_rounds(mtp, [0, 0]) == [5, 5, 5]
    assert [s[1] for s in mh._steps(mtp)] == [(1, 5)] * 3
    assert warned == [
        "Gemma4MTP adaptive K: no cost table (the warmup measured none and GEMMA4_SPEC_PLAIN_STEP_MS is unset); "
        "speculation runs as without the policy"
    ]


def _mtp_adaptive_off_keeps_the_rows_based_record_whatever_the_table_says(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    mtp = mh.build_mtp_model(adapter, monkeypatch, env={"GEMMA4_SPEC_ADAPTIVE": "0"})
    mh.install_decoder(mtp, step_ms=MTP_STEP_MS)
    mtp._spec_policy.prior.cond = [0.0] * 5
    assert _mtp_rounds(mtp, [0, 0]) == [5, 5, 5]


def _mtp_a_bad_policy_knob_is_refused_when_the_adapter_is_built(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    with expect_error(ValueError, "GEMMA4_SPEC_MIN_GAIN must be at least 1.0, got 0.5"):
        mh.build_mtp_model(adapter, monkeypatch, env={"GEMMA4_SPEC_MIN_GAIN": "0.5"})


def _mtp_every_record_switch_is_logged_with_the_ewma_the_cost_ratio_and_the_gain(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    # the switch ladder of the batched records ((8,3) then (16,1) on the retry) at the 1.10 bar it documents
    adaptive = _mtp_adaptive(
        adapter, monkeypatch, env={"GEMMA4_SPEC_K_BY_ROWS": "5:5,8:3,16:1", "GEMMA4_SPEC_MIN_GAIN": "1.10"}
    )
    messages = []
    monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: messages.append(str(message)))
    monkeypatch.setattr(adapter, "SPEC_POLICY_LOG_INTERVAL_S", 0.0)
    _mtp_rounds(adaptive, [5] + [0] * 21)
    lines = [m for m in messages if m.startswith("Gemma4MTP adaptive K:")]
    assert lines[0] == (
        "Gemma4MTP adaptive K: 1 rows, cap (1, 5) -> record (1, 5) S_k=1024 (gain; was none); EWMA accepted prior, "
        "per-position 1.00/1.00/1.00/1.00/1.00, expected 6.00 tokens/row at 50.0 ms (2.50x plain 20.0, pool none), gain 2.40x"
    )
    assert [re.search(r"-> (record \(\d+, \d+\) S_k=\d+|plain step)", line).group(1) for line in lines] == [
        "record (1, 5) S_k=1024",
        "record (8, 3) S_k=1024",
        "plain step",
        "record (16, 1) S_k=1024",
        "plain step",
    ]
    assert (
        "(retry; was K=0)" in lines[3]
        and "at 30.0 ms (1.50x plain 20.0, pool " in lines[3]
        and "), gain 0.79x" in lines[3]
    )
    assert lines[2] == (
        "Gemma4MTP adaptive K: 1 rows, cap (1, 5) -> plain step (exit; was K=3); EWMA accepted 0.93, "
        "per-position 0.19/1.00/1.00/1.00/1.00, plain 20.0 ms"
    )
    assert [m for m in messages if "proposing nothing" in m] == [], "the policy's plain steps use their own line"
    # the stats line every GEMMA4_SPEC_STATS_STEPS proposals (window counters, then they start over)
    stats = []
    monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: stats.append(str(message)))
    windowed = _mtp_adaptive(adapter, monkeypatch, env={"GEMMA4_SPEC_STATS_STEPS": "4"})
    _mtp_rounds(windowed, [5, 5, 5, 5, 5, 5, 5])
    lines = [m for m in stats if m.startswith("Gemma4MTP adaptive stats:")]
    assert len(lines) == 2 and lines[0].startswith("Gemma4MTP adaptive stats: window 4 steps | (1,5)@1024: 3 steps")
    assert re.search(
        r"\(1,5\)@1024: 3 steps, 18 tokens, 2 clean, \d+\.\d ms/step, 6\.00 tokens/step, \d+\.\d ms/token, "
        r"total \d+\.\d ms/step \d+\.\d ms/token, ratio \d+\.\d\dx \| reseeds 1 at \d+\.\d ms \| "
        r"dirty 1 steps, \d+ ms: reseed 1/\d+ \| total 3 steps, "
        r"\d+ ms, 18 tokens, \d+\.\d ms/token \| phases ms/call: (?:[a-z_]+ \d+\.\d(?:, )?)+ \| prefills 1 \| "
        r"calibrated: none$",
        lines[0],
    )
    phases = lines[0].split("phases ms/call: ")[1].split(" | prefills")[0]
    # the first proposal follows the prefill's (plain) step; then the proposal's own phases and the runner's
    # boundaries around the verify step
    assert {name for name, _ms in (p.split(" ") for p in phases.split(", "))} == {
        "gap_vp_p",
        "tables",
        "refresh",
        "reseed",
        "replay",
        "host",
        "gap_pv_v",
        "verify",
        "gap_vp_v",
    }, phases
    assert lines[1].startswith("Gemma4MTP adaptive stats: window 4 steps | (1,5)@1024: 4 steps, 24 tokens, 4 clean")
    assert "| reseeds 0 | total 4 steps, " in lines[1]
    # the window opens on the previous verify's boundary gap, so its phases list does not start with "tables"
    assert "tables" in lines[1].split("| phases ms/call: ")[1].split(" | prefills")[0]
    assert "| prefills 0 | calibrated: none" in lines[1] and "dirty" not in lines[1]


def _mtp_record_switches_are_rate_limited_to_one_line_per_interval(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    adaptive = _mtp_adaptive(adapter, monkeypatch)
    messages = []
    monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: messages.append(str(message)))
    monkeypatch.setattr(adapter, "SPEC_POLICY_LOG_INTERVAL_S", 1.0)  # the module default; a block above zeroed it
    _mtp_rounds(adaptive, [5] + [0] * 21)  # four switches after the first decision, within the 1 s interval
    assert len([m for m in messages if m.startswith("Gemma4MTP adaptive K:")]) == 1
    assert adaptive._spec_policy_logged[0] == 5 and adaptive._spec_policy_logged[2] == 4


def _mtp_the_session_summary_reports_the_accepted_drafts_ewma(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    adaptive = _mtp_adaptive(adapter, monkeypatch)
    messages = []
    monkeypatch.setattr(adapter.logger, "info", lambda message, *a, **k: messages.append(str(message)))
    _mtp_rounds(adaptive, [5, 3])
    adaptive.release_request(0)
    assert [m for m in messages if "contract summary" in m] == [
        "Gemma4MTP contract summary: slot 0 (release) 3 steps, 11 tokens, 3.67 tokens/step, "
        "4.80 accepted drafts/verify (EWMA over 2)"
    ]
    # the engine-life summary on release
    adaptive.release_persistent_capture()
    summary = [m for m in messages if m.startswith("Gemma4MTP adaptive summary:")]
    assert len(summary) == 1 and summary[0].startswith(
        "Gemma4MTP adaptive summary: engine life 2 steps | (1,5)@1024: 2 steps, 10 tokens"
    )
    assert adaptive._spec_policy.request_steps == 3.0, "the released session's steps feed the entry horizon"
    assert "| reseeds 1 at " in summary[0] and "| prefills 1 | calibrated: none" in summary[0]


def _mtp_warmup_measures_the_records_and_times_the_plain_step_after_the_capture(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    mtp = mh.build_mtp_model(adapter, monkeypatch, warm=False, env={"GEMMA4_ASSISTANT_MODEL": assistant_dir})
    costs = {("step", (1, 5), 1024): (0, 50.0, 1018, 70.0), ("step", (1, 5), 2048): 52.0, ("reseed", (1, 5)): 18.0}
    mh.stub_drafter_modules(monkeypatch, mtp.events, costs=costs, plain_ms=21.5)
    mh._warmup(mtp, True)
    assert [event[0] for event in mtp.events] == [
        "base_warmup",
        "prepare",
        "capture",
        "refresh",
        "measure",
        "time_plain",
        "time_plain",
    ]
    top = 1024 - 6  # the context minus one verify block: every cost is a line from position 0 to here
    assert mtp._spec_costs == {key: CostLine.of(value) for key, value in costs.items()}
    assert mtp._spec_plain_cost == {32: CostLine(0, 21.5, top, 21.5)}
    assert _names(mtp, "measure") == [("measure", top, sorted(costs))]
    zero = [[0] * 8] * 32  # the decode warmup's zero table: 32 rows, 8 blocks (the null block)
    assert _names(mtp, "refresh") == [("refresh", [zero, zero])], "a bounded target's ring tables are installed first"
    assert _names(mtp, "time_plain") == [("time_plain", 32, 0, zero), ("time_plain", 32, top, zero)]
    assert mtp._mtp_step_tables is None, "the warmup's table is not a decode step the next proposal may read"
    assert len(mtp._mtp_installed_tables) == 2
    assert all(torch.equal(t, torch.tensor(zero, dtype=torch.int32)) for t in mtp._mtp_installed_tables)
    mtp.release_persistent_capture()
    assert mtp._spec_costs is None and mtp._spec_plain_cost is None
    # every warmed decode bucket gets its own plain line, timed at 0 and at the top
    mtp = mh.build_mtp_model(adapter, monkeypatch, warm=False, env={"GEMMA4_ASSISTANT_MODEL": assistant_dir})
    mtp.tt_supported_decode_batch_sizes = (1, 32)
    mh.stub_drafter_modules(monkeypatch, mtp.events, costs=costs, plain_ms=21.5)
    mh._warmup(mtp, True)
    assert [e[1:3] for e in _names(mtp, "time_plain")] == [(1, 0), (1, top), (32, 0), (32, top)]
    assert _names(mtp, "time_plain")[0][3] == [[0] * 8]
    assert mtp._spec_plain_cost == {1: CostLine(0, 20.5, top, 20.5), 32: CostLine(0, 21.5, top, 21.5)}


def _mtp_the_plain_step_cost_from_the_env_skips_the_timed_replay(
    adapter, monkeypatch, assistant_dir=None, expect_error=None
):
    for raw, line in (("29", CostLine(0, 29.0, 1018, 29.0)), ("29,80", CostLine(0, 29.0, 1018, 80.0))):
        env = {"GEMMA4_ASSISTANT_MODEL": assistant_dir, "GEMMA4_SPEC_PLAIN_STEP_MS": raw}
        mtp = mh.build_mtp_model(adapter, monkeypatch, warm=False, env=env)
        mh.stub_drafter_modules(monkeypatch, mtp.events, costs={("step", (1, 5), 1024): 50.0})
        mh._warmup(mtp, True)
        assert _names(mtp, "time_plain") == [] and mtp._spec_plain_cost == {32: line}, raw


def _mtp_adaptive_off_measures_nothing_at_warmup(adapter, monkeypatch, assistant_dir=None, expect_error=None):
    mtp = mh.build_mtp_model(
        adapter, monkeypatch, warm=False, env={"GEMMA4_ASSISTANT_MODEL": assistant_dir, "GEMMA4_SPEC_ADAPTIVE": "0"}
    )
    mh.stub_drafter_modules(monkeypatch, mtp.events)
    mh._warmup(mtp, True)
    assert [event[0] for event in mtp.events] == ["base_warmup", "prepare", "capture"]
    assert mtp._spec_costs is None and mtp._spec_plain_cost is None
