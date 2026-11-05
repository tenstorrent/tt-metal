# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The MTP speculative adapter (``tools/qwen38_vllm_mtp.py``) on the scripted MTP chain of the sampled-session tests (no device).

A fake plugin runner walks the contract the way ``TTModelRunner`` does (the candidate block, ``accept_greedy_drafts``,
``propose_draft_tokens`` after every commit) over ``Qwen38MTPForCausalLM`` and the real chat session on that fake chain.
Pinned: the committed stream is the greedy stream for any acceptance pattern, with fewer passes than tokens; the
plan and the capabilities; ``propose_draft_tokens`` declines when no pass is open or when the plugin committed less than
the device accepted; a plugin cut (stop, ``max_tokens``) and trimmed drafts resync through leave and enter and keep the
stream; the next prefill leaves an open pass loop; every refusal; the plain class is unchanged.
The tests need the plugin's ``spec_decode`` module (no vLLM): ``PYTHONPATH`` includes ``vllm-tt-plugin/src``; they skip without it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, VOCAB_SIZE
from models.demos.blackhole.qwen38_flash_next.tests import test_mtp_sampled_session_no_device as fakes
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm_mtp as mtp_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import Qwen38ChatChainError
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import Qwen38ForCausalLM, greedy_batch_sampling_params
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm_mtp import (
    Qwen38MTPForCausalLM,
    mtp_drafts_from_env,
    mtp_ple_early_from_env,
)

_PLUGIN_SRC = Path("/home/ttuser/atupe/qwen38-flash-next/vllm-tt-plugin/src")
if _PLUGIN_SRC.is_dir() and str(_PLUGIN_SRC) not in sys.path:
    sys.path.append(str(_PLUGIN_SRC))
spec_decode = pytest.importorskip("vllm_tt_plugin.spec_decode")

K = fakes.K
PLACEHOLDER = spec_decode.PLACEHOLDER_TOKEN_ID
ALLOCATED = 32_768


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("QWEN38_HOST_FAST", "0")
    monkeypatch.setenv("QWEN38_STEP_SPLIT", "0")
    monkeypatch.delenv("QWEN38_VLLM_MTP_K", raising=False)
    monkeypatch.delenv("QWEN38_VLLM_MTP_PLE_EARLY", raising=False)
    monkeypatch.delenv("QWEN38_MTP_HYBRID_SINGLE", raising=False)


class BootstrapChain(fakes.FakeMTPChain):
    """The scripted chain with ``mtp_enter(bootstrap_tokens=...)`` of the served chain (the entry pass on given rows)."""

    def mtp_enter(self, first_token, ple_context, *, decide=None, bootstrap_tokens=None):
        if bootstrap_tokens is None:
            return super().mtp_enter(first_token, ple_context, decide=decide)
        assert len(bootstrap_tokens) == K + 1 and bootstrap_tokens[0] == first_token
        self.mtp.chain = self
        self.log.append("enter:given")
        return self.mtp.record(self._pass(list(bootstrap_tokens)))


def _model(**chain_fields) -> tuple[Qwen38MTPForCausalLM, fakes.FakeMTPChain]:
    chain = BootstrapChain(**chain_fields)
    chain.closed = False
    chain.misses_forbidden = True
    return Qwen38MTPForCausalLM(chain, allocated_context=ALLOCATED), chain


def _prompt(length: int, base: int = 100) -> list[int]:
    return [base + 7 * index for index in range(length)]


def _prefill(model, prompt, **extra):
    return model.prefill_forward(
        tokens=torch.tensor([prompt], dtype=torch.int32),
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[len(prompt)],
        start_pos=[0],
        empty_slots=[0],
        **extra,
    )


def _verify(model, tokens, positions, num_valid, accepted, **extra):
    return model.decode_forward(
        tokens=torch.tensor([tokens], dtype=torch.int32),
        start_pos=torch.tensor([positions], dtype=torch.int32),
        spec_mode="argmax_ids",
        num_valid_drafts=torch.tensor([num_valid], dtype=torch.int32),
        accepted_counts=torch.tensor([accepted], dtype=torch.int32),
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        **extra,
    )


def _propose(model, committed: list[int], accepted: int):
    block = committed + [PLACEHOLDER] * (K + 1 - len(committed))
    return model.propose_draft_tokens(
        K,
        torch.tensor([block], dtype=torch.int32),
        torch.zeros(1, K + 1, dtype=torch.int32),
        torch.tensor([accepted], dtype=torch.int32),
        hidden=None,
    )


class Runner:
    """The plugin runner's step loop for one request (batch 1): the candidate block from the committed token and the proposed
    drafts, one verify, the greedy walk, the commit (cut at ``max_new``, or at ``cut_at`` tokens of one step when given), then
    the proposal over the committed block."""

    def __init__(self, model, prompt: list[int], max_new: int):
        self.model, self.prompt, self.max_new = model, prompt, max_new
        out, _ = _prefill(model, prompt)
        self.committed = [int(out[0, 0].argmax())]
        self.drafts: list[int] = []
        self.accepted = 1
        self.steps = 0
        self.verify_blocks: list[tuple[list[int], int]] = []

    def step(self, *, cut: int | None = None, trim: int | None = None) -> None:
        drafts = self.drafts if trim is None else self.drafts[:trim]
        valid = len(drafts)
        position = len(self.prompt) + len(self.committed) - 1
        tokens = [self.committed[-1], *drafts] + [PLACEHOLDER] * (K - valid)
        positions = [position + column if column <= valid else -1 for column in range(K + 1)]
        out = _verify(self.model, tokens, positions, valid, self.accepted)
        assert out.spec_mode == "argmax_ids" and out.hidden is None and out.argmax_ids.shape == (1, K + 1)
        walked, counts = spec_decode.accept_greedy_drafts(
            out.argmax_ids,
            torch.tensor([tokens[1:]], dtype=torch.int32),
            torch.tensor([valid], dtype=torch.int32),
        )
        count = int(counts[0])
        new = walked[0, :count].tolist()
        if cut is not None:
            new = new[:cut]
        new = new[: self.max_new - len(self.committed)]
        self.verify_blocks.append((tokens, valid))
        self.committed.extend(new)
        self.accepted = len(new)
        self.steps += 1
        block = new + [PLACEHOLDER] * (K + 1 - len(new))
        offered = self.model.propose_draft_tokens(
            K,
            torch.tensor([block], dtype=torch.int32),
            torch.zeros(1, K + 1, dtype=torch.int32),
            torch.tensor([self.accepted], dtype=torch.int32),
            hidden=None,
        )
        assert offered.draft_token_ids.dtype == torch.int32 and offered.draft_token_ids.shape == (1, K)
        assert offered.num_valid.dtype == torch.int32 and offered.num_valid.shape == (1,)
        self.drafts = offered.draft_token_ids[0, : int(offered.num_valid[0])].tolist()

    def run(self) -> list[int]:
        while len(self.committed) < self.max_new:
            self.step()
        return self.committed


# -- the stream --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("length", [1, 5, 12])
def test_stream_is_the_greedy_stream_with_fewer_passes_than_tokens(length):
    model, chain = _model()
    prompt = _prompt(length)
    runner = Runner(model, prompt, 60)
    tokens = runner.run()
    assert tokens == fakes._greedy_stream(prompt, 60)
    driver = model.driver
    assert driver.resyncs == 0
    assert driver.passes == runner.steps
    assert runner.steps < 60 - 1  # the drafter is right two positions of three: several tokens per pass
    assert driver.accepted_drafts > 0
    assert chain.log.count("enter") == 1
    assert runner.verify_blocks[0][1] == 0 and all(valid == K for _, valid in runner.verify_blocks[1:])


def test_first_verify_is_the_bootstrap_pass_and_proposes_the_chain_drafts():
    model, chain = _model()
    prompt = _prompt(6)
    out, rope = _prefill(model, prompt)
    first = int(out[0, 0].argmax())
    assert out.shape == (1, 1, VOCAB_SIZE) and out.dtype == torch.float32 and rope.shape == (1,)
    assert first == fakes._greedy_stream(prompt, 1)[0]
    assert chain.log[-1] == "read_token_row"
    # nothing to propose before a pass is open
    declined = _propose(model, [first], 1)
    assert declined.num_valid.tolist() == [0]
    position = len(prompt)
    verified = _verify(model, [first, *[PLACEHOLDER] * K], [position] + [-1] * K, 0, 1)
    assert chain.log.count("enter") == 1 and chain.log[-1].startswith("pass:fused:")
    assert verified.argmax_ids[0, 0] == fakes._greedy_stream(prompt, 2)[1]
    offered = _propose(model, [int(verified.argmax_ids[0, 0])], 1)
    assert offered.num_valid.tolist() == [K]
    assert offered.draft_token_ids[0].tolist() == list(chain.next_tokens[1:])


def test_prefill_returns_the_device_greedy_token_for_a_greedy_contract():
    model, _ = _model()
    prompt = _prompt(4)
    out, _ = _prefill(model, prompt, sampling_params=greedy_batch_sampling_params(1))
    assert out.dtype == torch.int32 and out.shape == (1, 1)
    assert int(out[0, 0]) == fakes._greedy_stream(prompt, 1)[0]


def test_one_hot_logits_row_masks_the_padding_rows():
    model, _ = _model()
    out, _ = _prefill(model, _prompt(3))
    row = out[0, 0]
    assert int((row == 1.0).sum()) == 1 and bool(torch.isinf(row[TOKENIZER_SIZE:]).all())
    assert int((row == 0.0).sum()) == TOKENIZER_SIZE - 1


# -- the plugin cuts a pass ---------------------------------------------------------------------------------------------


def test_a_plugin_cut_resyncs_through_leave_and_enter_and_keeps_the_stream():
    model, chain = _model()
    prompt = _prompt(5)
    runner = Runner(model, prompt, 40)
    runner.step()  # the bootstrap pass
    runner.step()
    # find a pass that accepts at least 2 so a cut to 1 token is a real cut
    while True:
        runner.step()
        if model.driver.run is not None and model.driver.run.accepted >= 2:
            break
    device_rows = model.driver.run.accepted + 1
    log_before = len(chain.log)
    runner.committed  # the stream so far is the reference's prefix
    # the plugin committed only the first token of the pass (a stop string found later in the step, say)
    runner.accepted = 1
    runner.committed = runner.committed[: len(runner.committed) - (device_rows - 1)]
    assert runner.committed == fakes._greedy_stream(prompt, len(runner.committed))
    runner.drafts = []  # the proposal after the cut declined (committed != device accepted)
    offered = _propose(model, runner.committed[-1:], 1)
    assert offered.num_valid.tolist() == [0]
    runner.run()
    assert runner.committed == fakes._greedy_stream(prompt, 40)
    assert model.driver.resyncs == 1
    tail = chain.log[log_before:]
    assert f"leave:{1}" in tail and "enter" in tail and tail.index("leave:1") < tail.index("enter")
    assert len(model.session.committed) <= len(prompt) + 40


def test_trimmed_drafts_resync_and_keep_the_stream():
    model, chain = _model()
    prompt = _prompt(7)
    runner = Runner(model, prompt, 30)
    runner.step()
    runner.step()
    runner.step(trim=2)  # the scheduler kept two of the four drafts (near max_model_len)
    assert model.driver.resyncs == 1 and "enter:given" in chain.log
    assert runner.committed == fakes._greedy_stream(prompt, len(runner.committed))
    runner.run()
    assert runner.committed == fakes._greedy_stream(prompt, 30)


def test_a_new_prefill_leaves_the_open_pass_loop_before_it_resets():
    model, chain = _model()
    first = _prompt(5, 100)
    runner = Runner(model, first, 25)
    runner.step()
    runner.step()
    assert model.driver.run is not None
    model.release_request(0)
    assert model.driver.run is not None  # release is lazy
    second = _prompt(9, 300)
    mark = len(chain.log)
    assert Runner(model, second, 30).run() == fakes._greedy_stream(second, 30)
    tail = chain.log[mark:]
    leave = next(index for index, entry in enumerate(tail) if entry.startswith("leave:"))
    assert tail.index("reset") > leave


def test_propose_declines_when_the_committed_block_is_not_the_device_pass():
    model, chain = _model()
    prompt = _prompt(5)
    out, _ = _prefill(model, prompt)
    first = int(out[0, 0].argmax())
    verified = _verify(model, [first, *[PLACEHOLDER] * K], [len(prompt)] + [-1] * K, 0, 1)
    good = int(verified.argmax_ids[0, 0])
    assert _propose(model, [good], 1).num_valid.tolist() == [K]
    assert _propose(model, [good + 1], 1).num_valid.tolist() == [0]  # a token the device did not choose
    assert _propose(model, [good, good], 2).num_valid.tolist() == [0]  # the count the device did not accept


# -- refusals ------------------------------------------------------------------------------------------------------------


def test_decode_refusals_name_the_cause_and_touch_no_chain_call(expect_error):
    model, chain = _model()
    prompt = _prompt(5)
    out, _ = _prefill(model, prompt)
    first = int(out[0, 0].argmax())
    block = [first, *[PLACEHOLDER] * K]
    positions = [len(prompt)] + [-1] * K
    chain.log.clear()
    with expect_error(ValueError, "no narrow decode"):
        model.decode_forward(tokens=torch.tensor([[first]], dtype=torch.int32), start_pos=[len(prompt)])
    with expect_error(ValueError, "argmax_ids"):
        model.decode_forward(
            tokens=torch.tensor([block], dtype=torch.int32),
            start_pos=torch.tensor([positions], dtype=torch.int32),
            spec_mode="logits",
            num_valid_drafts=torch.zeros(1, dtype=torch.int32),
            accepted_counts=torch.ones(1, dtype=torch.int32),
        )
    with expect_error(ValueError, "block is"):
        _verify(model, block[:-1], positions[:-1], 0, 1)
    with expect_error(ValueError, "accepted_counts"):
        _verify(model, block, positions, 0, 0)
    with expect_error(ValueError, "num_valid_drafts"):
        _verify(model, block, positions, K + 1, 1)
    with expect_error(ValueError, "slot_remap"):
        _verify(model, block, positions, 0, 1, slot_remap=torch.tensor([1, 0], dtype=torch.int32))
    with expect_error(ValueError, "padding row"):
        _verify(model, block, [-1] * (K + 1), 0, 1)
    assert chain.log == []
    assert not model.slot.failed


def test_a_verify_without_a_pass_loop_must_follow_a_prefill(expect_error):
    model, _ = _model()
    prompt = _prompt(5)
    out, _ = _prefill(model, prompt)
    first = int(out[0, 0].argmax())
    with expect_error(Qwen38ChatChainError, "follow a prefill"):
        _verify(model, [first, *[PLACEHOLDER] * K], [len(prompt)] + [-1] * K, 0, 3)
    assert model.slot.failed  # a state error after device calls poisons the slot; the next prefill recovers
    out, _ = _prefill(model, prompt)
    assert not model.slot.failed


def test_position_that_differs_from_the_committed_sequence_is_refused(expect_error):
    model, _ = _model()
    prompt = _prompt(5)
    out, _ = _prefill(model, prompt)
    first = int(out[0, 0].argmax())
    with expect_error(Qwen38ChatChainError, "position"):
        _verify(model, [first, *[PLACEHOLDER] * K], [len(prompt) + 3] + [-1] * K, 0, 1)


def test_no_room_for_a_pass_is_refused_by_name(expect_error):
    model, _ = _model(room=7)
    prompt = _prompt(5)
    out, _ = _prefill(model, prompt)
    first = int(out[0, 0].argmax())
    with expect_error(Qwen38ChatChainError, "no room"):
        _verify(model, [first, *[PLACEHOLDER] * K], [len(prompt)] + [-1] * K, 0, 1)


def test_prefill_refusals(expect_error):
    model, _ = _model()
    tokens = torch.tensor([_prompt(4)], dtype=torch.int32)
    common = {"page_table": None, "kv_cache": None, "enable_trace": True, "prompt_lens": [4]}
    with expect_error(ValueError, "position 0"):
        model.prefill_forward(tokens=tokens, start_pos=[4], empty_slots=[0], **common)
    with expect_error(ValueError, "empty_slots"):
        model.prefill_forward(tokens=tokens, start_pos=[0], empty_slots=[1], **common)
    with expect_error(ValueError, "tokens must be"):
        model.prefill_forward(tokens=tokens.repeat(2, 1), start_pos=[0, 0], empty_slots=[0], **common)
    bad = tokens.clone()
    bad[0, 1] = TOKENIZER_SIZE
    with expect_error(ValueError, "prompt ids"):
        model.prefill_forward(tokens=bad, start_pos=[0], empty_slots=[0], **common)
    with expect_error(ValueError, "slot must be 0"):
        model.release_request(1)


def test_a_padding_row_token_in_the_pass_is_an_error(expect_error):
    model, chain = _model()
    prompt = _prompt(5)
    out, _ = _prefill(model, prompt)
    first = int(out[0, 0].argmax())
    original = chain.mtp_enter

    def bad_enter(*args, **kwargs):
        record = original(*args, **kwargs)
        return record.__class__(**{**record.__dict__, "argmaxes": (TOKENIZER_SIZE, *record.argmaxes[1:])})

    chain.mtp_enter = bad_enter
    with expect_error(Qwen38ChatChainError, "tokenizer"):
        _verify(model, [first, *[PLACEHOLDER] * K], [len(prompt)] + [-1] * K, 0, 1)


# -- plan, capabilities, class selection -----------------------------------------------------------------------------


def _config(max_model_len=16_384, mtp_layers=1):
    text = SimpleNamespace(mtp_num_hidden_layers=mtp_layers)
    hf = SimpleNamespace(get_text_config=lambda: text)
    return SimpleNamespace(model_config=SimpleNamespace(max_model_len=max_model_len, hf_config=hf))


def test_spec_plan_for_batch_one_and_k4():
    plan = Qwen38MTPForCausalLM.spec_plan(_config(), 1, 4)
    assert isinstance(plan, spec_decode.SpecPlan)
    assert (plan.effective_k, plan.accept_modes, plan.drafter_state, plan.supports_narrow_decode) == (
        4,
        ("argmax_ids",),
        "internal",
        False,
    )
    assert plan.lanes_per_request == 5 and plan.block_width == 5
    assert Qwen38MTPForCausalLM.spec_plan(_config(), 1, 7).effective_k == 4  # never above the chain's K


@pytest.mark.parametrize(
    "seqs, requested, config, text",
    [
        (9, 4, _config(), "max_num_seqs 1 or 2..8"),
        (8, 4, _config(), "exceed the 32-row tile"),
        (1, 3, _config(), "below the chain's draft count"),
        (1, 4, _config(max_model_len=10**9), "exceeds"),
        (1, 4, _config(mtp_layers=0), "no MTP layer"),
    ],
)
def test_spec_plan_rejects_without_raising(seqs, requested, config, text):
    plan = Qwen38MTPForCausalLM.spec_plan(config, seqs, requested)
    assert isinstance(plan, spec_decode.SpecReject) and text in plan.reason


def test_spec_plan_follows_the_draft_count_variable(monkeypatch):
    monkeypatch.setenv("QWEN38_VLLM_MTP_K", "3")
    assert Qwen38MTPForCausalLM.spec_plan(_config(), 1, 4).effective_k == 3
    assert isinstance(Qwen38MTPForCausalLM.spec_plan(_config(), 1, 2), spec_decode.SpecReject)
    monkeypatch.setenv("QWEN38_VLLM_MTP_K", "7")
    refused = Qwen38MTPForCausalLM.spec_plan(_config(), 1, 7)
    assert isinstance(refused, spec_decode.SpecReject) and "QWEN38_VLLM_MTP_K" in refused.reason


def test_capabilities_declare_the_contract_and_the_plain_class_does_not(monkeypatch):
    monkeypatch.setenv("QWEN38_RING", "0")
    caps = Qwen38MTPForCausalLM.model_capabilities
    assert caps["supports_spec_decode"] is True
    assert caps["spec_requirements"] == ["device_propose"] and caps["spec_hidden_handoff"] == []
    assert caps["supports_async_decode"] is False and caps["supports_async_spec_decode"] is False
    assert caps["supports_sample_on_device"] is False and caps["output_tokens_per_step"] == 1
    assert caps["supports_chunked_prefill"] is False and caps["supports_prefix_caching"] is False
    plain = Qwen38ForCausalLM.model_capabilities
    assert "supports_spec_decode" not in plain and "spec_requirements" not in plain
    assert plain["supports_sample_on_device"] is True


def test_plain_chain_options_are_unchanged_and_the_mtp_class_adds_its_own(monkeypatch):
    assert Qwen38ForCausalLM._chain_open_options(object()) == {}
    assert Qwen38MTPForCausalLM._chain_open_options(object()) == {"sampling": False, "mtp": 4, "mtp_ple_early": False}
    monkeypatch.setenv("QWEN38_VLLM_MTP_K", "5")
    monkeypatch.setenv("QWEN38_VLLM_MTP_PLE_EARLY", "1")
    assert Qwen38MTPForCausalLM._chain_open_options(object()) == {"sampling": False, "mtp": 5, "mtp_ple_early": True}


def test_env_parsing_refuses_by_name(expect_error):
    assert mtp_drafts_from_env({}) == 4 and mtp_drafts_from_env({"QWEN38_VLLM_MTP_K": "3"}) == 3
    for bad in ("2", "6", "x", ""):
        with expect_error(ValueError, "QWEN38_VLLM_MTP_K"):
            mtp_drafts_from_env({"QWEN38_VLLM_MTP_K": bad})
    assert mtp_ple_early_from_env({}) is False and mtp_ple_early_from_env({"QWEN38_VLLM_MTP_PLE_EARLY": "1"}) is True
    with expect_error(ValueError, "PLE_EARLY"):
        mtp_ple_early_from_env({"QWEN38_VLLM_MTP_PLE_EARLY": "2"})


def test_initialize_refuses_a_batch_before_any_library_call(expect_error):
    with expect_error(ValueError, "max_num_seqs 1 or 2..8"):
        Qwen38MTPForCausalLM.initialize_vllm_model(SimpleNamespace(), None, 9, 4096)
    with expect_error(ValueError, "32-row tile"):  # B x (K + 1) <= 32: K 4 takes at most 6 lanes
        Qwen38MTPForCausalLM.initialize_vllm_model(SimpleNamespace(), None, 7, 4096)


def test_a_chain_without_mtp_is_refused(expect_error):
    chain = SimpleNamespace(mtp=None)
    with expect_error(ValueError, "mtp=K"):
        Qwen38MTPForCausalLM(chain, allocated_context=ALLOCATED)


def test_class_is_reachable_by_the_override_path():
    path = f"{mtp_module.__name__}:{Qwen38MTPForCausalLM.__name__}"
    module_name, _, class_name = path.partition(":")
    assert getattr(sys.modules[module_name], class_name) is Qwen38MTPForCausalLM
    assert (
        qwen38_vllm.Qwen38ForCausalLM.initialize_vllm_model.__func__
        is not Qwen38MTPForCausalLM.initialize_vllm_model.__func__
    )


# -- warmup and shutdown ----------------------------------------------------------------------------------------------


def test_warmup_runs_the_bootstrap_and_a_following_pass_then_resets(monkeypatch, expect_error):
    monkeypatch.setattr(mtp_module, "WARMUP_PROMPT_TOKEN_IDS", tuple(_prompt(10)))
    model, chain = _model()
    model.warmup_model_decode(
        enable_trace=False, kv_cache=None, max_batch_size=1, num_blocks=4, can_sample_on_device=False
    )
    assert chain.log == []
    model.warmup_model_decode(
        enable_trace=True, kv_cache=None, max_batch_size=1, num_blocks=4, can_sample_on_device=False
    )
    assert chain.log.count("enter") == 1 and sum(entry.startswith("pass:") for entry in chain.log) == 2
    assert model.driver.run is None and chain.log[-1] == "reset"
    with expect_error(ValueError, "max_batch_size must be 1"):
        model.warmup_model_decode(
            enable_trace=True, kv_cache=None, max_batch_size=2, num_blocks=4, can_sample_on_device=False
        )
    # the warmed model serves a request from the reset
    prompt = _prompt(6, 500)
    assert Runner(model, prompt, 20).run() == fakes._greedy_stream(prompt, 20)


def test_shutdown_leaves_the_pass_loop_then_closes_the_chain():
    model, chain = _model()
    chain.close = lambda: chain.log.append("close")
    Runner(model, _prompt(5), 12).step()
    model.release_persistent_capture()
    assert chain.log[-2].startswith("leave:") and chain.log[-1] == "close"


# -- batch 2..8: the lane chain under the plugin's calls ---------------------------------------------------------------

from models.demos.blackhole.qwen38_flash_next.tests import (  # noqa: E402
    test_qwen38_lane_scheduler_no_device as lane_fakes,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lane_scheduler import Qwen38LaneAdmitted  # noqa: E402
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm_mtp import (  # noqa: E402
    Qwen38VLLMMTPLanesDriver,
    VerifyRow,
)

LK = 3  # the lanes' draft count: B x (K + 1) <= 32 with B = 8
LROWS = LK + 1


class LaneRecord:
    def __init__(self, positions, tokens, accepted, committed, argmaxes, next_tokens):
        self.positions = tuple(positions)
        self.tokens = tuple(tuple(block) for block in tokens)
        self.accepted = tuple(accepted)
        self.committed = tuple(tuple(block) for block in committed)
        self.argmaxes = tuple(tuple(row) for row in argmaxes)
        self.next_tokens = tuple(tuple(block) for block in next_tokens)


class FakeLanes(lane_fakes.FakeLanesDevice):
    """The scheduler tests' lane device (commit at the start of the next pass, landed counts, parked lanes hold) with K = 3
    and the fields of a real pass record (``tokens``, ``next_tokens``)."""

    def __init__(self, lanes: int):
        super().__init__(lanes)
        self.rows_log: list[list[int]] = []

    def admit(self, lane, ticket, between=None):
        self.calls.append(("admit", lane))
        self.sequence[lane] = list(ticket.prompt_ids)
        self.last_rows[lane] = None
        self.positions[lane] = len(ticket.prompt_ids)
        return Qwen38LaneAdmitted(
            pending=lane_fakes.next_token(self.sequence[lane]),
            position=len(ticket.prompt_ids),
            ple_context=(1, 2),
            seconds={"prefill": 0.0},
        )

    def set_blocks(self, blocks):
        self.calls.append(("set_blocks", [None if b is None else list(b) for b in blocks]))
        for lane, block in enumerate(blocks):
            if block is not None:
                assert len(block) == LROWS, block
                self.blocks[lane] = list(block)

    def step(self):
        self.passes += 1
        for lane in range(self.lanes):
            if self.commit_mask[lane] and self.last_rows[lane] is not None and self.counts[lane] >= 0:
                self.sequence[lane].extend(self.last_rows[lane][: self.counts[lane] + 1])
        self.commit_mask = list(self.active)
        positions = list(self.positions)
        tokens, committed, accepted, argmaxes, next_blocks = [], [], [], [], []
        for lane in range(self.lanes):
            block = self.blocks[lane] if self.blocks[lane] is not None else [0] * LROWS
            tokens.append(list(block))
            rows = [lane_fakes.next_token(self.sequence[lane] + block[: j + 1]) for j in range(LROWS)]
            a = 0
            while a < LK and block[a + 1] == rows[a]:
                a += 1
            argmaxes.append(rows)
            if self.active[lane]:
                accepted.append(a)
                committed.append(rows[: a + 1])
                self.last_rows[lane] = list(block)
                self.counts[lane] = a
                self.positions[lane] += a + 1
                consumed = self.sequence[lane] + block[: a + 1]
                next_block = [rows[a]]
                for index in range(LK):
                    next_block.append(lane_fakes.draft_token(consumed + next_block, index))
                self.blocks[lane] = next_block
            else:
                accepted.append(0)
                committed.append(())
            next_blocks.append(list(self.blocks[lane]) if self.blocks[lane] is not None else [0] * LROWS)
        return LaneRecord(positions, tokens, accepted, committed, argmaxes, next_blocks)

    def room(self, position):
        return (position & ~31) + 64 <= 8192


def _lanes_model(monkeypatch, lanes: int, hybrid: bool = False):
    monkeypatch.setenv("QWEN38_VLLM_MTP_K", str(LK))
    monkeypatch.setenv("QWEN38_MTP_HYBRID_SINGLE", "1" if hybrid else "0")  # the lane tests run the lane path
    device = FakeLanes(lanes)
    model = Qwen38MTPForCausalLM(None, allocated_context=8192, batch_lanes=lanes, chain_opener=lambda hook: None)
    model.lanes_session = device
    model.lane_driver = Qwen38VLLMMTPLanesDriver(device, lanes, LK)
    model.chain = SimpleNamespace(closed=False, misses_forbidden=True)
    return model, device


def _lane_stream(prompt: list[int], count: int) -> list[int]:
    sequence, out = list(prompt), []
    for _ in range(count):
        out.append(lane_fakes.next_token(sequence))
        sequence.append(out[-1])
    return out


class LaneRequest:
    """One request as the plugin keeps it: its state slot, the committed tokens, the proposed drafts, the tokens of its last
    commit.  ``block`` is its row of a verify call."""

    def __init__(self, slot: int, prompt: list[int], max_new: int):
        self.slot, self.prompt, self.max_new = slot, prompt, max_new
        self.committed: list[int] = []
        self.drafts: list[int] = []
        self.accepted = 1
        self.done = False
        self.passes = 0
        self.resyncs_expected = 0

    @property
    def position(self) -> int:
        return len(self.prompt) + len(self.committed) - 1


class LaneRunner:
    """The plugin runner's loop over several requests: prefill (one call per join), one verify per step over every live
    request, the per-request greedy walk and commit (cut at ``max_new``, or at ``cut_at`` tokens, or with trimmed drafts), the
    proposal over the rows, the release of a finished request."""

    def __init__(self, model, lanes: int):
        self.model, self.lanes = model, lanes
        self.requests: list[LaneRequest] = []
        self.steps = 0

    def join(self, slot: int, prompt: list[int], max_new: int) -> LaneRequest:
        request = LaneRequest(slot, prompt, max_new)
        out, rope = self.model.prefill_forward(
            tokens=torch.tensor([prompt], dtype=torch.int32),
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[len(prompt)],
            start_pos=[0],
            empty_slots=[slot],
        )
        assert out.shape == (1, 1, VOCAB_SIZE) and rope.shape == (1,)
        request.committed = [int(out[0, 0].argmax())]
        self.requests.append(request)
        return request

    def step(self, *, cut: dict | None = None, trim: dict | None = None, remap: list[int] | None = None) -> None:
        cut, trim = cut or {}, trim or {}
        live = [r for r in self.requests if not r.done]
        tokens = torch.full((self.lanes, LROWS), PLACEHOLDER, dtype=torch.int32)
        positions = torch.full((self.lanes, LROWS), -1, dtype=torch.int32)
        valid = torch.zeros(self.lanes, dtype=torch.int32)
        counts = torch.ones(self.lanes, dtype=torch.int32)
        row_of = {}
        if remap is not None:  # row i reads the state now in slot remap[i]: the request in slot remap[i] runs in row i
            row_of = {remap[i]: i for i in range(self.lanes)}
        for r in live:
            row = row_of.get(r.slot, r.slot)
            drafts = r.drafts[: trim[r.slot]] if r.slot in trim else r.drafts
            n = len(drafts)
            tokens[row, 0] = r.committed[-1]
            for j, d in enumerate(drafts):
                tokens[row, 1 + j] = d
            positions[row] = torch.tensor([r.position + c if c <= n else -1 for c in range(LROWS)], dtype=torch.int32)
            valid[row], counts[row] = n, r.accepted
            r.row, r.block_drafts = row, drafts
        out = self.model.decode_forward(
            tokens=tokens,
            start_pos=positions,
            spec_mode="argmax_ids",
            num_valid_drafts=valid,
            accepted_counts=counts,
            slot_remap=None if remap is None else torch.tensor(remap, dtype=torch.int32),
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            read_from_device=True,
        )
        assert out.argmax_ids.shape == (self.lanes, LROWS)
        if remap is not None:
            self.model.note_state_slots_moved({remap[i]: i for i in range(self.lanes) if remap[i] != i})
            for r in live:
                r.slot = r.row  # the plugin's ownership map follows
        committed_block = torch.full((self.lanes, LROWS), PLACEHOLDER, dtype=torch.int32)
        committed_counts = torch.ones(self.lanes, dtype=torch.int32)
        for r in live:
            walked, n_ok = spec_decode.accept_greedy_drafts(
                out.argmax_ids[r.row : r.row + 1],
                torch.tensor([r.block_drafts + [PLACEHOLDER] * (LK - len(r.block_drafts))], dtype=torch.int32),
                torch.tensor([len(r.block_drafts)], dtype=torch.int32),
            )
            new = walked[0, : int(n_ok[0])].tolist()
            if r.slot in cut:
                new = new[: cut[r.slot]]
            new = new[: r.max_new - len(r.committed)]
            r.committed.extend(new)
            r.accepted = len(new)
            r.passes += 1
            committed_block[r.row, : len(new)] = torch.tensor(new, dtype=torch.int32)
            committed_counts[r.row] = len(new)
        offered = self.model.propose_draft_tokens(
            LK, committed_block, torch.zeros(self.lanes, LROWS, dtype=torch.int32), committed_counts, hidden=None
        )
        assert offered.draft_token_ids.shape == (self.lanes, LK) and offered.num_valid.shape == (self.lanes,)
        for r in live:
            r.drafts = offered.draft_token_ids[r.row, : int(offered.num_valid[r.row])].tolist()
            if len(r.committed) >= r.max_new:
                r.done = True
                self.model.release_request(r.slot)
        self.steps += 1

    def run(self, **kwargs) -> None:
        while any(not r.done for r in self.requests):
            self.step(**kwargs)


def test_lane_spec_plan_applies_the_lanes_rule(monkeypatch):
    monkeypatch.setenv("QWEN38_VLLM_MTP_K", "3")
    for seqs in range(2, 9):
        plan = Qwen38MTPForCausalLM.spec_plan(_config(), seqs, 3)
        assert isinstance(plan, spec_decode.SpecPlan) and plan.effective_k == 3 and plan.lanes_per_request == 4
        assert plan.supports_narrow_decode is False and plan.accept_modes == ("argmax_ids",)
    rejected = Qwen38MTPForCausalLM.spec_plan(_config(), 9, 3)
    assert isinstance(rejected, spec_decode.SpecReject) and "2..8" in rejected.reason
    monkeypatch.setenv("QWEN38_VLLM_MTP_K", "4")
    assert isinstance(Qwen38MTPForCausalLM.spec_plan(_config(), 6, 4), spec_decode.SpecPlan)
    refused = Qwen38MTPForCausalLM.spec_plan(_config(), 7, 4)  # 7 x 5 = 35 rows
    assert isinstance(refused, spec_decode.SpecReject) and "32-row tile" in refused.reason
    monkeypatch.setenv("QWEN38_VLLM_MTP_K", "5")
    assert isinstance(Qwen38MTPForCausalLM.spec_plan(_config(), 5, 5), spec_decode.SpecPlan)
    assert isinstance(Qwen38MTPForCausalLM.spec_plan(_config(), 6, 5), spec_decode.SpecReject)
    assert isinstance(Qwen38MTPForCausalLM.spec_plan(_config(), 4, 3), spec_decode.SpecReject)  # requested_k < K


def test_lane_adapter_refuses_a_bad_geometry_at_construction(monkeypatch, expect_error):
    monkeypatch.setenv("QWEN38_VLLM_MTP_K", "4")
    with expect_error(ValueError, "32-row tile"):
        Qwen38MTPForCausalLM(None, allocated_context=8192, batch_lanes=7, chain_opener=lambda hook: None)
    with expect_error(ValueError, "chain opener"):
        Qwen38MTPForCausalLM(None, allocated_context=8192, batch_lanes=2)


def test_lane_chain_options_drop_the_second_queue_read(monkeypatch):
    monkeypatch.setenv("QWEN38_VLLM_MTP_K", "3")
    monkeypatch.setenv("QWEN38_VLLM_MTP_PLE_EARLY", "1")
    monkeypatch.setattr(Qwen38MTPForCausalLM, "_open_lanes", 8)
    assert Qwen38MTPForCausalLM._chain_open_options(object()) == {"sampling": False, "mtp": 3, "mtp_ple_early": False}
    monkeypatch.setattr(Qwen38MTPForCausalLM, "_open_lanes", 1)
    assert Qwen38MTPForCausalLM._chain_open_options(object())["mtp_ple_early"] is True


def test_four_requests_of_different_lengths_finish_at_different_times_with_the_greedy_streams(monkeypatch):
    model, device = _lanes_model(monkeypatch, 4)
    runner = LaneRunner(model, 4)
    prompts = [[100 + 7 * i + 1000 * u for i in range(3 + 4 * u)] for u in range(4)]
    lengths = [9, 31, 17, 50]
    requests = [runner.join(u, prompts[u], lengths[u]) for u in range(4)]
    finished_at = {}
    while any(not r.done for r in requests):
        runner.step()
        for u, r in enumerate(requests):
            if r.done and u not in finished_at:
                finished_at[u] = runner.steps
    for u, r in enumerate(requests):
        assert r.committed == _lane_stream(prompts[u], lengths[u]), u
    assert len(set(finished_at.values())) > 1 and finished_at[0] < finished_at[3]
    driver = model.lane_driver
    assert driver.resyncs == 0 and driver.holders == []
    assert driver.accepted_drafts > 0 and runner.steps < max(lengths) - 1  # several tokens per pass
    accepted = {r.passes for r in requests}
    assert len(accepted) > 1  # mixed accept counts: the lanes needed different numbers of passes


def test_requests_join_and_leave_between_steps(monkeypatch):
    model, device = _lanes_model(monkeypatch, 4)
    runner = LaneRunner(model, 4)
    a = runner.join(0, [11, 12, 13], 14)
    b = runner.join(1, [21, 22, 23, 24, 25], 40)
    for _ in range(3):
        runner.step()
    c = runner.join(2, [31, 32], 20)  # joins while two lanes decode
    for _ in range(2):
        runner.step()
    d = runner.join(3, [41, 42, 43, 44, 45, 46, 47], 12)  # a fourth joins; then a finishes and its slot is reused
    runner.run()
    e = runner.join(0, [51, 52, 53, 54], 25)  # the slot of the finished request, a new request
    runner.run()
    for r in (a, b, c, d, e):
        assert r.committed == _lane_stream(r.prompt, r.max_new), r.slot
    assert model.lane_driver.holders == []


def test_a_request_finishing_in_the_middle_of_a_pass_releases_its_lane_and_the_others_continue(monkeypatch):
    model, device = _lanes_model(monkeypatch, 3)
    runner = LaneRunner(model, 3)
    reqs = [runner.join(u, [100 * (u + 1) + i for i in range(4 + u)], 7 + 3 * u) for u in range(3)]
    runner.run()  # max_new cuts a pass inside its accepted drafts for at least one request
    for r in reqs:
        assert r.committed == _lane_stream(r.prompt, r.max_new)
    # the cut left no lane held and the next pass of a new request on a freed lane is clean
    f = runner.join(1, [9, 8, 7], 15)
    runner.run()
    assert f.committed == _lane_stream(f.prompt, 15)


def test_a_plugin_cut_on_a_live_request_rewinds_the_lane_and_keeps_the_stream(monkeypatch):
    model, device = _lanes_model(monkeypatch, 3)
    runner = LaneRunner(model, 3)
    reqs = [runner.join(u, [50 * (u + 1) + 3 * i for i in range(5)], 30) for u in range(3)]
    for _ in range(2):
        runner.step()
    driver = model.lane_driver
    before = driver.resyncs
    for _ in range(6):
        runner.step(cut={1: 1})  # the plugin commits one token of slot 1's pass whatever the device accepted
    runner.run()
    for r in reqs:
        assert r.committed == _lane_stream(r.prompt, 30)
    cut_calls = [call for call in device.calls if call[0] == "override_commit"]
    assert cut_calls and driver.resyncs > before  # slot 1 drafts right some of the time: those cuts rewind the lane
    assert all(call[1] == 1 and call[2] == 1 for call in cut_calls)


def test_trimmed_drafts_of_one_lane_resync_that_lane_only(monkeypatch):
    model, device = _lanes_model(monkeypatch, 3)
    runner = LaneRunner(model, 3)
    reqs = [runner.join(u, [70 * (u + 1) + 2 * i for i in range(6)], 40) for u in range(3)]
    for _ in range(3):
        runner.step()
    driver = model.lane_driver
    before = driver.resyncs
    runner.step(trim={0: 1, 2: 0})
    assert driver.resyncs == before + 2
    runner.run()
    for r in reqs:
        assert r.committed == _lane_stream(r.prompt, 40)


def test_a_slot_remap_permutes_the_lane_map_and_keeps_the_streams(monkeypatch):
    model, device = _lanes_model(monkeypatch, 4)
    runner = LaneRunner(model, 4)
    reqs = [runner.join(u, [13 * (u + 1) + i for i in range(4)], 24) for u in range(4)]
    runner.step()
    runner.step(remap=[1, 0, 2, 3])
    assert model.lane_of_slot == [1, 0, 2, 3]
    runner.step()
    runner.step(remap=[3, 1, 2, 0])
    runner.run()
    for r in reqs:
        assert r.committed == _lane_stream(r.prompt, 24)


def test_a_live_lane_without_a_live_row_is_refused_by_name(monkeypatch, expect_error):
    model, device = _lanes_model(monkeypatch, 3)
    runner = LaneRunner(model, 3)
    a = runner.join(0, [1, 2, 3], 20)
    b = runner.join(1, [4, 5, 6], 20)
    runner.step()
    b.done = True  # the plugin stops scheduling b without releasing it
    with expect_error(Qwen38ChatChainError, "hold requests but no live row"):
        runner.step()
    assert model.batch_failed


def test_lane_prefill_and_decode_refusals(monkeypatch, expect_error):
    model, device = _lanes_model(monkeypatch, 3)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    kwargs = dict(page_table=None, kv_cache=None, enable_trace=True, prompt_lens=[3], start_pos=[0])
    with expect_error(ValueError, "distinct slots"):
        model.prefill_forward(tokens=prompt, empty_slots=[5], **kwargs)
    with expect_error(ValueError, "start at position 0"):
        model.prefill_forward(tokens=prompt, empty_slots=[0], **{**kwargs, "start_pos": [4]})
    with expect_error(ValueError, "one entry per row"):
        model.prefill_forward(tokens=prompt, empty_slots=[0, 1], **kwargs)
    runner = LaneRunner(model, 3)
    runner.join(0, [1, 2, 3], 10)
    z = torch.zeros(3, LROWS, dtype=torch.int32)
    ok = dict(
        num_valid_drafts=torch.zeros(3, dtype=torch.int32),
        accepted_counts=torch.ones(3, dtype=torch.int32),
        spec_mode="argmax_ids",
    )
    with expect_error(ValueError, "no narrow decode"):
        model.decode_forward(tokens=z[:, :1], start_pos=torch.zeros(3, dtype=torch.int32))
    with expect_error(ValueError, "block is"):
        model.decode_forward(tokens=z[:2], start_pos=z[:2], **ok)
    with expect_error(ValueError, "live row"):
        model.decode_forward(tokens=z, start_pos=z - 1, **ok)
    with expect_error(ValueError, "permutation"):
        model.decode_forward(tokens=z, start_pos=z, slot_remap=torch.tensor([0, 0, 1]), **ok)


def test_first_verify_after_a_lane_prefill_must_match_the_prefill(monkeypatch, expect_error):
    model, device = _lanes_model(monkeypatch, 2)
    prompt = [5, 6, 7, 8]
    out, _ = model.prefill_forward(
        tokens=torch.tensor([prompt], dtype=torch.int32),
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[4],
        start_pos=[0],
        empty_slots=[0],
    )
    first = int(out[0, 0].argmax())
    row = VerifyRow(0, [first + 1, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER], 4, 0, 1)
    with expect_error(Qwen38ChatChainError, "first verify after a prefill"):
        model.lane_driver.verify([row])


def test_lane_greedy_prefill_contract_returns_int32_tokens(monkeypatch):
    model, device = _lanes_model(monkeypatch, 2)
    params = greedy_batch_sampling_params(2) if callable(greedy_batch_sampling_params) else None
    prompts = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    out, rope = model.prefill_forward(
        tokens=prompts,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[3, 3],
        start_pos=[0, 0],
        empty_slots=[0, 1],
        sampling_params=params,
    )
    assert out.dtype == torch.int32 and out.shape == (2, 1) and rope.shape == (2,)
    assert [int(t) for t in out[:, 0]] == [_lane_stream(p, 1)[0] for p in prompts.tolist()]


def test_lane_warmup_runs_two_requests_through_the_serving_calls(monkeypatch, expect_error):
    monkeypatch.setattr(mtp_module, "WARMUP_PROMPT_TOKEN_IDS", tuple(_prompt(10)))
    model, device = _lanes_model(monkeypatch, 4)
    model.warmup_model_decode(
        enable_trace=False, kv_cache=None, max_batch_size=4, num_blocks=4, can_sample_on_device=False
    )
    assert device.passes == 0
    with expect_error(ValueError, "max_batch_size must be the 4 lanes"):
        model.warmup_model_decode(
            enable_trace=True, kv_cache=None, max_batch_size=2, num_blocks=4, can_sample_on_device=False
        )
    model.warmup_model_decode(
        enable_trace=True, kv_cache=None, max_batch_size=4, num_blocks=4, can_sample_on_device=False
    )
    assert device.passes == 2 and model.lane_driver.holders == []
    runner = LaneRunner(model, 4)
    r = runner.join(0, [3, 1, 4, 1, 5], 20)
    runner.run()
    assert r.committed == _lane_stream(r.prompt, 20)


def test_lane_shutdown_closes_the_lanes_then_the_chain(monkeypatch):
    model, device = _lanes_model(monkeypatch, 2)
    order = []
    device.close = lambda: order.append("lanes")
    model.chain.close = lambda: order.append("chain")
    model.release_persistent_capture()
    assert order == ["lanes", "chain"]


# -- hybrid single: a lone request on the chain's pass loop in a batch-mode server -------------------------------------------


class HybridLanes(FakeLanes):
    """The lane device with the chain-state import of the real session replaced by a record (the committed rows the chain
    held become the lane's sequence)."""

    def __init__(self, lanes: int):
        super().__init__(lanes)
        self.imports: list[tuple] = []


def _hybrid_model(monkeypatch, lanes: int = 3, flag: str | None = None):
    """A lanes-mode adapter over the lane fake and a scripted B=1 chain with the lanes' K."""

    monkeypatch.setattr(fakes, "K", LK)
    monkeypatch.setattr(sys.modules[__name__], "K", LK)
    if flag is not None:
        monkeypatch.setenv("QWEN38_MTP_HYBRID_SINGLE", flag)
    monkeypatch.setenv("QWEN38_VLLM_MTP_K", str(LK))
    device = HybridLanes(lanes)
    chain = BootstrapChain()
    chain.closed = False
    chain.misses_forbidden = True
    chain.mtp.drafts = LK
    model = Qwen38MTPForCausalLM(None, allocated_context=ALLOCATED, batch_lanes=lanes, chain_opener=lambda hook: None)
    model.lanes_session = device
    model.lane_driver = Qwen38VLLMMTPLanesDriver(device, lanes, LK)
    model.chain = chain
    model.session = mtp_module.Qwen38ChatSession(chain, None, allocated_context=ALLOCATED)
    model.driver = mtp_module.Qwen38VLLMMTPDriver(chain, model.session, LK)

    def import_chain_state(lane, position, ple_context):
        device.imports.append((lane, position, ple_context, len(model.session.committed)))
        device.sequence[lane] = list(model.session.committed)
        device.positions[lane] = position
        device.last_rows[lane] = None

    model.lane_driver._import_chain_state = import_chain_state
    return model, device, chain


def _hybrid_prefill(model, prompt, slot):
    return model.prefill_forward(
        tokens=torch.tensor([prompt], dtype=torch.int32),
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[len(prompt)],
        start_pos=[0],
        empty_slots=[slot],
    )


def _hybrid_verify(model, lanes, row, block, position, num_valid, accepted, remap=None):
    tokens = torch.full((lanes, LROWS), PLACEHOLDER, dtype=torch.int32)
    positions = torch.full((lanes, LROWS), -1, dtype=torch.int32)
    valid = torch.zeros(lanes, dtype=torch.int32)
    counts = torch.ones(lanes, dtype=torch.int32)
    tokens[row] = torch.tensor(block, dtype=torch.int32)
    positions[row] = torch.arange(position, position + LROWS, dtype=torch.int32)
    valid[row], counts[row] = num_valid, accepted
    return model.decode_forward(
        tokens=tokens,
        start_pos=positions,
        spec_mode="argmax_ids",
        num_valid_drafts=valid,
        accepted_counts=counts,
        slot_remap=remap,
    )


def _hybrid_step(model, lanes, row, committed, position, drafts, accepted):
    """One plugin step of the lone request: the verify of ``[committed[-1], *drafts]``, the greedy walk, the proposal."""

    block = [committed[-1], *drafts] + [PLACEHOLDER] * (LK - len(drafts))
    out = _hybrid_verify(model, lanes, row, block, position, len(drafts), accepted)
    walked, n_ok = spec_decode.accept_greedy_drafts(
        out.argmax_ids[row : row + 1],
        torch.tensor([drafts + [PLACEHOLDER] * (LK - len(drafts))], dtype=torch.int32),
        torch.tensor([len(drafts)], dtype=torch.int32),
    )
    new = walked[0, : int(n_ok[0])].tolist()
    block_out = torch.full((lanes, LROWS), PLACEHOLDER, dtype=torch.int32)
    block_out[row, : len(new)] = torch.tensor(new, dtype=torch.int32)
    counts = torch.ones(lanes, dtype=torch.int32)
    counts[row] = len(new)
    offered = model.propose_draft_tokens(
        LK, block_out, torch.zeros(lanes, LROWS, dtype=torch.int32), counts, hidden=None
    )
    return out, new, offered


def _run_hybrid_steps(model, lanes, slot, prompt, steps):
    """Prefill the lone request and run ``steps`` plugin steps; returns (committed, drafts, accepted, position)."""

    out, _ = _hybrid_prefill(model, prompt, slot)
    committed = [int(out[0, 0].argmax())]
    drafts, accepted = [], 1
    for _ in range(steps):
        position = len(prompt) + len(committed) - 1
        _, new, offered = _hybrid_step(model, lanes, slot, committed, position, drafts, accepted)
        committed += new
        accepted = len(new)
        drafts = offered.draft_token_ids[slot, : int(offered.num_valid[slot])].tolist()
    return committed, drafts, accepted, len(prompt) + len(committed) - 1


def test_hybrid_flag_parsing(monkeypatch, expect_error):
    assert mtp_module.mtp_hybrid_single_from_env({}) is True
    assert mtp_module.mtp_hybrid_single_from_env({"QWEN38_MTP_HYBRID_SINGLE": "1"}) is True
    assert mtp_module.mtp_hybrid_single_from_env({"QWEN38_MTP_HYBRID_SINGLE": "0"}) is False
    for bad in ("2", "", "true", "on"):
        with expect_error(ValueError, "QWEN38_MTP_HYBRID_SINGLE"):
            mtp_module.mtp_hybrid_single_from_env({"QWEN38_MTP_HYBRID_SINGLE": bad})
    monkeypatch.setenv("QWEN38_VLLM_MTP_K", "3")
    monkeypatch.setenv("QWEN38_MTP_HYBRID_SINGLE", "x")
    with expect_error(ValueError, "QWEN38_MTP_HYBRID_SINGLE"):
        Qwen38MTPForCausalLM.initialize_vllm_model(None, None, 4, 4096)


def test_a_lone_prompt_with_no_live_lane_starts_on_the_chain_in_any_slot(monkeypatch):
    model, device, chain = _hybrid_model(monkeypatch, 3)
    prompt = _prompt(6)
    out, rope = _hybrid_prefill(model, prompt, 2)
    assert model.mtp_hybrid_slot == 2 and device.calls == [] and chain.inputs == prompt
    assert out.shape == (1, 1, VOCAB_SIZE) and int(out[0, 0].argmax()) == fakes._greedy_stream(prompt, 1)[0]
    assert model.lane_driver.holders == []


def test_start_conditions_not_met_use_the_lanes(monkeypatch):
    model, device, chain = _hybrid_model(monkeypatch, 3)  # two prompts in one call
    prompts = torch.tensor([_prompt(4), _prompt(4, 300)], dtype=torch.int32)
    model.prefill_forward(
        tokens=prompts,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[4, 4],
        start_pos=[0, 0],
        empty_slots=[0, 1],
    )
    assert model.mtp_hybrid_slot is None and model.lane_driver.holders == [0, 1] and chain.inputs == []
    # a request already on a lane: the next lone prompt takes a lane too
    _hybrid_prefill(model, _prompt(5, 900), 2)
    assert model.mtp_hybrid_slot is None and model.lane_driver.holders == [0, 1, 2]
    # flag off: the lone prompt takes a lane
    model, device, chain = _hybrid_model(monkeypatch, 3, flag="0")
    assert model.mtp_hybrid_single is False
    _hybrid_prefill(model, _prompt(5), 1)
    # admission takes the LOWEST free lane, not the slot's own: slot 1 runs in lane 0
    assert model.mtp_hybrid_slot is None and model.lane_driver.holders == [0] and chain.inputs == []


def test_a_lone_request_stays_on_the_chain_and_pads_the_other_rows(monkeypatch):
    model, device, chain = _hybrid_model(monkeypatch, 3)
    prompt = _prompt(6)
    out, _ = _hybrid_prefill(model, prompt, 1)
    first = int(out[0, 0].argmax())
    block = [first, PLACEHOLDER, PLACEHOLDER, PLACEHOLDER]
    verified = _hybrid_verify(model, 3, 1, block, len(prompt), 0, 1)
    assert verified.argmax_ids.shape == (3, LROWS) and not verified.argmax_ids[[0, 2]].any()
    assert chain.passes == 1 and device.passes == 0 and device.calls == [] and model.mtp_hybrid_slot == 1
    committed = [first, int(verified.argmax_ids[1, 0])]
    block_out = torch.full((3, LROWS), PLACEHOLDER, dtype=torch.int32)
    block_out[1, 0] = committed[1]
    offered = model.propose_draft_tokens(
        LK, block_out, torch.zeros(3, LROWS, dtype=torch.int32), torch.tensor([1, 1, 1], dtype=torch.int32), hidden=None
    )
    assert (
        int(offered.num_valid[1]) == LK
        and not offered.num_valid[[0, 2]].any()
        and not offered.draft_token_ids[[0, 2]].any()
    )
    assert offered.draft_token_ids[1].tolist() == list(chain.next_tokens[1:])


def test_a_slot_remap_moves_the_chain_request_with_its_row(monkeypatch):
    model, device, chain = _hybrid_model(monkeypatch, 3)
    prompt = _prompt(6)
    out, _ = _hybrid_prefill(model, prompt, 2)
    block = [int(out[0, 0].argmax()), PLACEHOLDER, PLACEHOLDER, PLACEHOLDER]
    _hybrid_verify(model, 3, 0, block, len(prompt), 0, 1, remap=torch.tensor([2, 1, 0], dtype=torch.int32))
    assert model.mtp_hybrid_slot == 0 and model.lane_of_slot == [2, 1, 0]
    model.note_state_slots_moved({2: 0, 0: 2})
    assert model.lane_of_slot == [2, 1, 0]
    model.release_request(0)
    assert model.mtp_hybrid_slot is None


def test_a_second_prefill_moves_the_chain_request_into_its_lane(monkeypatch):
    model, device, chain = _hybrid_model(monkeypatch, 3)
    prompt = _prompt(6)
    committed, drafts, accepted, position = _run_hybrid_steps(model, 3, 0, prompt, 3)
    assert len(drafts) == LK and model.mtp_hybrid_slot == 0
    run = model.driver.run
    rows = run.accepted + 1
    assert accepted == rows
    resets = chain.log.count("reset")
    _hybrid_prefill(model, _prompt(5, 900), 1)  # the second request
    # the move: the chain was settled at the rows the last pass committed and imported without a reset or a new prefill
    assert model.mtp_hybrid_slot is None and model.driver.run is None
    assert f"leave:{rows}" in chain.log and chain.log.count("reset") == resets
    assert device.imports == [(0, run.position + rows, (chain.inputs[-1], 0), run.position + rows)]
    lane_run = model.lane_driver.runs[0]
    assert lane_run.fresh and (lane_run.position, lane_run.pending, lane_run.settled) == (
        position,
        run.next_tokens[0],
        rows,
    )
    assert (
        model.lane_driver.counts[0] == -1
        and model.lane_driver.blocks[0] == [run.next_tokens[0]] + [mtp_module.PLACEHOLDER_DRAFT_TOKEN] * LK
    )
    assert ("admit", 0) not in device.calls and ("admit", 1) in device.calls and model.lane_driver.holders == [0, 1]


def test_a_two_row_decode_moves_the_chain_request_and_runs_on_the_lanes(monkeypatch, expect_error):
    model, device, chain = _hybrid_model(monkeypatch, 3)
    prompt = _prompt(6)
    committed, drafts, accepted, position = _run_hybrid_steps(model, 3, 0, prompt, 2)
    run = model.driver.run
    pending = run.next_tokens[0]
    assert committed[-1] == pending
    # a verify that carries the chain request and a row nobody prefilled: the move runs, the stranger row is refused by name
    tokens = torch.full((3, LROWS), PLACEHOLDER, dtype=torch.int32)
    positions = torch.full((3, LROWS), -1, dtype=torch.int32)
    tokens[0] = torch.tensor([pending, *drafts], dtype=torch.int32)
    positions[0] = torch.arange(position, position + LROWS, dtype=torch.int32)
    tokens[1, 0], positions[1, 0] = 7, 3
    with expect_error(Qwen38ChatChainError, "no prefilled request"):
        model.decode_forward(
            tokens=tokens,
            start_pos=positions,
            spec_mode="argmax_ids",
            num_valid_drafts=torch.tensor([LK, 0, 0], dtype=torch.int32),
            accepted_counts=torch.tensor([accepted, 1, 1], dtype=torch.int32),
        )
    assert model.mtp_hybrid_slot is None and len(device.imports) == 1 and model.lane_driver.runs[0] is not None
    assert model.batch_failed


def test_the_first_lane_verify_after_the_move_takes_the_settled_rows(monkeypatch, expect_error):
    model, device, chain = _hybrid_model(monkeypatch, 3)
    prompt = _prompt(6)
    committed, drafts, accepted, position = _run_hybrid_steps(model, 3, 0, prompt, 2)
    pending = model.driver.run.next_tokens[0]
    _hybrid_prefill(model, _prompt(5, 900), 1)
    driver = model.lane_driver
    block = [pending, *drafts]
    assert driver.runs[0].settled == accepted
    row = VerifyRow(0, block, position, LK, accepted)
    driver._plan(0, row)  # accepted == the rows the chain was settled with
    assert driver.blocks[0] == block and driver.writes
    for bad in (
        VerifyRow(0, block, position, LK, accepted + 1 if accepted < LK + 1 else accepted - 1),
        VerifyRow(0, block, position + 1, LK, accepted),
        VerifyRow(0, [pending + 1, *drafts], position, LK, accepted),
    ):
        with expect_error(Qwen38ChatChainError, "first verify after a prefill"):
            driver._plan(0, bad)


def test_an_unsettled_chain_request_moves_with_the_first_token(monkeypatch):
    model, device, chain = _hybrid_model(monkeypatch, 3)
    prompt = _prompt(6)
    out, _ = _hybrid_prefill(model, prompt, 0)
    first = int(out[0, 0].argmax())
    _hybrid_prefill(model, _prompt(5, 900), 1)
    run = model.lane_driver.runs[0]
    assert (run.position, run.pending, run.settled) == (len(prompt), first, 1) and "leave:1" not in chain.log


def test_a_mismatched_commit_after_the_move_is_refused(monkeypatch, expect_error):
    model, device, chain = _hybrid_model(monkeypatch, 3)
    committed, drafts, accepted, position = _run_hybrid_steps(model, 3, 0, _prompt(6), 2)
    pending = model.driver.run.next_tokens[0]
    _hybrid_prefill(model, _prompt(5, 900), 1)
    tokens = torch.full((3, LROWS), PLACEHOLDER, dtype=torch.int32)
    positions = torch.full((3, LROWS), -1, dtype=torch.int32)
    tokens[0] = torch.tensor([pending, *drafts], dtype=torch.int32)
    positions[0] = torch.arange(position, position + LROWS, dtype=torch.int32)
    tokens[1, 0], positions[1, 0] = 5, 5  # the second request's first verify (its prompt has 5 tokens)
    valid = torch.tensor([LK, 0, 0], dtype=torch.int32)
    wrong = 1 if accepted != 1 else 2
    with expect_error(Qwen38ChatChainError, "first verify after a prefill"):
        model.decode_forward(
            tokens=tokens,
            start_pos=positions,
            spec_mode="argmax_ids",
            num_valid_drafts=valid,
            accepted_counts=torch.tensor([wrong, 1, 1], dtype=torch.int32),
        )


def test_hybrid_release_leaves_the_pass_loop_and_clears_the_state(monkeypatch):
    model, device, chain = _hybrid_model(monkeypatch, 3)
    _run_hybrid_steps(model, 3, 1, _prompt(6), 1)
    assert model.driver.run is not None
    model.release_request(1)
    assert model.mtp_hybrid_slot is None and model.driver.run is None and chain.log[-1].startswith("leave:")
    assert device.calls == []
    # the next lone request starts on the chain again; a prefill into the held slot releases the old request first
    _hybrid_prefill(model, _prompt(5, 400), 1)
    assert model.mtp_hybrid_slot == 1
    _hybrid_prefill(model, _prompt(5, 500), 1)
    assert model.mtp_hybrid_slot == 1 and model.lane_driver.holders == []


def test_shutdown_leaves_the_hybrid_pass_loop_before_the_closes(monkeypatch):
    model, device, chain = _hybrid_model(monkeypatch, 3)
    order = []
    device.close = lambda: order.append("lanes")
    chain.close = lambda: order.append("chain")
    _run_hybrid_steps(model, 3, 0, _prompt(6), 1)
    model.release_persistent_capture()
    assert chain.log[-1].startswith("leave:") and order == ["lanes", "chain"] and model.mtp_hybrid_slot is None


def test_hybrid_single_off_is_the_lane_path_for_a_lone_request(monkeypatch):
    model, device, chain = _hybrid_model(monkeypatch, 3, flag="0")
    runner = LaneRunner(model, 3)
    r = runner.join(0, [3, 1, 4, 1, 5], 20)
    runner.run()
    assert r.committed == _lane_stream(r.prompt, 20) and chain.passes == 0 and chain.inputs == []
    assert model.mtp_hybrid_slot is None and device.passes > 0


def test_hybrid_warmup_runs_the_hybrid_path_in_order_and_leaves_a_fresh_server(monkeypatch):
    from loguru import logger

    monkeypatch.setattr(mtp_module, "WARMUP_PROMPT_TOKEN_IDS", tuple(_prompt(10)))
    model, device, chain = _hybrid_model(monkeypatch, 3)
    events, lines = [], []
    import_state = model.lane_driver._import_chain_state

    def spy_import(lane, position, ple_context):
        events.append(("move", device.passes, len(chain.log)))
        import_state(lane, position, ple_context)

    model.lane_driver._import_chain_state = spy_import
    sink = logger.add(lambda message: lines.append(str(message)), level="INFO")
    try:
        model.warmup_model_decode(
            enable_trace=True, kv_cache=None, max_batch_size=3, num_blocks=4, can_sample_on_device=False
        )
    finally:
        logger.remove(sink)
    ((_, passes_at_move, log_at_move),) = events
    log = chain.log
    # the chain ran the bootstrap pass and two more before the move, left its pass loop at the move, and no lane admitted the
    # hybrid request (the two admissions are the lane warmup's)
    assert log.count("enter") == 1 and sum(e.startswith("pass:") for e in log[:log_at_move]) == 3
    assert log[log_at_move - 1].startswith("leave:") and not any(e.startswith("pass:") for e in log[log_at_move:])
    assert [call for call in device.calls if call[0] == "admit"] == [("admit", 0), ("admit", 1)]
    assert device.passes == passes_at_move + 1  # one lane pass after the move
    # a fresh server: no hybrid, no lane run, the identity map, the chain out of verify mode
    assert model.mtp_hybrid_slot is None and model.lane_driver.holders == [] and model.lane_of_slot == [0, 1, 2]
    assert model.driver.run is None and chain.mtp.chain is None and not model.batch_failed
    assert any("mtp hybrid warmup" in line for line in lines)
    # it serves a hybrid request afterwards
    _hybrid_prefill(model, _prompt(5, 400), 0)
    assert model.mtp_hybrid_slot == 0


def test_hybrid_warmup_is_skipped_with_the_flag_off(monkeypatch):
    monkeypatch.setattr(mtp_module, "WARMUP_PROMPT_TOKEN_IDS", tuple(_prompt(10)))
    model, device, chain = _hybrid_model(monkeypatch, 3, flag="0")
    model.warmup_model_decode(
        enable_trace=True, kv_cache=None, max_batch_size=3, num_blocks=4, can_sample_on_device=False
    )
    assert chain.passes == 0 and device.imports == []
