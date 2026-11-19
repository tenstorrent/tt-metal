# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The sampled step order, its finish semantics and the request mapping, on a fake chain (no device)."""

from __future__ import annotations

import inspect

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.helpers.sampling import (
    SamplingFakeChain,
    SamplingFakeExtension,
    sampling_history_logits,
    sampling_parameters,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling as step
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_protocol import THINK_END_ID
from models.demos.blackhole.qwen38_flash_next.ttnn import device_sampler as ds
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import LOCAL_VOCAB_SIZE, TP_SIZE, VOCAB_SIZE
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import (
    Qwen38CandidateRow,
    Qwen38SamplingParameters,
    Qwen38SamplingProfile,
    sample_full_vocabulary,
)

EOS = (248_046, 248_044)
TOKENIZER_SIZE = 248_077


class FakeSession:
    chain_class, sampling_class = SamplingFakeChain, SamplingFakeExtension

    def __init__(self, prefix: list[int], *, eos_at: int | None = None) -> None:
        self.chain = self.chain_class(prefix, eos_at=eos_at)
        self.sampling = self.sampling_class(self.chain)
        self.committed = list(prefix)
        self.ple_context = None

    def _forced_step(self, token_id: int) -> None:
        residue = len(self.committed) % step.RESIDUE_CLASSES
        self.chain.write_token_row(token_id)
        self.chain.execute_head(residue)
        self.ple_context = self.chain.refresh_ple_row(token_id, self.ple_context)
        self.chain.execute_tail(residue)
        self.committed.append(token_id)


THINKING = Qwen38SamplingParameters.official_thinking(seed=77)


def _request(parameters=THINKING, **fields) -> step.Qwen38SamplingRequest:
    return step.Qwen38SamplingRequest(parameters, **fields)


def _run(session: FakeSession, request: step.Qwen38SamplingRequest, max_new_tokens: int, **extra):
    extra.setdefault("forced_step", session._forced_step)
    return list(
        step.generate_sampled(session, request, max_new_tokens, stop_ids=EOS, tokenizer_size=TOKENIZER_SIZE, **extra)
    )


def test_step_order_is_blocking_read_sample_write_head_ple_tail() -> None:
    session, request = FakeSession([1, 2, 3]), _request()
    items = _run(session, request, 4)
    assert [finish for _, finish in items] == [None, None, None, "length"]
    first = session.chain.log[:5]
    token = items[0][0]
    assert first == ["read_row", f"write:{token}", "head:3", f"ple:{token}", "tail:3"]
    assert "event" not in session.chain.log and "sync" not in session.chain.log
    # The residue follows the committed length (3 prefix tokens, then 4, 5).
    assert [entry for entry in session.chain.log if entry.startswith("head:")] == ["head:3", "head:0", "head:1"]
    # max_tokens: the last token comes from a blocking row read, is not consumed, and is written into the row.
    last = items[-1][0]
    assert session.chain.log[-2:] == ["read_row", f"write:{last}"] and session.chain.tails == 3
    assert session.chain.row == last
    assert session.committed == [1, 2, 3] + [token for token, _ in items[:3]]
    assert session.ple_context is not None and session.ple_context[0] == items[2][0]
    assert [sample.token_id for sample in request.samples] == [token for token, _ in items]


def test_eos_is_consumed_then_tail_is_completed_by_a_blocking_read() -> None:
    session = FakeSession([5, 6], eos_at=4)
    items = _run(session, _request(), 32)
    tokens = [token for token, _ in items]
    assert tokens[-1] == EOS[0] and items[-1][1] == "stop" and len(tokens) == 3
    assert session.chain.log[-3:] == [f"ple:{EOS[0]}", "tail:0", "read_token_row"]
    assert session.committed[-1] == EOS[0]  # consumed, like the greedy loop


def test_same_seed_same_stream_and_the_sample_carries_logprobs() -> None:
    left = _run(FakeSession([9]), _request(top_logprobs=3), 24)
    right_request = _request(top_logprobs=3)
    right = _run(FakeSession([9]), right_request, 24)
    assert [t for t, _ in left] == [t for t, _ in right]
    assert _run(FakeSession([9]), _request(Qwen38SamplingParameters.official_thinking(seed=78)), 24) != left
    sample = right_request.samples[0]
    assert sample.logprob <= 0 and len(sample.top_logprobs) == 3
    assert sample.top_logprobs[0][1] >= sample.top_logprobs[-1][1]
    assert right_request.clocks.fallbacks == 0 and right_request.clocks.candidate_misses == 0


def test_top_k_zero_takes_the_full_vocabulary_fallback_every_step() -> None:
    session = FakeSession([4])
    parameters = sampling_parameters(temperature=0.8, top_p=0.9, top_k=0, seed=3)
    request = _request(parameters)
    items = _run(session, request, 6, clock_ns=lambda: 0)
    assert session.sampling.full_reads == 6 and request.clocks.fallbacks == 6
    # The fallback's residue is TAIL(t-1)'s: one committed token before the step's residue.
    gathers = [entry for entry in session.chain.log if entry.startswith("full_gather:")]
    assert gathers[:4] == ["full_gather:0", "full_gather:1", "full_gather:2", "full_gather:3"]
    # And it is the reference sampler on the same logits and generator state.
    expected = sample_full_vocabulary(
        sampling_history_logits([4]).float(), parameters, token_history=[4], generator=torch.Generator().manual_seed(3)
    )
    assert items[0][0] == expected.token_id


def test_presence_penalty_counts_the_requests_output_not_the_prompt() -> None:
    # top_k 1 makes the sampled token the penalized argmax.  The fake row depends on the prompt's length and last
    # token only, so the prompt can hold the top peak of its own row.
    last = 3
    top = int(torch.argmax(sampling_history_logits([0, last]).to(torch.float32)))
    prefix = [top, last]
    parameters = sampling_parameters(top_k=1, presence_penalty=2.0, seed=1)
    session, request = FakeSession(prefix), _request(parameters)
    tokens = [token for token, _ in _run(session, request, 3)]
    # The prompt's `top` is not penalized: the first token is the row's argmax.  Under the whole-history rule it
    # would have lost 2.0 and the argmax would have moved.
    assert tokens[0] == top
    whole_history = sample_full_vocabulary(sampling_history_logits(prefix).float(), parameters, token_history=prefix)
    assert whole_history.token_id != top
    # Every step is the reference over the committed history with the prompt exempt (the last token is the
    # unconsumed length read, chosen the same way).
    generator = torch.Generator().manual_seed(1)
    for index, token in enumerate(tokens):
        history = prefix + tokens[:index]
        expected = sample_full_vocabulary(
            sampling_history_logits(history).to(torch.float32),
            parameters,
            token_history=history,
            prompt_tokens=len(prefix),
            generator=generator,
        )
        assert token == expected.token_id
    assert session.committed == prefix + tokens[:-1]


def test_should_stop_ends_the_loop_between_steps_without_a_sample() -> None:
    session, request = FakeSession([1]), _request()
    polls = []
    items = _run(session, request, 8, should_stop=lambda: (polls.append(1), "halt" if len(polls) == 3 else None)[1])
    assert items[-1] == (None, "halt") and len(items) == 3 and len(request.samples) == 2
    assert session.chain.log[-1].startswith("tail:") and len(session.committed) == 3


def test_think_budget_forces_think_end_through_the_forced_step(expect_error) -> None:
    session, request = FakeSession([1]), _request()
    items = _run(session, request, 6, think_budget=2)
    tokens = [token for token, _ in items]
    assert tokens[2] == THINK_END_ID and request.samples[2] is None and len(request.samples) == 6
    forced = session.chain.log.index(f"write:{THINK_END_ID}")
    assert session.chain.log[forced - 1].startswith("tail:") and session.chain.log[forced + 1] == "head:3"
    assert session.committed[3] == THINK_END_ID
    with expect_error(TypeError, match="forced step"):
        list(step.generate_sampled(FakeSession([1]), _request(), 2, stop_ids=EOS, tokenizer_size=1, think_budget=1))


def test_temperature_zero_is_refused_by_the_request(expect_error) -> None:
    with expect_error(ValueError, match="temperature > 0"):
        step.Qwen38SamplingRequest(Qwen38SamplingParameters.greedy())
    with expect_error(TypeError, match="Qwen38SamplingRequest"):
        list(step.generate_sampled(FakeSession([1]), THINKING, 4, stop_ids=EOS, tokenizer_size=TOKENIZER_SIZE))


def test_clocks_summary_reports_period_and_host_segment() -> None:
    clock = iter(range(0, 10_000_000, 100_000))
    request = _request()
    _run(FakeSession([2]), request, 5, clock_ns=lambda: next(clock))
    summary = request.clocks.summary()
    assert summary["tokens"] == 4 and summary["fallbacks"] == 0 and summary["candidate_misses"] == 0
    assert summary["period_median_ms"] == pytest.approx(0.4) and summary["host_segment_median_ms"] == pytest.approx(0.1)
    assert summary["sample_median_ms"] == pytest.approx(0.1)
    assert set(request.as_dict()) == {
        "profile",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "seed",
        "fallbacks",
        "candidate_misses",
        "logprobs_normalizer",
        "device_path",
        "draws",
        "first_token_rewrites",
        "verified_steps",
        "mtp_drafting",
        "mtp_acceptance_arithmetic",
        "mtp",
    }
    assert request.as_dict()["logprobs_normalizer"] == "candidate_row"  # the reported logprobs are row-relative
    assert (
        request.as_dict()["mtp_drafting"] is None and request.as_dict()["mtp"] is None
    )  # the 1-row loop: no pass loop verdict


# --- the request mapping ----------------------------------------------------------------------------------------


def test_no_sampling_field_is_greedy_and_absent_temperature_maps_to_the_card_profile_keyed_on_thinking() -> None:
    # The launchers serve with --sampling, so a request naming no sampling field must stay the bitwise greedy loop;
    # a sampling field without a temperature (seed alone included) takes the card profile.
    assert step.parameters_from_request({"messages": []}, enable_thinking=True, seed=11) is None
    assert step.parameters_from_request({}, enable_thinking=False, seed=12) is None
    thinking = step.parameters_from_request({"seed": 11}, enable_thinking=True, seed=99)
    assert thinking.profile is Qwen38SamplingProfile.THINKING and thinking.seed == 11
    assert (thinking.temperature, thinking.top_p, thinking.top_k) == (1.0, 0.95, 20)
    instruct = step.parameters_from_request({"seed": 12}, enable_thinking=False, seed=99)
    assert instruct.profile is Qwen38SamplingProfile.NON_THINKING and instruct.presence_penalty == 1.5
    # A partial request keeps the profile's other values but is a custom policy.
    partial = step.parameters_from_request({"top_p": 0.5}, enable_thinking=True, seed=1)
    assert partial.profile is Qwen38SamplingProfile.CUSTOM and (partial.temperature, partial.top_p, partial.top_k) == (
        1.0,
        0.5,
        20,
    )


def test_present_temperature_uses_openai_defaults_with_the_card_top_k() -> None:
    parameters = step.parameters_from_request(
        {"temperature": 0.6, "seed": 5, "repetition_penalty": 1.1}, enable_thinking=True, seed=1
    )
    assert (parameters.temperature, parameters.top_p, parameters.top_k, parameters.presence_penalty) == (
        0.6,
        1.0,
        20,
        0.0,
    )
    assert parameters.repetition_penalty == 1.1 and parameters.seed == 5


def test_temperature_zero_and_greedy_take_the_greedy_loop() -> None:
    assert step.parameters_from_request({"temperature": 0}, enable_thinking=False, seed=1) is None
    assert step.parameters_from_request({"temperature": 0.0, "top_k": 5}, enable_thinking=True, seed=1) is None
    assert step.parameters_from_request({"greedy": True}, enable_thinking=True, seed=1) is None
    assert step.parameters_from_request({"greedy": True, "temperature": 0}, enable_thinking=True, seed=1) is None
    assert step.parameters_from_request({"greedy": False}, enable_thinking=True, seed=1) is None  # asks for nothing
    assert step.parameters_from_request({"greedy": False, "top_p": 0.5}, enable_thinking=True, seed=1) is not None


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({"top_k": 33}, "top_k must be at most 32"),
        ({"temperature": "hot"}, "temperature must be a number"),
        ({"top_k": 2.5}, "top_k must be an integer"),
        ({"n": 2}, "n must be 1"),
        ({"seed": -1}, "seed must be in"),
        ({"temperature": -0.1}, "temperature must be nonnegative"),
        ({"top_p": 0.0}, "top_p must be in"),
        ({"presence_penalty": 3}, "presence_penalty must be in"),
        ({"min_p": 2}, "min_p must be in"),
        ({"greedy": "yes"}, "greedy must be a boolean"),
        ({"greedy": True, "temperature": 0.7}, "greedy is true but temperature is 0.7"),
    ],
)
def test_bad_fields_are_request_errors_naming_the_field_first(document, message, expect_error) -> None:
    error_type = step.Qwen38SamplingRequestError
    with pytest.raises(error_type, match=message) as info:  # allow-pytest.raises: reads the exception
        step.parameters_from_request(document, enable_thinking=True, seed=0)
    assert str(info.value).split(" ", 1)[0] in document


def test_logprobs_fields_and_the_content_item_shape(expect_error) -> None:
    assert step.logprobs_from_request({}) == (False, 0)
    assert step.logprobs_from_request({"logprobs": True, "top_logprobs": 5}) == (True, 5)
    for document in ({"logprobs": "yes"}, {"logprobs": True, "top_logprobs": 21}, {"top_logprobs": 2}):
        with expect_error(step.Qwen38SamplingRequestError):
            step.logprobs_from_request(document)
    sample = sample_full_vocabulary(sampling_history_logits([1]).to(torch.float32), THINKING, top_logprobs=2)
    item = step.logprobs_content_item(sample, sample.token_id, lambda token: f"<{token}>")
    assert item["token"] == f"<{sample.token_id}>" and item["logprob"] == sample.logprob
    assert item["bytes"] == list(f"<{sample.token_id}>".encode()) and len(item["top_logprobs"]) == 2
    forced = step.logprobs_content_item(None, THINK_END_ID, lambda token: "</think>")
    assert forced == {"token": "</think>", "logprob": None, "bytes": list(b"</think>"), "top_logprobs": []}
    assert set(step.SAMPLING_REQUEST_FIELDS) >= {"temperature", "greedy", "top_p", "top_k", "seed", "logprobs", "n"}


# --- the chain extension's shape ------------------------------------------------------------------------------------


def test_chain_extension_runs_the_greedy_epilogue_unchanged_before_the_row_and_the_sampler_after_it() -> None:
    body = inspect.getsource(step.Qwen38SamplingChainExtension._epilogue)  # eager in the warm pass, captured in TAIL
    host_branch, device_branch = body.split("if self.sampler is None:")
    order = [
        host_branch.index(f)
        for f in (
            "greedy_candidates(logits)",
            "resolve_greedy_on_device(candidates, into=token_row_io)",  # the greedy row lands in place, one program
            "sampling_candidates(",
        )
    ]
    assert order == sorted(order) and "self.sample(" not in host_branch and "ttnn.copy(" not in host_branch
    # without the sampler the greedy row is the token row
    assert "return candidates, row, greedy_row, greedy_row" in device_branch
    # the composite or the one-program kernel, by the registry, then its token into the resident row
    assert device_branch.index("self.sample(row, greedy_row, self.sampler)") < device_branch.index(
        "ttnn.copy(token_row, token_row_io)"
    )
    assert step.RESIDUE_CLASSES == 4
    primitives = ("read_candidate_row", "read_full_logits", "warm", "release", "mark_corruptible")
    assert all(hasattr(step.Qwen38SamplingChainExtension, name) for name in primitives)
    assert not hasattr(step.Qwen38SamplingChainExtension, "read_candidate_row_nonblocking")
    assert "ttnn.to_torch(ttnn.get_device_tensors(self.constants.readback_row)[0])" in inspect.getsource(
        step.Qwen38SamplingChainExtension.read_candidate_row
    )
    assert "ttnn.deallocate(gathered)" in inspect.getsource(step.Qwen38SamplingChainExtension._gather)
    capture = inspect.getsource(step.Qwen38SamplingChainExtension.capture_epilogue)
    warm = inspect.getsource(step.Qwen38SamplingChainExtension.warm)
    assert "self._epilogue(trace_output.logits, token_row_io)" in capture
    assert "self._epilogue(logits, token_row_io)" in warm and "self._gather(logits)" in warm  # the capture's epilogue
    assert "ids_equal_up_to_boundary_ties" in warm and "raise RuntimeError" in warm
    loop = inspect.getsource(step.generate_sampled_on_device)
    assert "sampler.write_uniform(request.next_uniform())" in loop and loop.index(
        "chain.execute_head(residue)"
    ) < loop.index("sampler.write_uniform") < loop.index("chain.execute_tail(residue)")
    assert (
        "read_candidate_row"
        not in loop.split("def reference_token")[1].split("produced = 0")[1].split("verify_each_step")[0]
    )


# --- the device-sampled loop: the greedy loop plus one draw write per step --------------------------------------------


class FakeDeviceSampler:
    """The device sampler's constants: records the policy and draw writes the loop makes."""

    def __init__(self, chain: SamplingFakeChain) -> None:
        self.chain = chain
        self.policy = ds.Qwen38DeviceSamplerPolicy.greedy_policy()
        self.uniform = 0.0
        self.writes: list[str] = []

    def write_policy(self, policy) -> dict:
        self.policy = policy
        self.chain.log.append(f"policy:{'greedy' if policy.greedy else 'sampled'}")
        return {}

    def write_uniform(self, uniform: float) -> None:
        self.uniform = uniform
        self.chain.log.append(f"u:{uniform}")


class FakeDeviceChain(SamplingFakeChain):
    """TAIL chooses the token the way the device composite does: the greedy id under the flag, else the reference."""

    def __init__(self, prefix: list[int], *, eos_at: int | None = None) -> None:
        super().__init__(prefix, eos_at=eos_at)
        self.sampler = None  # the sampling's constants
        self.row_history: list[torch.Tensor] = []  # the candidate row TAIL(t) computed, for the host replay

    def execute_tail(self, residue: int) -> None:
        self.log.append(f"tail:{residue}")
        self.tails += 1
        logits = self.logits()
        host_row = Qwen38CandidateRow.emulate(logits).to_host_row()
        self.row_history.append(host_row)
        sampler = self.sampler
        if sampler.policy.greedy:
            self.row = int(torch.argmax(logits.to(torch.float32)))
        else:
            values, ids = ds.candidate_row_lanes(host_row)
            self.row = ds.device_sampler_reference(values, ids, sampler.policy, sampler.uniform).token_id


class FakeDeviceSampling(SamplingFakeExtension):
    presence_on_device = False  # the composite's constants: no device history
    sampler_class = FakeDeviceSampler

    def __init__(self, chain: FakeDeviceChain) -> None:
        super().__init__(chain)
        self.sampler = chain.sampler = self.sampler_class(chain)

    def device_policy_of(self, request):
        return request.device_policy(presence_on_device=self.presence_on_device)

    def rewrite_history(self, emitted) -> None:
        return None

    def read_candidate_row(self) -> Qwen38CandidateRow:
        self.chain.log.append("read_row")
        return Qwen38CandidateRow.from_host_row(self.chain.row_history[-1])

    def begin_request(self, request) -> None:
        step.Qwen38SamplingChainExtension.begin_request(self, request)


class FakeDeviceSession(FakeSession):
    chain_class, sampling_class = FakeDeviceChain, FakeDeviceSampling

    def __init__(self, prefix: list[int], *, eos_at: int | None = None) -> None:
        super().__init__(prefix, eos_at=eos_at)
        self.row_token = None

    def start(self, request, prompt: list[int]) -> None:
        """The request start (the policy and the first draw), then the prompt: its last TAIL chooses x_0."""

        self.sampling.begin_request(request)
        for token in prompt:
            self._forced_step(token)


def _run_device(session: FakeDeviceSession, request, max_new_tokens: int, *, prefilled: bool = True, **extra):
    extra.setdefault("forced_step", session._forced_step)
    return list(
        step.generate_sampled_on_device(
            session,
            request,
            max_new_tokens,
            stop_ids=EOS,
            tokenizer_size=TOKENIZER_SIZE,
            prefilled=prefilled,
            **extra,
        )
    )


def test_device_loop_is_the_greedy_loop_plus_a_draw_write_and_replays_the_reference() -> None:
    session, request = FakeDeviceSession([1, 2]), _request()
    session.start(request, [])  # request start: the policy and the first draw, before the prompt
    assert session.chain.log == ["policy:sampled", f"u:{request.uniforms[0]}"] and len(request.uniforms) == 1
    session._forced_step(3)  # the prompt's last TAIL chooses x_0 under the request's policy and u_0
    start = len(session.chain.log)
    items = _run_device(session, request, 5)
    tokens = [token for token, _ in items]
    assert [finish for _, finish in items] == [None, None, None, None, "length"]
    log = session.chain.log[start:]
    # Entry: the row read and the token read (both complete the prompt's TAIL); no rewrite when prefilled.
    assert log[:2] == ["read_row", "read_token_row"] and request.first_token_rewrites == 0
    first = tokens[0]
    assert log[2:9] == [
        "read_nonblocking",
        "event",
        "head:3",
        f"u:{request.uniforms[1]}",
        "sync",
        f"ple:{first}",
        "tail:3",
    ]
    assert "write:" not in " ".join(log[2:])  # no host token write: the device chose every token
    assert log[-1] == "read_token_row" and session.chain.row == tokens[-1]  # max_tokens: the row keeps the token
    assert len(request.uniforms) == 5 and request.samples == [None] * 4
    # The host replay: x_k = reference(row of TAIL(k-1), u_k) over the rows the fake TAIL computed.
    policy = request.device_policy()
    for k, token in enumerate(tokens):
        values, ids = ds.candidate_row_lanes(session.chain.row_history[k])
        assert token == ds.device_sampler_reference(values, ids, policy, request.uniforms[k]).token_id
    assert session.committed == [1, 2, 3] + tokens[:-1]
    assert request.as_dict()["device_path"] and request.as_dict()["draws"] == 5
    # The same seed gives the same stream on a fresh session; a different seed differs.
    again, again_request = FakeDeviceSession([1, 2]), _request()
    again.start(again_request, [3])
    assert [t for t, _ in _run_device(again, again_request, 5)] == tokens
    other, other_request = FakeDeviceSession([1, 2]), _request(Qwen38SamplingParameters.official_thinking(seed=78))
    other.start(other_request, [3])
    assert [t for t, _ in _run_device(other, other_request, 5)] != tokens


def test_device_loop_verifies_each_step_when_asked_and_rewrites_a_stale_continuation(expect_error) -> None:
    session, request = FakeDeviceSession([4]), _request()
    session.start(request, [5])
    items = _run_device(session, request, 6, verify_each_step=True)
    assert len(items) == 6 and request.verified_steps == 4  # tokens 1..4 of the loop; x_0 at entry, x_5 by the read
    assert session.chain.log.count("read_row") == 5
    # A continuation: the row was chosen under the previous request's policy (greedy here); the host rewrites x_0
    # (a seed whose first draw leaves the argmax).
    session.sampling.sampler.write_policy(ds.Qwen38DeviceSamplerPolicy.greedy_policy())
    session._forced_step(9)
    values, ids = ds.candidate_row_lanes(session.chain.row_history[-1])
    for seed in range(5, 200):
        request = _request(Qwen38SamplingParameters.official_thinking(seed=seed))
        session.sampling.begin_request(request)
        expected = ds.device_sampler_reference(values, ids, request.device_policy(), request.uniforms[0]).token_id
        if expected != session.chain.row:
            break
    else:
        raise AssertionError("no seed left the argmax")
    start = len(session.chain.log)
    items = _run_device(session, request, 3, prefilled=False)
    assert items[0][0] == expected and request.first_token_rewrites == 1
    assert session.chain.log[start : start + 3] == ["read_row", "read_token_row", f"write:{expected}"]
    # Prefilled and mismatching is a device error.
    session.sampling.sampler.write_policy(ds.Qwen38DeviceSamplerPolicy.greedy_policy())
    session._forced_step(9)
    request = _request(Qwen38SamplingParameters.official_thinking(seed=6))
    session.sampling.begin_request(request)
    with expect_error(RuntimeError, match="first token"):
        _run_device(session, request, 3, prefilled=True)


def test_device_loop_finishes_like_the_greedy_loop(expect_error) -> None:
    session, request = FakeDeviceSession([5, 6], eos_at=4), _request()
    session.start(request, [7])
    items = _run_device(session, request, 32)
    assert items[-1] == (EOS[0], "stop") and session.chain.log[-1] == "read_token_row"
    assert session.committed[-1] == EOS[0]
    session, request = FakeDeviceSession([1]), _request()
    session.start(request, [2])
    items = _run_device(session, request, 6, think_budget=2)
    tokens = [token for token, _ in items]
    assert tokens[2] == THINK_END_ID and request.samples[2] is None and len(request.samples) == 5
    polls = []
    session, request = FakeDeviceSession([1]), _request()
    session.start(request, [2])
    items = _run_device(
        session, request, 8, should_stop=lambda: (polls.append(1), "halt" if len(polls) == 3 else None)[1]
    )
    assert items[-1] == (None, "halt") and len(items) == 3
    with expect_error(RuntimeError, match="begin_request"):
        _run_device(FakeDeviceSession([1]), _request(), 2)


def _ramp_logits() -> torch.Tensor:
    """Shard 0's first 64 ids on a ramp down from 5.0 in steps of 1/32, every other shard's from -10, the rest -30:
    once the request has emitted a dozen of shard 0's read ids, the penalty pushes them under its unread ones."""

    logits = torch.full((VOCAB_SIZE,), -30.0)
    for shard in range(TP_SIZE):
        start = shard * LOCAL_VOCAB_SIZE
        logits[start : start + 64] = (5.0 if shard == 0 else -10.0) - torch.arange(64) / 32
    return logits.to(torch.bfloat16)


class FakeFusedSampler(FakeDeviceSampler):
    def __init__(self, chain: "FakeDeviceChain") -> None:
        super().__init__(chain)
        self.history: set[int] = set()

    def reset_history(self) -> None:
        self.history = set()


class FakeFusedChain(FakeDeviceChain):
    """The fused sampler's TAIL: the presence penalty from the device history, its draw's bit set, the row's guard."""

    guard = False

    def logits(self) -> torch.Tensor:
        return _ramp_logits()

    def execute_tail(self, residue: int) -> None:
        self.log.append(f"tail:{residue}")
        self.tails += 1
        host_row = Qwen38CandidateRow.emulate(self.logits()).to_host_row()
        self.row_history.append(host_row)
        sampler = self.sampler
        values, ids = ds.candidate_row_lanes(host_row)
        seen = torch.tensor([int(token_id) in sampler.history for token_id in ids.tolist()], dtype=torch.bool)
        sample = ds.device_sampler_reference(values, ids, sampler.policy, sampler.uniform, seen=seen)
        self.row, self.guard = sample.token_id, sample.guard
        sampler.history.add(sample.token_id)


class FakeFusedSampling(FakeDeviceSampling):
    presence_on_device = True
    sampler_class = FakeFusedSampler

    def rewrite_history(self, emitted) -> None:
        self.chain.log.append("history")
        self.sampler.history = set(emitted)

    def read_guard_nonblocking(self):
        self.chain.log.append("read_guard")
        return self.chain.guard

    def read_guard(self, pending=None) -> bool:
        return self.chain.guard if pending is None else pending


class FakeFusedSession(FakeDeviceSession):
    chain_class, sampling_class = FakeFusedChain, FakeFusedSampling


def test_device_loop_samples_a_guarded_presence_row_over_the_full_vocabulary() -> None:
    session = FakeFusedSession([1, 2])
    request = _request(Qwen38SamplingParameters.official_non_thinking(seed=5))
    session.start(request, [3])
    penalty = request.device_policy(presence_on_device=True).presence_penalty
    assert penalty == 1.5, penalty
    start = len(session.chain.log)
    tokens = [token for token, _ in _run_device(session, request, 40, verify_each_step=True)]
    log = session.chain.log[start:]
    fallbacks = request.clocks.fallbacks
    counts = (fallbacks, log.count("history"), session.sampling.full_reads, request.verified_steps)
    assert 0 < fallbacks == counts[1] == counts[2] < 39 and counts[3] == 38, counts
    assert request.as_dict()["fallbacks"] == fallbacks, request.as_dict()
    # A penalised row's guard is read with its token, before HEAD(t): the guarded row is the host sampler's draw over
    # TAIL(t-1)'s full logits, in the token row and the history before HEAD consumes it.
    steps = [entry for entry in log if entry in ("read_nonblocking", "read_guard", "event", "sync")]
    assert steps[:3] == ["read_nonblocking", "event", "sync"], steps[:3]  # x_0, decided at entry: no guard read
    assert steps[3:] == ["read_nonblocking", "read_guard", "event", "sync"] * 38, steps[3:11]
    generator = torch.Generator().manual_seed(5)
    history = []
    for k, token in enumerate(tokens[:-1]):
        values, ids = ds.candidate_row_lanes(session.chain.row_history[k])
        seen = torch.tensor([int(token_id) in history for token_id in ids.tolist()], dtype=torch.bool)
        policy = request.device_policy(presence_on_device=True)
        reference = ds.device_sampler_reference(values, ids, policy, request.uniforms[k], seen=seen)
        if reference.guard:
            expected = sample_full_vocabulary(
                _ramp_logits().to(torch.float32),
                request.parameters,
                token_history=[1, 2, 3, *history],
                prompt_tokens=3,
                generator=generator,
            ).token_id
            gather = f"full_gather:{(2 + k) % step.RESIDUE_CLASSES}"
            head = log.index(f"head:{(3 + k) % step.RESIDUE_CLASSES}", log.index(gather))
            assert log[head - 3 : head] == [gather, f"write:{expected}", "history"], (k, log[head - 3 : head])
        else:
            expected = reference.token_id
        assert token == expected, (k, token, expected)
        history.append(token)
        log = log[log.index(f"tail:{(3 + k) % step.RESIDUE_CLASSES}") + 1 :]
    assert any(32 <= token < 64 for token in tokens), tokens  # unread ids of shard 0: never the device's draw


def test_device_policy_routing_and_begin_request_for_greedy_and_host_loop_requests() -> None:
    thinking = _request()
    assert thinking.device_policy() == ds.Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0)
    assert _request(top_logprobs=2).device_policy() is None and _request(logprobs=True).device_policy() is None
    assert _request(Qwen38SamplingParameters.official_non_thinking(seed=1)).device_policy() is None  # presence 1.5
    assert _request(sampling_parameters(temperature=0.8, top_p=0.9, top_k=0, seed=3)).device_policy() is None
    session = FakeDeviceSession([1])
    session.sampling.begin_request(None)  # a greedy request: the flag, no draw
    assert session.chain.log == ["policy:greedy"]
    host_loop = _request(Qwen38SamplingParameters.official_non_thinking(seed=1))
    session.sampling.begin_request(host_loop)
    assert session.chain.log == ["policy:greedy", "policy:greedy"] and host_loop.uniforms == []
    assert not host_loop.as_dict()["device_path"]
    assert set(_request().as_dict()) >= {"device_path", "draws", "first_token_rewrites", "verified_steps"}
