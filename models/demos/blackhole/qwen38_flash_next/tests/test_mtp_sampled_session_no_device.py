# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The sampled MTP pass loop of the chat session on a scripted chain (no device).

A fake device model (deterministic logits per history) and a fake drafter (the model's argmax, wrong every third
draft) stand behind the chain primitives the session calls; the chain runs both verify forms' contracts as the served
chain routes them (``mtp_enter``): a greedy request's passes run the fused form (the device's verdict inside the body,
no head readback, no host decision), a sampled request's the split form (the head's readback: argmaxes, the device's
greedy verdict, the per-row candidate rows, goes to the host's ``decide``, whose verdict the tail lands); either way the
decision's rows are committed at the next pass or at the leave.  What is pinned: the greedy pass loop still gives
the greedy stream; the sampled pass loop gives the stream the design's algorithm produces from the same rows and
draws (the host re-derivation), every emitted token has positive probability under the target's conditional at its
position, the committed sequence and the row match the 1-row loops' finish semantics; the hand-offs (no room, EOS,
the thinking budget, a hook stop) feed the pending token by a forced step and continue on ``generate_sampled``;
the admission refuses what the rows cannot bound.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from models.demos.blackhole.qwen38_flash_next.chat import EOS_TOKEN_IDS
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling_step as step
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_protocol import THINK_END_ID
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn import speculative_sampling as spec
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import VOCAB_SIZE, ZERO_EMBEDDING_TOKEN
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import (
    Qwen38CandidateFallback,
    Qwen38CandidateRow,
    Qwen38SamplingParameters,
    candidate_distribution,
    full_distribution,
)

EOS = EOS_TOKEN_IDS[0]
PEAK_LIMIT = 200_000
K = 4
THINKING = Qwen38SamplingParameters.official_thinking(seed=77)


def _logits_for(history: list[int], *, eos_at: int | None = None) -> torch.Tensor:
    """The fake target: a bf16 row determined by the history's length and last token; EOS dominates from ``eos_at``."""

    generator = torch.Generator().manual_seed(len(history) * 7919 + (history[-1] if history else 0))
    row = torch.randn(VOCAB_SIZE, generator=generator) * 2.0
    peaks = torch.randperm(PEAK_LIMIT, generator=generator)[:8]
    row[peaks] = 12.0 + torch.rand(8, generator=generator) * 6.0
    if eos_at is not None and len(history) >= eos_at:
        row[EOS] = 60.0
    return row.to(torch.bfloat16)


def _argmax(history: list[int], *, eos_at: int | None = None) -> int:
    return int(torch.argmax(_logits_for(history, eos_at=eos_at).to(torch.float32)))


def _draft_for(history: list[int], *, eos_at: int | None = None) -> int:
    """The fake drafter: the target's argmax, the runner-up every third position (a miss)."""

    logits = _logits_for(history, eos_at=eos_at).to(torch.float32)
    top = torch.topk(logits, 2).indices.tolist()
    return top[1] if len(history) % 3 == 0 else top[0]


class FakeMTPChain:
    """The traced chain's primitives over the fake model: the 1-row steps, the pass loop in both verify forms
    (``decide`` None: the fused one; a callable: the split one), logged as ``pass:<form>:<accepted>``."""

    allocated_context = 32_768
    chunk_trace_id = 1  # the chunked mode is available; prompts below CHUNK_PREFILL_MIN_ROWS take the forced path

    def __init__(self, *, sampled: bool = True, room: int = 10_000, eos_at: int | None = None) -> None:
        self.inputs: list[int] = []  # every token the device consumed (HEAD steps and committed pass rows)
        self.row: int | None = None
        self.log: list[str] = []
        self.eos_at = eos_at
        self.room = room
        self.sampling = FakeSampling(self)
        self.mtp = session_module.Qwen38ChainMTP(
            drafts=K, anchor="off", components=None, verify=None, draft=None, step_inputs=None, chunk_extension=None
        )
        self.mtp.sampled = sampled
        self.decide = None
        self.pending_rows: list[int] = []
        self.next_tokens: list[int] | None = None
        self.head_full_rows: torch.Tensor | None = None
        self.passes = 0

    # -- the 1-row primitives ----------------------------------------------------------------------------------

    def write_token_row(self, token_id: int) -> None:
        self.log.append(f"write:{token_id}")
        self.row = token_id

    def write_mtp_next_token(self, next_token_id: int | None) -> None:
        self.mtp.step_written = True

    def execute_head(self, residue: int) -> None:
        self.log.append(f"head:{residue}")
        self.inputs.append(self.row)

    def execute_tail(self, residue: int) -> None:
        self.log.append(f"tail:{residue}")
        self.mtp.step_written = False
        self.row = _argmax(self.inputs, eos_at=self.eos_at)

    def refresh_ple_row(self, token_id: int, context):
        return (token_id, 0 if context is None else context[0])

    def read_token_row(self) -> int:
        self.log.append("read_token_row")
        return self.row

    def read_token_row_nonblocking(self):
        return self.row

    @staticmethod
    def pending_value(pending) -> int:
        return pending

    def record_event(self):
        return object()

    @staticmethod
    def event_synchronize(event) -> None:
        pass

    def position(self) -> int:
        return len(self.inputs)

    @staticmethod
    def loop_guard():
        return nullcontext()

    def reset_and_seed(self, token_id: int) -> None:
        self.inputs = []
        self.row = token_id
        self.pending_rows = []
        self.mtp.chain = None
        self.log.append("reset")

    # -- the pass loop ------------------------------------------------------------------------------------------

    def mtp_pass_fits(self, position: int) -> bool:
        return position + K + 1 <= self.room

    def _pass(self, tokens: list[int]) -> mtp_v2.Qwen38TTNNMTPPassRecord:
        history = list(self.inputs)
        rows_logits = [_logits_for(history + tokens[: row + 1], eos_at=self.eos_at) for row in range(K + 1)]
        argmaxes = [int(torch.argmax(logits.to(torch.float32))) for logits in rows_logits]
        accepted = 0
        while accepted < K and argmaxes[accepted] == tokens[accepted + 1]:
            accepted += 1
        candidate_rows = torch.stack(
            [Qwen38CandidateRow.emulate(logits).to_host_row().reshape(-1) for logits in rows_logits]
        )
        self.head_full_rows = torch.stack([logits.to(torch.float32) for logits in rows_logits])
        if self.decide is None:
            # The fused verify: the device's (a, t') and its argmax lanes land in the body; no head readback.
            decision = None
            accepted_star, next_token, alignment = accepted, argmaxes[accepted], tuple(argmaxes)
            self.head_full_rows = None
        else:
            # The split verify: the head's readback to the host's decide, its verdict written back before the tail.
            head = mtp_v2.Qwen38TTNNVerifyHeadReadback(accepted, argmaxes[accepted], tuple(argmaxes), candidate_rows)
            decision = self.decide(tokens, head)
            accepted_star, next_token = decision.accepted, decision.next_token
            alignment = tuple(decision.alignment_tokens)
        self.pending_rows = tokens[: accepted_star + 1]  # committed at the next pass or at the leave
        chain_history = history + self.pending_rows + [next_token]
        drafts: list[int] = []
        for _ in range(K):
            drafts.append(_draft_for(chain_history + drafts, eos_at=self.eos_at))
        self.next_tokens = [next_token, *drafts]
        self.passes += 1
        self.log.append(f"pass:{'fused' if decision is None else 'split'}:{accepted_star}")
        return mtp_v2.Qwen38TTNNMTPPassRecord(
            index=self.passes - 1,
            position=len(history),
            tokens=tuple(tokens),
            accepted=accepted_star,
            committed=(*tokens[1 : accepted_star + 1], next_token),
            argmaxes=alignment,
            next_token=next_token,
            first_draft=drafts[0],
            finished=False,
            segments_ns={},
            decision=decision,
        )

    def mtp_enter(self, first_token: int, ple_context, *, decide=None) -> mtp_v2.Qwen38TTNNMTPPassRecord:
        if self.mtp.chain is not None:
            raise session_module.Qwen38ChatChainError("the MTP pass loop is already active")
        if decide is not None and not self.mtp.sampled:
            raise session_module.Qwen38ChatChainError("a host decision needs the split verify")
        self.decide = decide  # None: the fused traces (no host decision); a callable: the split traces
        self.mtp.chain = self
        self.log.append("enter")
        return self.mtp.record(self._pass([first_token] + [session_module.MTP_BOOTSTRAP_DRAFT_TOKEN] * K))

    def mtp_step(self) -> mtp_v2.Qwen38TTNNMTPPassRecord:
        if self.mtp.chain is None:
            raise session_module.Qwen38ChatChainError("the MTP pass loop is not active")
        self.inputs.extend(self.pending_rows)  # the commit trace: the previous pass's accepted rows
        self.pending_rows = []
        return self.mtp.record(self._pass(self.next_tokens))

    def mtp_leave(self, *, position: int, committed_rows: int):
        self.inputs.extend(self.pending_rows[:committed_rows])
        self.pending_rows = []
        self.mtp.chain = None
        if len(self.inputs) != position:
            raise session_module.Qwen38ChatChainError(f"leave at {len(self.inputs)} vs {position}")
        self.row = _argmax(self.inputs, eos_at=self.eos_at)  # the 1-row buffers rebuilt: the model's next token
        self.log.append(f"leave:{committed_rows}")
        return (self.inputs[-1], 0)

    def mtp_read_full_logits_rows(self) -> torch.Tensor:
        self.log.append("full_rows")
        return self.head_full_rows


class FakeSampling:
    def __init__(self, chain: FakeMTPChain) -> None:
        self.chain = chain

    def read_candidate_row(self) -> Qwen38CandidateRow:
        self.chain.log.append("read_row")
        return Qwen38CandidateRow.emulate(_logits_for(self.chain.inputs, eos_at=self.chain.eos_at))

    def read_full_logits(self, residue: int) -> torch.Tensor:
        self.chain.log.append(f"full_gather:{residue}")
        return _logits_for(self.chain.inputs, eos_at=self.chain.eos_at).to(torch.float32)


def _session(**chain_fields) -> tuple[session_module.Qwen38ChatSession, FakeMTPChain]:
    chain = FakeMTPChain(**chain_fields)
    return session_module.Qwen38ChatSession(chain, template=None), chain


PROMPT = [11, 22, 33, 44, 55]


def _greedy_stream(prompt: list[int], count: int) -> list[int]:
    history = list(prompt)
    for _ in range(count):
        history.append(_argmax(history))
    return history[len(prompt) :]


def _expected_sampled_stream(prompt: list[int], parameters: Qwen38SamplingParameters, count: int) -> list[int]:
    """The design's algorithm on the fake model from the request's seed: the entry token sampled from the row, then
    passes (the bootstrap's placeholder drafts first) decided by the point-mass acceptance over the rows' candidate
    distributions with the pass's histories; the first ``count`` emitted tokens."""

    generator = torch.Generator().manual_seed(parameters.seed)
    uniform = lambda: float(torch.rand((), generator=generator, dtype=torch.float32))  # noqa: E731

    def distribution(row_logits: torch.Tensor, history: list[int]):
        try:
            return candidate_distribution(
                Qwen38CandidateRow.emulate(row_logits), parameters, token_history=history, prompt_tokens=len(prompt)
            )
        except Qwen38CandidateFallback:
            return full_distribution(
                row_logits.to(torch.float32), parameters, token_history=history, prompt_tokens=len(prompt)
            )

    committed = list(prompt)
    emitted = [distribution(_logits_for(committed), committed).draw(uniform())]
    tokens = [emitted[0]] + [session_module.MTP_BOOTSTRAP_DRAFT_TOKEN] * K
    while len(emitted) < count:
        rows = [_logits_for(committed + tokens[: row + 1]) for row in range(K + 1)]
        result = spec.accept_point_mass(
            lambda row: distribution(rows[row], committed + tokens[: row + 1]), tokens[1:], uniform
        )
        committed += tokens[: result.accepted + 1]
        emitted += [*tokens[1 : result.accepted + 1], result.token]
        drafts: list[int] = []
        for _ in range(K):
            drafts.append(_draft_for(committed + [result.token] + drafts))
        tokens = [result.token, *drafts]
    return emitted[:count]


# --- the greedy pass loop runs the fused verify on the sampled-capable chain --------------------------------------------


def _pass_forms(chain: FakeMTPChain) -> set[str]:
    return {entry.split(":")[1] for entry in chain.log if entry.startswith("pass:")}


def test_greedy_request_on_the_sampled_capable_chain_runs_the_fused_verify_and_gives_the_greedy_stream() -> None:
    session, chain = _session()
    completion = session.complete(PROMPT, 14, stop_ids=())
    assert completion.token_ids == _greedy_stream(PROMPT, 14)
    assert completion.finish_reason == "length" and completion.mtp["passes"] >= 2
    # Every pass ran the fused form: no host decision, so no greedy split pass was checked (accept_checks 0; the
    # field stays in the response of a sampled-capable chain).
    assert _pass_forms(chain) == {"fused"} and chain.decide is None and chain.head_full_rows is None
    assert completion.mtp["sampled"] is True and completion.mtp["accept_checks"] == 0
    assert session.committed == PROMPT + completion.token_ids[:-1] and chain.inputs == session.committed
    assert chain.row == completion.token_ids[-1]  # length: the last token unconsumed in the row
    assert not any(entry == "read_row" for entry in chain.log)  # the greedy loop never reads the candidate row


def test_greedy_request_on_a_switch_off_chain_cannot_take_a_host_decision(expect_error) -> None:
    session, chain = _session(sampled=False)
    completion = session.complete(PROMPT, 6, stop_ids=())
    assert completion.token_ids == _greedy_stream(PROMPT, 6) and completion.mtp["sampled"] is False
    assert "accept_checks" not in completion.mtp
    with expect_error(session_module.Qwen38ChatChainError, match="host decision"):
        chain.mtp_enter(1, None, decide=lambda tokens, head: None)


# --- the sampled pass loop ----------------------------------------------------------------------------------------------


def _law_holds(prompt: list[int], tokens: list[int], parameters: Qwen38SamplingParameters, *, eos_at=None) -> None:
    """Every emitted token has positive probability under the target's conditional at its position."""

    history = list(prompt)
    for token in tokens:
        target = full_distribution(
            _logits_for(history, eos_at=eos_at).to(torch.float32),
            parameters,
            token_history=history,
            prompt_tokens=len(prompt),
        )
        assert target.probability(token) > 0, (history, token)
        history.append(token)


def test_sampled_request_drafts_and_gives_the_host_re_derived_stream() -> None:
    for seed in (77, 78, 1234):
        parameters = Qwen38SamplingParameters.official_thinking(seed=seed)
        session, chain = _session()
        request = step.Qwen38SamplingRequest(parameters)
        completion = session.complete(PROMPT, 16, stop_ids=(), sampling=request)
        assert request.mtp_drafting == "drafted" and completion.finish_reason == "length"
        assert completion.token_ids == _expected_sampled_stream(PROMPT, parameters, 16), seed
        assert _pass_forms(chain) == {"split"}  # every pass of a sampled request through the split form
        _law_holds(PROMPT, completion.token_ids, parameters)
        assert completion.mtp["passes"] == request.mtp.passes == chain.passes >= 2
        assert completion.mtp["sampled_passes"] == chain.passes and completion.mtp["accept_checks"] == 0
        # The draw ledger: every pass consumed a* + 2 or k + 1 draws, so at least two; the chain's counters agree.
        assert request.mtp.draws == chain.mtp.sampled_draws == completion.mtp["sampled_draws"] >= 2 * chain.passes
        assert len(request.mtp.acceptance_probabilities) >= chain.passes
        assert session.committed == PROMPT + completion.token_ids[:-1] and chain.inputs == session.committed
        assert chain.row == completion.token_ids[-1]  # length: the token the client saw sits in the row
        assert chain.log.count("read_row") == 1  # the entry sample; the passes read the head rows instead
        assert request.as_dict()["mtp"]["passes"] == chain.passes
        # None samples for the pass tokens, one real sample for the entry token: index-aligned with the stream.
        assert (
            len(request.samples) == len(completion.token_ids) and request.samples[0].token_id == completion.token_ids[0]
        )
        assert all(sample is None for sample in request.samples[1:])
    first = step.Qwen38SamplingRequest(Qwen38SamplingParameters.official_thinking(seed=5))
    other = step.Qwen38SamplingRequest(Qwen38SamplingParameters.official_thinking(seed=6))
    assert (
        _session()[0].complete(PROMPT, 12, stop_ids=(), sampling=first).token_ids
        != _session()[0].complete(PROMPT, 12, stop_ids=(), sampling=other).token_ids
    )
    assert (
        _greedy_stream(PROMPT, 12)
        != _session()[0].complete(PROMPT, 12, stop_ids=(), sampling=step.Qwen38SamplingRequest(THINKING)).token_ids
    )


def test_sampled_request_hands_off_to_the_one_row_loop_by_a_forced_step_when_no_pass_fits() -> None:
    session, chain = _session(room=len(PROMPT) + K + 1 + 6)  # room for the first passes only
    request = step.Qwen38SamplingRequest(THINKING)
    completion = session.complete(PROMPT, 20, stop_ids=(), sampling=request)
    assert completion.finish_reason == "length" and len(completion.token_ids) == 20
    assert request.mtp_drafting == "drafted" and 1 <= request.mtp.passes < 6
    _law_holds(PROMPT, completion.token_ids, THINKING)
    assert session.committed == PROMPT + completion.token_ids[:-1] and chain.inputs == session.committed
    leave = next(index for index, entry in enumerate(chain.log) if entry.startswith("leave:"))
    after = chain.log[leave + 1 :]
    # The pending token is fed by a forced step (a write and a HEAD) before the 1-row loop reads any candidate row:
    # never left in the row for that loop, which samples from fresh rows for the rest of the request.
    first_head = next(index for index, entry in enumerate(after) if entry.startswith("head:"))
    assert after[first_head - 1].startswith("write:") and "read_row" not in after[:first_head]
    assert "read_row" in after[first_head:] and "enter" not in after
    assert chain.row == completion.token_ids[-1]


def test_sampled_hand_off_keeps_the_prompt_length_for_the_penalties(monkeypatch) -> None:
    # The non-thinking profile penalizes the output: after the hand-off the 1-row loop must still count every token
    # the pass loop emitted as output (prompt_tokens = the prompt's length), as plain sampling does at that position.
    session, chain = _session(room=len(PROMPT) + K + 1 + 6)
    seen: list[int] = []
    choose = step.choose_token

    def recording(session_, row, request, *, tail_residue, prompt_tokens):
        seen.append(prompt_tokens)
        return choose(session_, row, request, tail_residue=tail_residue, prompt_tokens=prompt_tokens)

    monkeypatch.setattr(step, "choose_token", recording)
    request = step.Qwen38SamplingRequest(Qwen38SamplingParameters.official_non_thinking(seed=21))
    completion = session.complete(PROMPT, 20, stop_ids=(), sampling=request)
    assert request.mtp_drafting == "drafted" and len(completion.token_ids) == 20 and request.mtp.passes >= 1
    assert any(entry.startswith("leave:") for entry in chain.log)  # the hand-off happened
    assert len(seen) > 1 and set(seen) == {len(PROMPT)}  # the entry sample, then every 1-row sample after the hand-off
    _law_holds(PROMPT, completion.token_ids, request.parameters)


def test_sampled_request_with_an_exhausted_thinking_budget_forces_think_end_first_and_keeps_samples_aligned() -> None:
    session, chain = _session()
    request = step.Qwen38SamplingRequest(THINKING)
    completion = session.complete(PROMPT, 8, stop_ids=(), think_budget=0, sampling=request)
    assert completion.token_ids[0] == THINK_END_ID and len(completion.token_ids) == 8
    # The entry token was sampled before the budget check and dropped for the forced </think> (as _generate drops the
    # row's token): its sample goes with it, so the samples stay index-aligned with the stream.
    assert request.samples[0] is None and len(request.samples) == len(completion.token_ids)
    assert request.samples[1] is not None and request.samples[1].token_id == completion.token_ids[1]
    assert chain.log.count("read_row") == 2  # the dropped entry sample, then the re-sample after the forced token
    assert session.committed == PROMPT + completion.token_ids[:-1] and chain.inputs == session.committed
    _law_holds(PROMPT + completion.token_ids[:1], completion.token_ids[1:], THINKING)


def test_sampled_request_stops_at_eos_inside_a_pass_and_consumes_it() -> None:
    session, chain = _session(eos_at=len(PROMPT) + 5)
    request = step.Qwen38SamplingRequest(THINKING)
    completion = session.complete(PROMPT, 40, sampling=request)
    assert completion.finish_reason == "stop" and completion.token_ids[-1] == EOS
    assert session.committed == PROMPT + completion.token_ids and chain.inputs == session.committed
    assert request.mtp_drafting == "drafted"
    _law_holds(PROMPT, completion.token_ids, THINKING, eos_at=chain.eos_at)


def test_sampled_request_forces_the_thinking_budget_and_re_enters() -> None:
    session, chain = _session()
    request = step.Qwen38SamplingRequest(THINKING)
    completion = session.complete(PROMPT, 14, stop_ids=(), think_budget=3, sampling=request)
    assert completion.finish_reason == "length" and len(completion.token_ids) == 14
    assert completion.token_ids[3] == THINK_END_ID and request.samples[3] is None
    forced = chain.log.index(f"write:{THINK_END_ID}")
    assert chain.log[forced + 1].startswith("head:") and "enter" in chain.log[forced:]  # a forced step, then re-entry
    assert chain.log[:forced].count("enter") == 1 and "read_row" in chain.log[forced:]  # the pending token re-sampled
    assert session.committed == PROMPT + completion.token_ids[:-1] and chain.inputs == session.committed
    assert len(request.samples) == len(completion.token_ids)


def test_sampled_request_hook_stop_feeds_the_streamed_token_by_a_forced_step() -> None:
    session, chain = _session()
    request = step.Qwen38SamplingRequest(THINKING)
    polls: list[int] = []
    completion = session.complete(
        PROMPT,
        30,
        stop_ids=(),
        sampling=request,
        should_stop=lambda: (polls.append(1), "halt" if len(polls) == 4 else None)[1],
    )
    assert completion.finish_reason == "halt" and 1 <= len(completion.token_ids) < 30
    assert session.committed == PROMPT + completion.token_ids and chain.inputs == session.committed
    assert session.row_unconsumed and chain.log[-1].startswith("tail:")


def test_admission_refuses_what_the_rows_cannot_bound_and_the_one_row_loop_serves_them() -> None:
    session, chain = _session()
    for parameters, reason in (
        (
            Qwen38SamplingParameters(temperature=0.8, top_p=0.9, top_k=0, presence_penalty=0.0, seed=3),
            "refused: top_k 0",
        ),
        (
            Qwen38SamplingParameters(temperature=1.0, top_p=1.0, top_k=20, presence_penalty=-0.5, seed=3),
            "refused: penalty raises logits",
        ),
    ):
        request = step.Qwen38SamplingRequest(parameters)
        completion = session.complete(PROMPT, 6, stop_ids=(), sampling=request)
        assert request.mtp_drafting == reason and completion.mtp is None and len(completion.token_ids) == 6
        assert "enter" not in chain.log and chain.log.count("read_row") == 6
        assert request.as_dict()["mtp_drafting"] == reason and request.as_dict()["mtp"] is None
        chain.log.clear()
    logprobs = step.Qwen38SamplingRequest(THINKING, logprobs=True, top_logprobs=2)
    completion = session.complete(PROMPT, 4, stop_ids=(), sampling=logprobs)
    assert logprobs.mtp_drafting == "refused: logprobs" and all(sample is not None for sample in logprobs.samples)
    session_off, _ = _session(sampled=False)
    request = step.Qwen38SamplingRequest(THINKING)
    completion = session_off.complete(PROMPT, 4, stop_ids=(), sampling=request)
    assert request.mtp_drafting == "refused: QWEN38_MTP_SAMPLED off" and completion.mtp is None
    # The teacher-forced prefill mode keeps every sampled request on the 1-row loop.
    request = step.Qwen38SamplingRequest(THINKING)
    completion = session.complete(PROMPT, 4, stop_ids=(), sampling=request, prefill_mode="teacher_forced")
    assert request.mtp_drafting == "refused: teacher-forced prefill" and completion.mtp is None
    request = step.Qwen38SamplingRequest(THINKING)
    completion = session.complete(PROMPT, 4, stop_ids=(), sampling=request, speculative=False)
    assert request.mtp_drafting == "refused: not speculative" and completion.mtp is None


def test_chain_mtp_summary_carries_the_split_counters() -> None:
    mtp = session_module.Qwen38ChainMTP(
        drafts=4, anchor="off", components=None, verify=None, draft=None, step_inputs=None, chunk_extension=None
    )
    assert set(mtp.summary()) == {"k", "anchor", "sampled", "passes", "accepted_drafts", "tokens_per_pass"}
    mtp.sampled = True
    # A greedy split pass decided on the host (decide_greedy's record): a diagnostic's, not the served path's, whose
    # greedy requests run the fused traces; the counter still counts it.
    greedy = mtp_v2.Qwen38TTNNVerifyDecision(2, 9, (7, 8, 9, 1, 2), {"accept_checks": 1})
    sampled = mtp_v2.Qwen38TTNNVerifyDecision(
        1,
        5,
        (7, 5, ZERO_EMBEDDING_TOKEN, ZERO_EMBEDDING_TOKEN, ZERO_EMBEDDING_TOKEN),
        {"sampled": True, "draws": 3, "fallbacks": 1, "resampled": True, "acceptance_probabilities": (0.9, 0.2)},
    )
    record = lambda decision: mtp.record(  # noqa: E731
        mtp_v2.Qwen38TTNNMTPPassRecord(0, 0, (1, 7, 8, 9, 1), decision.accepted, (), (), 0, 0, False, {}, decision)
    )
    record(greedy)
    snapshot = mtp.counters()  # a request starting here
    record(sampled)
    summary = mtp.summary()
    assert (summary["passes"], summary["accepted_drafts"], summary["accept_checks"]) == (2, 3, 1)
    assert (
        summary["sampled_passes"],
        summary["sampled_accepted_drafts"],
        summary["sampled_draws"],
        summary["sampled_fallbacks"],
    ) == (1, 1, 3, 1)
    assert summary["sampled_tokens_per_pass"] == 2.0 and summary["tokens_per_pass"] == 2.5
    # Since the snapshot: the sampled pass alone, every field.
    since = mtp.summary(since=snapshot)
    assert set(since) == set(summary) == MTP_RESPONSE_KEYS
    assert (since["passes"], since["accepted_drafts"], since["accept_checks"]) == (1, 1, 0)
    assert (since["sampled_passes"], since["sampled_accepted_drafts"], since["sampled_draws"]) == (1, 1, 3)
    assert since["sampled_fallbacks"] == 1 and since["tokens_per_pass"] == since["sampled_tokens_per_pass"] == 2.0
    assert set(snapshot) == set(session_module.Qwen38ChainMTP.COUNTERS) and snapshot["passes"] == 1
    assert mtp.captured_trace_ids() == []
    # A device-decided pass (the third form): a sampled pass whose draws are the uniforms the program consumed, the
    # device never falling back; its guard deviations counted on their own.
    device = mtp_v2.Qwen38TTNNMTPPassRecord(
        2, 0, (1, 7, 8, 9, 1), 1, (), (), 0, 0, False, {}, None, tuple(range(16)), "device-theta", 2, None
    )
    before = mtp.counters()
    mtp.record(device)
    added = {name: value - before[name] for name, value in mtp.counters().items()}
    assert added == {
        "passes": 1,
        "accepted_drafts": 1,
        "accept_checks": 0,
        "sampled_passes": 1,
        "sampled_accepted_drafts": 1,
        "sampled_draws": 3,  # a* + 2 = u_0, u_1 and v
        "sampled_fallbacks": 0,
        "device_accept_passes": 1,
        "device_accept_guard_deviations": 2,
    }
    every = mtp_v2.Qwen38TTNNMTPPassRecord(
        3, 0, (1, 7, 8, 9, 1), 4, (), (), 0, 0, False, {}, None, tuple(range(16)), "device-theta", 0, None
    )
    mtp.record(every)
    assert mtp.sampled_draws - before["sampled_draws"] == 3 + 5  # every draft accepted: k + 1 draws
    summary = mtp.summary(since=before)
    assert (summary["device_accept"], summary["device_accept_passes"], summary["device_accept_guard_deviations"]) == (
        False,
        2,
        2,
    )


def test_chain_mtp_captured_trace_ids_cover_both_forms_and_the_shared_commit_once() -> None:
    mtp = session_module.Qwen38ChainMTP(
        drafts=4, anchor="off", components=None, verify=None, draft=None, step_inputs=None, chunk_extension=None
    )
    assert mtp.captured_trace_ids() == []
    mtp.traces = mtp_v2.Qwen38TTNNMTPTraces(verify_first=1, draft=2, commit=3)
    assert mtp.captured_trace_ids() == [1, 3, 2]  # the switch off: the fused form alone
    mtp.sampled = True
    mtp.split_traces = mtp_v2.Qwen38TTNNMTPTraces(verify_first=None, draft=6, commit=3, verify_head=4, verify_tail=5)
    # The switch on: the fused form's traces first, then the split form's, the shared commit once (close() releases
    # every id of this list exactly once).
    assert mtp.captured_trace_ids() == [1, 3, 2, 4, 5, 6]
    assert len(set(mtp.captured_trace_ids())) == 6
    # The device acceptance beside them: the third form's trace and its draft, the commit still once.
    mtp.device_accept = True
    mtp.sampled_traces = mtp_v2.Qwen38TTNNMTPTraces(verify_first=None, draft=8, commit=3, verify_sampled=7)
    assert mtp.captured_trace_ids() == [1, 3, 2, 4, 5, 6, 7, 8]


def test_traced_chain_mtp_enter_routes_greedy_to_the_fused_traces_and_sampled_to_the_split_ones(
    expect_error, monkeypatch
) -> None:
    """The routing point of the hardware chain (its device calls replaced): a greedy request's pass loop is built on
    the fused traces with no head output and no host decision, a sampled request's on the split traces with the
    head output and its ``decide``; the eager switch in happens either way."""

    built: list[dict] = []
    entered: list[dict] = []

    class RecordingChain:
        def __init__(
            self,
            model,
            verify,
            draft,
            traces,
            verify_output,
            *,
            replay,
            position,
            enqueue,
            head_output,
            decide,
            before_verify_sampled=None,
            record_candidate_rows=False,
            early_reader=None,
        ):
            built.append(
                dict(
                    model=model,
                    verify=verify,
                    draft=draft,
                    traces=traces,
                    verify_output=verify_output,
                    replay=replay,
                    position=position,
                    enqueue=enqueue,
                    head_output=head_output,
                    decide=decide,
                    before_verify_sampled=before_verify_sampled,
                    record_candidate_rows=record_candidate_rows,
                    early_reader=early_reader,
                )
            )

        def bootstrap(self, tokens):
            return SimpleNamespace(
                accepted=0, decision=None, tokens=tuple(tokens), arithmetic=None, guard_deviations=None
            )

    monkeypatch.setattr(mtp_v2, "Qwen38TTNNMTPChain", RecordingChain)
    monkeypatch.setattr(mtp_v2, "enter_verify_mode", lambda *args, **kwargs: entered.append(kwargs))
    fused = mtp_v2.Qwen38TTNNMTPTraces(verify_first=1, draft=2, commit=3)
    split = mtp_v2.Qwen38TTNNMTPTraces(verify_first=None, draft=6, commit=3, verify_head=4, verify_tail=5)
    mtp = session_module.Qwen38ChainMTP(
        drafts=K,
        anchor="off",
        components=None,
        verify="verify",
        draft="draft",
        step_inputs=None,
        chunk_extension=None,
        traces=fused,
        verify_output="fused-row",
        sampled=True,
        split_traces=split,
        split_verify_output="tail-row",
        head_output="head",
    )
    chain = object.__new__(session_module.Qwen38TracedChain)
    chain.mtp = mtp
    chain.built_target = SimpleNamespace(model="model")
    chain.state = SimpleNamespace(position=SimpleNamespace(read=lambda: 7))
    record = chain.mtp_enter(11, (3, 4))
    assert record.tokens == (11, *[session_module.MTP_BOOTSTRAP_DRAFT_TOKEN] * K) and mtp.passes == 1
    assert entered == [{"position": 7, "ple_context": (3, 4)}] and isinstance(mtp.chain, RecordingChain)
    greedy = built[-1]
    assert greedy["traces"] is fused and greedy["verify_output"] == "fused-row" and greedy["head_output"] is None
    assert greedy["decide"] is mtp_v2.decide_greedy and greedy["position"] == 7
    assert (greedy["model"], greedy["verify"], greedy["draft"]) == ("model", "verify", "draft")
    assert greedy["replay"] == chain._replay and greedy["enqueue"] == chain._enqueue
    with expect_error(session_module.Qwen38ChatChainError, match="already active"):
        chain.mtp_enter(11, None)
    mtp.chain = None
    decide = lambda tokens, head: None  # noqa: E731
    chain.mtp_enter(11, None, decide=decide)
    sampled = built[-1]
    assert sampled["traces"] is split and sampled["verify_output"] == "tail-row" and sampled["head_output"] == "head"
    assert sampled["decide"] is decide and mtp.passes == 2 and len(entered) == 2
    # A host decision needs the split form: refused with the switch off, and without the split traces.
    mtp.chain = None
    mtp.sampled = False
    with expect_error(session_module.Qwen38ChatChainError, match="host decision"):
        chain.mtp_enter(11, None, decide=decide)
    mtp.sampled, mtp.split_traces = True, None
    with expect_error(session_module.Qwen38ChatChainError, match="host decision"):
        chain.mtp_enter(11, None, decide=decide)
    assert len(built) == 2 and len(entered) == 2  # neither refusal reached the switch or built a chain
    chain.mtp_enter(11, None)  # the greedy route needs no split traces
    assert built[-1]["traces"] is fused and built[-1]["head_output"] is None


PROMPT_B = [12, 23, 34, 45, 56, 67]
PROMPT_C = [13, 24, 35]
# The response's qwen38.mtp object on a split-verify chain (the health object has the same keys, cumulative).
MTP_RESPONSE_KEYS = {
    "k",
    "anchor",
    "sampled",
    "passes",
    "accepted_drafts",
    "tokens_per_pass",
    "accept_checks",
    "sampled_passes",
    "sampled_accepted_drafts",
    "sampled_tokens_per_pass",
    "sampled_draws",
    "sampled_fallbacks",
    "device_accept",
    "device_accept_passes",
    "device_accept_guard_deviations",
}


def test_consecutive_requests_report_their_own_counters_and_the_chain_summary_the_totals() -> None:
    """Every field of a response's ``qwen38.mtp`` counts that request alone, the split counters included (they
    were chain-cumulative once: a second request reported the first's accept_checks and sampled draws on top of
    its own); the chain's summary (``/health.mtp``) carries the totals over the requests."""

    session, chain = _session()
    greedy = session.complete(PROMPT, 14, stop_ids=())
    request = step.Qwen38SamplingRequest(THINKING)
    sampled = session.complete(PROMPT_B, 16, stop_ids=(), sampling=request)
    again = session.complete(PROMPT_C, 10, stop_ids=())
    assert set(greedy.mtp) == set(sampled.mtp) == set(again.mtp) == MTP_RESPONSE_KEYS
    assert greedy.mtp["passes"] >= 2 and sampled.mtp["passes"] >= 2 and again.mtp["passes"] >= 1
    assert greedy.mtp["passes"] + sampled.mtp["passes"] + again.mtp["passes"] == chain.passes
    # The greedy requests: every pass through the fused form (no host decision to check), nothing sampled.
    for completion in (greedy, again):
        assert completion.mtp["accept_checks"] == 0 and completion.mtp["passes"] >= 1
        assert completion.mtp["sampled_passes"] == completion.mtp["sampled_accepted_drafts"] == 0
        assert completion.mtp["sampled_draws"] == completion.mtp["sampled_fallbacks"] == 0
        assert completion.mtp["sampled_tokens_per_pass"] is None
    # The sampled request: its own passes, accepted drafts, draws and fallbacks (= the request's ledger), no check.
    assert sampled.mtp["passes"] == sampled.mtp["sampled_passes"] == request.mtp.passes
    assert sampled.mtp["accepted_drafts"] == sampled.mtp["sampled_accepted_drafts"] == request.mtp.accepted_drafts
    assert sampled.mtp["sampled_draws"] == request.mtp.draws >= 2 * request.mtp.passes
    assert sampled.mtp["sampled_fallbacks"] == request.mtp.fallbacks and sampled.mtp["accept_checks"] == 0
    assert (
        sampled.mtp["tokens_per_pass"]
        == sampled.mtp["sampled_tokens_per_pass"]
        == request.mtp.as_dict()["tokens_per_pass"]
        == round((request.mtp.passes + request.mtp.accepted_drafts) / request.mtp.passes, 4)
    )
    assert _pass_forms(chain) == {"fused", "split"}  # both forms served on one chain
    # The chain's summary: the totals, field by field.
    health = chain.mtp.summary()
    assert set(health) == MTP_RESPONSE_KEYS and health["sampled"] is True and health["passes"] == chain.passes
    for name in session_module.Qwen38ChainMTP.COUNTERS:
        assert health[name] == greedy.mtp[name] + sampled.mtp[name] + again.mtp[name], name
    assert health["tokens_per_pass"] == round((health["passes"] + health["accepted_drafts"]) / health["passes"], 4)
