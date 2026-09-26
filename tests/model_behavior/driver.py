# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Scripted requests, independent of the model, device library, and serving framework."""

import math
from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.0
    top_k: int = 32
    top_p: float = 1.0
    seed: int | None = None
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    enable_log_probs: bool = False


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt: str
    sampling: Sampling = field(default_factory=Sampling)
    max_tokens: int = 12
    prefill_chunk_ends: tuple[int, ...] = ()


@dataclass(frozen=True)
class Sample:
    token_id: int
    logprob: float
    reference_logprob: float


@dataclass
class RequestState:
    request: Request
    slot: int
    prompt_tokens: tuple[int, ...]
    output_tokens: list[int] = field(default_factory=list)
    samples: list[Sample] = field(default_factory=list)

    @property
    def position(self) -> int:
        """Position of the last sampled token, which is the next decode input."""
        return len(self.prompt_tokens) + len(self.output_tokens) - 1


class ModelAdapter(Protocol):
    capacity: int
    vocab_size: int
    duplicate_seed_policy: str
    long_context_slots: tuple[int, ...]
    prefill_events: list[dict]
    stochastic_prefill: bool
    logprob_phases: tuple[str, ...]
    supports_chunked_prefill: bool

    def encode(self, prompt: str) -> tuple[int, ...]:
        ...

    def decode_tokens(self, tokens: list[int]) -> str:
        ...

    def prefill(self, admitted: list[RequestState]) -> dict[int, int | Sample]:
        """Return the first token for each admitted physical slot."""
        ...

    def decode(self, active: list[RequestState], *, reset_batch: bool) -> dict[int, int | Sample]:
        """Return one token per active slot; histories include the last decode input."""
        ...

    def describe(self) -> dict:
        ...


class RequestDriver:
    """Explicit admission and fixed token budgets; no timing, retries, or EOS scheduling.

    Completion notifies the optional public request-release hook, as serving does.
    Adapters receive the changed layout on the next decode, and new requests use
    normal prefill on the same KV pages. No private model state is reset here.
    """

    def __init__(self, adapter: ModelAdapter):
        self.adapter = adapter
        self.active: dict[int, RequestState] = {}
        self.requests: dict[str, RequestState] = {}
        self.events: list[dict] = []
        self.checks: list[dict] = []
        self._layout_changed = False

    def admit(self, placements: list[tuple[int, Request]]) -> None:
        if not placements:
            raise ValueError("An admission must contain at least one request")
        slots = [slot for slot, _ in placements]
        ids = [request.request_id for _, request in placements]
        if len(set(slots)) != len(slots) or len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate slots or request IDs in admission: {slots=}, {ids=}")
        admitted = []
        for slot, request in placements:
            if not 0 <= slot < self.adapter.capacity or slot in self.active:
                raise ValueError(f"Slot {slot} is unavailable; active slots: {sorted(self.active)}")
            if request.request_id in self.requests or type(request.max_tokens) is not int or request.max_tokens < 1:
                raise ValueError(f"Request needs a new ID and a positive token budget: {request}")
            tokens = tuple(self.adapter.encode(request.prompt))
            if not tokens:
                raise ValueError(f"Empty encoded prompt for request {request.request_id}")
            admitted.append(RequestState(request, slot, tokens))
        for state in admitted:
            self.active[state.slot] = state
            self.requests[state.request.request_id] = state
        self._layout_changed = True
        self._record(admitted, self.adapter.prefill(admitted), phase="prefill")

    def step(self) -> None:
        if not self.active:
            raise ValueError("Cannot decode an empty batch")
        active = [self.active[slot] for slot in sorted(self.active)]
        reset_batch = self._layout_changed
        tokens = self.adapter.decode(active, reset_batch=reset_batch)
        self._layout_changed = False
        self._record(active, tokens, phase="decode", reset_batch=reset_batch)

    def drain(self) -> None:
        while self.active:
            self.step()

    def _record(self, states, tokens, *, phase, reset_batch=False):
        expected = {state.slot for state in states}
        if set(tokens) != expected:
            raise ValueError(f"{phase} returned slots {set(tokens)}; expected {expected}")
        for state in states:
            slot = state.slot
            sample = tokens[slot]
            if state.request.sampling.enable_log_probs and phase in getattr(
                self.adapter, "logprob_phases", ("prefill", "decode")
            ):
                if not isinstance(sample, Sample):
                    raise ValueError(f"Missing sampled-token logprob at {slot=} during {phase}")
                if any(not math.isfinite(value) or value > 0 for value in (sample.logprob, sample.reference_logprob)):
                    raise ValueError(f"Invalid sampled-token logprob {sample!r} at {slot=} during {phase}")
            token = sample.token_id if isinstance(sample, Sample) else sample
            if type(token) is not int or not 0 <= token < self.adapter.vocab_size:
                raise ValueError(f"Invalid sampled token {token!r} at {slot=} during {phase}")
        for state in states:
            sample = tokens[state.slot]
            token = sample.token_id if isinstance(sample, Sample) else sample
            if isinstance(sample, Sample):
                state.samples.append(sample)
            self.events.append(
                dict(
                    request_id=state.request.request_id,
                    slot=state.slot,
                    phase=phase,
                    token_index=len(state.output_tokens),
                    position=len(state.prompt_tokens) + len(state.output_tokens),
                    token_id=token,
                    reset_batch=reset_batch,
                )
            )
            state.output_tokens.append(token)
            if len(state.output_tokens) == state.request.max_tokens:
                release = getattr(self.adapter, "release_request", None)
                if callable(release):
                    release(state.slot)
                del self.active[state.slot]
                self._layout_changed = True

    def report(self) -> dict:
        return {
            "requests": {key: asdict(state) for key, state in self.requests.items()},
            "events": self.events,
            "checks": self.checks,
        }


def run_batch(driver: RequestDriver, placements: list[tuple[int, Request]]) -> list[RequestState]:
    """Run a complete batch and return its states in admission order."""
    if driver.active:
        raise ValueError("A complete-batch comparison requires the previous batch to finish")
    driver.admit(placements)
    driver.drain()
    return [driver.requests[request.request_id] for _, request in placements]


def assert_varied_tokens(states: list[RequestState], *, first_token: bool = False) -> None:
    """Require variation for repeated instances of the SAME prompt/configuration.

    Seed values may differ. Comparing unrelated prompts would let an ignored
    sampling parameter pass merely because the model responds to its input.
    """
    if len(states) < 2 or len({state.prompt_tokens for state in states}) != 1:
        raise ValueError("Diversity requires at least two requests with identical prompt tokens")
    for state in states:
        if len(state.output_tokens) != state.request.max_tokens:
            raise AssertionError(f"Incomplete output for {state.request.request_id}")
    outputs = [tuple(state.output_tokens[:1] if first_token else state.output_tokens) for state in states]
    phase = "prefill token" if first_token else "generated sequence"
    assert len(set(outputs)) > 1, f"No {phase} diversity across {len(states)} requests: {outputs}"


def assert_same_tokens(first: RequestState, second: RequestState) -> None:
    """Strict comparison, including lengths; never infer a near-tie from text."""
    a, b = first.output_tokens, second.output_tokens
    if a == b and len(a) == first.request.max_tokens and len(b) == second.request.max_tokens:
        return
    index = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    phase = "prefill" if index == 0 else "decode"
    raise AssertionError(
        f"{first.request.request_id} vs {second.request.request_id}: first difference/incomplete output "
        f"at generated token {index} ({phase}); slots {first.slot}/{second.slot}; "
        f"prompt lengths {len(first.prompt_tokens)}/{len(second.prompt_tokens)}; "
        f"sampling {first.request.sampling}/{second.request.sampling}\n"
        f"tokens: {a}\nversus: {b}"
    )
