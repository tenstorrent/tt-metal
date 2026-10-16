# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sampled decode steps for the 1-row chain: the request mapping, the chain additions, the step order, the discriminator.

The greedy loop of ``tools/qwen38_chat_session.py`` (``_generate``) stays as it is: ``temperature 0``
never reads the candidate row and is bitwise today's chain.  A sampled step is a forced step whose
token the host chooses after reading the candidate row that TAIL(t-1) wrote (the ordering F2 of
``REFRESH-B5B-TRACE-SPLIT-SPEC-20260902.md`` section 4.4 with the forced-step write in the middle)::

    D   row = sampling.read_candidate_row()                    # blocking: completes TAIL(t-1); the device idles
        x_t = sample(row)
    A'  chain.write_token_row(x_t)                             # overwrites the device's greedy id
    C   chain.execute_head(residue)
    F   ple_context = chain.refresh_ple_row(x_t, ple_context)  # hidden under HEAD(t) as today
    G   chain.execute_tail(residue)

The read is blocking on purpose: nothing could overlap it (HEAD(t) needs x_t), and the blocking read is
the cheaper wake-up (61-70 us vs the evented 71-79 us in the epilogue micro-test).  The price is the
lost HEAD overlap of the token read (the exposed segment the discriminator measures: read wake-up, parse
and sample, token write, HEAD launch); the PLE refresh still hides under HEAD.  Finish semantics are the greedy loop's: EOS is consumed, ``max_tokens``
leaves the last token unconsumed (written into the token row, so the row holds the token the client saw),
a token at or above the tokenizer size ends with ``error``, the ``should_stop`` hook ends the request
between steps, and a thinking budget forces ``</think>`` through the session's forced step.  A pure
continuation of a sampled request (nothing to prefill, the row unconsumed) samples that position again.

What the chain gains (``Qwen38SamplingChainExtension``): the epilogue constants and readback row, the
epilogue that runs the greedy resolve unchanged and then ``Qwen38TTNNLMHead.sampling_candidates``, the
warm-pass check of the row against the eager full gather, and two primitives (the blocking row read, the
fallback's eager full-vocabulary gather of the residue's trace logits).

Request fields (``parameters_from_request``; ``extra_body`` keys are merged by the server first):

    field                                    absent                                       notes
    (no sampling field at all)               the greedy loop, bitwise                     the default request on any server
    temperature                              card profile keyed on enable_thinking        0 = the greedy loop, bitwise
                                             (thinking 1.0/0.95/20/0, instruct 0.7/0.8/20/1.5)
                                             when another sampling field is named
    greedy                                   false                                        true = the greedy loop (temperature must be absent or 0)
    top_p, top_k, min_p                      1.0 / 20 / 0 (profile values when             top_k > 32 refused (400);
                                             temperature is absent too)                   top_k 0 = full-vocabulary fallback
    presence_penalty, frequency_penalty      0 / 0 (instruct profile: presence 1.5)        OpenAI additive semantics over the
                                                                                          request's output (not the prompt)
    repetition_penalty                       1.0                                          transformers rule over prompt + output
    seed                                     the server's draw, echoed in the response    one torch.Generator per request
    logprobs, top_logprobs                   false / 0                                    log-softmax over the read candidates (above
                                                                                          the full value by -log of their mass);
                                                                                          sampled requests only
    n                                        1                                            n != 1 refused (one traced chain)

MTP drafting for sampled requests (the default on an ``--mtp --sampling`` server; ``QWEN38_MTP_SAMPLED=0`` turns it
off): the pass loop of ``tools/qwen38_chat_session.py`` runs the split verify and asks :func:`accept_pass` for the
verdict on every pass, the point-mass acceptance (``ttnn/speculative_sampling.py``) over the k + 1 rows' candidate
distributions (the same processors as :func:`choose_token`, the row's history the committed stream plus the pass's
earlier rows; built for every row at once by ``sampling.candidate_distributions``, bitwise the per-row function); a
row the candidate guard cannot bound is sampled over the eagerly gathered verify logits.
:func:`drafting_admission` names the requests the pass loop does not serve (``top_k`` 0, a boosting penalty, logprobs,
the device-sampler loop): they take the 1-row loop above with the reason in ``qwen38.sampling.mtp_drafting``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from statistics import median
from typing import Any

import torch
from ttnn.tools.trace_allocation_tracker import acknowledge_corruptible

import ttnn
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_protocol import THINK_END_ID
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_mtp_device_accept import ARITHMETIC_HOST
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_mtp_device_accept import SWITCH as DEVICE_ACCEPT_SWITCH
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_mtp_device_accept import Qwen38DeviceAcceptance
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_mtp_device_accept import (
    for_chain as device_acceptance_for_chain,
)
from models.demos.blackhole.qwen38_flash_next.tools.resident_decode import (
    SINGLE_TRACE_RESIDUE_CLASS_TRACES as RESIDUE_CLASSES,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import fused, mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn.device_sampler import (
    Qwen38DeviceSamplerPolicy,
    Qwen38TTNNDeviceSamplerConstants,
    candidate_row_lanes,
    device_sampler_reference,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    SAMPLING_CANDIDATE_ROW_SHAPE,
    ZERO_EMBEDDING_TOKEN,
    Qwen38ShardedLogits,
    Qwen38TTNNLMHead,
    Qwen38TTNNSamplingCandidateConstants,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import (
    CANDIDATE_TOP_K_LIMIT,
    MAX_SEED,
    MAX_TOP_LOGPROBS,
    Qwen38CandidateFallback,
    Qwen38CandidateRow,
    Qwen38CandidateRows,
    Qwen38CandidateSample,
    Qwen38RowDistribution,
    Qwen38SamplingParameters,
    UniformStream,
    candidate_distributions,
    full_distribution,
    sample_candidates,
    sample_full_vocabulary,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.speculative_sampling import (
    Qwen38SpeculativeAcceptance,
    accept_point_mass,
)

SAMPLING_REQUEST_FIELDS = (
    "temperature",
    "greedy",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "seed",
    "logprobs",
    "top_logprobs",
    "n",
)
_PROFILE_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "min_p",
    "frequency_penalty",
    "repetition_penalty",
)
# The chain discriminator: the seeded card-profile stream, the row check over the four residue classes, the period bar.
DISCRIMINATOR_TOKENS = 128
DISCRIMINATOR_SEED = 20260904
DISCRIMINATOR_ROW_TOKENS = 33
DISCRIMINATOR_PERIOD_TARGET_MS = 50.4
DEVICE_PERIOD_NOISE_MS = 0.25  # the device loop's bar: within this of the greedy loop's period on the same chain


class Qwen38SamplingRequestError(ValueError):
    """A sampling field the server refuses with 400; the message starts with the field's name."""


# -- the request mapping -------------------------------------------------------------------------------------


def parameters_from_request(
    document: Mapping[str, Any], *, enable_thinking: bool, seed: int
) -> Qwen38SamplingParameters | None:
    """The request's sampling fields as one validated policy (the table in the module docstring); ``None`` = greedy.

    A request naming no sampling field, ``temperature 0`` and ``greedy: true`` take the greedy
    loop (``None``), so the default request is the bitwise greedy stream on a sampling server
    too; ``greedy`` with a positive temperature is a contradiction and refused.
    """

    def number(name: str, default: Real, *, integer: bool = False) -> Real:
        value = document.get(name)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, Real) or (integer and not isinstance(value, int)):
            kind = "an integer" if integer else "a number"
            raise Qwen38SamplingRequestError(f"{name} must be {kind}, got {value!r}")
        return value

    if number("n", 1, integer=True) != 1:
        raise Qwen38SamplingRequestError("n must be 1: the server runs one traced chain")
    greedy = document.get("greedy", False)
    if type(greedy) is not bool:
        raise Qwen38SamplingRequestError(f"greedy must be a boolean, got {greedy!r}")
    temperature = number("temperature", None)
    if temperature is not None and temperature < 0:
        raise Qwen38SamplingRequestError(f"temperature must be nonnegative, got {temperature}")
    if greedy and temperature:
        raise Qwen38SamplingRequestError(f"greedy is true but temperature is {temperature}: greedy needs 0 or none")
    if greedy or (temperature is not None and temperature == 0):
        return None
    if all(document.get(name) is None for name in (*_PROFILE_FIELDS, "seed")):
        return None  # nothing asked for sampling: the greedy loop, whatever the server captured
    request_seed = number("seed", seed, integer=True)
    if not 0 <= request_seed <= MAX_SEED:
        raise Qwen38SamplingRequestError(f"seed must be in [0,{MAX_SEED}], got {request_seed}")
    profile = (
        Qwen38SamplingParameters.official_thinking(seed=request_seed)
        if enable_thinking
        else Qwen38SamplingParameters.official_non_thinking(seed=request_seed)
    )
    if temperature is None:
        if all(document.get(name) is None for name in _PROFILE_FIELDS):
            return profile  # a seed alone: the card profile, seeded
        defaults = {name: getattr(profile, name) for name in _PROFILE_FIELDS}
    else:
        defaults = {
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": 20,
            "presence_penalty": 0.0,
            "min_p": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
        }
    fields = {name: number(name, default, integer=name == "top_k") for name, default in defaults.items()}
    if fields["top_k"] > CANDIDATE_TOP_K_LIMIT:
        raise Qwen38SamplingRequestError(f"top_k must be at most {CANDIDATE_TOP_K_LIMIT}, got {fields['top_k']}")
    try:
        return Qwen38SamplingParameters(seed=request_seed, **fields)
    except (TypeError, ValueError) as error:
        raise Qwen38SamplingRequestError(str(error)) from error


def logprobs_from_request(document: Mapping[str, Any]) -> tuple[bool, int]:
    """``(logprobs, top_logprobs)``: OpenAI's boolean and the 0..20 alternatives count."""

    logprobs = document.get("logprobs", False)
    if type(logprobs) is not bool:
        raise Qwen38SamplingRequestError("logprobs must be a boolean")
    top = document.get("top_logprobs", 0)
    if top is None:
        top = 0
    if isinstance(top, bool) or type(top) is not int or not 0 <= top <= MAX_TOP_LOGPROBS:
        raise Qwen38SamplingRequestError(f"top_logprobs must be an integer in [0,{MAX_TOP_LOGPROBS}], got {top!r}")
    if top and not logprobs:
        raise Qwen38SamplingRequestError("top_logprobs requires logprobs: true")
    return logprobs, top


def logprobs_content_item(
    sample: Qwen38CandidateSample | None, token_id: int, decode: Callable[[int], str]
) -> dict[str, Any]:
    """One ``choices[].logprobs.content[]`` item (OpenAI shape); a forced token (no sample) carries ``logprob`` null."""

    def item(token: int, logprob: float | None) -> dict[str, Any]:
        text = decode(token)
        return {"token": text, "logprob": logprob, "bytes": list(text.encode("utf-8"))}

    if sample is None:
        return {**item(token_id, None), "top_logprobs": []}
    document = item(sample.token_id, sample.logprob)
    document["top_logprobs"] = [item(token, logprob) for token, logprob in sample.top_logprobs]
    return document


# -- the chain additions -------------------------------------------------------------------------------------


WARM_POLICY = Qwen38DeviceSamplerPolicy(temperature=1.0, top_k=20, top_p=0.95, min_p=0.0)
WARM_UNIFORM = 0.37500011920928955  # 6291458 * 2**-24


class Qwen38SamplingChainExtension:
    """What a sampling chain adds to the traced chain: the epilogue, its warm-pass check and three per-step primitives.

    ``device_sampler`` adds the on-device sampler (``ttnn/device_sampler.py``) after the candidate row: TAIL then
    writes the sampled token (or the greedy one, by the request's flag) into the token row itself, and a sampled
    request runs the greedy loop plus one draw write per step (:func:`generate_sampled_on_device`).
    """

    def __init__(
        self,
        lm_head: Qwen38TTNNLMHead,
        mesh: Any,
        *,
        device_sampler: bool = False,
        device_accept: Qwen38DeviceAcceptance | None = None,
    ) -> None:
        self.lm_head = lm_head
        # QWEN38_MTP_DEVICE_ACCEPT: the MTP pass loop's on-device acceptance (its constants, admission and per-pass
        # uniforms live in tools/qwen38_mtp_device_accept.py); None keeps the host-decided pass.
        self.device_accept = device_accept
        self.mesh = mesh
        self.constants = Qwen38TTNNSamplingCandidateConstants.build(mesh, lm_head.mesh_contract)
        # The sampler: the one-program fused kernel when the registry serves it (its constants carry the presence
        # history), else the composite on its constants; both take (row, greedy row, constants).
        self.sample = fused.resolve(fused.sampler_tail.NAME)
        # QWEN38_FUSED=candidate_row: the scan's shard row feeds the candidate row's gather (greedy_tail on)
        self.candidate_row = fused.enabled("candidate_row")
        fused_sampler = self.sample is fused.kernel(fused.sampler_tail.NAME).fused
        if not device_sampler:
            self.sampler = None
        elif fused_sampler:
            self.sampler = fused.sampler_tail.Qwen38TTNNSamplerTailConstants.build(mesh, lm_head.mesh_contract)
        else:
            self.sampler = Qwen38TTNNDeviceSamplerConstants.build(mesh, lm_head.mesh_contract)
        self.presence_on_device = device_sampler and fused_sampler
        self.trace_logits: list[Qwen38ShardedLogits] = []  # TAIL residue r's logits (trace-stable), for the fallback
        self.trace_rows: list[Any] = []

    def _epilogue(self, logits: Qwen38ShardedLogits, token_row_io: Any) -> tuple[Any, Any, Any, Any]:
        """TAIL's last ops, eager in the warm pass and captured in the trace, one function so the capture asks only for
        programs the warm compiled (a cache miss inside a trace capture is fatal, and a program's cache key is its
        compile-time form, not its buffer addresses): the greedy path exactly as today (its resolve into the resident
        row, greedy_tail's ``copy_into`` is a compile-time argument), the candidate row into the readback buffer, and
        with the device sampler the sampler on the row and the resolve's own row, its token row (greedy or sampled, by
        the flag) copied into the resident row.  Returns ``(candidates, row, greedy_row, token_row)``; without the
        sampler the token row is the greedy row."""

        if self.candidate_row:
            candidates = self.lm_head.greedy_candidates(logits, candidate_row=self.constants)
        else:
            candidates = self.lm_head.greedy_candidates(logits)
        greedy_row = self.lm_head.resolve_greedy_on_device(candidates, into=token_row_io)
        row = self.lm_head.sampling_candidates(logits, self.constants, candidates=candidates)
        if self.sampler is None:
            return candidates, row, greedy_row, greedy_row
        token_row = self.sample(row, greedy_row, self.sampler)
        ttnn.copy(token_row, token_row_io)
        return candidates, row, greedy_row, token_row

    def capture_epilogue(self, trace_output: Any, token_row_io: Any) -> tuple[Any, Any]:
        """The epilogue inside the TAIL capture: its row and logits kept for the residue (the host read, the fallback
        gather); the resolve's own row freed when the sampler's token row is the trace's."""

        if trace_output.logits is None:
            raise RuntimeError("TAIL capture returned no logits")
        candidates, row, greedy_row, trace_token_row = self._epilogue(trace_output.logits, token_row_io)
        if trace_token_row is not greedy_row:
            ttnn.deallocate(greedy_row)
        self.trace_rows.append(row)
        self.trace_logits.append(trace_output.logits)
        return candidates, trace_token_row

    def mark_corruptible(self) -> None:
        """Before the miss guard closes: the readback row is rewritten by every replay, like the token row; the
        sampler's host-written scalars and table between replays."""
        if getattr(self, "device_accept", None) is not None:
            self.device_accept.mark_corruptible()

        acknowledge_corruptible(self.constants.readback_row)
        if self.sampler is not None:
            self.sampler.mark_corruptible()

    def mark_trace_rows_corruptible(self) -> None:
        """After a capture: its row is rewritten by every replay, like the greedy candidates."""

        acknowledge_corruptible(self.trace_rows[-1])

    def warm(self, logits: Qwen38ShardedLogits, token_row_io: Any, *, label: str) -> dict[str, Any]:
        """The capture's epilogue run eagerly (its programs compile here) and the fallback gather; the row must equal
        torch.topk of the gather.

        With the device sampler the epilogue runs under the warm policy and then under the greedy flag: each token must
        equal the host reference on the read row, the greedy one the resolve's own row's id, and the resident row
        (``token_row_io``, holding the warm step's resolved token on entry) holds it again on exit.
        """

        full = self._gather(logits)
        passes: tuple[tuple[str | None, Any], ...] = ((None, None),)
        if self.sampler is not None:
            passes = (("sampled", WARM_POLICY), ("greedy", Qwen38DeviceSamplerPolicy.greedy_policy()))
        agreement: dict[str, Any] | None = None
        checks: dict[str, dict[str, int]] = {}
        greedy = None
        for name, policy in passes:
            if policy is not None:
                self.sampler.write_policy(policy)
                self.sampler.write_uniform(WARM_UNIFORM)
            candidates, row, greedy_row, token_row = self._epilogue(logits, token_row_io)
            actual = self.read_candidate_row()
            if agreement is None:
                agreement = actual.agreement(Qwen38CandidateRow.emulate(full.to(torch.bfloat16)))
                if not all(agreement["values_bitwise"]) or not all(agreement["ids_equal_up_to_boundary_ties"]):
                    raise RuntimeError(
                        f"{label}: candidate row {actual.values.tolist()} {actual.ids.tolist()} vs torch.topk of the "
                        f"full gather beyond boundary ties: {agreement}"
                    )
            if policy is not None:
                greedy = self._token_of(greedy_row)
                if name == "greedy":
                    expected = greedy
                else:
                    values, ids = candidate_row_lanes(actual.to_host_row())
                    expected = device_sampler_reference(values, ids, WARM_POLICY, WARM_UNIFORM).token_id
                device = self._token_of(token_row)
                checks[name] = {"device": device, "expected": expected}
                if device != expected:
                    raise RuntimeError(f"{label}: device sampler {name} token {device} vs expected {expected}")
                ttnn.deallocate(token_row)
            ttnn.deallocate(greedy_row)
            ttnn.deallocate(row)
            ttnn.deallocate(candidates.local_indices)
            ttnn.deallocate(candidates.local_values)
        if self.sampler is not None:
            resident = self._token_of(token_row_io)
            if resident != greedy:
                raise RuntimeError(
                    f"{label}: the resident token row holds {resident} after the warm passes, not {greedy}"
                )
            agreement["device_sampler"] = checks
        return agreement

    @staticmethod
    def _token_of(token_row: Any) -> int:
        host = ttnn.to_torch(ttnn.get_device_tensors(token_row)[0]).reshape(-1)
        if not torch.equal(host[1:], torch.zeros_like(host[1:])) or float(host[0]) != int(host[0]):
            raise RuntimeError(f"token row is not one id at column 0: {host.tolist()}")
        return int(host[0])

    def release(self) -> None:
        for tensor in (*self.trace_rows, self.constants.readback_row, self.constants.shard_vocab_start):
            ttnn.deallocate(tensor)
        if self.sampler is not None:
            self.sampler.release()
        self.trace_rows.clear()
        self.trace_logits.clear()
        if getattr(self, "device_accept", None) is not None:
            self.device_accept.release()

    def begin_request(self, request: "Qwen38SamplingRequest | None") -> None:
        """Request start, before the prompt's steps: the device policy (the greedy flag for a greedy or host-loop
        request) and, on the device path, the first draw; eager writes ordered before the request's first TAIL."""

        device_accept = getattr(self, "device_accept", None)  # an extension built without one decides on the host
        if device_accept is not None and request is not None and device_accept.admission(request) is None:
            device_accept.begin_request(request)
        if self.sampler is None:
            return
        policy = None if request is None else self.device_policy_of(request)
        if policy is None:
            self.sampler.write_policy(Qwen38DeviceSamplerPolicy.greedy_policy())
            return
        self.sampler.write_policy(policy)
        if self.presence_on_device:
            self.sampler.reset_history()  # no token emitted yet
        request.uniforms.clear()
        request.stream = UniformStream(request.parameters.seed)
        self.sampler.write_uniform(request.next_uniform())

    def device_acceptance_for(
        self, request: "Qwen38SamplingRequest | None"
    ) -> tuple[str | None, Callable[[Sequence[int]], None] | None]:
        """The pass loop's device acceptance for this request: ``(None, the before-verify hook)`` when the
        device decides its passes, else ``(the refusal, None)`` and the host decides them."""

        device_accept = getattr(self, "device_accept", None)
        if device_accept is None:
            return f"refused: {DEVICE_ACCEPT_SWITCH} off", None
        why = device_accept.admission(request)
        if why is not None:
            return why, None
        return None, device_accept.before_verify(request)

    def device_policy_of(self, request: "Qwen38SamplingRequest") -> Qwen38DeviceSamplerPolicy | None:
        """The request's device policy on this chain's sampler (the presence penalty only where the sampler keeps
        the history), or ``None`` for the host loop."""

        return request.device_policy(presence_on_device=self.presence_on_device)

    def rewrite_history(self, emitted: Sequence[int]) -> None:
        """The request's emitted tokens as the host knows them, into the device history (a step whose token is not
        the device's draw: a forced ``</think>``, a first-token rewrite); a sampler without a history ignores it."""

        if self.presence_on_device:
            self.sampler.rewrite_history([list(emitted)])

    def read_candidate_row(self) -> Qwen38CandidateRow:
        """Blocking: completes the queued TAIL and parses its row (device 0's replica; the four agree)."""

        return Qwen38CandidateRow.from_host_row(ttnn.to_torch(ttnn.get_device_tensors(self.constants.readback_row)[0]))

    def read_full_logits(self, residue: int) -> torch.Tensor:
        """The fallback's eager full-vocabulary gather of TAIL(residue)'s logits, about 2 MB, as fp32 ``[VOCAB_SIZE]``."""

        return self._gather(self.trace_logits[residue])

    def read_full_logits_rows(self, logits: Qwen38ShardedLogits) -> torch.Tensor:
        """The verify rows' fallback: the eager gather of the split verify's retained logits (about 2.5 MB at k = 4),
        fp32 ``[rows, VOCAB_SIZE]``."""

        return self._gather(logits).reshape(int(logits.global_shape[2]), -1)

    def warm_rows(
        self, logits: Qwen38ShardedLogits, candidate_rows: torch.Tensor, *, label: str
    ) -> list[dict[str, Any]]:
        """The verify head's candidate rows against torch.topk of the eager rows gather (the fallback's program
        compiles here): per row values bitwise, ids equal up to ties at a shard's k-th value."""

        full = self.read_full_logits_rows(logits)
        if tuple(candidate_rows.shape) != (full.shape[0], SAMPLING_CANDIDATE_ROW_SHAPE[3]):
            raise RuntimeError(f"{label}: {tuple(candidate_rows.shape)} candidate rows for {full.shape[0]} logits rows")
        reports = []
        for row, (host_row, full_row) in enumerate(zip(candidate_rows, full)):
            actual = Qwen38CandidateRow.from_host_row(host_row.reshape(SAMPLING_CANDIDATE_ROW_SHAPE))
            agreement = actual.agreement(Qwen38CandidateRow.emulate(full_row.to(torch.bfloat16)))
            if not all(agreement["values_bitwise"]) or not all(agreement["ids_equal_up_to_boundary_ties"]):
                raise RuntimeError(
                    f"{label}: verify row {row} candidates {actual.values.tolist()} {actual.ids.tolist()} vs torch.topk "
                    f"of the full gather beyond boundary ties: {agreement}"
                )
            reports.append(agreement)
        return reports

    def _gather(self, logits: Qwen38ShardedLogits) -> torch.Tensor:
        gathered = self.lm_head.gather_full_logits(logits)
        try:
            host = ttnn.to_torch(ttnn.get_device_tensors(gathered)[0])
        finally:
            ttnn.deallocate(gathered)
        return host.reshape(-1).to(torch.float32)


# -- the sampled loop -----------------------------------------------------------------------------------------


@dataclass
class Qwen38SamplingStepClocks:
    """Per-token instants of the sampled loop (``clock_ns``) and its counters, for the period arm of the discriminator."""

    row_available_ns: list[int] = field(default_factory=list)  # the blocking row read returned
    token_available_ns: list[int] = field(default_factory=list)  # the host chose x_t
    head_enqueued_ns: list[int] = field(default_factory=list)
    tail_enqueued_ns: list[int] = field(default_factory=list)
    fallbacks: int = 0  # steps sampled over the full vocabulary (top_k 0, a boosting penalty, the guard)
    candidate_misses: int = 0  # sampled tokens outside the read candidate row (only a fallback can produce one)

    def periods_ms(self) -> list[float]:
        instants = self.token_available_ns
        return [(later - earlier) / 1e6 for earlier, later in zip(instants[:-1], instants[1:])]

    def summary(self) -> dict[str, Any]:
        periods = self.periods_ms()
        # The device waits for all of this: parse + sample (row -> token), token-row write + HEAD launch
        # (token -> head).  The read wake-up before ``row_available`` is the rest of the period delta.
        sample = [(token - row) / 1e6 for row, token in zip(self.row_available_ns, self.token_available_ns)]
        exposed = [(head - token) / 1e6 for token, head in zip(self.token_available_ns, self.head_enqueued_ns)]
        return {
            "tokens": len(self.token_available_ns),
            "period_median_ms": median(periods) if periods else None,
            "period_min_ms": min(periods) if periods else None,
            "sample_median_ms": median(sample) if sample else None,
            "host_segment_median_ms": median(exposed) if exposed else None,
            "fallbacks": self.fallbacks,
            "candidate_misses": self.candidate_misses,
        }


ACCEPTANCE_HISTOGRAM_BINS = 10


@dataclass
class Qwen38SampledDraftingStats:
    """The pass loop's counters of one sampled request: passes decided on the host, drafts accepted, draws consumed,
    rows sampled over the full vocabulary (the candidate guard failed), tokens drawn from a rejection's residual,
    and the acceptance probabilities ``p_j(d_{j+1})`` of every row drawn on (the per-row log the measurement reads)."""

    passes: int = 0
    accepted_drafts: int = 0
    draws: int = 0
    fallbacks: int = 0
    resampled: int = 0
    guard_deviations: int = 0  # device-decided rows whose kept minimum did not clear the shard floor
    acceptance_probabilities: list[float] = field(default_factory=list)

    def record_device(self, statistics: Sequence[float], drafts: int) -> None:
        """One device-decided pass from its 16 statistics lanes (``fused/mtp_accept``): ``a*`` drafts accepted,
        ``k + 1`` uniforms consumed, the resample flag, the guard bits, ``p_j = w_j / S_j`` of the evaluated rows."""

        from models.demos.blackhole.qwen38_flash_next.tools import qwen38_mtp_device_accept as da
        from models.demos.blackhole.qwen38_flash_next.ttnn.fused import mtp_accept as ma

        self.passes += 1
        self.accepted_drafts += int(statistics[ma.STAT_ACCEPTED])
        self.draws += drafts + 1
        self.resampled += int(float(statistics[ma.STAT_RESAMPLED]) != 0)
        self.guard_deviations += da.guard_deviations(statistics)
        self.acceptance_probabilities.extend(da.acceptance_probabilities(statistics, drafts))

    def record(self, acceptance: Qwen38SpeculativeAcceptance, fallbacks: int) -> None:
        self.passes += 1
        self.accepted_drafts += acceptance.accepted
        self.draws += acceptance.draws
        self.fallbacks += fallbacks
        self.resampled += int(acceptance.resampled)
        self.acceptance_probabilities.extend(acceptance.acceptance_probabilities)

    def as_dict(self) -> dict[str, Any]:
        probabilities = self.acceptance_probabilities
        histogram = [0] * ACCEPTANCE_HISTOGRAM_BINS
        for probability in probabilities:
            histogram[min(int(probability * ACCEPTANCE_HISTOGRAM_BINS), ACCEPTANCE_HISTOGRAM_BINS - 1)] += 1
        return {
            "passes": self.passes,
            "accepted_drafts": self.accepted_drafts,
            "tokens_per_pass": None
            if not self.passes
            else round((self.passes + self.accepted_drafts) / self.passes, 4),
            "draws": self.draws,
            "fallbacks": self.fallbacks,
            "resampled": self.resampled,
            "guard_deviations": self.guard_deviations,
            "rows_drawn": len(probabilities),
            "acceptance_probability_mean": None
            if not probabilities
            else round(sum(probabilities) / len(probabilities), 4),
            "acceptance_probability_histogram": histogram,  # equal bins over [0, 1]; the last one holds 1.0
        }


@dataclass
class Qwen38SamplingRequest:
    """One sampled request: its policy, the generator its seed starts, the alternatives it wants, what it produced.

    ``samples`` is index-aligned with the completion's token ids; a token the thinking
    budget forced has no sample (``None``), and so has a token the MTP pass loop emitted.
    ``mtp_drafting`` is the pass loop's verdict on the request (``drafted``, or the refusal
    reason) once the session decided it; ``mtp`` its counters.
    """

    parameters: Qwen38SamplingParameters
    top_logprobs: int = 0
    logprobs: bool = False
    generator: torch.Generator = field(init=False, repr=False)
    samples: list[Qwen38CandidateSample | None] = field(default_factory=list)
    clocks: Qwen38SamplingStepClocks = field(default_factory=Qwen38SamplingStepClocks)
    # The device path (``generate_sampled_on_device``): its splitmix64 stream, the draws written (the ledger), how
    # many first tokens the host had to rewrite (a continuation whose row the previous request's policy chose).
    stream: UniformStream = field(init=False, repr=False)
    uniforms: list[float] = field(default_factory=list)
    first_token_rewrites: int = 0
    verified_steps: int = 0
    mtp_drafting: str | None = None
    mtp: Qwen38SampledDraftingStats = field(default_factory=Qwen38SampledDraftingStats)
    mtp_arithmetic: str | None = None  # the pass decisions' law realisation: host-fp32 or device-theta
    prompt_tokens: int = 0  # the device loop's history starts after these (set at its entry)

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, Qwen38SamplingParameters) or self.parameters.temperature == 0:
            raise ValueError("a sampling request needs Qwen38SamplingParameters with temperature > 0")
        if isinstance(self.top_logprobs, bool) or type(self.top_logprobs) is not int:
            raise TypeError(f"top_logprobs must be an integer, got {self.top_logprobs!r}")
        if type(self.logprobs) is not bool:
            raise TypeError(f"logprobs must be a bool, got {self.logprobs!r}")
        self.generator = torch.Generator(device="cpu").manual_seed(self.parameters.seed)
        self.stream = UniformStream(self.parameters.seed)

    def device_policy(self, *, presence_on_device: bool = False) -> Qwen38DeviceSamplerPolicy | None:
        """The device sampler's policy for this request, or ``None`` when it takes the host loop: penalties, a
        temperature above the table's range, ``top_k`` 0, or the logprobs the device loop does not read."""

        if self.logprobs or self.top_logprobs:
            return None
        return Qwen38DeviceSamplerPolicy.from_parameters(self.parameters, presence_on_device=presence_on_device)

    def next_uniform(self) -> float:
        uniform = self.stream.next_uniform()
        self.uniforms.append(uniform)
        return uniform

    def draw(self) -> float:
        """One fp32 uniform from the request's generator: the draw the host samplers make (``sampling._draw``), so
        the pass loop's acceptance and the 1-row sampled tail consume one stream."""

        return float(torch.rand((), generator=self.generator, dtype=torch.float32))

    def as_dict(self) -> dict[str, Any]:
        """The response's ``qwen38.sampling`` object: the policy, its seed and the loop's counters."""

        return {
            **parameters_as_dict(self.parameters),
            "fallbacks": self.clocks.fallbacks,
            "candidate_misses": self.clocks.candidate_misses,
            # The reported logprobs are normalised over the read candidates, not the vocabulary (the row's mass).
            "logprobs_normalizer": "candidate_row",
            "device_path": bool(self.uniforms),
            "draws": len(self.uniforms),
            "first_token_rewrites": self.first_token_rewrites,
            "verified_steps": self.verified_steps,
            "mtp_drafting": self.mtp_drafting,
            "mtp_acceptance_arithmetic": self.mtp_arithmetic,
            "mtp": None if self.mtp_drafting != "drafted" else self.mtp.as_dict(),
        }


def parameters_as_dict(p: Qwen38SamplingParameters) -> dict[str, Any]:
    return {
        "profile": p.profile.value,
        "temperature": p.temperature,
        "top_p": p.top_p,
        "top_k": p.top_k,
        "min_p": p.min_p,
        "presence_penalty": p.presence_penalty,
        "frequency_penalty": p.frequency_penalty,
        "repetition_penalty": p.repetition_penalty,
        "seed": p.seed,
    }


def choose_token(
    session: Any, row: Qwen38CandidateRow, request: Qwen38SamplingRequest, *, tail_residue: int, prompt_tokens: int
) -> Qwen38CandidateSample:
    """The candidate sampler, or the full-vocabulary sampler over TAIL(tail_residue)'s logits when the row cannot prove
    exactness; ``session.committed`` is the penalties' history, its first ``prompt_tokens`` the request's prompt."""

    try:
        sample = sample_candidates(
            row,
            request.parameters,
            token_history=session.committed,
            prompt_tokens=prompt_tokens,
            generator=request.generator,
            top_logprobs=request.top_logprobs,
        )
    except Qwen38CandidateFallback:
        request.clocks.fallbacks += 1
        full = session.sampling.read_full_logits(tail_residue)
        sample = sample_full_vocabulary(
            full,
            request.parameters,
            token_history=session.committed,
            prompt_tokens=prompt_tokens,
            generator=request.generator,
            top_logprobs=request.top_logprobs,
        )
    if sample.token_id not in row.ids.reshape(-1).tolist():
        request.clocks.candidate_misses += 1
    return sample


# -- the MTP pass loop's sampled decision --------------------------------------------------------------------


def drafting_admission(mtp: Any, request: Qwen38SamplingRequest | None, *, device_loop: bool = False) -> str | None:
    """Why the MTP pass loop does not serve a sampled request, or ``None`` when it drafts for it: the chain's switch
    (``mtp.sampled``, ``QWEN38_MTP_SAMPLED``), ``top_k`` 0 and a boosting penalty (the candidate rows cannot bound
    them: the 1-row loop's full-vocabulary fallback every step), logprobs (the row normaliser is not carried per
    verify row), the device-sampler loop.  A greedy request (``None``) is always admitted."""

    if request is None:
        return None
    if mtp is None:
        return "refused: no MTP chain"
    if not getattr(mtp, "sampled", False):
        return "refused: QWEN38_MTP_SAMPLED off"
    if device_loop:
        return "refused: device sampler loop"
    if request.parameters.top_k == 0:
        return "refused: top_k 0"
    if request.parameters.raises_logits:
        return "refused: penalty raises logits"
    if request.logprobs or request.top_logprobs:
        return "refused: logprobs"
    return None


def accept_pass(
    session: Any,
    request: Qwen38SamplingRequest,
    prompt_tokens: int,
    tokens: Sequence[int],
    head: mtp_v2.Qwen38TTNNVerifyHeadReadback,
) -> mtp_v2.Qwen38TTNNVerifyDecision:
    """The sampled request's verdict on one split pass (the chain's ``decide``): row j of ``tokens`` ``[t_P, d_1 ..
    d_k]`` gives ``p_j`` from its candidate row under the request's processors with the history
    ``session.committed + tokens[:j + 1]`` (the drafts accepted earlier in the pass count, as plain decode's output
    would; ``prompt_tokens`` exempts the prompt from the additive penalties), or over the eagerly gathered verify
    logits when the guard fails (``session.chain.mtp_read_full_logits_rows``); the point-mass acceptance draws
    from ``request.draw``.  The alignment tokens are ``[d_1 .. d_a*, x*]`` then the zero-embedding sentinel.

    The k + 1 candidate distributions come from one batched pass (``candidate_distributions``, bitwise the per-row
    ``candidate_distribution``); the acceptance still asks for them row by row, so a fallback is counted, and the
    verify logits read, only for a row the acceptance reaches."""

    rows = len(tokens)
    if tuple(head.candidate_rows.shape) != (rows, SAMPLING_CANDIDATE_ROW_SHAPE[3]):
        raise ValueError(f"{tuple(head.candidate_rows.shape)} candidate rows for a {rows}-row pass")
    pass_tokens = [int(token) for token in tokens]
    committed = session.committed
    distributions = candidate_distributions(
        Qwen38CandidateRows.from_host_rows(head.candidate_rows),
        request.parameters,
        token_history=committed,
        row_tokens=pass_tokens,
        prompt_tokens=prompt_tokens,
    )
    fallbacks = 0
    full_rows: torch.Tensor | None = None

    def distribution(row: int) -> Qwen38RowDistribution:
        nonlocal fallbacks, full_rows
        try:
            return distributions.row(row)
        except Qwen38CandidateFallback:
            fallbacks += 1
            if full_rows is None:
                full_rows = session.chain.mtp_read_full_logits_rows()
            return full_distribution(
                full_rows[row],
                request.parameters,
                token_history=[*committed, *pass_tokens[: row + 1]],
                prompt_tokens=prompt_tokens,
            )

    acceptance = accept_point_mass(distribution, pass_tokens[1:], request.draw)
    request.mtp.record(acceptance, fallbacks)
    if request.mtp_arithmetic is None:
        request.mtp_arithmetic = ARITHMETIC_HOST
    alignment = pass_tokens[1 : acceptance.accepted + 1] + [acceptance.token]
    alignment += [ZERO_EMBEDDING_TOKEN] * (rows - len(alignment))
    return mtp_v2.Qwen38TTNNVerifyDecision(
        acceptance.accepted,
        acceptance.token,
        tuple(alignment),
        {
            "sampled": True,
            "draws": acceptance.draws,
            "fallbacks": fallbacks,
            "resampled": acceptance.resampled,
            "acceptance_probabilities": acceptance.acceptance_probabilities,
        },
    )


def sample_next_token(session: Any, request: Qwen38SamplingRequest, *, prompt_tokens: int) -> Qwen38CandidateSample:
    """The pass loop's row-0 token: sampled from the candidate row the last TAIL wrote (blocking: completes it), as
    the 1-row sampled step would at this position; appended to ``request.samples``."""

    tail_residue = (len(session.committed) - 1) % RESIDUE_CLASSES
    row = session.sampling.read_candidate_row()
    sample = choose_token(session, row, request, tail_residue=tail_residue, prompt_tokens=prompt_tokens)
    request.samples.append(sample)
    return sample


def generate_sampled(
    session: Any,
    request: Qwen38SamplingRequest,
    max_new_tokens: int,
    *,
    stop_ids: Sequence[int],
    tokenizer_size: int,
    think_budget: int | None = None,
    should_stop: Callable[[], str | None] | None = None,
    forced_step: Callable[[int], None] | None = None,
    clock_ns: Callable[[], int] | None = None,
    prompt_tokens: int | None = None,
) -> Iterator[tuple[int | None, str | None]]:
    """The sampled A-G loop over ``session.chain`` and ``session.sampling``; yields ``(x_t, finish)`` like ``_generate``.

    ``session`` carries ``chain`` (the traced chain's primitives), ``sampling`` (the extension),
    ``committed`` (the host list of every input token: the prompt when the loop starts, then this
    request's output; the penalties' history, presence and frequency over the output only) and
    ``ple_context``.  ``prompt_tokens`` is the request's prompt length, the committed length when
    the loop starts unless the caller already emitted output (the pass loop's hand-off).  Every
    sampled token appends its sample to ``request.samples`` before it is yielded.  A hook stop
    yields ``(None, reason)``; the thinking budget forces ``</think>`` through ``forced_step`` (the
    session's) and appends ``None`` for it.
    """

    if not isinstance(request, Qwen38SamplingRequest):
        raise TypeError("generate_sampled needs a Qwen38SamplingRequest")
    if think_budget is not None and forced_step is None:
        raise TypeError("a thinking budget needs the session's forced step")
    chain, clocks = session.chain, request.clocks
    now = clock_ns or (lambda: 0)
    if prompt_tokens is None:
        prompt_tokens = len(session.committed)
    elif type(prompt_tokens) is not int or not 0 <= prompt_tokens <= len(session.committed):
        raise ValueError(f"prompt_tokens must be an int in [0, {len(session.committed)}], got {prompt_tokens!r}")
    produced = 0
    reasoning_tokens = 0
    thinking_open = think_budget is not None
    while True:
        reason = None if should_stop is None else should_stop()
        if reason is not None:
            yield None, reason
            return
        if thinking_open and reasoning_tokens >= think_budget:
            forced_step(THINK_END_ID)
            request.samples.append(None)
            thinking_open = False
            produced += 1
            yield THINK_END_ID, "length" if produced == max_new_tokens else None
            if produced == max_new_tokens:
                return
            continue
        tail_residue = (len(session.committed) - 1) % RESIDUE_CLASSES  # TAIL(t-1) ran one committed token ago
        if produced + 1 == max_new_tokens:
            row = session.sampling.read_candidate_row()  # completes TAIL(t-1)
            sample = choose_token(session, row, request, tail_residue=tail_residue, prompt_tokens=prompt_tokens)
            request.samples.append(sample)
            chain.write_token_row(sample.token_id)  # the row holds the token the client saw, as after the greedy loop
            session.row_token = sample.token_id
            yield sample.token_id, "length"
            return
        residue = len(session.committed) % RESIDUE_CLASSES
        row = session.sampling.read_candidate_row()  # completes TAIL(t-1)
        clocks.row_available_ns.append(now())
        sample = choose_token(session, row, request, tail_residue=tail_residue, prompt_tokens=prompt_tokens)
        token_id = sample.token_id
        clocks.token_available_ns.append(now())
        chain.write_token_row(token_id)
        chain.execute_head(residue)
        clocks.head_enqueued_ns.append(now())
        produced += 1
        if thinking_open:
            thinking_open = token_id != THINK_END_ID
            reasoning_tokens += 1
        session.ple_context = chain.refresh_ple_row(token_id, session.ple_context)
        chain.execute_tail(residue)
        clocks.tail_enqueued_ns.append(now())
        session.committed.append(token_id)
        request.samples.append(sample)
        if token_id >= tokenizer_size or token_id in stop_ids:
            chain.read_token_row()  # completes TAIL(t); x_{t+1} is discarded
            yield token_id, "error" if token_id >= tokenizer_size else "stop"
            return
        yield token_id, None


def generate_sampled_on_device(
    session: Any,
    request: Qwen38SamplingRequest,
    max_new_tokens: int,
    *,
    stop_ids: Sequence[int],
    tokenizer_size: int,
    prefilled: bool,
    think_budget: int | None = None,
    should_stop: Callable[[], str | None] | None = None,
    forced_step: Callable[[int], None] | None = None,
    clock_ns: Callable[[], int] | None = None,
    verify_each_step: bool = False,
) -> Iterator[tuple[int | None, str | None]]:
    """The device-sampled loop: the greedy A-G loop plus one draw write per step; yields ``(x_t, finish)`` like
    :func:`generate_sampled`.

    Entry: the request's policy and first draw were written by ``begin_request`` before the prompt's steps, so the
    last prompt TAIL chose the first token on the device.  The row is read once here (blocking, completing that
    TAIL) and the host reference recomputes the token from it and the first draw: equal when ``prefilled`` (a
    mismatch is a device error), rewritten into the row when the request continues an unconsumed row the previous
    request's policy chose (``first_token_rewrites``).  Every step then writes the next draw behind HEAD(t) (ordered
    before TAIL(t), which consumes it); a forced ``</think>`` step reuses the resident draw.  ``verify_each_step``
    (the discriminator's arm) also reads the row every step and checks the device token against the reference
    (``verified_steps``); the production loop reads nothing but the token.
    """

    if not isinstance(request, Qwen38SamplingRequest):
        raise TypeError("generate_sampled_on_device needs a Qwen38SamplingRequest")
    if think_budget is not None and forced_step is None:
        raise TypeError("a thinking budget needs the session's forced step")
    if not request.uniforms:
        raise RuntimeError("the device path needs begin_request's policy and first draw before the prompt")
    chain, sampler, clocks = session.chain, session.sampling.sampler, request.clocks
    now = clock_ns or (lambda: 0)
    request.prompt_tokens = len(session.committed)

    def reference_token() -> int:
        return device_sampler_reference_of(session, request, -1)

    # The first token: the device's choice against the host reference on the row it was chosen from.
    expected = reference_token()
    device_token = chain.read_token_row()
    if device_token != expected:
        if prefilled:
            raise RuntimeError(
                f"device sampler first token {device_token} vs host reference {expected} on the read row "
                f"(u={request.uniforms[-1]}, policy={session.sampling.device_policy_of(request)})"
            )
        chain.write_token_row(expected)
        session.sampling.rewrite_history([expected])  # the emitted first token, not the device's draw
        request.first_token_rewrites += 1
    produced = 0
    reasoning_tokens = 0
    thinking_open = think_budget is not None
    while True:
        reason = None if should_stop is None else should_stop()
        if reason is not None:
            yield None, reason
            return
        if thinking_open and reasoning_tokens >= think_budget:
            # the device's draw for this position is discarded and </think> emitted in its place: the history image
            # (the output so far plus </think>) lands before the forced step's TAIL adds its own draw
            session.sampling.rewrite_history([*session.committed[request.prompt_tokens :], THINK_END_ID])
            forced_step(THINK_END_ID)
            request.samples.append(None)
            thinking_open = False
            produced += 1
            yield THINK_END_ID, "length" if produced == max_new_tokens else None
            if produced == max_new_tokens:
                return
            continue
        if produced + 1 == max_new_tokens:
            token_id = chain.read_token_row()  # completes TAIL(t-1); the row keeps the unconsumed token
            if token_id >= tokenizer_size:
                raise RuntimeError(f"token row holds {token_id}, at or above the tokenizer size {tokenizer_size}")
            session.row_token = token_id
            yield token_id, "length"
            return
        residue = len(session.committed) % RESIDUE_CLASSES
        pending = chain.read_token_row_nonblocking()
        event = chain.record_event()
        chain.execute_head(residue)
        sampler.write_uniform(request.next_uniform())  # behind HEAD(t), before TAIL(t)
        clocks.head_enqueued_ns.append(now())
        chain.event_synchronize(event)
        token_id = chain.pending_value(pending)
        clocks.token_available_ns.append(now())
        if verify_each_step and produced > 0:
            if token_id != (check := device_sampler_reference_of(session, request, -2)):
                raise RuntimeError(f"device sampler step {produced} token {token_id} vs host reference {check}")
            request.verified_steps += 1
        produced += 1
        if thinking_open:
            thinking_open = token_id != THINK_END_ID
            reasoning_tokens += 1
        session.ple_context = chain.refresh_ple_row(token_id, session.ple_context)
        chain.execute_tail(residue)
        clocks.tail_enqueued_ns.append(now())
        session.committed.append(token_id)
        request.samples.append(None)
        if token_id >= tokenizer_size or token_id in stop_ids:
            chain.read_token_row()  # completes TAIL(t); x_{t+1} is discarded
            yield token_id, "error" if token_id >= tokenizer_size else "stop"
            return
        yield token_id, None


def device_sampler_reference_of(session: Any, request: Qwen38SamplingRequest, draw_index: int) -> int:
    """The host reference on the readback row (the last completed TAIL's) with the request's draw ``draw_index`` and
    the request's emitted tokens so far as the presence history (``session.committed`` past the prompt)."""

    values, ids = candidate_row_lanes(session.sampling.read_candidate_row().to_host_row())
    emitted = set(session.committed[request.prompt_tokens :])
    seen = torch.tensor([int(i) in emitted for i in ids.tolist()], dtype=torch.bool)
    policy = session.sampling.device_policy_of(request)
    return device_sampler_reference(values, ids, policy, request.uniforms[draw_index], seen=seen).token_id


# -- the discriminator over a live session ------------------------------------------------------------------


def compare_candidate_rows_with_full_gathers(
    session: Any, forced_step: Callable[[int], None], tokens: Sequence[int]
) -> list[dict[str, Any]]:
    """Arm a: force ``tokens`` one by one; after each TAIL compare its row with the eager full gather (torch.topk)."""

    records = []
    for token in tokens:
        residue = len(session.committed) % RESIDUE_CLASSES
        forced_step(token)
        row = session.sampling.read_candidate_row()  # blocking: completes this TAIL
        full = session.sampling.read_full_logits(residue)
        greedy_argmax = int(torch.argmax(full))
        row_greedy = int(row.ids.reshape(-1)[int(torch.argmax(row.values.reshape(-1)))])
        records.append(
            {
                "input_token": token,
                "position": len(session.committed),
                "residue": residue,
                "row_vs_torch": row.agreement(Qwen38CandidateRow.emulate(full.to(torch.bfloat16))),
                "greedy_argmax": greedy_argmax,
                "row_greedy": row_greedy,
                "greedy_agrees": row_greedy == greedy_argmax,
            }
        )
    return records


def run_discriminator(
    session: Any,
    prompt_ids: Sequence[int],
    *,
    tokens: int = DISCRIMINATOR_TOKENS,
    seed: int = DISCRIMINATOR_SEED,
    row_tokens: int = DISCRIMINATOR_ROW_TOKENS,
    period_target_ms: float = DISCRIMINATOR_PERIOD_TARGET_MS,
) -> dict[str, Any]:
    """The chain discriminator on one prompt: rows vs truth (a), the greedy period (d), two seeded card-profile streams
    (c: identical), the sampled period against the target (d) and every sampled token inside its candidate row (e).

    Arm b (``temperature 0`` = the CPU greedy record, 96/96) is the server's acceptance replay, run before this.
    """

    prompt_ids = list(prompt_ids)
    session.reset()
    rows = compare_candidate_rows_with_full_gathers(session, session._forced_step, prompt_ids[:row_tokens])
    rows_pass = all(
        all(record["row_vs_torch"]["values_bitwise"])
        and all(record["row_vs_torch"]["ids_equal_up_to_boundary_ties"])
        and record["greedy_agrees"]
        for record in rows
    )
    greedy = session.complete(prompt_ids, tokens, stop_ids=())  # extends the forced prefix
    streams, clocks, completions = [], [], []
    device_sampler = getattr(session.sampling, "sampler", None) is not None
    # On a device-sampler chain a third stream runs with the per-step host verification (arm d): the same tokens.
    for verify in (False, False, *([True] if device_sampler else [])):
        request = Qwen38SamplingRequest(Qwen38SamplingParameters.official_thinking(seed=seed))
        completion = session.complete(prompt_ids, tokens, stop_ids=(), sampling=request, verify_each_step=verify)
        streams.append(completion.token_ids)
        clocks.append(request.clocks.summary())
        logprobs = [sample.logprob for sample in request.samples if sample is not None]
        completions.append(
            {
                "tokens_per_second": completion.tokens_per_second,
                "period_ms": None if completion.tokens_per_second is None else 1e3 / completion.tokens_per_second,
                "finish_reason": completion.finish_reason,
                "samples": len(request.samples),
                "sampled_logprob_median": median(logprobs) if logprobs else None,
                "device_path": bool(request.uniforms),
                "draws": len(request.uniforms),
                "first_token_rewrites": request.first_token_rewrites,
                "verified_steps": request.verified_steps,
                "verify_each_step": verify,
            }
        )
    sampled_periods = [summary["period_median_ms"] for summary in clocks[:2]]
    # The host loop's bar is the absolute target; the device loop's is "no exposed host cost": its period within
    # noise of the greedy loop's period on the same chain (the sampler ops run for greedy requests too).
    greedy_period_ms = None if greedy.tokens_per_second is None else 1e3 / greedy.tokens_per_second
    period_pass = all(
        period is not None
        and (
            period <= greedy_period_ms + DEVICE_PERIOD_NOISE_MS
            if device_sampler and greedy_period_ms is not None
            else period <= period_target_ms
        )
        for period in sampled_periods
    )
    exposed = []
    for summary in clocks:
        sample, host = summary["sample_median_ms"], summary["host_segment_median_ms"]
        exposed.append(None if sample is None or host is None else sample + host)
    result = {
        "prompt_tokens": len(prompt_ids),
        "tokens": tokens,
        "seed": seed,
        "profile": Qwen38SamplingParameters.official_thinking(seed=seed).profile.value,
        "rows": rows,
        "rows_pass": rows_pass,
        "greedy": {
            "token_ids": greedy.token_ids,
            "tokens_per_second": greedy.tokens_per_second,
            "period_ms": greedy_period_ms,
            "finish_reason": greedy.finish_reason,
        },
        "sampled": {"streams": streams, "clocks": clocks, "completions": completions},
        "device_sampler": device_sampler,
        "streams_identical": all(stream == streams[0] for stream in streams) and len(streams[0]) == tokens,
        "sampled_differs_from_greedy": streams[0] != greedy.token_ids,
        "sampled_period_median_ms": sampled_periods,
        "period_target_ms": period_target_ms,
        "period_pass": period_pass,
        "period_delta_vs_greedy_ms": [
            None if period is None or greedy_period_ms is None else period - greedy_period_ms
            for period in sampled_periods
        ],
        # period delta = read wake-up (this residual) + parse and sample + token write and HEAD launch
        "exposed_host_median_ms": exposed,
        "read_wakeup_residual_ms": [
            None if period is None or greedy_period_ms is None or host is None else period - greedy_period_ms - host
            for period, host in zip(sampled_periods, exposed)
        ],
        "fallbacks": [summary["fallbacks"] for summary in clocks],
        "candidate_misses": [summary["candidate_misses"] for summary in clocks],
        "tokens_inside_candidates": all(summary["candidate_misses"] == 0 for summary in clocks),
        "device_verified": (
            None if not device_sampler else completions[-1]["verified_steps"] == max(0, len(streams[-1]) - 2)
        ),
    }
    result["pass"] = bool(
        rows_pass
        and result["streams_identical"]
        and result["tokens_inside_candidates"]
        and period_pass
        and result["device_verified"] is not False
    )
    return result


__all__ = [
    "DEVICE_ACCEPT_SWITCH",  # re-exported for the chain open (the import-pruning hook keeps __all__ names)
    "Qwen38DeviceAcceptance",  # re-exported for the chain open (the import-pruning hook keeps __all__ names)
    "device_acceptance_for_chain",  # re-exported for the chain open (the import-pruning hook keeps __all__ names)
    "ACCEPTANCE_HISTOGRAM_BINS",
    "DISCRIMINATOR_PERIOD_TARGET_MS",
    "DISCRIMINATOR_ROW_TOKENS",
    "DISCRIMINATOR_SEED",
    "DISCRIMINATOR_TOKENS",
    "RESIDUE_CLASSES",
    "SAMPLING_REQUEST_FIELDS",
    "Qwen38SampledDraftingStats",
    "Qwen38SamplingChainExtension",
    "Qwen38SamplingRequest",
    "Qwen38SamplingRequestError",
    "Qwen38SamplingStepClocks",
    "accept_pass",
    "choose_token",
    "compare_candidate_rows_with_full_gathers",
    "device_sampler_reference_of",
    "drafting_admission",
    "generate_sampled",
    "generate_sampled_on_device",
    "logprobs_content_item",
    "logprobs_from_request",
    "parameters_as_dict",
    "parameters_from_request",
    "run_discriminator",
    "sample_next_token",
]
