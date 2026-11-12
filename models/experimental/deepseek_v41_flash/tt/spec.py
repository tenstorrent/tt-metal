# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Speculative decode for :class:`~.model.DeepSeekV41Model` with the DSpark drafter.

A round verifies ``[next, draft_1 .. draft_k]`` at positions ``p .. p + k`` in one pipelined
:meth:`decode_prompt_traced`: step ``j``'s logits give the target's distribution ``q_j`` for position
``p + j + 1``. With ``m`` drafts accepted the round emits ``draft_1 .. draft_m`` and one more token
``a_m``, and the next round starts at ``p + m + 1`` with ``next = a_m``. The rejected rows are undone
(:meth:`~.model.DeepSeekV41Model.spec_fixup`) before anything reads them.

The accept rule is DeepInfer's speculative sampling (its ``fused_sample``): ``q_j`` is the target's
logits at ``temperature``, then top-p, then top-k; ``p_j`` is the distribution the drafter drew
``draft_{j+1}`` from. ``draft_{j+1}`` is accepted with probability ``min(1, q_j(d) / p_j(d))``; the first
rejected row ``j`` draws ``a_j`` from ``max(q_j - exp(-DRAFT_PROB_PENALTY) * p_j, 0)`` instead, and a round
that accepts all ``k`` drafts draws ``a_k`` from ``q_k``. The output is then distributed as sampling the
target alone. The DSpark drafter samples each Markov step at ``temperature`` on device (Gumbel-max over
uniforms drawn on device, :meth:`~.model.DeepSeekV41Model.spec_draft_async`) without top-p / top-k, so ``p_j =
softmax(logits_j / temperature)``; a ``propose`` override is deterministic, a one-hot ``p_j``.

The pipelined verify is streamed (:meth:`~.model.DeepSeekV41Model.spec_verify_async`): the host applies the
rule to each step's logits as they come out, and at the first reject (or after the last row) it queues the
draft of the next round, then, while the drafter runs, the fixup and that round's first row ``a_m``, which
needs no draft, then reads the drafts that follow ``a_m`` and dispatches them behind it, while the rows past
the reject are still in the pipeline; their outputs are dropped and the next round's steps queue behind them.
So a round takes the pipeline's latency plus a step per accepted draft, plus whatever of the draft is not hidden
behind ``a_m``'s step, and the caller's work between rounds runs while the next round is in the pipeline (only
the caller's last round, see :meth:`DSparkSpeculator.step`, leaves its fixup to the next round). Each decode step's row reaches the drafter's rings only when the host releases its
receive, one row ahead of the accept, so the draft sees the round's rows up to the reject and no further.

With ``device_accept`` (the drafter, pipelined verify and no top-p / top-k) the accept runs on the drafter
instead: each decode step hands it its logits, and one trace accepts the round and drafts the next from the
result (:meth:`~.model.DeepSeekV41Model.spec_accept_draft`), so the host reads back only ``m``, ``a_m`` and
the proposals, but only after the whole round.

``temperature = 0`` is greedy: the drafter takes its argmax, ``draft_{j+1}`` is accepted iff it is
``argmax q_j`` and ``a_m`` is the argmax too, so the output is the greedy decode's token for token. (DeepInfer
clamps the temperature to ``1e-5``, the same rule but for exact logit ties, which it splits at random.)

Every decode step, prompt and verified rows alike, hands its row to the drafter device to device, which
writes its ``main_kv`` into the rings (the draft masks the rows past its frontier, and the next round
rewrites them), so the round has no write of its own. The drafter then drafts the next block from ``a_m``;
the proposals for ``p + m + 2 ..`` become the next round's drafts.

With ``parallel_verify`` the round's ``k + 1`` rows go through every layer in one step instead
(:meth:`~.model.DeepSeekV41Model.decode_verify`, a model built with ``verify_rows = k + 1``, replayed from
the traces :meth:`~.model.DeepSeekV41Model.capture_spec_traces` captures); the accept rule, the fixup and the
hand-off are the same.
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
from loguru import logger

from models.experimental.deepseek_v41_flash.tt.decode.dspark import DRAFT_PROB_PENALTY

# Log every round's phases (and the round's accepted count) as it finishes. Off: only the end-of-run summary.
_VERBOSE = os.environ.get("DEEPSEEK_V4_SPEC_VERBOSE", "1") not in ("0", "", "false", "False")
# Seconds a single round phase may take before the watchdog dumps every thread's stack. 0 disables it.
_STALL_SECONDS = float(os.environ.get("DEEPSEEK_V4_SPEC_STALL_SECONDS", "30"))


@dataclass
class SpecStats:
    """Per-round accepted draft counts, the time spent in rounds and its split over a round's phases.

    The phases are host wall time: ``fixup`` only uploads and enqueues, so its device time lands in
    whichever later phase waits on the device (``verify`` / ``draft``). A streamed verify's ``verify`` runs up
    to its first reject, the accept included. Otherwise ``accept`` is the host's accept rule over the verified
    logits, or a device accept has ``accept+draft``, the drafter's one trace for both.
    """

    accepted: list[int] = field(default_factory=list)
    seconds: float = 0.0
    phases: dict[str, float] = field(default_factory=dict)

    def add(self, phase: str, seconds: float) -> None:
        self.phases[phase] = self.phases.get(phase, 0.0) + seconds

    @property
    def tokens(self) -> int:
        return sum(self.accepted) + len(self.accepted)

    def summary(self, drafts: int) -> str:
        rounds = len(self.accepted)
        if not rounds:
            return "no rounds"
        histogram = [self.accepted.count(m) for m in range(drafts + 1)]
        split = ", ".join(f"{name} {1000 * t / rounds:.1f}" for name, t in self.phases.items())
        return (
            f"{self.tokens} tokens in {rounds} rounds: {self.tokens / rounds:.2f} tokens/round, "
            f"accepted per round {histogram}, {self.tokens / self.seconds:.2f} tok/s; "
            f"ms/round {1000 * self.seconds / rounds:.1f} ({split})"
        )


class _StallWatch:
    """Dump every thread's stack when one phase of a round runs past :data:`_STALL_SECONDS`.

    A speculative round is a chain of host waits -- a fixup's uploads, the verify's output reads, the
    drafter's readback -- and a stall in any of them looks identical from outside: the test simply stops
    printing. The watchdog names the phase and shows which queue every thread is parked on, which is what
    distinguishes a deadlocked device from a slow one.

    ``enter`` arms it for one phase; ``cancel`` disarms it. A phase that trips the timer logs once and does
    not re-arm, so a genuinely slow phase reports its stack once instead of spamming.
    """

    def __init__(self, model=None) -> None:
        self._timer: Optional[threading.Timer] = None
        self._phase = ""
        self._model = model

    def enter(self, phase: str) -> None:
        if _STALL_SECONDS <= 0:
            return
        self.cancel()
        self._phase = phase
        timer = threading.Timer(_STALL_SECONDS, self._dump, args=(phase,))
        timer.daemon = True
        self._timer = timer
        timer.start()

    def cancel(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()
        self._phase = ""

    def _dump(self, phase: str) -> None:
        """Log the model's dispatch state and then the stack of every thread.

        The state line comes first: it says which queue the round is stuck on (undispatched replays,
        unwritten packets, unread outputs), and the stacks then say which thread is holding it. The dump
        also shows the watchdog's own frame, harmless because it runs after the state line is built.
        """
        frames = sys._current_frames()
        state = ""
        if self._model is not None:
            try:
                state = f" state {self._model.spec_state()}"
            except Exception as exc:  # a probe must never be the thing that kills a run
                state = f" state unavailable ({exc!r})"
        lines = [f"[spec stall] no progress for {_STALL_SECONDS:.0f}s in phase {phase!r};{state}"]
        for thread in threading.enumerate():
            frame = frames.get(thread.ident)
            if frame is None:
                continue
            where = "".join(traceback.format_stack(frame))
            lines.append(f"\n--- thread {thread.name} (daemon={thread.daemon}) ---\n{where.rstrip()}")
        logger.error("".join(lines))


def _log_round(spec: "DSparkSpeculator", p: int, emitted: Sequence[int], phases: dict[str, float]) -> None:
    """Per-round one-liner: the round's start position, how many drafts it accepted and each phase's time."""
    split = ", ".join(f"{name} {1000 * t:.2f}" for name, t in phases.items())
    logger.info(
        f"[spec] round {len(spec.stats.accepted)}: p {p} -> {spec._pos} ({len(emitted)} tokens, "
        f"m={len(emitted) - 1}), {split}; {spec.model.spec_state()}"
    )


class DSparkSpeculator:
    """Drive speculative decode on ``model``, prepared with :meth:`prepare_spec_decode`.

    ``drafts`` (``k``) is how many of the drafter's ``block_size`` proposals a round verifies.
    ``propose(token, frontier)`` overrides the drafter: it returns the proposals for positions
    ``frontier + 2 ..`` given ``token`` at ``frontier + 1`` (the exactness test feeds corrupted
    greedy tokens through it, with ``prepare_spec_decode(drafter=False)``).

    ``parallel_verify`` verifies each round in one :meth:`~.model.DeepSeekV41Model.decode_verify` step,
    which needs ``model.verify_rows == drafts + 1``. ``temperature`` (0 for greedy), ``top_p`` (1 disables)
    and ``top_k`` (-1 disables) are the target's sampling parameters, as DeepInfer's ``SamplingParams``;
    ``seed`` seeds the draws (the host's, and the seeds of the drafter's uniforms). The drafter samples at
    ``temperature`` from here on. ``device_accept`` accepts each whole round on the drafter rather than
    streaming the pipelined verify (:attr:`streaming`, the default with traced decode).
    """

    def __init__(
        self,
        model,
        drafts: int = 4,
        propose: Optional[Callable[[int, int], Sequence[int]]] = None,
        parallel_verify: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        seed: int = 0,
        device_accept: bool = False,
    ) -> None:
        if temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if top_k == 0 or top_k < -1:
            raise ValueError(f"top_k must be -1 (disabled) or >= 1, got {top_k}")
        if model.spec_window_rows is None:
            raise ValueError("call model.prepare_spec_decode() first")
        if drafts + 1 >= model.spec_window_rows:
            raise ValueError(
                f"a round verifies {drafts + 1} rows, but the compressor windows keep {model.spec_window_rows}"
            )
        if propose is None and model.dspark is None:
            raise ValueError("the model has no drafter; pass propose or prepare_spec_decode(drafter=True)")
        if parallel_verify and model.verify_rows != drafts + 1:
            raise ValueError(f"a round verifies {drafts + 1} rows, but the model was built for {model.verify_rows}")
        if model.dspark is not None:
            if drafts > model.dspark.block:
                raise ValueError(f"{drafts} drafts per round, but the drafter proposes {model.dspark.block}")
            if drafts + 1 > model.dspark.main_rows:
                raise ValueError(f"a round verifies {drafts + 1} rows, but write_main takes {model.dspark.main_rows}")
        self.model = model
        self.drafts = drafts
        self.parallel_verify = parallel_verify
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self._rng = torch.Generator().manual_seed(seed)
        self._propose = propose
        # The drafter accepts and drafts on in one trace only with the round's inputs: its own drafts, the
        # pipelined steps' logits (the parallel verify sends none), and no top-p / top-k, which it does not apply.
        if device_accept and not (
            model.dspark is not None
            and model.spec_device_accept
            and propose is None
            and not parallel_verify
            and top_p == 1.0
            and top_k == -1
            and drafts < model.dspark.block
        ):
            raise ValueError(
                "device_accept needs the drafter prepared with device_accept, pipelined verify, no top-p / top-k "
                "and drafts < block"
            )
        self.device_accept = device_accept
        self.streaming = not parallel_verify and not device_accept and not model._eager_decode
        if model.dspark is not None and propose is None:
            model.dspark.set_sampling(temperature)
        self.stats = SpecStats()
        self._stall = _StallWatch(model)
        self._next = None
        self._pos = None
        self._proposals: list[int] = []
        # The drafter's logits [k, V] for the proposals, when it sampled them; None for one-hot proposals.
        self._proposal_logits: Optional[torch.Tensor] = None
        # A streamed round reads those logits only once its verify is dispatched, while the pipeline runs it.
        self._logits_pending = False
        self._rejected: list[int] = []
        # How many of a streamed round's rows (from ``self._next`` at ``self._pos``) are already in the pipeline,
        # with the fixup of the round before it: the previous round dispatched the first at its reject, ahead of
        # its draft, and the drafts once it had read them. 0, 1 or all of them.
        self._ahead = 0
        # The round dispatch() queued for collect(): (first position, inputs, start time, time after its fixup).
        self._dispatched: Optional[tuple[int, list[int], float, float]] = None

    def _seed(self) -> Optional[int]:
        """A nonzero seed for the drafter's next uniforms when sampling, else None (greedy draws none)."""
        if self.temperature == 0:
            return None
        return int(torch.randint(1, 2**31 - 1, (), generator=self._rng))

    def _draft(self, token: int, frontier: int) -> list[int]:
        self._queue_draft(token, frontier)
        return self._drafted()

    def _queue_draft(self, token: int, frontier: int) -> None:
        """Start the proposals after ``token`` at ``frontier + 1``; :meth:`_drafted` returns them."""
        self._proposal_logits = None
        self._drafting = (token, frontier)
        if self._propose is None:
            drafter = self.model.dspark
            # Several sessions' speculators share the drafter, each at its own request's temperature.
            if drafter.temperature != self.temperature:
                drafter.set_sampling(self.temperature)
            self._logits_pending = self.temperature > 0 and not self.device_accept and self.streaming
            self.model.spec_draft_async(token, frontier, self._seed())

    def _drafted(self) -> list[int]:
        if self._propose is not None:
            return list(self._propose(*self._drafting))[: self.drafts]
        host_accept = self.temperature > 0 and not self.device_accept
        ids, _, logits = self.model.spec_read_draft(logits=host_accept and not self._logits_pending)
        if logits is not None:
            self._proposal_logits = logits[: self.drafts]
        return ids[: self.drafts]

    def _target_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """The target's distribution for one row: ``logits`` at the temperature, then top-p, then top-k (stable
        sorts, DeepInfer's ``apply_top_p_top_k_ref``)."""
        logits = logits.reshape(-1).float() / self.temperature
        if self.top_p < 1.0:
            probs, order = torch.sort(torch.softmax(logits, dim=-1), descending=True, stable=True)
            logits[order[probs.cumsum(dim=-1) - probs > self.top_p]] = float("-inf")
        if self.top_k > 0:
            order = torch.argsort(logits, descending=True, stable=True)
            logits[order[self.top_k :]] = float("-inf")
        return torch.softmax(logits, dim=-1)

    def _sample(self, probs: torch.Tensor) -> int:
        """A draw from ``probs`` (unnormalized) by inverse CDF: ~20x faster than ``torch.multinomial`` over the
        vocabulary, and it is on the round's critical path once per round."""
        cdf = probs.cumsum(dim=0)
        u = torch.rand(1, generator=self._rng, dtype=cdf.dtype) * cdf[-1]
        return min(int(torch.searchsorted(cdf, u, right=True)), probs.numel() - 1)

    def _next_token(self, logits: torch.Tensor) -> int:
        """The target's token from one row of logits: the argmax when greedy, else a draw from the row."""
        if self.temperature == 0:
            return int(logits.reshape(-1).float().argmax().item())
        return self._sample(self._target_probs(logits))

    def _verify_row(self, inputs: list[int], j: int, row: torch.Tensor) -> Optional[int]:
        """Row ``j`` of a verified round, given the rows before it accepted: ``None`` if it accepts the draft
        ``inputs[j + 1]``, else the round's last token ``a_j`` (always one for the last row)."""
        k = len(inputs) - 1
        if j == k:
            return self._next_token(row)
        if self.temperature == 0:
            target = int(row.argmax().item())
            return None if inputs[j + 1] == target else target
        q, d = self._target_probs(row), inputs[j + 1]
        if self._proposal_logits is None:
            p = torch.zeros_like(q)
            p[d] = 1.0
        else:
            p = torch.softmax(self._proposal_logits[j] / self.temperature, dim=-1)
        accept = min(1.0, float(q[d]) / float(p[d])) if p[d] > 0 else 0.0
        if accept > float(torch.rand((), generator=self._rng)):
            return None
        return self._sample((q - math.exp(-DRAFT_PROB_PENALTY) * p).clamp_min_(0.0))

    def _accept(self, inputs: list[int], rows: list[torch.Tensor]) -> tuple[int, int]:
        """``(m, a_m)`` for a verified round: the drafts ``inputs[1 .. m]`` it accepts and the token after them."""
        for j, row in enumerate(rows):
            token = self._verify_row(inputs, j, row)
            if token is not None:
                return j, token
        raise ValueError(f"{len(rows)} rows for a round of {len(inputs)}")

    def prefill(self, prompt: Sequence[int], start_pos: int = 0) -> int:
        """Feed ``prompt`` from ``start_pos``, which also fills the drafter's rings; returns the first generated token.

        Captures the traces first if needed (their compile runs write rows prefill overwrites).
        """
        model = self.model
        prompt = list(prompt)
        logger.info(
            f"[spec] prefill: {len(prompt)} tokens from {start_pos} (traces captured: {model._traced_captured})"
        )
        if not model._traced_captured:
            self._stall.enter("prefill: capture_spec_traces")
            try:
                model.capture_spec_traces(prompt[0], start_pos)
            finally:
                self._stall.cancel()
        self._stall.enter("prefill: decode_prompt_traced")
        try:
            out = model.decode_prompt_traced(prompt, start_pos)
        finally:
            self._stall.cancel()
        return self.start(self._next_token(out), start_pos + len(prompt))

    def start(self, token: int, pos: int) -> int:
        """Start from what plain decode left: every position before ``pos`` is in the caches and the drafter's
        rings, and ``token`` (the target's pick from position ``pos - 1``) goes in at ``pos``. Drafts its
        successors; returns ``token``. :meth:`prefill` ends here; a caller that fed the prompt itself (a server's
        scheduler) starts here."""
        self._next = token
        self._pos = pos
        self._rejected = []
        self._ahead = 0
        self._stall.enter("start: draft")
        try:
            self._proposals = self._draft(token, pos - 1)
        finally:
            self._stall.cancel()
        logger.info(f"[spec] started at {pos}: first token {token}, drafts {self._proposals}")
        return token

    def finish(self) -> None:
        """Hand the model back to plain decode, which feeds the last returned token at :attr:`position` next:
        drop the next round's rows the last :meth:`step` dispatched ahead, if any, read every output still
        unread and undo the rejected rows (:meth:`~.model.DeepSeekV41Model.spec_fixup`). A :meth:`start` (or
        :meth:`prefill`) may follow."""
        model = self.model
        self._drop_ahead()
        self._stall.enter(f"finish at {self._pos}: drain_stale_outputs, spec_fixup({self._rejected})")
        try:
            model.drain_stale_outputs()
            model.spec_fixup(self._rejected)
        finally:
            self._stall.cancel()
        self._rejected = []

    def pause(self) -> None:
        """Leave the model to another session between two rounds; :meth:`step` carries on later from where this
        one stopped, with this session resident again (``activate_sessions``) and its drafts kept.

        The next round's rows the last :meth:`step` dispatched ahead are dropped (and redone by the next round;
        ``step(ahead=False)`` dispatches none), and the draft's logits a streamed round reads late are read now:
        the other session's draft replaces the drafter's outputs. The rejected rows wait for this session's next
        fixup, which restores them from its own snapshot (:meth:`~.model.DeepSeekV41Model.spec_fixup`)."""
        if self.device_accept:
            raise NotImplementedError("a device accept accepts against the drafter's last draft, another session's")
        model = self.model
        self._drop_ahead()
        if self._logits_pending:
            self._proposal_logits = model.spec_draft_logits()[: self.drafts]
            self._logits_pending = False

    def _drop_ahead(self) -> None:
        """Count the next round's rows already dispatched as rejected: release their drafter receives and mark
        their outputs unwanted, so the next fixup restores the ring rows they wrote."""
        n, self._ahead = self._ahead, 0
        if not n:
            return
        if self.model.dspark is not None:
            self.model.spec_release_writes(n)
        self.model.spec_discard_outputs(n)
        self._rejected += list(range(self._pos, self._pos + n))

    def dispatch(self) -> None:
        """The first half of a streamed round: queue the fixup and the round's verify; :meth:`collect` takes its
        outputs. Other sessions may dispatch their rounds in between, each with its session resident: their rows
        follow this one's through the pipeline, so the rounds are collected in the order they were dispatched."""
        if not self.streaming:
            raise RuntimeError("only a streamed verify is dispatched apart from its accept")
        if self._dispatched is not None:
            raise RuntimeError("the last dispatched round has not been collected")
        model = self.model
        t0 = time.perf_counter()
        p = self._pos
        inputs = [self._next] + self._proposals
        if not self._ahead:
            self._stall.enter(f"round p={p}: spec_fixup({self._rejected})")
            try:
                model.spec_fixup(self._rejected)
            finally:
                self._stall.cancel()
        t1 = time.perf_counter()
        # The previous round dispatched this one's first row at its reject, and its drafts once they were read.
        first = self._ahead
        if first < len(inputs):
            self._stall.enter(f"round p={p}: spec_verify_async ({len(inputs) - first} steps)")
            try:
                model.spec_verify_async(inputs[first:], p + first)
            finally:
                self._stall.cancel()
        self._ahead = 0
        self._dispatched = (p, inputs, t0, t1)

    def collect(self, budget: int, stop_ids: Sequence[int] = ()) -> tuple[list[int], bool]:
        """The second half of the round :meth:`dispatch` queued, as :meth:`advance`'s: ``(tokens, done)``. It
        dispatches no row ahead and reads the draft's logits before it returns, so another session's round may
        follow."""
        stop = set(stop_ids)
        return self._cut(self._collect(budget, stop, ahead=False), budget, stop)

    def _collect(self, budget: Optional[int], stop_ids: Sequence[int], ahead: bool) -> list[int]:
        """Accept the dispatched round's rows as they come out, queue the next draft (and, ``ahead`` and not the
        caller's last round, the next round's first row) and read the draft; returns the round's tokens."""
        if self._dispatched is None:
            raise RuntimeError("dispatch() a round first")
        (p, inputs, t0, t1), self._dispatched = self._dispatched, None
        model, k = self.model, self.drafts
        held = model.dspark is not None
        if self._logits_pending:
            self._stall.enter(f"round p={p}: read the draft's logits")
            try:
                self._proposal_logits = model.spec_draft_logits()[:k]
            finally:
                self._stall.cancel()
            self._logits_pending = False
        for m in range(k + 1):
            if held:
                self._stall.enter(f"round p={p}: release_writes(row {m})")
                try:
                    model.spec_release_writes()
                finally:
                    self._stall.cancel()
            self._stall.enter(f"round p={p}: read output of row {m}")
            try:
                row = model.read_decoded_output().reshape(-1).float()
            finally:
                self._stall.cancel()
            token = self._verify_row(inputs, m, row)
            if _VERBOSE:
                # Row ``k`` is the bonus row: it has no draft to accept, it emits the round's last token.
                if m < k:
                    logger.info(
                        f"[spec] round p={p}: row {m} draft {inputs[m + 1]} -> "
                        f"{'accepted' if token is None else f'rejected, token {token}'}"
                    )
                else:
                    logger.info(f"[spec] round p={p}: row {m} bonus (all drafts accepted) -> token {token}")
            if token is not None:
                break
        t2 = time.perf_counter()
        self._rejected = list(range(p + m + 1, p + len(inputs)))
        self._next = token
        self._pos = p + m + 1
        emitted = inputs[1 : m + 1] + [token]
        last = not ahead or any(t in stop_ids for t in emitted) or (budget is not None and len(emitted) >= budget)
        self._stall.enter(f"round p={p}: queue the draft (frontier {p + m})")
        try:
            self._queue_draft(token, p + m)
        finally:
            self._stall.cancel()
        t_queued = time.perf_counter()
        if not last:
            # The next round's first row needs no draft: queue it (behind the fixup of this round's
            # rejects) while the drafter drafts its successors, so it runs through the pipeline meanwhile.
            self._stall.enter(f"round p={p}: early row 0 at {self._pos} (fixup {self._rejected})")
            try:
                model.spec_fixup(self._rejected)
                model.spec_verify_async([token], self._pos)
            finally:
                self._stall.cancel()
            self._rejected = []
            self._ahead = 1
        t_row0 = time.perf_counter()
        self._stall.enter(f"round p={p}: read the draft (frontier {p + m})")
        try:
            self._proposals = self._drafted()
            if not ahead and self._logits_pending:
                # Another session's draft may come next, and replace these on the drafter.
                self._proposal_logits = model.spec_draft_logits()[:k]
                self._logits_pending = False
        finally:
            self._stall.cancel()
        t_draft = time.perf_counter()
        # After the draft's readback, which would otherwise wait behind these rows' receives.
        if held:
            self._stall.enter(f"round p={p}: release_writes({k - m} trailing rows)")
            try:
                model.spec_release_writes(k - m)
            finally:
                self._stall.cancel()
        model.spec_discard_outputs(k - m)
        if self._ahead:
            # The rest of the next round, its drafts behind its first row: whatever the caller does between rounds
            # (a server streams the tokens) then runs while they go through the pipeline.
            self._stall.enter(f"round p={p}: next round's drafts at {self._pos + 1}")
            try:
                model.spec_verify_async(self._proposals, self._pos + 1)
            finally:
                self._stall.cancel()
            self._ahead += len(self._proposals)
        t3 = time.perf_counter()
        phases = {
            "fixup": t1 - t0,
            "verify": t2 - t1,
            "queue draft": t_queued - t2,
            "row0": t_row0 - t_queued,
            "draft": t_draft - t_row0,
            "release": t3 - t_draft,
        }
        for phase, seconds in phases.items():
            self.stats.add(phase, seconds)
        self.stats.accepted.append(m)
        self.stats.seconds += t3 - t0
        if _VERBOSE:
            _log_round(self, p, emitted, phases)
        return emitted

    def step(self, budget: Optional[int] = None, stop_ids: Sequence[int] = (), ahead: bool = True) -> list[int]:
        """One verify round; returns the tokens it emits (1 to ``drafts + 1``).

        ``budget`` (tokens the caller still wants) and ``stop_ids`` say which round is the caller's last: a
        streamed round that is not dispatches the next round's first row at its reject and its drafts once they
        are read (see the module docstring).
        ``ahead=False`` makes it dispatch none, as before a :meth:`pause`. A streamed round is :meth:`dispatch`
        then :meth:`collect`'s half.
        """
        if self.streaming:
            self.dispatch()
            return self._collect(budget, stop_ids, ahead)
        model, k = self.model, self.drafts
        t0 = time.perf_counter()
        p = self._pos
        inputs = [self._next] + self._proposals
        if not self._ahead:
            self._stall.enter(f"round p={p}: spec_fixup({self._rejected})")
            try:
                model.spec_fixup(self._rejected)
            finally:
                self._stall.cancel()
        t1 = time.perf_counter()
        if self.device_accept:
            self._stall.enter(f"round p={p}: device-accept verify")
            try:
                model.decode_prompt_traced(inputs, p)
            finally:
                self._stall.cancel()
            t2 = time.perf_counter()
            self._stall.enter(f"round p={p}: spec_accept_draft")
            try:
                m, token, ids = model.spec_accept_draft(p, k, self.temperature > 0, self._seed())
            finally:
                self._stall.cancel()
            self._rejected = list(range(p + m + 1, p + len(inputs)))
            self._next = token
            self._pos = p + m + 1
            self._proposals = ids[:k]
            t3 = time.perf_counter()
            phases = {"fixup": t1 - t0, "verify": t2 - t1, "accept+draft": t3 - t2}
            for phase, seconds in phases.items():
                self.stats.add(phase, seconds)
            self.stats.accepted.append(m)
            self.stats.seconds += t3 - t0
            emitted = inputs[1 : m + 1] + [token]
            if _VERBOSE:
                _log_round(self, p, emitted, phases)
            return emitted
        rows: list[torch.Tensor] = []
        self._stall.enter(f"round p={p}: {'parallel' if self.parallel_verify else 'pipelined'} verify")
        try:
            if self.parallel_verify:
                rows = [row.reshape(-1).float() for row in model.decode_verify(inputs, p)]
            else:
                model.decode_prompt_traced(
                    inputs, p, on_output=lambda i, out: rows.append(out.reshape(-1).float().clone())
                )
        finally:
            self._stall.cancel()
        t2 = time.perf_counter()
        m, token = self._accept(inputs, rows)
        self._rejected = list(range(p + m + 1, p + len(inputs)))
        t3 = time.perf_counter()
        self._next = token
        self._pos = p + m + 1
        self._stall.enter(f"round p={p}: draft (frontier {self._pos - 1})")
        try:
            self._proposals = self._draft(self._next, self._pos - 1)[:k]
        finally:
            self._stall.cancel()
        t4 = time.perf_counter()
        phases = {"fixup": t1 - t0, "verify": t2 - t1, "accept": t3 - t2, "draft": t4 - t3}
        for phase, seconds in phases.items():
            self.stats.add(phase, seconds)
        self.stats.accepted.append(m)
        self.stats.seconds += t4 - t0
        emitted = inputs[1 : m + 1] + [token]
        if _VERBOSE:
            _log_round(self, p, emitted, phases)
        return emitted

    def generate(
        self,
        max_new_tokens: int,
        stop_ids: Sequence[int] = (),
        on_tokens: Optional[Callable[[list[int]], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> list[int]:
        """Rounds until ``max_new_tokens`` (beyond the first token :meth:`prefill` returned), a stop id, or
        ``should_stop()`` (asked before every round) is true; ``on_tokens`` gets each round's tokens.

        A round cut short by a stop id or the count keeps only the tokens up to the cut: the rows it accepted
        past the cut count as rejected, so :meth:`extend` can carry on from the last returned token. The
        streamed verify's unwanted outputs are read off before it returns. After ``should_stop`` the last round
        may have dispatched the next one's rows, which :meth:`extend` and :meth:`finish` account for.
        """
        out: list[int] = []
        stop = set(stop_ids)
        done = self._next in stop
        while not done and len(out) < max_new_tokens:
            if should_stop is not None and should_stop():
                break
            kept, done = self.advance(max_new_tokens - len(out), stop)
            out.extend(kept)
            if on_tokens is not None:
                on_tokens(kept)
            if _VERBOSE and (len(out) % 10 < len(kept) or done):
                logger.info(f"[spec] generate: {len(out)}/{max_new_tokens} tokens, pos {self._pos}")
        self._stall.enter("generate: drain_stale_outputs")
        try:
            self.model.drain_stale_outputs()
        finally:
            self._stall.cancel()
        logger.info(
            f"[spec] generate done: {len(out)} tokens, pos {self._pos}, stats {self.stats.summary(self.drafts)}"
        )
        return out

    def advance(self, budget: int, stop_ids: Sequence[int] = (), ahead: bool = True) -> tuple[list[int], bool]:
        """One round of at most ``budget`` tokens: ``(tokens, done)``, ``done`` once they reach a stop id or the
        budget. A round cut short there keeps only the tokens up to the cut, as :meth:`generate`'s rounds do.
        ``ahead`` is :meth:`step`'s."""
        stop = set(stop_ids)
        return self._cut(self.step(budget, stop, ahead), budget, stop)

    def _cut(self, emitted: list[int], budget: int, stop: set[int]) -> tuple[list[int], bool]:
        """Cut the round that emitted ``emitted`` at its first stop id or at ``budget`` tokens."""
        for keep, token in enumerate(emitted, start=1):
            if token in stop or keep == budget:
                self._cut_round(emitted, keep)
                return emitted[:keep], True
        return emitted, False

    def _cut_round(self, emitted: list[int], keep: int) -> None:
        """Make the last round as if it had accepted only ``keep - 1`` drafts, emitting ``emitted[:keep]``."""
        if keep == len(emitted):
            return
        p = self._pos - len(emitted)  # the round's first position
        self._rejected = list(range(p + keep, self._pos)) + self._rejected
        self._pos = p + keep
        self._next = emitted[keep - 1]
        self.stats.accepted[-1] = keep - 1
        # The rows the drafter has yet to receive each wait for the host to read their output first.
        self._stall.enter(f"cut at p={p} keep={keep}: drain_stale_outputs")
        try:
            self.model.drain_stale_outputs()
        finally:
            self._stall.cancel()
        self._stall.enter(f"cut at p={p} keep={keep}: draft")
        try:
            self._proposals = self._draft(self._next, self._pos - 1)[: self.drafts]
        finally:
            self._stall.cancel()
        logger.info(
            f"[spec] cut the round at p={p} to {keep} token(s): pos {self._pos}, next {self._next}, "
            f"drafts {self._proposals}, rejected {self._rejected}"
        )

    def extend(self, tokens: Sequence[int]) -> int:
        """Feed ``tokens`` after the last returned token (say, the next user turn after an end-of-sentence id)
        and seed the drafter on them; returns the first token generated after them, as :meth:`prefill`."""
        start = self._pos
        logger.info(f"[spec] extend: {len(tokens)} tokens from {start}, next {self._next}, rejected {self._rejected}")
        self.finish()
        return self.prefill([self._next, *tokens], start)

    @property
    def position(self) -> int:
        """Where the next input token goes."""
        return self._pos
