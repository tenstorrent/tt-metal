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
uniforms drawn on device, :meth:`~.model.DeepSeekV41Model.spec_draft`) without top-p / top-k, so ``p_j =
softmax(logits_j / temperature)``; a ``propose`` override is deterministic, a one-hot ``p_j``.

The pipelined verify is streamed (:meth:`~.model.DeepSeekV41Model.spec_verify_async`): the host applies the
rule to each step's logits as they come out, and at the first reject (or after the last row) it drafts the
next round at once, while the rows past the reject are still in the pipeline; their outputs are dropped and
the next round's steps queue behind them. So a round takes the pipeline's latency plus a step per accepted
draft, plus the draft. Each decode step's row reaches the drafter's rings only when the host releases its
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
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from models.experimental.deepseek_v41_flash.tt.decode.dspark import DRAFT_PROB_PENALTY


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
            and propose is None
            and not parallel_verify
            and top_p == 1.0
            and top_k == -1
            and drafts < model.dspark.block
        ):
            raise ValueError("device_accept needs the drafter, pipelined verify, no top-p / top-k and drafts < block")
        self.device_accept = device_accept
        self.streaming = not parallel_verify and not device_accept and not model._eager_decode
        if model.dspark is not None and propose is None:
            model.dspark.set_sampling(temperature)
        self.stats = SpecStats()
        self._next = None
        self._pos = None
        self._proposals: list[int] = []
        # The drafter's logits [k, V] for the proposals, when it sampled them; None for one-hot proposals.
        self._proposal_logits: Optional[torch.Tensor] = None
        self._rejected: list[int] = []

    def _seed(self) -> Optional[int]:
        """A nonzero seed for the drafter's next uniforms when sampling, else None (greedy draws none)."""
        if self.temperature == 0:
            return None
        return int(torch.randint(1, 2**31 - 1, (), generator=self._rng))

    def _draft(self, token: int, frontier: int) -> list[int]:
        self._proposal_logits = None
        if self._propose is not None:
            return list(self._propose(token, frontier))[: self.drafts]
        host_accept = self.temperature > 0 and not self.device_accept
        ids, _, logits = self.model.spec_draft(token, frontier, self._seed(), logits=host_accept)
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
        return int(torch.multinomial(probs, 1, generator=self._rng).item())

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
        if not model._traced_captured:
            model.capture_spec_traces(prompt[0], start_pos)
        out = model.decode_prompt_traced(prompt, start_pos)
        end = start_pos + len(prompt)
        self._next = self._next_token(out)
        self._pos = end
        self._rejected = []
        self._proposals = self._draft(self._next, end - 1)
        return self._next

    def step(self) -> list[int]:
        """One verify round; returns the tokens it emits (1 to ``drafts + 1``)."""
        model, k = self.model, self.drafts
        t0 = time.perf_counter()
        p = self._pos
        inputs = [self._next] + self._proposals
        model.spec_fixup(self._rejected)
        t1 = time.perf_counter()
        if self.device_accept:
            model.decode_prompt_traced(inputs, p)
            t2 = time.perf_counter()
            m, token, ids = model.spec_accept_draft(p, k, self.temperature > 0, self._seed())
            self._rejected = list(range(p + m + 1, p + len(inputs)))
            self._next = token
            self._pos = p + m + 1
            self._proposals = ids[:k]
            t3 = time.perf_counter()
            for phase, seconds in (("fixup", t1 - t0), ("verify", t2 - t1), ("accept+draft", t3 - t2)):
                self.stats.add(phase, seconds)
            self.stats.accepted.append(m)
            self.stats.seconds += t3 - t0
            return inputs[1 : m + 1] + [token]
        if self.streaming:
            held = model.dspark is not None
            model.spec_verify_async(inputs, p)
            for m in range(k + 1):
                if held:
                    model.spec_release_writes()
                token = self._verify_row(inputs, m, model.read_decoded_output().reshape(-1).float())
                if token is not None:
                    break
            t2 = time.perf_counter()
            self._rejected = list(range(p + m + 1, p + len(inputs)))
            self._next = token
            self._pos = p + m + 1
            self._proposals = self._draft(token, p + m)
            # After the draft's readback, which would otherwise wait behind these rows' receives.
            if held:
                model.spec_release_writes(k - m)
            model.spec_discard_outputs(k - m)
            t3 = time.perf_counter()
            for phase, seconds in (("fixup", t1 - t0), ("verify", t2 - t1), ("draft", t3 - t2)):
                self.stats.add(phase, seconds)
            self.stats.accepted.append(m)
            self.stats.seconds += t3 - t0
            return inputs[1 : m + 1] + [token]
        rows: list[torch.Tensor] = []
        if self.parallel_verify:
            rows = [row.reshape(-1).float() for row in model.decode_verify(inputs, p)]
        else:
            model.decode_prompt_traced(inputs, p, on_output=lambda i, out: rows.append(out.reshape(-1).float().clone()))
        t2 = time.perf_counter()
        m, token = self._accept(inputs, rows)
        self._rejected = list(range(p + m + 1, p + len(inputs)))
        t3 = time.perf_counter()
        self._next = token
        self._pos = p + m + 1
        self._proposals = self._draft(self._next, self._pos - 1)[:k]
        t4 = time.perf_counter()
        for phase, seconds in (("fixup", t1 - t0), ("verify", t2 - t1), ("accept", t3 - t2), ("draft", t4 - t3)):
            self.stats.add(phase, seconds)
        self.stats.accepted.append(m)
        self.stats.seconds += t4 - t0
        return inputs[1 : m + 1] + [token]

    def generate(self, max_new_tokens: int, stop_ids: Sequence[int] = ()) -> list[int]:
        """Rounds until ``max_new_tokens`` (beyond the first token :meth:`prefill` returned) or a stop id.

        A round cut short by either keeps only the tokens up to the cut: the rows it accepted past the
        cut count as rejected, so :meth:`extend` can carry on from the last returned token. The streamed
        verify's unwanted outputs are read off before it returns.
        """
        out: list[int] = []
        stop = set(stop_ids)
        done = self._next in stop
        while not done and len(out) < max_new_tokens:
            emitted = self.step()
            for keep, token in enumerate(emitted, start=1):
                out.append(token)
                if token in stop or len(out) == max_new_tokens:
                    self._cut_round(emitted, keep)
                    done = True
                    break
        self.model.drain_stale_outputs()
        return out

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
        self.model.drain_stale_outputs()
        self._proposals = self._draft(self._next, self._pos - 1)[: self.drafts]

    def extend(self, tokens: Sequence[int]) -> int:
        """Feed ``tokens`` after the last returned token (say, the next user turn after an end-of-sentence id)
        and seed the drafter on them; returns the first token generated after them, as :meth:`prefill`."""
        start = self._pos
        self.model.spec_fixup(self._rejected)
        return self.prefill([self._next, *tokens], start)

    @property
    def position(self) -> int:
        """Where the next input token goes."""
        return self._pos
