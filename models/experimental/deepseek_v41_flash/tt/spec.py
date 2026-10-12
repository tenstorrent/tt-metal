# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Greedy speculative decode for :class:`~.model.DeepSeekV41Model` with the DSpark drafter.

A round verifies ``[next, draft_1 .. draft_k]`` at positions ``p .. p + k`` in one pipelined
:meth:`decode_prompt_traced`: step ``j``'s logits give the target's token ``a_j`` for position
``p + j + 1``. The drafts are accepted while ``draft_{j+1} == a_j``; with ``m`` accepted the round
emits ``a_0 .. a_m`` and the next round starts at ``p + m + 1`` with ``next = a_m``. The output is
the greedy decode's token for token, since every accepted draft is the target's own argmax and the
rejected rows are undone (:meth:`~.model.DeepSeekV41Model.spec_fixup`) before anything reads them.

Per round the drafter writes the verified rows' ``main_kv`` into its rings and drafts the next block
from ``a_m``; the proposals for ``p + m + 2 ..`` become the next round's drafts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch


@dataclass
class SpecStats:
    """Per-round accepted draft counts, the time spent in rounds and its split over a round's phases.

    The phases are host wall time: ``fixup`` and ``write`` only upload and enqueue, so their device time
    lands in whichever later phase waits on the device (``verify`` / ``draft``).
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
    """Drive greedy speculative decode on ``model``, prepared with :meth:`prepare_spec_decode`.

    ``drafts`` (``k``) is how many of the drafter's ``block_size`` proposals a round verifies.
    ``propose(token, frontier)`` overrides the drafter: it returns the proposals for positions
    ``frontier + 2 ..`` given ``token`` at ``frontier + 1`` (the exactness test feeds corrupted
    greedy tokens through it, with ``prepare_spec_decode(drafter=False)``).
    """

    def __init__(self, model, drafts: int = 4, propose: Optional[Callable[[int, int], Sequence[int]]] = None) -> None:
        if propose is None and model.dspark is None:
            raise ValueError("the model has no drafter; pass propose or prepare_spec_decode(drafter=True)")
        if model.dspark is not None:
            if drafts > model.dspark.block:
                raise ValueError(f"{drafts} drafts per round, but the drafter proposes {model.dspark.block}")
            if drafts + 1 > model.dspark.main_rows:
                raise ValueError(f"a round verifies {drafts + 1} rows, but write_main takes {model.dspark.main_rows}")
        self.model = model
        self.drafts = drafts
        self._propose = propose
        self.stats = SpecStats()
        self._next = None
        self._pos = None
        self._proposals: list[int] = []
        self._rejected: list[int] = []

    def _draft(self, token: int, frontier: int) -> list[int]:
        if self._propose is not None:
            return list(self._propose(token, frontier))[: self.drafts]
        ids, _ = self.model.spec_draft(token, frontier)
        return ids[: self.drafts]

    @staticmethod
    def _argmax(logits: torch.Tensor) -> int:
        return int(logits.reshape(-1).float().argmax().item())

    def prefill(self, prompt: Sequence[int], start_pos: int = 0) -> int:
        """Feed ``prompt`` from ``start_pos`` and seed the drafter; returns the first generated token.

        Captures the traces first if needed (their compile runs write rows prefill overwrites).
        """
        model = self.model
        prompt = list(prompt)
        if not model._traced_captured:
            model.capture_spec_traces(prompt[0], start_pos)
        out = model.decode_prompt_traced(prompt, start_pos)
        end = start_pos + len(prompt)
        if model.dspark is not None:
            model.spec_seed(range(max(start_pos, end - model.sliding_window), end))
        self._next = self._argmax(out)
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
        model.spec_fixup(self._rejected, frontier=p - 1)
        t1 = time.perf_counter()
        targets: list[int] = []
        model.decode_prompt_traced(inputs, p, on_output=lambda i, out: targets.append(self._argmax(out)))
        t2 = time.perf_counter()
        m = 0
        while m < len(inputs) - 1 and inputs[m + 1] == targets[m]:
            m += 1
        verified = len(inputs)
        self._rejected = list(range(p + m + 1, p + verified))
        if model.dspark is not None:
            model.spec_write_main(range(p, p + verified), [j <= m for j in range(verified)])
        t3 = time.perf_counter()
        self._next = targets[m]
        self._pos = p + m + 1
        self._proposals = self._draft(self._next, self._pos - 1)[:k]
        t4 = time.perf_counter()
        for phase, seconds in (("fixup", t1 - t0), ("verify", t2 - t1), ("write", t3 - t2), ("draft", t4 - t3)):
            self.stats.add(phase, seconds)
        self.stats.accepted.append(m)
        self.stats.seconds += t4 - t0
        return targets[: m + 1]

    def generate(self, max_new_tokens: int, stop_ids: Sequence[int] = ()) -> list[int]:
        """Rounds until ``max_new_tokens`` (beyond the first token :meth:`prefill` returned) or a stop id."""
        out: list[int] = []
        stop = set(stop_ids)
        if self._next in stop:
            return out
        while len(out) < max_new_tokens:
            emitted = self.step()
            for token in emitted:
                out.append(token)
                if token in stop or len(out) == max_new_tokens:
                    return out
        return out

    @property
    def position(self) -> int:
        """Where the next input token goes."""
        return self._pos
