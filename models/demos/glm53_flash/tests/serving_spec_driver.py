# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The vllm-tt-plugin runner's per-step calls on the GLM53 adapter's speculative path, host-side (torch only):
``SpecRunnerSim`` builds the candidate blocks the runner builds (``TTModelRunner._spec_candidate_block``: column 0
the last committed token, columns ``1 .. num_valid`` the drafts, the placeholder past them, positions ``p + j``
/ ``-1``), decides verify vs ordinary decode as ``_step_verifies`` does for a narrow-decode model (a verify when
some live row carries drafts or committed more than one token last step), walks acceptance with the host copy of
``accept_greedy_drafts`` (``tt/vllm_spec.py``), commits the prefixes and calls ``propose_draft_tokens`` after every
step with the committed block, its positions (``_committed_positions``) and the counts. Shared by the host test
(``test_serving_spec_host.py``, a torch fake behind the adapter) and the (2,4) block test
(``test_serving_spec_mesh.py``, the real adapter).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from models.demos.glm53_flash.tt.vllm_spec import PLACEHOLDER_TOKEN_ID, accept_greedy_walk


def greedy_params(n: int):
    """The plugin's device-sampling parameters of ``n`` greedy rows (``TTSamplingParams`` lists)."""
    return SimpleNamespace(temperature=[0.0] * n, top_k=[1] * n, top_p=[1.0] * n)


def decode_kw() -> dict:
    """The v1 reload commands of a non-async adapter (every decode reloads its inputs)."""
    return dict(reload_inputs=True, reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False)


def committed_positions(input_positions: torch.Tensor, width: int) -> torch.Tensor:
    """``TTModelRunner._committed_positions``: the committed block starts one past the input's first column and runs
    consecutively; a padding row (input position -1) is -1 throughout."""
    first = (input_positions if input_positions.dim() == 1 else input_positions[:, 0]).to(torch.int32)
    offsets = torch.arange(1, width + 1, dtype=torch.int32)
    positions = first.unsqueeze(1) + offsets.unsqueeze(0)
    return positions.masked_fill((first < 0).unsqueeze(1), -1)


@dataclass
class SlotSim:
    """One request on the runner's side: its last committed token and where the next one is consumed."""

    token: int
    position: int
    out: list[int] = field(default_factory=list)
    drafts: list[int] = field(default_factory=list)  # the outstanding proposal (already capped)
    count: int = 1  # the previous step's accepted count (1 after a prefill / an ordinary step)
    done: bool = False


@dataclass
class StepLog:
    kind: str  # "verify" | "plain"
    anchors: dict[int, int]
    counts: dict[int, int] = field(default_factory=dict)
    width: int = 1
    num_valid: dict[int, int] = field(default_factory=dict)


class SpecRunnerSim:
    """Drives ``adapter`` like the runner: ``wire`` padded rows (row ``i`` = logical slot ``i``, no remaps),
    ``num_drafts`` the launch's ``num_speculative_tokens`` (may be below the adapter's K), device sampling
    (greedy) unless ``device_sampling`` is False (host logits: the ordinary step returns logits, the verify ids)."""

    def __init__(self, adapter, num_drafts: int, wire: int, *, device_sampling: bool = True, page_table=None):
        self.a = adapter
        self.K = int(num_drafts)
        self.wire = int(wire)
        self.device_sampling = device_sampling
        self.page_table = page_table
        self.slots: dict[int, SlotSim] = {}
        self.log: list[StepLog] = []

    # ---- requests
    def add(self, slot: int, first_token: int, position: int) -> None:
        """A prefilled request: ``first_token`` is consumed at ``position`` (the prompt length) on its first step."""
        if not 0 <= slot < self.wire:
            raise ValueError(f"slot {slot} outside the {self.wire} wire rows")
        self.slots[slot] = SlotSim(token=int(first_token), position=int(position))

    def live(self) -> list[int]:
        return [s for s, st in sorted(self.slots.items()) if not st.done]

    def drop_drafts(self, slot: int) -> None:
        """What the scheduler does with a proposal it decides not to verify (preemption, a grammar cut to nothing):
        the row joins the next verify with ``num_valid_drafts`` 0."""
        self.slots[slot].drafts = []

    def finish(self, slot: int) -> None:
        st = self.slots[slot]
        st.done = True
        st.drafts = []
        self.a.release_request(slot)

    # ---- steps
    def step_verifies(self) -> bool:
        """``_step_verifies`` for a narrow-decode model whose live rows are all speculable."""
        live = self.live()
        return any(self.slots[s].drafts for s in live) or any(self.slots[s].count > 1 for s in live)

    def step(self) -> StepLog:
        return self._verify() if self.step_verifies() else self._ordinary()

    def run(self, n_tokens: int, max_steps: int = 10_000) -> dict[int, list[int]]:
        """Every live request until it holds ``n_tokens`` committed tokens (then released, like a finished request)."""
        steps = 0
        while self.live():
            self.step()
            steps += 1
            for s in self.live():
                if len(self.slots[s].out) >= n_tokens:
                    self.finish(s)
            if steps > max_steps:
                raise RuntimeError(f"no convergence after {max_steps} steps")
        return {s: st.out[:n_tokens] for s, st in sorted(self.slots.items())}

    def _ordinary(self) -> StepLog:
        live = self.live()
        tokens = torch.zeros(self.wire, 1, dtype=torch.int32)
        positions = torch.full((self.wire,), -1, dtype=torch.int32)
        for s in live:
            st = self.slots[s]
            tokens[s, 0] = st.token
            positions[s] = st.position
        kwargs = dict(decode_kw(), page_table=self.page_table)
        if self.device_sampling:
            out = self.a.decode_forward(tokens, positions, sampling_params=greedy_params(self.wire), **kwargs)
            ids = out.reshape(self.wire, -1)[:, 0]
        else:
            logits, _ = self.a.decode_forward(tokens, positions, **kwargs)
            ids = logits.reshape(self.wire, -1).argmax(-1)
        committed = torch.zeros(self.wire, 1, dtype=torch.int32)
        counts = torch.ones(self.wire, dtype=torch.int32)
        for s in live:
            st = self.slots[s]
            tok = int(ids[s])
            st.out.append(tok)
            st.token, st.position, st.count = tok, st.position + 1, 1
            committed[s, 0] = tok
        log = StepLog("plain", {s: int(positions[s]) for s in live}, {s: 1 for s in live})
        self._propose(committed, committed_positions(positions, 1), counts, live)
        self.log.append(log)
        return log

    def _verify(self) -> StepLog:
        live = self.live()
        width = self.K + 1
        tokens = torch.zeros(self.wire, width, dtype=torch.int32)
        positions = torch.full((self.wire, width), -1, dtype=torch.int32)
        num_valid = torch.zeros(self.wire, dtype=torch.int32)
        counts_in = torch.ones(self.wire, dtype=torch.int32)
        drafts = torch.full((self.wire, self.K), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        for s in live:
            st = self.slots[s]
            d = list(st.drafts[: self.K])
            tokens[s, 0] = st.token
            tokens[s, 1:] = PLACEHOLDER_TOKEN_ID
            for j, t in enumerate(d):
                tokens[s, 1 + j] = int(t)
                drafts[s, j] = int(t)
            positions[s, : 1 + len(d)] = st.position + torch.arange(1 + len(d), dtype=torch.int32)
            num_valid[s] = len(d)
            counts_in[s] = st.count
        kwargs = dict(decode_kw(), page_table=self.page_table)
        if self.device_sampling:
            kwargs["sampling_params"] = greedy_params(self.wire)
        out = self.a.decode_forward(
            tokens,
            positions,
            num_valid_drafts=num_valid,
            accepted_counts=counts_in,
            spec_mode="argmax_ids",
            **kwargs,
        )
        if getattr(out, "spec_mode", None) != "argmax_ids" or getattr(out, "argmax_ids", None) is None:
            raise AssertionError(f"a verify must return VerifyOutput(argmax_ids), got {type(out).__name__}")
        argmax = out.argmax_ids
        if argmax.dtype != torch.int32 or argmax.dim() != 2 or int(argmax.shape[0]) != self.wire:
            raise AssertionError(
                f"argmax_ids must be int32 [{self.wire}, 1+K'], got {argmax.dtype} {tuple(argmax.shape)}"
            )
        answered = int(argmax.shape[1]) - 1
        committed, counts = accept_greedy_walk(argmax, drafts[:, :answered], num_valid)
        log = StepLog(
            "verify",
            {s: int(positions[s, 0]) for s in live},
            width=int(argmax.shape[1]),
            num_valid={s: int(num_valid[s]) for s in live},
        )
        for s in live:
            st = self.slots[s]
            c = int(counts[s])
            block = [int(t) for t in committed[s, :c]]
            assert all(t >= 0 for t in block), (s, block)
            st.out.extend(block)
            st.token, st.position, st.count = block[-1], st.position + c, c
            st.drafts = []
            log.counts[s] = c
        if int(committed.shape[1]) < width:  # a narrow answer: the runner pads the block to the uniform 1+K
            pad = torch.full((self.wire, width - int(committed.shape[1])), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
            committed = torch.cat([committed, pad], dim=1)
        self._propose(committed, committed_positions(positions, width), counts, live)
        self.log.append(log)
        return log

    def _propose(self, committed, positions, counts, live) -> None:
        d = self.a.propose_draft_tokens(self.K, committed, positions, counts, hidden=None)
        ids = d.draft_token_ids
        if ids.dtype != torch.int32 or tuple(ids.shape) != (self.wire, self.K):
            raise AssertionError(
                f"draft_token_ids must be int32 [{self.wire}, {self.K}], got {ids.dtype} {tuple(ids.shape)}"
            )
        valid = d.num_valid
        if valid is None:
            valid = torch.full((self.wire,), self.K, dtype=torch.int32)
        if valid.dtype != torch.int32 or tuple(valid.shape) != (self.wire,):
            raise AssertionError(f"num_valid must be int32 [{self.wire}], got {valid.dtype} {tuple(valid.shape)}")
        if bool(((valid < 0) | (valid > self.K)).any()):
            raise AssertionError(f"num_valid outside [0, {self.K}]: {valid.tolist()}")
        for s in live:
            n = int(valid[s])
            row = ids[s, :n]
            if n and bool((row < 0).any()):
                raise AssertionError(f"slot {s}: a negative draft id inside num_valid {n}: {row.tolist()}")
            self.slots[s].drafts = [int(t) for t in row]
