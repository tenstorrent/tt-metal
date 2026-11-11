# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fakes of the chunked prefill driver's model and ttnn, shared by the driver's no-device tests."""

from __future__ import annotations

import itertools
from types import SimpleNamespace

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module


def prefill_next_context(context, token: int) -> tuple[int, int]:
    """The n-gram context after one token: ``(c1, token)`` (the resident lookup's rule)."""
    return (2 if context is None else context[1], token)


def prefill_context_after(tokens, context):
    for token in tokens:
        context = prefill_next_context(context, token)
    return context


class PrefillFakeModel:
    """Records the driver's calls in ``calls``, a chunk state by its ``rows``; ``prepare_chunk_inputs`` returns the
    rows + 1 n-gram contexts the PLE lookup would."""

    def __init__(self, allocated_context: int) -> None:
        self.allocated_context = allocated_context
        self.calls: list[tuple] = []
        self.vision: list[tuple] = []  # (positions, features) per chunk and ("finish", rope_shift) at the hand-off

    def reset_chunk_state_inplace(self, state, chunk_state) -> None:
        self.calls.append(("reset", state, chunk_state.rows))

    def write_chunk_accepted(self, chunk_state, accepted: int) -> None:
        self.calls.append(("accepted", chunk_state.rows, accepted))

    def prepare_chunk_inputs(self, chunk_state, token_ids, *, ple_context, positions=None, features=None):
        tokens = tuple(token_ids)
        assert len(tokens) == chunk_state.rows
        self.calls.append(("prepare", chunk_state.rows, tokens, ple_context))
        self.vision.append((positions, features))
        contexts = [ple_context]
        for token in tokens:
            contexts.append(prefill_next_context(contexts[-1], token))
        return SimpleNamespace(chunk_state=chunk_state, tokens=tokens, contexts=tuple(contexts))

    def upload_chunk_inputs(self, chunk_state, prepared) -> None:
        assert prepared.chunk_state is chunk_state  # the upload lands in the state the rows were prepared for
        self.calls.append(("upload", chunk_state.rows, prepared.tokens))

    def finish_prefill(self, state, chunk_state, prefilled: int, *, rope_shift: int = 0) -> None:
        self.calls.append(("finish", chunk_state.rows, prefilled))
        self.vision.append(("finish", rope_shift))

    def forward_prefill_chunk_generic(self, chunk_state, state, *, gdn_step_anchor: bool = False, mtp=None) -> None:
        self.calls.append(("eager", chunk_state.rows, gdn_step_anchor, mtp))


def install_prefill_fakes(monkeypatch):
    """A ``PrefillFakeModel`` whose ``calls`` (also ``log``) record the driver's ttnn calls on the mesh ``"mesh"``
    (replays, numbered events, synchronizes) and trace verifications too, and a ``forced_step`` that logs each
    alignment step."""
    model = PrefillFakeModel(allocated_context=70000)
    log = model.calls
    events = itertools.count()

    def execute(mesh, trace_id, *, cq_id, blocking):
        assert mesh == "mesh" and cq_id == 0
        log.append(("replay", trace_id, blocking))

    def record_event(mesh, cq_id):
        log.append(("record", next(events)))
        return log[-1][1]

    class Tracker:
        @staticmethod
        def verify_before_replay(mesh, trace_id) -> None:
            assert mesh == "mesh"
            log.append(("verify", trace_id))

    def forced_step(token: int, context):
        log.append(("forced_step", token, context))
        return prefill_next_context(context, token)

    fake_ttnn = SimpleNamespace(
        _ttnn_execute_trace=execute,
        record_event=record_event,
        event_synchronize=lambda event: log.append(("wait", event)),
        synchronize_device=lambda mesh: log.append(("synchronize",)),
    )
    monkeypatch.setattr(driver_module, "ttnn", fake_ttnn)
    monkeypatch.setattr(driver_module, "TraceAllocationTracker", Tracker)
    return SimpleNamespace(model=model, log=log, forced_step=forced_step)
