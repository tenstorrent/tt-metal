# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The interface every model's CPU reference implements, and the block-graph runner built on it.

A model's ``hooks.reference(spec, layers=None, dtype=torch.float32)`` returns an object with:

    layer_ids: list[int]
    new_state(max_seq) -> state                      per-layer chunked-prefill state (KV cache, recurrent state, ...)
    forward_chunk(tokens, start, state, rec, logits_last_n=0) -> (final_hidden [S, H], logits [n, V] or None)
    state_tensors(state, layer, length) -> {name: tensor}     names = spec.state.tensors, positions [0, length)
    load_state(state, layer, tensors, length) -> None         inverse of state_tensors (golden prefix -> state)
    block_graph(layer) -> list[Step]                          the block's components in execution order
    component(layer, step_name) -> fn(ctx, *inputs) -> tensor the CPU implementation of one step
    chunk_context(layer, start, length, state) -> Ctx        what stateful / positional steps need (rope tables, ...)

Recorder names. ``forward_chunk`` calls ``rec(name, tensor)`` at every boundary: ``embed``, ``final_norm``,
``logits``, and per layer ``L{i}.in`` (block input), ``L{i}.<step output>`` for every step, where the last step's
output is ``out``. Extra names (e.g. ``L{i}.q``) are allowed. The golden step stores these; component, swap and
ladder tests read them.

The reference's own forward should run each block through ``run_block`` so the graph is the code, not a
description of it. The reference gate replays the graph from recorded boundaries and requires an exact match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

Recorder = Callable[[str, torch.Tensor], None]


def noop(name: str, t: torch.Tensor) -> None:
    pass


@dataclass(frozen=True)
class Step:
    """One component of a block: reads named boundary tensors, writes one."""

    name: str  # unique in the block: "attn_norm", "attention", "attn_residual", ...
    inputs: tuple[str, ...]  # boundary names read: "in", "attn_norm", ...
    output: str  # boundary name written; the block's last step writes "out"
    kind: str = "op"  # template family: norm, attention, mlp, moe, router, residual, embedding, op
    stateful: bool = False  # reads or writes the per-layer state (the swap test then carries a state prefix)


@dataclass
class Ctx:
    """Per (layer, chunk) context handed to every step."""

    layer: int
    start: int
    length: int
    state: Any = None  # the implementation's own state object for this layer
    extra: dict = field(default_factory=dict)


def validate_graph(steps: list[Step]) -> list[str]:
    errs, known, names = [], {"in"}, set()
    for s in steps:
        if s.name in names:
            errs.append(f"duplicate step {s.name}")
        names.add(s.name)
        for i in s.inputs:
            if i not in known:
                errs.append(f"step {s.name} reads {i!r} before any step writes it")
        if s.output in known:
            errs.append(f"step {s.name} overwrites boundary {s.output!r}")
        known.add(s.output)
    if not steps or steps[-1].output != "out":
        errs.append("the last step must write 'out'")
    return errs


def run_block(
    steps: list[Step],
    component: Callable[[str], Callable],
    ctx: Ctx,
    h_in: torch.Tensor,
    rec: Recorder = noop,
    overrides: dict[str, Callable] | None = None,
    prefix: str = "",
) -> torch.Tensor:
    """Execute a block graph. ``overrides[step] = fn(ctx, *inputs)`` replaces a step (device module, stub)."""
    overrides = overrides or {}
    env = {"in": h_in}
    rec(f"{prefix}in", h_in)
    for s in steps:
        fn = overrides.get(s.name) or component(s.name)
        env[s.output] = fn(ctx, *[env[i] for i in s.inputs])
        rec(f"{prefix}{s.output}", env[s.output])
    return env["out"]


def boundary_names(steps: list[Step]) -> list[str]:
    return ["in"] + [s.output for s in steps]
