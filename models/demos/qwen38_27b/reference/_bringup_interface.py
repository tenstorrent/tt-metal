# Vendored by bringup_fw.scaffold from bringup-sc tools/prefill/bringup_fw/reference/interface.py (sha256 efb6f023e5b7944e5cbe5f096786c42bc00f0f42eeca5746bd46436dbd47b6d4). Do not edit: rerun the scaffold.
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The interface every model's CPU reference implements, and the block-graph runner built on it.

A model's ``hooks.reference(spec, layers=None, dtype=torch.float32)`` returns an object with the methods below.

Dtypes. ``dtype`` is the compute dtype. Every weight tensor is held in ``weight_dtype(spec)`` (spec
``reference.weight_dtype``, bf16 by default: half the host memory, and the run's RAM checks count on it) and cast to
the activation's dtype in the op that uses it, ``F.linear(x, as_compute(w, x))``, never kept as an fp32 copy. fp32
weights slip in through ``.float()`` at load, ``nn.Linear`` / ``nn.Embedding`` built without a dtype, and dequantizing
loaders that return fp32 (cast their result). The model's own fp32 work (norm variance, softmax, router logits, RoPE
tables, recurrent state) stays fp32 as upstream does it.

    layer_ids: list[int]
    new_state(max_seq) -> state                      per-layer chunked-prefill state (KV cache, recurrent state, ...)
    forward_chunk(tokens, start, state, rec, logits_last_n=0) -> (final_hidden [S, H], logits [n, V] or None)
    state_tensors(state, layer, length) -> {name: tensor}     names = spec.state_names(layer), positions [0, length);
                                                              a fixed-size tensor (spec state.fixed) is the current one
    load_state(state, layer, tensors, length) -> None         inverse of state_tensors (golden prefix -> state; a
                                                              fixed-size tensor is the state as of position length)
    block_graph(layer) -> list[Step]                          the block's components in execution order
    component(layer, step_name) -> fn(ctx, *inputs) -> tensor the CPU implementation of one step
    chunk_context(layer, start, length, state) -> Ctx        what stateful / positional steps need (rope tables, ...)

Recorder names. ``forward_chunk`` calls ``rec(name, tensor)`` at every boundary: ``embed``, ``final_norm``,
``logits``, and per layer ``L{i}.in`` (block input), ``L{i}.<step output>`` for every step, where the last step's
output is ``out``. Extra names (e.g. ``L{i}.q``) are allowed. The golden step stores these; component, swap and
ladder tests read them.

The reference's own forward should run each block through ``run_block`` so the graph is the code, not a
description of it. The reference gate replays the graph from recorded boundaries and requires an exact match.

Profile sections. ``run_block`` marks one section per step (``signpost(step name)``); device code may mark finer ones
itself (``signpost("attention.sdpa")``, imported from this module). A signpost does nothing unless the framework's
profiler (bringup_fw/testing/profiler.py) has set ``SECTION_HOOK`` while it profiles, so a handed-off package runs
it for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

Recorder = Callable[[str, torch.Tensor], None]


def noop(name: str, t: torch.Tensor) -> None:
    pass


SECTION_HOOK: Callable[[str], None] | None = None  # set by the framework's profiler while it profiles


def signpost(name: str) -> None:
    """Start a profile section (module docstring); free when nothing profiles."""
    if SECTION_HOOK is not None:
        SECTION_HOOK(name)


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
        signpost(s.name)  # a profile section per step
        fn = overrides.get(s.name) or component(s.name)
        env[s.output] = fn(ctx, *[env[i] for i in s.inputs])
        rec(f"{prefix}{s.output}", env[s.output])
    return env["out"]


def weight_dtype(spec) -> torch.dtype:
    """The dtype the reference keeps its weights in (spec ``reference.weight_dtype``)."""
    return getattr(torch, spec.get("reference.weight_dtype"))


def as_compute(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """A stored weight in the activation's dtype, for the op that uses it: ``F.linear(x, as_compute(w, x))``."""
    return w if w.dtype == x.dtype else w.to(x.dtype)


def boundary_names(steps: list[Step]) -> list[str]:
    return ["in"] + [s.output for s in steps]
