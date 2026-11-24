# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 execution order (torch-free): the CPU reference stack and the device transformer both run it, and the
runner places its passes on pipeline stages (``pass_schedule``)."""

import copy
from dataclasses import dataclass


@dataclass(frozen=True)
class LoopStep:
    """One step of the GLM-5.4 schedule: a decoder layer, or the damp gate closing loop pass ``loop_pass``.

    ``loop_pass`` is 0 for prelude and coda layers and 1..loop_count inside the loop.
    """

    kind: str  # "layer" | "gate"
    loop_pass: int
    layer: int | None = None


def loop_schedule(num_layers: int, loop_first_layer: int, loop_last_layer: int, loop_count: int) -> list[LoopStep]:
    """Execution order: prelude, then loop_count x (loop layers, gate), then coda."""
    if not 0 <= loop_first_layer <= loop_last_layer < num_layers or loop_count < 1:
        raise ValueError(f"bad loop {loop_first_layer}..{loop_last_layer} x{loop_count} in {num_layers} layers")
    steps = [LoopStep("layer", 0, i) for i in range(loop_first_layer)]
    for k in range(1, loop_count + 1):
        steps += [LoopStep("layer", k, i) for i in range(loop_first_layer, loop_last_layer + 1)]
        steps.append(LoopStep("gate", k))
    steps += [LoopStep("layer", 0, i) for i in range(loop_last_layer + 1, num_layers)]
    return steps


def loop_count(config) -> int:
    """How many times ``config`` runs its loop layers: ``config.loop_count``, or 0 for a model without a loop (GLM-5.3
    and every other model). The one reading of the field the model, the cache allocator and the runner share."""
    return getattr(config, "loop_count", None) or 0


# --- Stage placement (pipeline ranks over the 106-pass schedule) ---


def pass_schedule(config):
    """The pipeline positions of ``config``: one per layer pass (``PassSchedule``), the loop's passes in order."""
    from models.demos.common.prefill.runners.pass_schedule import PassSchedule

    steps = loop_schedule(config.num_hidden_layers, config.loop_first_layer, config.loop_last_layer, loop_count(config))
    return PassSchedule(
        passes=tuple((s.layer, s.loop_pass) for s in steps if s.kind == "layer"), num_layers=config.num_hidden_layers
    )


def runs_later_loop_pass(loop_passes) -> bool:
    """A stage running some loop pass >= 2 (the whole loop on one stage, or GLM-5.4 G2): it holds pass-2 state, a
    window ring past each cache slot's pass-1 positions (tt.mla.two_bank). ``loop_passes`` is the stage's loop passes
    (``PassSchedule.rank_slice``); None means every pass. The model and the cache allocator both decide by this."""
    return loop_passes is None or any(p >= 2 for p in loop_passes)


def runs_only_later_loop_passes(loop_passes) -> bool:
    """A stage running the loop only in passes >= 2 (GLM-5.4 G2): it runs no indexer (every pass-2 layer reuses its
    pass-1 layer's top-k, A5) and receives pass 1's KV and top-k from the pass-1 stage (the loop handoff). None means
    every pass."""
    return loop_passes is not None and bool(loop_passes) and 1 not in loop_passes


def stage_indexer_types(config, loop_passes: tuple) -> list:
    """The full/shared map a stage builds its blocks with: the model's, except that a stage running the loop only
    in passes >= 2 marks every loop layer ``shared`` (no indexer weights, no index-cache slot)."""
    types = list(config.indexer_types)
    if runs_only_later_loop_passes(loop_passes):
        for layer in range(config.loop_first_layer, config.loop_last_layer + 1):
            types[layer] = "shared"
    return types


def stage_config(config, loop_passes):
    """The config a stage builds its blocks and caches with: ``config`` itself, except on a stage running the loop
    only in passes >= 2 (G2), which gets a shallow copy with its stage map (``stage_indexer_types``)."""
    if not runs_only_later_loop_passes(loop_passes):
        return config
    staged = copy.copy(config)
    staged.indexer_types = stage_indexer_types(config, loop_passes)
    return staged
