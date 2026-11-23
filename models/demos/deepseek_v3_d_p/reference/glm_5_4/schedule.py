# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 execution order (torch-free): the CPU reference stack and the device transformer both run it, and the
runner places its passes on pipeline stages (``pass_schedule``)."""

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


# --- Stage placement (pipeline ranks over the 106-pass schedule) ---


def pass_schedule(config):
    """The pipeline positions of ``config``: one per layer pass (``PassSchedule``), the loop's passes in order."""
    from models.demos.common.prefill.runners.pass_schedule import PassSchedule

    steps = loop_schedule(config.num_hidden_layers, config.loop_first_layer, config.loop_last_layer, config.loop_count)
    return PassSchedule(
        passes=tuple((s.layer, s.loop_pass) for s in steps if s.kind == "layer"), num_layers=config.num_hidden_layers
    )


def runs_loop_without_indexer(config, loop_passes: tuple) -> bool:
    """A stage running the loop only in passes >= 2 (GLM-5.4 G2) runs no indexer: every pass-2 layer reuses its
    pass-1 layer's top-k (A5), received from the pass-1 stage."""
    return bool(loop_passes) and 1 not in loop_passes


def stage_indexer_types(config, loop_passes: tuple) -> list:
    """The full/shared map a stage builds its blocks with: the model's, except that a stage running the loop only
    in passes >= 2 marks every loop layer ``shared`` (no indexer weights, no index-cache slot)."""
    types = list(config.indexer_types)
    if runs_loop_without_indexer(config, loop_passes):
        for layer in range(config.loop_first_layer, config.loop_last_layer + 1):
            types[layer] = "shared"
    return types
