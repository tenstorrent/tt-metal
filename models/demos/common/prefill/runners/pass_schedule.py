# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Pipeline positions of a looped model: one position per layer PASS, in execution order. ttnn-free.

A plain model runs every layer once, so the runner splits its layers over the pipeline ranks. A looped model
(GLM-5.4: layers 26..53 run twice with shared weights) runs some layers more than once, so the runner splits its
passes instead: GLM-5.4's 78 layers are 106 positions (26 prelude, 2 x 28 loop, 24 coda), and its four stages are
the position ranges [0, 26), [26, 54), [54, 82), [82, 106). ``PREFILL_NUM_LAYERS`` and ``PREFILL_PP_LAYER_COUNTS``
then count positions.

Contracts kept here, because the runner, the producer and the runtime must agree on them:

* A rank holds whole loop passes: a pass's gate reads the pass input, and the next pass needs every layer's KV and
  top-k of this one. So no rank starts strictly inside a loop pass (``valid_starts``).
* Layer acks: a position of loop pass ``k >= 2`` acks as ``layer + (k - 1) * num_layers``, never as its layer, so it
  cannot collide with pass 1, the coda or the MTP rows (ids ``num_layers ..``).
* KV migration: a looped model publishes no KV chunk table (``builds_kv_chunk_table``). Its decode KV handoff is not
  defined yet (GLM-5.4 Q2.3), and no table could tell a layer's own KV from a rank's copy of it (G2's bank A) or
  describe a pass-2 cache's window ring.
"""

from __future__ import annotations

from dataclasses import dataclass


def builds_kv_chunk_table(pass_schedule, *, migration_requested: bool, model_name: str) -> bool:
    """Whether the runner builds a KV chunk table: always for a plain model (``pass_schedule`` None), never for a looped
    one, which then refuses a run that asks for KV migration (publish or file export) rather than skip it silently."""
    if pass_schedule is None:
        return True
    if migration_requested:
        raise RuntimeError(
            f"{model_name}: KV migration of a looped model is not supported (its decode KV handoff is deferred); "
            "run with PREFILL_ENABLE_MIGRATION=0 PREFILL_MIGRATION_EXPORT_TO_FILE=0"
        )
    return False


def loop_ack_id(layer: int, loop_pass: int, num_layers: int) -> int:
    """Global layer id a pass of ``layer`` acks under: its own for passes 0 and 1, ``layer + (k - 1) * num_layers``
    for loop pass ``k >= 2``."""
    return layer + (loop_pass - 1) * num_layers if loop_pass >= 2 else layer


@dataclass(frozen=True)
class RankSlice:
    """What one rank builds for a position range: blocks ``[first_layer, first_layer + num_layers)`` and the loop
    passes it runs them in (empty when it holds no loop layer)."""

    first_layer: int
    num_layers: int
    loop_passes: tuple


@dataclass(frozen=True)
class PassSchedule:
    """``passes[pos] = (layer, loop_pass)``; ``loop_pass`` is 0 outside the loop and ``1..`` inside it."""

    passes: tuple
    num_layers: int

    def __post_init__(self):
        layers = sorted({layer for layer, _ in self.passes})
        if layers != list(range(self.num_layers)):
            raise ValueError(f"a pass schedule must run each of the {self.num_layers} layers, got layers {layers}")

    def __len__(self) -> int:
        return len(self.passes)

    def _pass_ranges(self) -> list:
        """[start, end) position range of every loop pass, in order."""
        ranges, pos = [], 0
        while pos < len(self.passes):
            k = self.passes[pos][1]
            end = pos + 1
            while end < len(self.passes) and self.passes[end][1] == k:
                end += 1
            if k:
                ranges.append((pos, end))
            pos = end
        return ranges

    def valid_starts(self) -> set:
        """Positions a rank may start at: any but those strictly inside a loop pass."""
        inside = {p for start, end in self._pass_ranges() for p in range(start + 1, end)}
        return set(range(len(self.passes))) - inside

    def rank_slice(self, first: int, count: int) -> RankSlice:
        """The blocks and loop passes of positions ``[first, first + count)``, which must hold whole loop passes."""
        if not (0 <= first and count >= 1 and first + count <= len(self.passes)):
            raise ValueError(f"positions [{first}, {first + count}) outside the {len(self.passes)}-pass schedule")
        for start, end in self._pass_ranges():
            if start < first < end or start < first + count < end:
                raise ValueError(
                    f"positions [{first}, {first + count}) cut loop pass {self.passes[start][1]} at positions "
                    f"[{start}, {end}); a rank holds whole loop passes"
                )
        held = self.passes[first : first + count]
        layers = [layer for layer, _ in held]
        first_layer, last_layer = min(layers), max(layers)
        if sorted(set(layers)) != list(range(first_layer, last_layer + 1)):
            raise ValueError(f"positions [{first}, {first + count}) hold non-contiguous layers {sorted(set(layers))}")
        loop_passes = tuple(sorted({k for _, k in held if k}))
        return RankSlice(first_layer, last_layer - first_layer + 1, loop_passes)

    def ack_id(self, pos: int) -> int:
        """Global layer id position ``pos`` acks under (see the module docstring)."""
        layer, k = self.passes[pos]
        return loop_ack_id(layer, k, self.num_layers)

    def ack_ids(self, first: int, count: int) -> list:
        return [self.ack_id(p) for p in range(first, first + count)]

    @property
    def ack_id_bound(self) -> int:
        """One past the largest ack id any position takes: the layer-id axis a completion reader must accept."""
        return max(self.ack_ids(0, len(self))) + 1
