# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The KDA carries, owned outside the layer and pinned to one address for the run.

`ttKDA.forward(hidden, state)` is purely functional: it reads the state it is given, retains
nothing, and hands back a freshly allocated replacement. Something has to own those carries across
a chunked prefill, and this is it — the KDA counterpart of `MlaKvCache`.

**Why the obvious version does not work.** Installing the returned state and deallocating the old
one is correct eagerly and wrong under trace, in a way that produces no error. A captured trace
records addresses: `forward` reads the carry at address A and writes its replacement to address B,
both baked into the command stream. Replay runs no Python — `SubDeviceTraceController.replay()`
only calls `ttnn.execute_trace` — so every replay reads A and writes B, and a host-side dict swap
between replays changes nothing the device will ever look at. The recurrence silently restarts from
whatever is at A on every chunk, which is a plausible-looking model that is wrong past the first
5120 tokens. Worse, deallocating the superseded pair *during capture* frees A while the recorded
stream still reads it, so the allocator may hand A to a later op inside the same capture.

So the carries are allocated once, before `begin_trace_capture`, and the write-back happens
**inside** the captured region: `ttnn.copy(new -> persistent)`, then free the return. `ttnn.copy` is
`ttnn::prim::copy` with a preallocated output — an ordinary capturable device program — and this is
the same shape the KV cache write already has, which is what the existing traced prefill is built
on. `models/demos/blackhole/qwen36/tt/model.py::_restore_gdn_scratch` reaches the same conclusion
for the same reason, in its own words: "preserving the addresses the trace baked in".

The cost is one DRAM-to-DRAM copy of the carry per KDA layer per chunk. At 8x4 with tp=4 the
recurrent carry is TP-sharded and SP-replicated at 1.50 MiB/chip and the convolution tail is
54 KiB/chip, so an 18-KDA-layer slice moves ~28 MiB/chip/chunk and holds the same again — against a
5120-token chunk whose MLA and MoE traffic is orders of magnitude larger.
"""

from __future__ import annotations

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA


class KdaStateCache:
    """One address-stable carry per (user slot, KDA layer).

    Keyed by GLOBAL layer index, matching the block's own `layer_idx`, so a pipeline rank holding a
    slice needs no second numbering — unlike the KV cache, whose slots are rank-local because the
    cache is sized to the rank. Nothing here is sized to the rank: a carry belongs to a layer.
    """

    def __init__(self, layers: dict[int, ttKDA], num_slots: int = 1):
        if num_slots < 1:
            raise ValueError(f"num_slots must be at least 1, got {num_slots}")
        self._layers = dict(layers)
        self._num_slots = num_slots
        # `allocate_state` zeros, which is where a stream begins; these are the buffers whose
        # addresses a trace will bake in, so they are allocated once and never replaced.
        self._states: list[dict[int, KdaState]] = [
            {idx: layer.allocate_state(batch_size=1) for idx, layer in self._layers.items()} for _ in range(num_slots)
        ]
        # Held for `reset`, so zeroing a slot is a copy rather than a reallocation. Reallocating
        # would move the addresses a capture depends on.
        self._zeros: dict[int, KdaState] = {
            idx: layer.allocate_state(batch_size=1) for idx, layer in self._layers.items()
        }

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._layers))

    def read(self, layer_idx: int, slot: int = 0) -> KdaState:
        """This layer's live carry. BORROWED — `ttKDA.forward` only reads it, and so must callers."""
        return self._states[slot][layer_idx]

    def commit(self, layer_idx: int, new_state: KdaState, slot: int = 0) -> None:
        """Advance the carry in place and free the replacement `forward` allocated.

        Both copies are ordinary device programs and belong inside a trace capture, so replay
        advances the recurrence with no host in the loop.
        """
        current = self._states[slot][layer_idx]
        ttnn.copy(new_state.recurrent, current.recurrent)
        ttnn.copy(new_state.convolution, current.convolution)
        ttnn.deallocate(new_state.recurrent)
        ttnn.deallocate(new_state.convolution)

    def reset(self, slot: int = 0) -> None:
        """Zero a slot's carries, ending whatever stream was in flight.

        A carry summarizes the whole prefix behind it, so a new request must not continue from the
        previous one's. Call this outside a captured region — at `actual_start == 0` — since a trace
        replays every chunk and would re-zero each time.
        """
        zeros = self._zeros
        for layer_idx, state in self._states[slot].items():
            ttnn.copy(zeros[layer_idx].recurrent, state.recurrent)
            ttnn.copy(zeros[layer_idx].convolution, state.convolution)

    def deallocate(self) -> None:
        for slot_states in self._states:
            for state in slot_states.values():
                ttnn.deallocate(state.recurrent)
                ttnn.deallocate(state.convolution)
        for state in self._zeros.values():
            ttnn.deallocate(state.recurrent)
            ttnn.deallocate(state.convolution)
        self._states = []
        self._zeros = {}
