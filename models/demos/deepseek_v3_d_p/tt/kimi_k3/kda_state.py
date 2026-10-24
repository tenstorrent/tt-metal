# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The KDA carries, owned outside the layer, with the user slot chosen on device.

`ttKDA.forward(hidden, state)` is purely functional: it reads the state it is given, retains
nothing, and hands back a freshly allocated replacement. Something has to own those carries across
a chunked prefill, for every resident user, and this is it — the KDA counterpart of `MlaKvCache`.

**Why the obvious version does not work.** Installing the returned state and deallocating the old
one is correct eagerly and wrong under trace, in a way that produces no error. A captured trace
records addresses: `forward` reads the carry at address A and writes its replacement to address B,
both baked into the command stream. Replay runs no Python — `SubDeviceTraceController.replay()`
only calls `ttnn.execute_trace` — so every replay reads A and writes B, and a host-side dict swap
between replays changes nothing the device will ever look at. The same holds for the user slot: a
slot picked in Python picks an address, and a capture keeps that address for every replay, so a
trace captured on slot 0 would advance slot 0 whatever slot the request is on (#59977).

So every address a capture touches is allocated once, before `begin_trace_capture`, and the slot is
resolved on device, the way MLA resolves its KV slot from `metadata[0]`:

* each layer keeps every slot's carry in one native buffer per state kind, `[num_slots, ...]`;
* each layer has one working carry, the only state `ttKDA.forward` ever reads;
* per chunk, inside the captured region: load `entries[slot] -> working`, run the layer, store the
  replacement into `entries[slot]` and export it into the engine's migration slabs at the same slot,
  then free it. `ttnn.experimental.indexed_entry_copy` reads `slot` on device for every copy.

The cost is three DRAM-to-DRAM copies of the carry per KDA layer per chunk (load, store, export). At
8x4 with tp=4 the recurrent carry is TP-sharded and SP-replicated at 1.50 MiB/chip and the
convolution tail is 54 KiB/chip, so an 18-KDA-layer slice moves ~85 MiB/chip/chunk — against a
5120-token chunk whose MLA and MoE traffic is orders of magnitude larger. Every slot's carries are
resident: ~28 MiB/chip per slot plus the working and zero carries.
"""

from __future__ import annotations

import ttnn
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaStates, slot_index


def _entries_like(state: KdaState, num_slots: int, device) -> KdaState:
    """Zeroed `[num_slots, ...]` buffers with `state`'s per-entry shape, dtype, layout and memory."""

    def entries(tensor: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.zeros(
            (num_slots, *tuple(tensor.shape)[1:]),
            dtype=tensor.dtype,
            layout=tensor.layout,
            device=device,
            memory_config=tensor.memory_config(),
        )

    return KdaState(recurrent=entries(state.recurrent), convolution=entries(state.convolution))


def _copy_entry(source: KdaState, destination: KdaState, slot: ttnn.Tensor) -> None:
    ttnn.experimental.indexed_entry_copy(source.recurrent, destination.recurrent, slot)
    ttnn.experimental.indexed_entry_copy(source.convolution, destination.convolution, slot)


class KdaStateCache:
    """One carry per (user slot, KDA layer), selected by a device-resident slot.

    Keyed by GLOBAL layer index, matching the block's own `layer_idx`, so a pipeline rank holding a
    slice needs no second numbering — unlike the KV cache, whose slots are rank-local because the
    cache is sized to the rank. Nothing here is sized to the rank: a carry belongs to a layer.
    """

    def __init__(self, layers: dict[int, ttKDA], num_slots: int = 1):
        if num_slots < 1:
            raise ValueError(f"num_slots must be at least 1, got {num_slots}")
        if not layers:
            raise ValueError("KdaStateCache needs at least one KDA layer")
        self._layers = dict(layers)
        self._num_slots = num_slots
        self._device = next(iter(self._layers.values())).device
        # `allocate_state` zeros, which is where a stream begins. These are the buffers whose addresses
        # a trace bakes in, so they are allocated once and never replaced.
        self._working: dict[int, KdaState] = {
            idx: layer.allocate_state(batch_size=1) for idx, layer in self._layers.items()
        }
        self._entries: dict[int, KdaState] = {
            idx: _entries_like(state, num_slots, self._device) for idx, state in self._working.items()
        }
        # Held for `reset`, so zeroing a slot is a copy rather than a reallocation.
        self._zeros: dict[int, KdaState] = {
            idx: layer.allocate_state(batch_size=1) for idx, layer in self._layers.items()
        }
        # The engine-owned contract copy (`KdaStates`), bound by the runtime before the first forward.
        # While bound, every commit also lands in it, so a migration reader sees each layer's state
        # as of the last committed chunk. Tests that build the transformer directly never bind one.
        self._slabs: KdaStates | None = None

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._layers))

    @property
    def slabs(self) -> KdaStates | None:
        return self._slabs

    def bind_slabs(self, slabs: KdaStates) -> None:
        """Attach the contract copy every commit exports into.

        Must happen before the first forward that is captured: the slab addresses get baked into the
        trace along with the carries'. Rebinding to a different object is refused for the same reason.
        """
        if self._slabs is not None and self._slabs is not slabs:
            raise RuntimeError("KDA state slabs are already bound; a capture may depend on their addresses")
        if tuple(slabs.layer_ids) != self.layer_ids:
            raise ValueError(f"slab layers {slabs.layer_ids} != this rank's KDA layers {self.layer_ids}")
        if slabs.num_slots != self._num_slots:
            raise ValueError(f"slabs hold {slabs.num_slots} slots but the carries hold {self._num_slots}")
        for layer_idx, state in self._working.items():
            if tuple(state.recurrent.shape) != slabs.geometry.recurrent_shape:
                raise ValueError(
                    f"layer {layer_idx} recurrent carry {tuple(state.recurrent.shape)} != "
                    f"contract geometry {slabs.geometry.recurrent_shape}"
                )
            if tuple(state.convolution.shape) != slabs.geometry.convolution_shape:
                raise ValueError(
                    f"layer {layer_idx} convolution carry {tuple(state.convolution.shape)} != "
                    f"contract geometry {slabs.geometry.convolution_shape}"
                )
        self._slabs = slabs

    def load(self, layer_idx: int, slot: ttnn.Tensor) -> KdaState:
        """Stage `slot`'s carry into this layer's working carry and return it. BORROWED, read-only.

        `slot` is a replicated one-element UINT32 device tensor (the traced metadata's slot word, or
        `slot_index`), read on device, so a captured load follows the slot written before each replay.
        """
        working = self._working[layer_idx]
        _copy_entry(self._entries[layer_idx], working, slot)
        return working

    def commit(self, layer_idx: int, new_state: KdaState, slot: ttnn.Tensor) -> None:
        """Store `new_state` as `slot`'s carry, export it to the slabs, and free it.

        Every copy is an ordinary device program that reads `slot` on device and belongs inside a trace
        capture, so replay advances the right user's recurrence with no host in the loop.
        """
        _copy_entry(new_state, self._entries[layer_idx], slot)
        if self._slabs is not None:
            # Same captured region, so the slab advances with the carry on every replay, and it lands
            # before any later MLA layer acks this chunk.
            self._slabs.export_layer(new_state, slot, layer_idx)
        ttnn.deallocate(new_state.recurrent)
        ttnn.deallocate(new_state.convolution)

    def reset(self, slot: int = 0) -> None:
        """Zero a slot's carries, ending whatever stream was in flight.

        A carry summarizes the whole prefix behind it, so a new request must not continue from the
        previous one's. Call this outside a captured region — at `actual_start == 0` — since a trace
        replays every chunk and would re-zero each time.
        """
        index = self._slot_index(slot)
        for layer_idx, entries in self._entries.items():
            _copy_entry(self._zeros[layer_idx], entries, index)
            if self._slabs is not None:
                self._slabs.export_layer(self._zeros[layer_idx], index, layer_idx)
        ttnn.deallocate(index)

    def import_layer(self, layer_idx: int, slot: int = 0) -> None:
        """Overwrite one carry from the contract copy (a migrated-in state). Outside any capture."""
        if self._slabs is None:
            raise RuntimeError("no KDA state slabs bound; nothing to import from")
        index = self._slot_index(slot)
        staged = self._layers[layer_idx].allocate_state(batch_size=1)
        self._slabs.import_layer(staged, slot, layer_idx)
        _copy_entry(staged, self._entries[layer_idx], index)
        for tensor in (index, staged.recurrent, staged.convolution):
            ttnn.deallocate(tensor)

    def read(self, layer_idx: int, slot: int = 0) -> KdaState:
        """A copy of `slot`'s committed carry for this layer. OWNED: the caller frees both tensors."""
        index = self._slot_index(slot)
        copy = self._layers[layer_idx].allocate_state(batch_size=1)
        _copy_entry(self._entries[layer_idx], copy, index)
        ttnn.deallocate(index)
        return copy

    def deallocate(self) -> None:
        for states in (self._working, self._entries, self._zeros):
            for state in states.values():
                ttnn.deallocate(state.recurrent)
                ttnn.deallocate(state.convolution)
        self._working, self._entries, self._zeros = {}, {}, {}

    def _slot_index(self, slot: int) -> ttnn.Tensor:
        if not 0 <= slot < self._num_slots:
            raise ValueError(f"slot {slot} out of range [0, {self._num_slots})")
        return slot_index(self._device, slot)
