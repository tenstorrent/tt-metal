# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-slot lifecycle and inactive-row protection around native TP4 decode.

Integration: allocate BatchState(model, cache) before capture, prepare its slot
ops, and keep it alive with the cache. In a masked model trace call before_layer
and after_layer around each existing decode_batch_packed; the all-active trace
must omit these calls. set_active is a request/scheduler transition, not a
per-token operation. Use plus_one(..., skip_negative_entries=True) for positions.

Full-attention kernels already skip cache/SDPA rows at position -1. Their RoPE
lookup must independently use safe_rope_indices while retaining the original
signed position for paged_fused_update_cache and SDPA. No decoder kernel or
projection/dtype policy is replaced here.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.inactive_state_copy import inactive_state_copy


def safe_rope_indices(positions):
    """Clamp only embedding indices; never replace the cache/SDPA positions."""
    return ttnn.typecast(ttnn.maximum(positions, 0), ttnn.uint32)


@dataclass
class _Scratch:
    conv_snapshot: object
    recurrent_snapshot: object
    conv_merge: object
    recurrent_merge: object
    zero_conv: object
    zero_recurrent: object


class BatchState:
    """Own scratch/masks for one complete ordered shared state-pool binding.

    Linear layers share scratch because execution is serial: before_layer copies
    layerN into snapshot, its native forward mutates its own pool, after_layer
    restores inactive rows before layerN+1 overwrites scratch. No tensor alias of
    snapshot/merge escapes an operation. All backing buffers predate captures.
    Sampling/RNG ownership is intentionally handled by the generator.
    """

    def __init__(self, model, cache):
        import torch

        self.model, self.cache = model, cache
        self.selective_restore = model.selective_inactive_restore
        if len(cache) != len(model.layers) or not cache:
            raise ValueError("Cache must cover exactly the configured model layers")
        self.capacity = len(cache[0])
        if not 1 <= self.capacity <= 32 or any(len(states) != self.capacity for states in cache):
            raise ValueError("Cache requires a consistent fixed-slot capacity from 1 through 32")
        self.scratch = {}
        self.layer_scratch = {}
        for index, (layer, states) in enumerate(zip(model.layers, cache)):
            if layer.kind != "linear_attention":
                continue
            conv, recurrent = states[0].conv, states[0].recurrent
            if not all(
                state.conv is conv and state.recurrent is recurrent and state.state_row == row
                for row, state in enumerate(states)
            ):
                raise ValueError("Masked native decode requires the complete ordered linear pool")
            if tuple(conv.shape) != (3, 1, self.capacity, 2048) or tuple(recurrent.shape) != (
                self.capacity,
                8,
                128,
                128,
            ):
                raise ValueError("Unexpected native linear pool geometry")
            if conv.dtype != ttnn.bfloat16 or recurrent.dtype != ttnn.float32:
                raise ValueError("Preserve BF16 convolution and FP32 recurrence pools")
            key = (tuple(conv.shape), tuple(recurrent.shape))
            if key not in self.scratch:
                self.scratch[key] = _Scratch(
                    ttnn.zeros_like(conv),
                    ttnn.zeros_like(recurrent),
                    ttnn.zeros_like(conv),
                    ttnn.zeros_like(recurrent),
                    model.upload(torch.zeros((1, 1, 3, 2048)), layout=ttnn.ROW_MAJOR_LAYOUT),
                    model.upload(torch.zeros((1, 8, 128, 128)), dtype=ttnn.float32),
                )
            self.layer_scratch[index] = self.scratch[key]
        # Singleton dimensions broadcast to channel/head dimensions. FP32 mask
        # avoids promoting/demoting recurrent state at the selection boundary.
        self.active_conv = model.upload(torch.ones((1, 1, self.capacity, 1)))
        self.active_recurrent = model.upload(torch.ones((self.capacity, 1, 1, 1)), dtype=ttnn.float32)
        self.active_rows = model.upload(
            torch.ones((1, 32), dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.active = (True,) * self.capacity
        self.mask_refreshes = 0

    def set_active(self, active):
        """Refresh persistent masks only when the scheduler changes active slots."""
        import torch

        active = tuple(bool(value) for value in active)
        if len(active) != self.capacity:
            raise ValueError("Active mask must cover every fixed slot")
        if active == self.active:
            return False
        values = torch.tensor(active, dtype=torch.float32)
        conv = self.model.upload(values.reshape(1, 1, self.capacity, 1), device=False)
        recurrent = self.model.upload(values.reshape(self.capacity, 1, 1, 1), dtype=ttnn.float32, device=False)
        ttnn.copy_host_to_device_tensor(conv, self.active_conv)
        ttnn.copy_host_to_device_tensor(recurrent, self.active_recurrent)
        rows = torch.ones((1, 32), dtype=torch.int32)
        rows[0, : self.capacity] = values.to(torch.int32)
        host_rows = self.model.upload(rows, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=False)
        ttnn.copy_host_to_device_tensor(host_rows, self.active_rows)
        self.active = active
        self.mask_refreshes += 1
        return True

    def before_layer(self, index):
        """Device-only snapshot before the unchanged pooled native layer call."""
        scratch = self.layer_scratch.get(index)
        if scratch is None:
            return
        state = self.cache[index][0]
        ttnn.copy(state.conv, scratch.conv_snapshot)
        if self.selective_restore:
            inactive_state_copy(state.recurrent, scratch.recurrent_snapshot, self.active_rows)
        else:
            ttnn.copy(state.recurrent, scratch.recurrent_snapshot)

    def after_layer(self, index):
        """Device-only exact selection, retaining inactive FP32 values as FP32."""
        scratch = self.layer_scratch.get(index)
        if scratch is None:
            return
        state = self.cache[index][0]
        ttnn.where(self.active_conv, state.conv, scratch.conv_snapshot, output_tensor=scratch.conv_merge)
        ttnn.copy(scratch.conv_merge, state.conv)
        if self.selective_restore:
            inactive_state_copy(scratch.recurrent_snapshot, state.recurrent, self.active_rows)
        else:
            ttnn.where(
                self.active_recurrent,
                state.recurrent,
                scratch.recurrent_snapshot,
                output_tensor=scratch.recurrent_merge,
            )
            ttnn.copy(scratch.recurrent_merge, state.recurrent)

    def _snapshot(self, index):
        scratch = self.layer_scratch[index]
        state = self.cache[index][0]
        # Scheduler permutations/reset warmup must snapshot every row, including
        # active ones. Selective snapshots are only for the masked decode guard.
        ttnn.copy(state.conv, scratch.conv_snapshot)
        ttnn.copy(state.recurrent, scratch.recurrent_snapshot)
        return SimpleNamespace(conv=scratch.conv_snapshot, recurrent=scratch.recurrent_snapshot, state_row=None)

    def _validate_slots(self, slots):
        slots = tuple(int(slot) for slot in slots)
        if len(set(slots)) != len(slots) or any(slot < 0 or slot >= self.capacity for slot in slots):
            raise ValueError("Slots must be distinct valid fixed-slot indices")
        return slots

    def permute_slots(self, remap):
        """Destination i takes old remap[i], simultaneously across the whole pool.

        No KV-page movement: vLLM owns page addresses and supplies remapped page
        tables separately. The operation moves slot-owned convolution/recurrence,
        including off-batch slots, before any destination consumes its new state.
        The generator must move its sampler state once under the same permutation.
        Call only between drained submissions, never during an in-flight read.
        """
        remap = self._validate_slots(remap)
        if len(remap) != self.capacity:
            raise ValueError("slot_remap must permute the entire capacity, including off-batch slots")
        for index in self.layer_scratch:
            layer, states = self.model.layers[index], self.cache[index]
            old = self._snapshot(index)
            for destination, source in enumerate(remap):
                old.state_row = source
                for name in ("conv", "recurrent"):
                    value = layer.read_linear_state(old, name)
                    layer.write_linear_state(states[destination], name, value)

    def reset_slots(self, slots):
        """Clear only selected recurrent/conv rows; full KV is overwritten by prefill."""
        slots = self._validate_slots(slots)
        for index, scratch in self.layer_scratch.items():
            layer, states = self.model.layers[index], self.cache[index]
            for slot in slots:
                layer.write_linear_state(states[slot], "conv", scratch.zero_conv)
                layer.write_linear_state(states[slot], "recurrent", scratch.zero_recurrent)

    def warmup_slot_ops(self):
        """Prepare every scalar slice/write signature before capture, preserving state.

        Identity permutation exercises every source and destination index. Indices
        belong to separate read/write programs, so no quadratic pair enumeration
        is needed. Reset operations are warmed against scratch-saved real state.
        """
        self.permute_slots(range(self.capacity))
        for index, scratch in self.layer_scratch.items():
            layer, states = self.model.layers[index], self.cache[index]
            self._snapshot(index)
            for state in states:
                layer.write_linear_state(state, "conv", scratch.zero_conv)
                layer.write_linear_state(state, "recurrent", scratch.zero_recurrent)
            ttnn.copy(scratch.conv_snapshot, states[0].conv)
            ttnn.copy(scratch.recurrent_snapshot, states[0].recurrent)
            # Compile exact BF16/FP32 selection variants too. Both sides currently
            # contain the same original state, so warmup has no numerical effect.
            self.after_layer(index)
