# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Trace-replayability of the device-valued slot slice.

Request-mode tracing needs one captured trace to serve any user: the slot begin rides a persistent device
tensor, and re-targeting that tensor between replays must move the slot WITHOUT recapture. The slice reader
NoC-reads the begin tensor at kernel runtime, so execute_trace observes fresh contents. This drives the
production path end to end: capture ``_slot_slice_device`` over a ``MiniMaxKVCache`` whose begin tensors
point at user A, replay -> A's slot; ``set_read_user(B)`` (the runtime's per-replay re-target), replay the
SAME trace -> B's slot; back to A rules out a one-way latch. Slot k is filled with the value k so a
mis-targeted read is caught by content, not just shape.
"""

import torch

import ttnn
from models.demos.minimax_m3.tt.attention.prefill import _slot_slice_device

from ..test_factory import parametrize_mesh_with_fabric
from .test_device_slot_slice import HEAD_DIM, MAX_ROWS, N_LAYERS, N_SLOTS, make_kv_cache, to_device


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_device_slot_slice_trace_retarget(mesh_device, device_params):
    filled = torch.arange(N_SLOTS, dtype=torch.float32).view(N_SLOTS, 1, 1, 1).expand(N_SLOTS, 1, MAX_ROWS, HEAD_DIM)
    packed = to_device(filled.contiguous(), mesh_device)
    kv_cache = make_kv_cache(packed)

    layer_idx, user_a, user_b = 1, 0, 2
    slot_a = user_a * N_LAYERS + layer_idx
    slot_b = user_b * N_LAYERS + layer_idx

    def host_ref(slot):
        return ttnn.to_torch(ttnn.slice(packed, (slot, 0, 0, 0), (slot + 1, 1, MAX_ROWS, HEAD_DIM)))

    def sliced():
        return _slot_slice_device(packed, kv_cache, layer_idx, slot_a, MAX_ROWS, HEAD_DIM, mesh_device)

    # Warm pass creates the persistent begin/end tensors (pointing at slot_a) and compiles the programs.
    sliced().deallocate(True)
    # Hold the slot tensors fixed exactly as capture_chunk_trace does: a host->device copy inside a capture
    # is illegal, so the captured forward must only read the begin tensor the warm pass set.
    with kv_cache.frozen_slots():
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        out = sliced()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)

    def replay_expecting(slot):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        got = ttnn.to_torch(out)
        assert torch.equal(
            got, host_ref(slot)
        ), f"expected slot {slot} (value {slot}.0), replay read value {got.flatten()[0].item():.1f}"

    try:
        replay_expecting(slot_a)
        kv_cache.set_read_user(user_b)  # host update outside the trace re-targets every layer's begin tensor
        replay_expecting(slot_b)
        kv_cache.set_read_user(user_a)
        replay_expecting(slot_a)
    finally:
        ttnn.release_trace(mesh_device, tid)
