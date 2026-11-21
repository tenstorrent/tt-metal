# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""A weight version that the ThreadedWeightBridge receiver gets while CQ0 captures a trace must reach the device.

Without the lock, the capture puts the receiver's CQ1 writes and event into the CQ0 trace: the pad keeps its old
value, and a CQ0 wait for the CQ1 event never ends.
"""

import collections
import contextlib
import json
import queue
import struct
import time

import pytest
import torch
import ttnn

from ttml.trainers.grpo_trainer.remote_rollout import threaded_weight_bridge as twb
from ttml.trainers.grpo_trainer.remote_rollout.weight_bridge import TTML_RANK, _torch_save_bytes

pytestmark = pytest.mark.requires_device

SHAPE = [1, 1, 32, 1024]


class _QueueContext:
    """The receiver's end of an in-memory communicator."""

    def __init__(self):
        self._queues = collections.defaultdict(queue.Queue)

    def put(self, blob, len_tag, body_tag):
        self._queues[len_tag].put(struct.pack("<Q", len(blob)))
        if blob:
            self._queues[body_tag].put(blob)

    def recv(self, nbytes, source, tag):
        return self._queues[tag].get(timeout=60)

    def empty(self, tag):
        return self._queues[tag].empty()


@pytest.mark.timeout(300)
def test_receiver_writes_wait_for_trace_capture(monkeypatch):
    ctx = _QueueContext()
    monkeypatch.setattr(twb, "duplicate_distributed_context", lambda: ctx)
    monkeypatch.setattr(twb, "exit_on_error", contextlib.nullcontext)
    monkeypatch.setattr(ttnn, "begin_trace_capture", twb._begin_trace_capture)
    monkeypatch.setattr(ttnn, "end_trace_capture", twb._end_trace_capture)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), num_command_queues=2, trace_region_size=1 << 20)
    bridge = twb.ThreadedWeightBridge.receiver(peer_rank=TTML_RANK, submeshes=[mesh])
    tid = None
    try:
        a = ttnn.from_torch(torch.full(SHAPE, 10.0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        ttnn.add(a, a)
        ttnn.synchronize_device(mesh)
        bridge.connect()

        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        ttnn.add(a, a)
        entry = {"key": "w", "shape": SHAPE, "dtype": "BFLOAT16", "layout": "TILE"}
        manifest = json.dumps({"weight_version": 1, "entries": [entry]}).encode()
        ctx.put(manifest, twb._TAG_MANIFEST_LEN, twb._TAG_MANIFEST_BODY)
        weight = _torch_save_bytes(torch.full(SHAPE, 2.0, dtype=torch.bfloat16))
        ctx.put(weight, twb._TAG_WEIGHT_LEN, twb._TAG_WEIGHT_BODY)
        while not ctx.empty(twb._TAG_MANIFEST_BODY):
            time.sleep(0.01)
        time.sleep(0.1)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)

        with bridge._cv:
            assert bridge._cv.wait_for(lambda: bridge._pending, timeout=60)
        assert ttnn.to_torch(bridge._recv_pads[0]["w"], cq_id=1).unique().tolist() == [2.0]
        with bridge.receive_weights(block=True) as (version, (pads,)):
            assert version == 1
            assert ttnn.to_torch(pads["w"], cq_id=0).unique().tolist() == [2.0]
    finally:
        ctx.put(b"", twb._TAG_MANIFEST_LEN, twb._TAG_MANIFEST_BODY)
        bridge.close()
        if tid is not None:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)
