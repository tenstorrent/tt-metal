# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Repro — B2: allocating device buffers while a captured trace is still live.

NOT a tt-metal framework bug: tt-metal correctly *warns* (allocator.cpp:110,
verify_safe_allocation) that buffers allocated while a trace is live may be corrupted
once the trace executes. The gpt-oss stage-05 stall was the MODEL harness violating this
rule: GPTOSSGenerator.reset() reset the KV cache but left both the decode model-trace and
sampling-trace live, so the next request's prefill allocated buffers under a live trace →
this warning → later host dispatch-queue stall (topk's internal pad hung in
fetch_queue_reserve_back while the device mesh was idle). Fix = release traces before reset.

This script demonstrates the RULE in isolation (no model): capture a trace, then allocate
a buffer while it is live (expect the unsafe-allocation warning), then show that releasing
the trace first makes allocation safe.

Requires only a mesh device (no watcher, no model weights).

Run:
  TT_METAL_HOME=$METAL PYTHONPATH=$METAL python3 repros/trace_unsafe_alloc.py
Watch stderr for:  "Allocating device buffers is unsafe due to the existence of an active trace"
"""
from __future__ import annotations

import torch
import ttnn

MESH_SHAPE = (1, 4)


def _big(mesh_device):
    """Allocate a fresh (non-trivial) DRAM buffer and return it."""
    host = torch.zeros((1, 1, 512, 4096), dtype=torch.bfloat16)
    return ttnn.from_torch(host, device=mesh_device, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main() -> int:
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE))
    try:
        a = _big(mesh_device)
        out = _big(mesh_device)                # preallocate the op output BEFORE capture
                                               # (allocation is illegal during trace capture)
        # Warm up the op once outside capture so no program build happens during capture.
        ttnn.add(a, a, output_tensor=out)

        # Capture a trivial trace so a trace buffer is live afterwards.
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        ttnn.add(a, a, output_tensor=out)      # writes into the preallocated buffer; recorded
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)

        # UNSAFE: allocate while the trace buffer is still live -> allocator.cpp:110 warning.
        print("[unsafe] allocating a buffer while a captured trace is LIVE "
              "(expect allocator.cpp:110 unsafe-allocation warning on stderr) ...")
        b = _big(mesh_device)
        ttnn.deallocate(b, True)

        # SAFE: release the trace first, then allocate.
        ttnn.release_trace(mesh_device, tid)
        print("[safe] released trace; allocating again (no unsafe warning expected) ...")
        c = _big(mesh_device)
        ttnn.deallocate(c, True)
        ttnn.deallocate(out, True)
        ttnn.deallocate(a, True)
        print("DONE — compare stderr: the warning should appear only in the [unsafe] phase.")
        return 0
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    raise SystemExit(main())
