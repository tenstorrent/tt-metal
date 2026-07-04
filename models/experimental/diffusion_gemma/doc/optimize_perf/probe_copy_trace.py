# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Minimal isolation: which uint32 in-place threading op is trace-safe?

The single-step traced denoise loop FATALs at ``ttnn.copy(next_canvas, canvas_buf)``
with "Writes are not supported during trace capture" (fd_mesh_command_queue.cpp:665),
while the bf16 ``ttnn.copy(new_signal, signal_buf)`` in the adapter passes. This probe
isolates the cause with NO model build: it tries several in-place threading forms
against destination buffers allocated different ways, each inside a trace, and reports
which are trace-safe. *** run only when QB2 is free. ***
"""
from __future__ import annotations

import torch

import ttnn


def _mk_from_torch(mesh, val=3):
    host = torch.full((1, 1, 256, 1), val, dtype=torch.int32)
    return ttnn.from_torch(
        host, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _try_trace(mesh, label, dst, op):
    """Warm the op eagerly, then attempt it inside a trace; report PASS/FATAL."""
    src = _mk_from_torch(mesh, 7)
    try:
        op(src, dst)  # warm (eager, program-cache fill)
        ttnn.synchronize_device(mesh)
    except Exception as e:  # noqa: BLE001
        print(f"RESULT {label}: EAGER_FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
        src.deallocate(True)
        return
    try:
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        op(src, dst)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        ttnn.release_trace(mesh, tid)
        print(f"RESULT {label}: TRACE_SAFE", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"RESULT {label}: TRACE_FATAL {type(e).__name__}: {str(e)[:140]}", flush=True)
    finally:
        src.deallocate(True)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    try:
        # (a) dst allocated via from_torch (the failing style)
        _try_trace(mesh, "copy_into_from_torch", _mk_from_torch(mesh, 0), lambda s, d: ttnn.copy(s, d))
        # (b) dst allocated via zeros (pure device alloc, like signal_buf)
        zeros_dst = ttnn.zeros([1, 1, 256, 1], dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=mesh)
        _try_trace(mesh, "copy_into_zeros", zeros_dst, lambda s, d: ttnn.copy(s, d))
        # (c) dst allocated via clone of a from_torch tensor (device op result)
        clone_dst = ttnn.clone(_mk_from_torch(mesh, 0))
        _try_trace(mesh, "copy_into_clone", clone_dst, lambda s, d: ttnn.copy(s, d))
        # (d) ttnn.assign into a from_torch dst
        _try_trace(mesh, "assign_into_from_torch", _mk_from_torch(mesh, 0), lambda s, d: ttnn.assign(s, d))
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
