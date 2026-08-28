# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The SOLE device opener for the `Qwen/Qwen3-Coder-Next` demo package.

NOTHING under `tt/` opens a device.  `build_pipeline(device, ...)` is handed the device its
caller opened and the pipeline runs on THAT device and no other -- a second, ad-hoc open would
create a competing mesh with its own command-queue count, which is precisely what makes
`ttnn.begin_trace_capture` die on `id < mesh_command_queues_.size()`.

Every entry point comes through `open_mesh()` here, exactly once per process:

  * `tests/e2e/conftest.py`            -- the session fixture the gate tests share
  * `tests/e2e/test_text_generation_perf.py`
  * `demo/demo_text_generation.py`
  * the module-level `host_op_selftest()` / `trace_capture_selftest()` hooks in `tt/pipeline.py`,
    which the observer probes run standalone

The mesh is always opened with `num_command_queues=1` and a trace region, so the trace lever is
available on whatever device the pipeline is then built on.

WHY A LADDER
------------
The manifest asks for `MeshShape(16, 2)`.  This host's SystemMesh is 8x4, and asking for a 16x2
view of it does not fail fast -- `open_mesh_device` HANGS in the topology mapper (measured
2026-08-27: >20 min, no progress), so it is not on the automatic ladder; `TT_QWEN3_MESH=16x2`
still forces an attempt.  Nothing is lost by opening 8x4 instead: it is the SAME 32 chips, and the
(1, 2) TP tiles `tt/mesh.py` carves out of it are the same 16 adjacent-chip pairs either way --
DP=16 x TP=2 either way.
"""
from __future__ import annotations

import os

import ttnn

from models.demos.qwen3_coder_next.tt import mesh as tt_mesh

REQUESTED_MESH = tuple(tt_mesh.MANIFEST.get("mesh", (16, 2)))

# Widest first. The machine's own SystemMesh shape is prepended at call time, so the full mesh is
# always attempted in the orientation the runtime can actually map. A bare `MeshShape(1, 2)` open
# is the LAST multi-chip rung on purpose: on a Galaxy a partial mesh dies in fabric bring-up
# ("Fabric Router Sync: Timeout ... Ethernet handshake likely failed") because the routers try to
# handshake over links whose remote partner is outside the mesh.
MESH_LADDER = (
    (8, 4),   # all 32 chips -> 16 x TP=2
    (4, 8),
    (4, 2),   # 8 chips  -> 4 x TP=2
    (2, 2),   # 4 chips  -> 2 x TP=2
    (1, 2),   # one bare TP group
    (1, 1),   # single device
)

DEFAULT_L1_SMALL_SIZE = 32768
# Trace regions are command buffers, not tensors, so this is cheap against a 32 GB chip.
DEFAULT_TRACE_REGION_SIZE = 128 * 1024 * 1024
# One command queue, matching the trace lever the perf path replays on.
NUM_COMMAND_QUEUES = 1


def system_mesh_shape():
    """The machine's full SystemMesh as `(rows, cols)`, or None when it cannot be queried."""
    try:
        dims = [int(d) for d in ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()]
    except Exception:
        return None
    return tuple(dims) if len(dims) == 2 else None


def open_mesh(
    shape=None,
    *,
    l1_small_size=DEFAULT_L1_SMALL_SIZE,
    trace_region_size=DEFAULT_TRACE_REGION_SIZE,
):
    """Enable the inter-chip fabric and open the widest mesh this host can actually map.

    Returns `(mesh_device, (rows, cols))`.  `shape` (or `TT_QWEN3_MESH=RxC`) forces one shape.
    """
    env = os.environ.get("TT_QWEN3_MESH")
    if shape is None and env:
        shape = tuple(int(v) for v in env.lower().split("x"))

    if shape is not None:
        candidates = [tuple(shape)]
    else:
        system = system_mesh_shape()
        candidates = list(MESH_LADDER)
        if system and system[0] * system[1] >= tt_mesh.TP_DEGREE:
            candidates = [system] + [c for c in candidates if c != system]
        print(f"[placement] SystemMesh={system}; requested {REQUESTED_MESH}; trying {candidates[0]}", flush=True)
    last = None
    for rows, cols in candidates:
        try:
            # The inter-chip fabric MUST be up before the mesh opens, or any CCL raises
            # `fabric_context_ != nullptr`.  It must equally be DOWN for a one-chip mesh, which
            # has no ethernet partner to hand-shake with and times out in the router sync.
            ttnn.set_fabric_config(
                ttnn.FabricConfig.FABRIC_1D if rows * cols > 1 else ttnn.FabricConfig.DISABLED
            )
            device = ttnn.open_mesh_device(
                ttnn.MeshShape(rows, cols),
                l1_small_size=l1_small_size,
                trace_region_size=trace_region_size,
                num_command_queues=NUM_COMMAND_QUEUES,
            )
        except Exception as exc:  # not mappable on this SystemMesh -- try the next rung
            last = exc
            print(f"[placement] MeshShape({rows}, {cols}) unavailable: {str(exc).splitlines()[0][:110]}")
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                pass
            continue
        actual = tt_mesh.rows_cols(device)
        chips = device.get_num_devices()
        want = tt_mesh.MANIFEST.get("chips", 32)
        note = "" if chips == want else f"  (FALLBACK from the requested {want} chips)"
        print(
            f"[placement] mesh {actual[0]}x{actual[1]} = {chips} chips  "
            f"DP={actual[0]} (axis 0)  TP={actual[1]} (axis 1)  cq={NUM_COMMAND_QUEUES}{note}",
            flush=True,
        )
        return device, actual
    raise RuntimeError(f"no mesh shape in {candidates} could be opened; last error: {last}")


def close_mesh(device):
    # RELEASE CHILD SUBMESHES FIRST.  `tt_mesh.tp_groups()` carves the open mesh into TP groups
    # with `create_submeshes`, and the pipeline keeps them alive for the life of the run.  Closing
    # the PARENT while a child still holds a command queue raises
    #     TT_THROW mesh_device.cpp: MeshDevice cq ID 0 is in use by child submesh ID 1
    #                               during close of mesh ID 0
    # and, when the device profiler is enabled (`tracy -p` sets TT_METAL_DEVICE_PROFILER=1), the
    # same conflict DEADLOCKS the teardown instead of raising -- which is what made every tracy
    # run hang after its last signpost and never emit ops_perf_results.
    # Best-effort: a submesh that is already gone must not stop the parent from closing.
    try:
        for sub in list(device.get_submeshes()):
            try:
                ttnn.close_mesh_device(sub)
            except Exception:  # noqa: BLE001 - already released, or not closeable on its own
                pass
    except Exception:  # noqa: BLE001 - older ttnn without get_submeshes
        pass
    ttnn.close_mesh_device(device)
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:
        pass
