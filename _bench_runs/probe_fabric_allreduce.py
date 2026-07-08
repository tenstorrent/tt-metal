# Throwaway fabric/topology probe for the 16-chip prefill TP=8 AllReduce blocker.
#
# The carve-based ttnn_16_decode prefill crashes in _tp_all_reduce with
#   "Could not find any forwarding direction from src (M0,D0) to dst (M0,D28)"
# i.e. the TP=8 reduce_scatter ring can't route on a carved (1,8) submesh under
# FABRIC_1D. This probe reproduces ONLY that CCL call (no adapter load) so we can
# cheaply test which (fabric_mode, mesh_mode) combination actually routes.
#
# Usage (one config per process; fabric state is global so don't mix in-process):
#   PROBE_FABRIC=1d|2d  PROBE_MESH=carve|toplevel  python _bench_runs/probe_fabric_allreduce.py
import os
import sys
import time
import subprocess

import torch
import ttnn

from models.experimental.pi0_5.tt.tt_bh_glx.stage_prefill_tp4 import _tp_all_reduce

FABRIC = os.environ.get("PROBE_FABRIC", "2d").lower()
MESH = os.environ.get("PROBE_MESH", "carve").lower()
TP = 8

fabric_cfg = ttnn.FabricConfig.FABRIC_2D if FABRIC == "2d" else ttnn.FabricConfig.FABRIC_1D
print(f"[probe] FABRIC={FABRIC} MESH={MESH} TP={TP}", flush=True)


def open_mesh():
    """Open either a carved (1,8) row of a (4,8) parent, or a top-level (1,8)
    at offset (0,0). Returns (parent_or_none, mesh)."""
    last_err = None
    for attempt in range(3):
        try:
            ttnn.set_fabric_config(fabric_cfg)
            if MESH == "toplevel":
                mesh = ttnn.open_mesh_device(
                    mesh_shape=ttnn.MeshShape(1, 8),
                    offset=ttnn.MeshCoordinate(0, 0),
                )
                return None, mesh
            parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))
            sub = parent.create_submesh(ttnn.MeshShape(1, 8), ttnn.MeshCoordinate(0, 0))
            return parent, sub
        except Exception as e:  # IndexError on set_fabric_config etc.
            last_err = e
            print(f"[probe] open attempt {attempt} failed: {e!r}", flush=True)
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                pass
            try:
                subprocess.run(["tt-smi", "-r"], check=False, timeout=180)
            except FileNotFoundError:
                pass
            time.sleep(10)
    raise last_err


t0 = time.time()
parent, mesh = open_mesh()
print(f"[probe] mesh opened in {time.time()-t0:.1f}s; num_devices={mesh.get_num_devices()}", flush=True)
g = mesh.compute_with_storage_grid_size()
print(f"[probe] compute_with_storage_grid_size = (x={g.x}, y={g.y})", flush=True)

# Replicated activation-shaped tensor; _tp_all_reduce picks the scatter dim from shape.
host = torch.randn(1, 1, 32, 2048, dtype=torch.bfloat16)
x = ttnn.from_torch(
    host,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    memory_config=ttnn.L1_MEMORY_CONFIG,
)

rc = 0
try:
    out = _tp_all_reduce(x, TP)
    ttnn.synchronize_device(mesh)
    print(f"[probe] RESULT=PASS  all_reduce routed; out.shape={tuple(out.shape)}", flush=True)
    ttnn.deallocate(out)
except Exception as e:
    rc = 1
    print(f"[probe] RESULT=FAIL  {type(e).__name__}: {e}", flush=True)

try:
    if parent is not None:
        ttnn.close_mesh_device(mesh)
        ttnn.close_mesh_device(parent)
    else:
        ttnn.close_mesh_device(mesh)
finally:
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    except Exception:
        pass
sys.exit(rc)
