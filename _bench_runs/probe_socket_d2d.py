# Throwaway probe: which denoise inter-stage socket hops route on a carved (1,8)
# row of a (4,8) Galaxy parent, under FABRIC_1D vs FABRIC_2D?
#
# The 8-stage matmul_decode denoise pipeline is proven only on a dedicated (1,8)
# ring (FABRIC_1D). In the 16-chip integration it's carved from row 1 of a (4,8)
# torus and the velocity-wrap socket (stage7->stage0) fails: "no forwarding
# direction". This probe isolates raw SocketTransport.send between carved (1,1)
# submeshes WITHOUT model weights, testing:
#   - adjacent forward hop   (R,0) -> (R,1)
#   - long wrap hop          (R,7) -> (R,0)
# under the chosen fabric. Reports PASS/FAIL per hop.
#
# Usage: PROBE_FABRIC=1d|2d  PROBE_ROW=0|1  python _bench_runs/probe_socket_d2d.py
import os
import sys
import time
import subprocess

import torch
import ttnn

from models.experimental.pi0_5.tt.tt_bh_glx.transport import SocketTransport

FABRIC = os.environ.get("PROBE_FABRIC", "1d").lower()
ROW = int(os.environ.get("PROBE_ROW", "1"))
fabric_cfg = ttnn.FabricConfig.FABRIC_2D if FABRIC == "2d" else ttnn.FabricConfig.FABRIC_1D
print(f"[sockprobe] FABRIC={FABRIC} ROW={ROW}", flush=True)


def open_parent():
    last = None
    for attempt in range(3):
        try:
            ttnn.set_fabric_config(fabric_cfg)
            return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8))
        except Exception as e:
            last = e
            print(f"[sockprobe] open attempt {attempt} failed: {e!r}", flush=True)
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                pass
            try:
                subprocess.run(["tt-smi", "-r"], check=False, timeout=180)
            except FileNotFoundError:
                pass
            time.sleep(10)
    raise last


def chip(parent, r, c):
    return parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(r, c))


def try_hop(name, src_mesh, dst_mesh):
    host = torch.randn(1, 1, 32, 2048, dtype=torch.bfloat16)
    src = ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=src_mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(src_mesh),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    tp = SocketTransport()
    try:
        out = tp.send(src, dst_mesh, tag=name)
        ttnn.synchronize_device(dst_mesh)
        print(f"[sockprobe] HOP {name}: PASS  out.shape={tuple(out.shape)}", flush=True)
        ok = True
    except Exception as e:
        print(f"[sockprobe] HOP {name}: FAIL  {type(e).__name__}: {e}", flush=True)
        ok = False
    finally:
        try:
            tp.close()
        except Exception:
            pass
        ttnn.deallocate(src)
    return ok


t0 = time.time()
parent = open_parent()
print(f"[sockprobe] (4,8) parent opened in {time.time()-t0:.1f}s", flush=True)

c0 = chip(parent, ROW, 0)
c1 = chip(parent, ROW, 1)
c7 = chip(parent, ROW, 7)

# adjacent forward hop (what every stage i->i+1 needs)
try_hop(f"adjacent_R{ROW}_0to1", c0, c1)
# long wrap hop (velocity_wrap stage7->stage0)
try_hop(f"wrap_R{ROW}_7to0", c7, c0)
# CROSS-ROW KV hop: prefill chip (0,c) -> denoise chip (1,c) (column hop) — the
# e2e socket KV handoff. Test (0,0)->(1,0) and (0,7)->(1,7).
xr_p0 = chip(parent, 0, 0)
xr_d0 = chip(parent, 1, 0)
xr_p7 = chip(parent, 0, 7)
xr_d7 = chip(parent, 1, 7)
try_hop("xrow_prefill0_to_denoise0", xr_p0, xr_d0)
try_hop("xrow_prefill7_to_denoise7", xr_p7, xr_d7)
for _m in (xr_p0, xr_d0, xr_p7, xr_d7):
    try:
        ttnn.close_mesh_device(_m)
    except Exception:
        pass

for m in (c1, c7, c0):
    try:
        ttnn.close_mesh_device(m)
    except Exception:
        pass
ttnn.close_mesh_device(parent)
try:
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
except Exception:
    pass
print("[sockprobe] done", flush=True)
sys.exit(0)
