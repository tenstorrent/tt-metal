# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the EP all-to-all prototype on a 2-chip simulator (ttsim bh_x2): correctness + device-profiler cycles.
env: SIM_T (tokens, default 8), SIM_L (links, 1), SIM_DTYPE (fp8|bf16), SIM_PAYLOAD (0=default), SIM_ITERS (2)."""
import os, sys, time

import torch
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_ep_a2a as B  # noqa: E402

t0 = time.time()
pay = int(os.environ.get("SIM_PAYLOAD", "0"))
if pay:
    rc = ttnn.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = pay
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None,
                           ttnn.FabricTensixConfig.DISABLED, ttnn.FabricUDMMode.DISABLED, ttnn.FabricManagerMode.DEFAULT, rc)
else:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2))
print(f"[sim] open {time.time()-t0:.1f}s payload={ttnn.get_tt_fabric_max_payload_size_bytes()}", flush=True)
c = dict(tokens=int(os.environ.get("SIM_T", "8")), links=int(os.environ.get("SIM_L", "1")),
         dtype=os.environ.get("SIM_DTYPE", "fp8"), epd=int(os.environ.get("SIM_EPD", "112")),
         n_ops=1, n_replays=1)
# Sim: no trace timing (too slow); correctness + profiler only.
B.time_trace = lambda mesh, fn, n_ops, n_rep: (fn(), ttnn.synchronize_device(mesh), {"wall_s": 0})[-1]
rec = B.run_case(mesh, c, ttnn.get_tt_fabric_max_payload_size_bytes(), "l1")
print(f"[sim] RESULT {rec}", flush=True)
ttnn.ReadDeviceProfiler(mesh)
print(f"[sim] done {time.time()-t0:.1f}s", flush=True)
sys.stdout.flush()
os._exit(0)
