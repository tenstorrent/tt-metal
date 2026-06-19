"""Drive the codeowners' standalone selective_reduce_combine test (run_combine_test
from models/demos/deepseek_v3/tests/test_combine_tg.py) on a self-opened (4,8) galaxy,
WITHOUT pytest (avoids the deepseek_v3 conftest's unrelated transformers import error).

Tests the combine OP in isolation (synthetic dense tensors, no moe_compute) on the same
4-device-ring config (cluster_axis=0, COL dispatch, FABRIC_1D_RING) where the fused
moe_compute combine deadlocks.

  COMBINE_TOPO=ring    (default) -> topology as the test ships it (Ring)
  COMBINE_TOPO=linear            -> monkeypatch the op to force Topology.Linear

PASS => the standalone combine works in isolation -> the deadlock is in moe_compute's
        FUSED integration, not the combine op (and compute_only + standalone combine is a
        viable workaround).
HANG => the combine op itself deadlocks on this build for this config/topology.

Run inside docker with PYTHONPATH including the tt-metal root:
  COMBINE_TOPO=linear python3 moe_compute_combine_deadlock_repro/standalone_combine_driver.py
"""
import os
import sys
import ttnn

MESH = (4, 8)
L1 = 1 << 15
TOPO = os.environ.get("COMBINE_TOPO", "ring").lower()

if TOPO in ("linear", "line"):
    _orig = ttnn.experimental.selective_reduce_combine

    def _patched(*a, **k):
        k["topology"] = ttnn.Topology.Linear
        return _orig(*a, **k)

    ttnn.experimental.selective_reduce_combine = _patched
    print(">>> patched selective_reduce_combine -> topology=Linear", flush=True)

# Import AFTER ttnn so the monkeypatch is in place; importing the module (not pytest)
# skips the deepseek_v3 conftest entirely.
from models.demos.deepseek_v3.tests.test_combine_tg import run_combine_test

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
cfg = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.COL)
dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(MESH), l1_small_size=L1, dispatch_core_config=cfg)
print(f">>> device opened; running standalone combine (topo={TOPO})", flush=True)
try:
    # deepseek test params (4-device ring, cluster_axis=0) -- matches our fused-hang config:
    # mesh, batch=64, experts_per_device=2, select_experts_k=8, hidden=7168, seq=1,
    # cluster_axis=0, worker ((0,0),(3,3)), token/data parallel 4/4, num_links=4, mux ((4,0),(5,7)).
    run_combine_test(dev, MESH, 64, 2, 8, 7168, 1, 0, ((0, 0), (3, 3)), 4, 4, 4, ((4, 0), (5, 7)), 1)
    print(f">>> STANDALONE COMBINE PASSED (topo={TOPO})", flush=True)
finally:
    ttnn.close_mesh_device(dev)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
