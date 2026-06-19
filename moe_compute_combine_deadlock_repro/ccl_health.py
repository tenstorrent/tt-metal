"""Standalone CCL / fabric health check — NO moe_compute, NO weights, ~30s.

Mirrors the smoke epilogue's cross-device collectives (reduce_scatter + all_gather on
both cluster axes) on the SAME device config (mesh (4,8), COL dispatch, FABRIC_1D_RING)
but with nothing else. Progressive markers + a synchronize_device after each step show
exactly how far the device gets.

  PASS (all steps + "CCL HEALTH OK") -> device fabric is fine; a smoke hang is
       moe_compute-induced (it poisons the subsequent axis-1 CCL).
  HANG (stuck after some step marker, exit 124) -> device fabric is degraded this
       session; the marker tells you which collective/axis died.

Run inside docker (env per the graph_0 `run` wrapper):
  timeout 180 python3 moe_compute_hang_repro/ccl_health.py
"""
import sys
import torch
import ttnn

MESH_SHAPE = (4, 8)
H = 5120
TOK = 16
L1 = 1 << 15


def step(msg):
    print(f">>> {msg}", flush=True, file=sys.stderr)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    dispatch_cfg = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.COL)
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(MESH_SHAPE), l1_small_size=L1,
                                   dispatch_core_config=dispatch_cfg)
    step("device opened")
    try:
        dram = ttnn.DRAM_MEMORY_CONFIG
        hifi4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 math_approx_mode=False, fp32_dest_acc_en=True,
                                                 packer_l1_acc=False)
        # 1. bare device sync (no fabric traffic)
        ttnn.synchronize_device(device)
        step("bare synchronize_device OK")

        # 2. alloc + sync (allocator/device write path)
        base = ttnn.from_torch(torch.randn(1, 1, TOK, H) * 0.05, device=device,
                               layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=dram,
                               mesh_mapper=ttnn.ReplicateTensorToMesh(device))
        ttnn.synchronize_device(device)
        step("alloc + sync OK")

        # 3. cross-col (cluster_axis=1) reduce_scatter -- the smoke epilogue's ep5 op
        rs1 = ttnn.reduce_scatter(input_tensor=base, dim=3, cluster_axis=1, subdevice_id=None,
                                  memory_config=dram, num_links=None, topology=ttnn.Topology.Linear,
                                  compute_kernel_config=hifi4)
        ttnn.synchronize_device(device)
        step(f"reduce_scatter axis1 OK {tuple(rs1.shape)}")

        # 4. cross-col (cluster_axis=1) all_gather -- the smoke epilogue's ep6 op
        ag1 = ttnn.all_gather(input_tensor=rs1, dim=3, cluster_axis=1, subdevice_id=None,
                              memory_config=dram, num_links=None, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(device)
        step(f"all_gather axis1 OK {tuple(ag1.shape)}")

        # 5. cross-row (cluster_axis=0) all_gather -- the dispatch axis moe_compute uses
        ag0 = ttnn.all_gather(input_tensor=base, dim=2, cluster_axis=0, subdevice_id=None,
                              memory_config=dram, num_links=None, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(device)
        step(f"all_gather axis0 OK {tuple(ag0.shape)}")

        # 6. cross-row (cluster_axis=0) Ring all_gather -- moe_compute's combine topology
        ag0r = ttnn.all_gather(input_tensor=base, dim=2, cluster_axis=0, subdevice_id=None,
                               memory_config=dram, num_links=None, topology=ttnn.Topology.Ring)
        ttnn.synchronize_device(device)
        step(f"all_gather axis0 Ring OK {tuple(ag0r.shape)}")

        print("CCL HEALTH OK", flush=True)
    finally:
        ttnn.close_mesh_device(device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
