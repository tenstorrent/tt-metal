# SPDX-License-Identifier: Apache-2.0
"""
Fabric-Manager live-demo workload (single host, one T3K = 2x4 mesh).

This is step 2 of the TTFM split-lifecycle demo. It runs AFTER
`run_fabric_manager --initialize-fabric` has already brought the fabric up and
LEFT IT RUNNING. Here we attach to that pre-built fabric with
FabricManagerMode.ENABLED (== neither INIT_FABRIC nor TERMINATE_FABRIC), so Metal
does NOT initialize or tear down the fabric -- it just uses it. Expect the log line
    "Fabric initialized through Fabric Manager"   (fabric_firmware_initializer.cpp:320)
instead of the DEFAULT-mode "Initializing Fabric" / "Fabric Initialized with config".

CRUCIAL: in ENABLED mode we must NOT call set_fabric_config(DISABLED) on the way out
-- that flips the mode back to DEFAULT (INIT|TERMINATE) and Metal's close path would
then tear down the fabric the manager owns. We just close the mesh and exit; the fabric
stays up for `run_fabric_manager --terminate-fabric` to tear down.

Run (single host, NO tt-run; env per prompt.md):
    python3 .../claude_job/tt_stack_walkthrough/scripts/fm_workload.py
"""

import os

import torch
import ttnn

TILE = 32


def pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def main():
    torch.manual_seed(0)

    # Attach to the fabric that run_fabric_manager already brought up. Match the CI
    # fabric-manager test path: FABRIC_2D + RELAXED_INIT + ENABLED.
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D,
        reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT,
        fabric_manager_mode=ttnn.FabricManagerMode.ENABLED,
    )

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4))
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(
        f"[FM demo] attached to Fabric-Manager fabric; mesh {tuple(device.shape)} = "
        f"{device.shape[0] * device.shape[1]} chips",
        flush=True,
    )

    # (a) compute sanity: replicated add across all 8 chips (proves dispatch works).
    a = torch.randn(1, 1, TILE, TILE).to(torch.bfloat16)
    b = torch.randn(1, 1, TILE, TILE).to(torch.bfloat16)
    golden_add = (a + b).to(torch.bfloat16)
    ta = ttnn.from_torch(
        a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    tb = ttnn.from_torch(
        b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )
    tadd = ttnn.add(ta, tb)
    ttnn.synchronize_device(device)
    add_shard = ttnn.to_torch(ttnn.get_device_tensors(tadd)[0])
    p_add = pcc(golden_add, add_shard)
    print(f"[FM demo] {'PASS' if p_add >= 0.99 else 'FAIL'} ADD        pcc={p_add:.6f}", flush=True)
    for t in (ta, tb, tadd):
        ttnn.deallocate(t)

    # (b) fabric collective on the manager-owned fabric: all_gather over a 1x4 line submesh.
    line = device.create_submesh(ttnn.MeshShape(1, 4))
    x = torch.randn(1, 1, TILE, TILE * 4).to(torch.bfloat16)
    tx = ttnn.from_torch(
        x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=line, mesh_mapper=ttnn.ShardTensorToMesh(line, dim=3)
    )
    tg = ttnn.all_gather(tx, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(line)
    gathered = ttnn.to_torch(ttnn.get_device_tensors(tg)[0])  # every chip holds full x
    p_ag = pcc(x, gathered)
    print(
        f"[FM demo] {'PASS' if p_ag >= 0.999999 else 'FAIL'} ALL_GATHER pcc={p_ag:.6f} "
        f"(fabric collective ran on the Fabric-Manager fabric)",
        flush=True,
    )
    ttnn.deallocate(tx)
    ttnn.deallocate(tg)

    ok = p_add >= 0.99 and p_ag >= 0.999999
    print("[FM demo] " + ("ALL PASS on Fabric-Manager (ENABLED) fabric" if ok else "FAILED"), flush=True)

    # Close the mesh but LEAVE THE FABRIC UP (do NOT set_fabric_config here) so that
    # `run_fabric_manager --terminate-fabric` performs the real teardown.
    ttnn.close_mesh_device(device)

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
