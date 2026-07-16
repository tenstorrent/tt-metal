# SPDX-License-Identifier: Apache-2.0
"""
TTFM split-lifecycle demo that WORKS on a T3K — driven in a single process.

Why in-process: the `run_fabric_manager` CLI drives the same FabricManagerMode split
lifecycle across THREE separate processes (init / use / terminate). On a T3K that fails
after the init: chips 4-7 are remote n300 halves reached over ethernet, so every fresh
process re-runs UMD topology discovery, which probes those chips for an eth heartbeat —
and the EDM routers the previous process left running occupy those eth cores, so the
device won't even re-open (`Timed out waiting for ETH heartbeat … Stuck at 0xabcd…`).
On a Galaxy every chip is PCIe-direct, so there's no remote-over-eth discovery to
conflict — which is why the in-repo FM CCL tests are Galaxy-8x4-only.

The fix on a T3K: exercise the *same* FabricManagerMode enum values in ONE process. The
UMD Cluster (and its topology discovery) is built exactly once — before any fabric is up
— and reused across every open_mesh/close_mesh, so the heartbeat conflict never arises.
`set_fabric_config` may be re-called after the mesh is closed and only the fabric_manager
mode changed (fabric_config stays FABRIC_2D); see metal_env.cpp:235-330.

Lifecycle exercised (single T3K, 2x4):
  1. INIT_FABRIC      open+close → routers come up and are LEFT UP (teardown is a no-op
                      without the TERMINATE flag; fabric_firmware_initializer.cpp:347).
  2. ENABLED          open → "Fabric initialized through Fabric Manager" (:320): ATTACH,
                      no re-init. Run add + all_gather, verify vs torch. close (leaves up).
  3. TERMINATE_FABRIC open → "Compiling fabric … for fabric termination"; close tears the
                      routers down. Then set_fabric_config(DISABLED) → clean exit, no reset.

Run (single host, NO tt-run; env per prompt.md):
    python3 .../claude_job/tt_stack_walkthrough/scripts/fm_lifecycle.py
"""

import os

import torch
import ttnn

TILE = 32
FC = ttnn.FabricConfig.FABRIC_2D
REL = ttnn.FabricReliabilityMode.RELAXED_INIT
MESH = ttnn.MeshShape(2, 4)


def pcc(a, b):
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return 1.0 if d == 0 else float((a @ b) / d)


def enabled_workload(device):
    """Real work on the manager-owned fabric: replicated add (dispatch) + a line all_gather
    (fabric collective). Returns True on PASS."""
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
    p_add = pcc(golden_add, ttnn.to_torch(ttnn.get_device_tensors(tadd)[0]))
    for t in (ta, tb, tadd):
        ttnn.deallocate(t)

    line = device.create_submesh(ttnn.MeshShape(1, 4))
    x = torch.randn(1, 1, TILE, TILE * 4).to(torch.bfloat16)
    tx = ttnn.from_torch(
        x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=line, mesh_mapper=ttnn.ShardTensorToMesh(line, dim=3)
    )
    tg = ttnn.all_gather(tx, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(line)
    p_ag = pcc(x, ttnn.to_torch(ttnn.get_device_tensors(tg)[0]))
    ttnn.deallocate(tx)
    ttnn.deallocate(tg)

    ok = p_add >= 0.99 and p_ag >= 0.999999
    print(f"[ENABLED] {'PASS' if p_add >= 0.99 else 'FAIL'} ADD        pcc={p_add:.6f}", flush=True)
    print(
        f"[ENABLED] {'PASS' if p_ag >= 0.999999 else 'FAIL'} ALL_GATHER pcc={p_ag:.6f} "
        f"(fabric collective ran on the ENABLED/attached fabric)",
        flush=True,
    )
    return ok


def phase(tag, mode, workload=False):
    """set_fabric_config(mode) → open → [workload] → close. Returns workload result (or True)."""
    print(f"\n=== PHASE {tag} (fabric_manager_mode={mode}) ===", flush=True)
    ttnn.set_fabric_config(FC, reliability_mode=REL, fabric_manager_mode=mode)
    device = ttnn.open_mesh_device(mesh_shape=MESH)
    result = True
    try:
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        print(f"[{tag}] mesh open {tuple(device.shape)} = {device.shape[0] * device.shape[1]} chips", flush=True)
        if workload:
            result = enabled_workload(device)
    finally:
        ttnn.close_mesh_device(device)
        print(f"[{tag}] mesh closed", flush=True)
    return result


def main():
    torch.manual_seed(0)
    workload_ok = False
    reached_enabled = False
    try:
        phase("INIT", ttnn.FabricManagerMode.INIT_FABRIC)  # bring up, leave up
        reached_enabled = True
        workload_ok = phase("ENABLED", ttnn.FabricManagerMode.ENABLED, workload=True)
    finally:
        # Always tear the fabric down so the box is left clean (no reset needed).
        try:
            phase("TERMINATE", ttnn.FabricManagerMode.TERMINATE_FABRIC)
        except Exception as e:  # noqa: BLE001
            print(f"[TERMINATE] error during teardown: {e}", flush=True)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            print("[TERMINATE] set_fabric_config(DISABLED) — fabric down, clean exit", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[TERMINATE] error disabling fabric: {e}", flush=True)

    print("\n" + "=" * 70, flush=True)
    if reached_enabled and workload_ok:
        print("[FM lifecycle] ALL PASS: INIT_FABRIC → ENABLED (workload green) → TERMINATE_FABRIC", flush=True)
        print("=" * 70, flush=True)
    else:
        print("[FM lifecycle] FAILED", flush=True)
        print("=" * 70, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
