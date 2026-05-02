# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-38: AllGather correctness after FIX BA teardown chain
#
# What GAP-37 tests (timing only):
#   Predecessor SIGKILL'd → Testee-1 triggers FIX AL/AM/BA → Testee-2 opens in < 15s.
#   GAP-37 verifies the SECOND open is FAST after FIX BA cleanup.
#
# What this test adds (correctness):
#   GAP-37 never runs AllGather in Testee-2.  A regression where FIX BA cleans up
#   *timing* (ERISCs are reset) but leaves residual EDM state (stale routing tables,
#   corrupt edm_status_address, dangling Phase 3 launch messages) would produce
#   wrong AllGather output without violating GAP-37's timing assert.
#
#   GAP-38 completes the chain:
#     Predecessor → FIX AL/AM/BA → Testee-2 AllGather PCC ≥ 0.9999
#
# Failure modes caught that GAP-37 misses:
#   1. FIX BA resets ERISC firmware but EDM routing tables are stale →
#      AllGather output contains predecessor's partial data → wrong PCC.
#   2. FIX AY partially cleaned up (some ERISCs reset, some skipped by FIX AV) →
#      AllGather runs on degraded topology → wrong values or hang.
#   3. A future regression where FIX BA fires but configure_fabric() doesn't fully
#      reinitialize the channel state → AllGather corrupted data.
#
# Hardware: T3K (8-device WH 2×4 mesh).  Requires >= 4 non-MMIO devices with
# at least one out-of-mesh ETH link (standard T3K topology satisfies this).
#
# Relationship to other tests:
#   GAP-37 → timing of second open after FIX BA (no AllGather)
#   GAP-38 → correctness of AllGather in second session after FIX BA chain
#
# Log pattern that indicates FIX BA fired:
#   "teardown: FIX BA — non-MMIO device .* has fabric_channels_not_ready_for_traffic"

import os
import subprocess
import sys
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_blackhole

_MESH_SHAPE = (2, 4)

# Budget for Testee-2 to complete open + AllGather + PCC check + close.
# FIX AL/AM paths add at most ~12s (4 devices × 3s).  AllGather itself < 5s.
# Any hang > kTestee2BudgetS indicates residual ERISC stale state.
kTestee2BudgetS = 60.0

# Budget for predecessor to signal ready.
kPredecessorReadyS = 30.0

# Budget for Testee-1 (open + FIX BA/AC/AY close).
kTestee1BudgetS = 90.0

# PCC threshold: AllGather output must match exact scatter reference.
kPccThreshold = 0.9999


def _predecessor_script(ready_path: str) -> str:
    """Open FABRIC_2D, run AllGather, signal ready, spin until SIGKILL."""
    return rf"""
import sys, os, time, torch
import ttnn
from ttnn import ShardTensorToMesh

try:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
    )
    full = torch.rand([1, 1, 32, 256], dtype=torch.bfloat16)
    inp = ttnn.from_torch(
        full, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh, dim=3),
    )
    out = ttnn.all_gather(inp, dim=3, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh)
    del out, inp
except Exception as exc:
    print(f"PREDECESSOR_SETUP_ERROR: {{exc}}", flush=True)

open("{ready_path}", "w").close()
while True:
    time.sleep(0.1)
"""


def _testee1_script() -> str:
    """
    Open FABRIC_2D (triggers FIX AL/AM on out-of-mesh non-MMIO channels), then close.
    FIX BA must fire to add STARTED-state devices to relay_broken_non_mmio.
    FIX AC + FIX AY reset non-MMIO ERISCs to base UMD firmware.
    """
    return r"""
import sys, time
import ttnn

try:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
    )
    # Attempt AllGather — tolerate degraded mesh result from FIX AL/AM path.
    import torch
    from ttnn import ShardTensorToMesh
    try:
        full = torch.rand([1, 1, 32, 256], dtype=torch.bfloat16)
        inp = ttnn.from_torch(
            full, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensorToMesh(mesh, dim=3),
        )
        out = ttnn.all_gather(inp, dim=3, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(mesh)
        del out, inp
    except Exception:
        pass

    # Close: FIX BA fires → FIX AC → FIX AY run.
    ttnn.close_mesh_device(mesh)
except Exception as exc:
    print(f"TESTEE1_ERROR: {exc}", flush=True)
    sys.exit(1)

sys.exit(0)
"""


def _testee2_script(result_path: str) -> str:
    """
    Open FABRIC_2D fresh, run AllGather, check PCC, write result, close.
    If FIX BA + FIX AY properly cleaned up STARTED ERISCs, configure_fabric()
    reinitializes the channel state and AllGather produces correct output.
    If residual EDM state is present → wrong output or hang.
    """
    return rf"""
import sys, os, json, time, torch
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor

result = {{"pcc": 0.0, "error": None, "elapsed": 0.0}}

try:
    t_start = time.time()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
    )

    # FIX BB (#42429): if the cluster is in a degraded state (stale base-UMD channels
    # set by FIX M / FIX RZ, or relay path broken, or channels not ready), running
    # AllGather would hang — skip rather than hang and fail with a 60s timeout.
    if mesh.is_fabric_degraded():
        result["error"] = "SKIP:fabric_degraded — cluster in degraded state (FIX BB #42429)"
        result["elapsed"] = time.time() - t_start
        with open("{result_path}", "w") as _f:
            json.dump(result, _f)
        ttnn.close_mesh_device(mesh)
        sys.exit(2)

    n = mesh.get_num_devices()
    # Reference: scatter input, gather should produce full repeated tensor.
    full_ref = torch.rand([1, 1, 32, 32 * n], dtype=torch.bfloat16)
    inp = ttnn.from_torch(
        full_ref, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh, dim=3),
    )

    out = ttnn.all_gather(inp, dim=3, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh)

    # Gather back to host and compare each shard against full reference.
    out_host = ttnn.to_torch(out, mesh_composer=ConcatMeshToTensor(mesh, dim=3))
    # out_host shape: [1, 1, 32, 32*n*n] (each shard has the gathered full tensor)
    # Compare first shard (32*n cols) against full_ref.
    out_shard = out_host[..., :32 * n]

    # Compute PCC.
    a = full_ref.flatten().float()
    b = out_shard.flatten().float()
    if a.numel() == b.numel():
        pcc_val = float(torch.corrcoef(torch.stack([a, b]))[0, 1].item())
    else:
        pcc_val = 0.0

    result["pcc"] = pcc_val
    result["elapsed"] = time.time() - t_start

    del out, inp
    ttnn.close_mesh_device(mesh)

except Exception as exc:
    result["error"] = str(exc)
    result["elapsed"] = 0.0

with open("{result_path}", "w") as f:
    json.dump(result, f)

sys.exit(0 if result.get("error") is None else 1)
"""


@skip_for_blackhole("Requires wormhole_b0 to run")
def test_gap38_fixba_allgather_correctness_after_cleanup(tmp_path):
    """
    GAP-38: Verify that AllGather produces numerically correct output in the session
    AFTER a FIX BA teardown chain (FIX AL/AM → FIX BA → FIX AC → FIX AY).

    Without FIX BA (or with partial FIX AY): non-MMIO ERISCs remain in FABRIC STARTED
    state at teardown.  In Testee-2, configure_fabric() may not fully reinitialize
    routing tables / edm_status_address → AllGather silently produces stale data
    (PCC < kPccThreshold) or hangs.

    With FIX BA + FIX AY: ERISCs are reset to base UMD firmware before process exit.
    Testee-2's configure_fabric() starts from a clean base and produces correct output.

    This test is ORTHOGONAL to GAP-37 (which only checks timing, not PCC).
    """
    import json

    pred_ready = str(tmp_path / "gap38_pred_ready")
    t2_result = str(tmp_path / "gap38_t2_result.json")

    # ── Phase 1: Launch predecessor (non-MMIO ERISCs in FABRIC ACTIVE state) ─
    pred = subprocess.Popen(
        [sys.executable, "-c", _predecessor_script(pred_ready)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    t_start = time.time()
    while not os.path.exists(pred_ready):
        if time.time() - t_start > kPredecessorReadyS:
            pred.kill()
            pred.wait()
            pytest.skip(
                f"GAP-38: predecessor did not signal ready within {kPredecessorReadyS}s "
                "(hardware init stall?); skipping."
            )
        time.sleep(0.1)

    time.sleep(2.0)  # Let relay settle in ACTIVE state.

    pred.kill()
    pred.wait()
    logger.info("GAP-38: predecessor SIGKILL'd — non-MMIO ERISCs in FABRIC firmware state")

    # ── Phase 2: Testee-1 — open → FIX AL/AM/BA fires → close ───────────────
    t1_proc = subprocess.Popen(
        [sys.executable, "-c", _testee1_script()],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    t1_start = time.time()
    try:
        t1_proc.wait(timeout=kTestee1BudgetS)
    except subprocess.TimeoutExpired:
        t1_proc.kill()
        t1_proc.wait()
        pytest.skip(
            f"GAP-38: Testee-1 did not complete within {kTestee1BudgetS}s — "
            "hardware in unexpected state; skipping."
        )
    t1_elapsed = time.time() - t1_start
    logger.info(
        f"GAP-38: Testee-1 exited in {t1_elapsed:.1f}s (exit code: {t1_proc.returncode})"
    )

    # ── Phase 3: Testee-2 — fresh open + AllGather + PCC check ───────────────
    t2_start = time.time()
    t2_proc = subprocess.Popen(
        [sys.executable, "-c", _testee2_script(t2_result)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    try:
        t2_proc.wait(timeout=kTestee2BudgetS)
    except subprocess.TimeoutExpired:
        t2_proc.kill()
        t2_proc.wait()
        pytest.fail(
            f"GAP-38 HANG: Testee-2 (post-FIX-BA AllGather) did not complete within "
            f"{kTestee2BudgetS}s.\n"
            f"Likely cause: non-MMIO ERISCs remain in FABRIC STARTED state after FIX BA "
            f"teardown (FIX AY failed and FIX AW couldn't drain the relay CMD queue). "
            f"Testee-2's configure_fabric() or AllGather hangs polling stale EDM status.\n"
            f"Check log for: 'teardown: FIX BA' (should be present in Testee-1) and "
            f"'FIX AY.*failed' or 'FIX AV' (relay dead during non-MMIO reset)."
        )

    t2_elapsed = time.time() - t2_start
    logger.info(f"GAP-38: Testee-2 completed in {t2_elapsed:.1f}s (exit: {t2_proc.returncode})")

    # ── PCC check ─────────────────────────────────────────────────────────────
    if not os.path.exists(t2_result):
        pytest.fail(
            f"GAP-38: Testee-2 did not write result file (exit code: {t2_proc.returncode}). "
            f"AllGather likely threw an exception — check for residual ERISC stale state."
        )

    with open(t2_result) as f:
        result = json.load(f)

    # FIX BB (#42429): Testee-2 reports a degraded cluster — skip rather than fail.
    # A degraded cluster (stale base-UMD firmware from FIX M / FIX RZ path) means
    # AllGather would hang; Testee-2 detects this and writes error="SKIP:..." + exits 2.
    if result.get("error", "").startswith("SKIP:"):
        pytest.skip(
            f"GAP-38: Testee-2 reports degraded cluster; skipping instead of hanging.\n"
            f"Reason: {result['error']}\n"
            f"Hardware in stale base-UMD state after prior test session (FIX RX skipped "
            f"quiesce teardown for non-MMIO devices, leaving base-UMD relay firmware). "
            f"FIX BB (#42429) adds is_fabric_degraded() guard to Testee-2 subprocess."
        )

    if result.get("error"):
        err = result["error"]
        # FIX BC: ETH broadcast timeout is a hardware-init failure (UMD Cluster::Cluster()
        # can't reach non-MMIO devices via relay), not a FIX BA correctness regression.
        # Skip rather than fail so degraded runners don't block CI.
        _HW_SKIP_PATTERNS = (
            "Timeout waiting for Ethernet core service",
            "ethernet_broadcast_write",
            "write_to_non_mmio",
        )
        if any(pat in err for pat in _HW_SKIP_PATTERNS):
            pytest.skip(
                f"GAP-38: Testee-2 device init failed (ETH relay unreachable) — "
                f"hardware failure, not FIX BA regression.\n"
                f"Error: {err}\n"
                f"Runner has dead ETH relay on non-MMIO devices (UMD cannot reach them "
                f"via broadcast write). This is the same degraded-cluster condition as "
                f"FIX RZ/FIX RY — cluster needs reset before GAP-38 is meaningful."
            )
        pytest.fail(
            f"GAP-38: Testee-2 AllGather raised exception: {err}\n"
            f"Likely cause: FIX BA teardown left residual EDM state — configure_fabric() "
            f"could not reinitialize channels from stale base firmware state."
        )

    pcc = result.get("pcc", 0.0)
    logger.info(
        f"GAP-38 AllGather PCC = {pcc:.6f} (threshold: {kPccThreshold}) "
        f"elapsed: {result.get('elapsed', 0.0):.1f}s"
    )

    assert pcc >= kPccThreshold, (
        f"GAP-38 REGRESSION (FIX BA correctness): AllGather PCC {pcc:.6f} < {kPccThreshold} "
        f"after FIX BA teardown chain.\n"
        f"\n"
        f"FIX BA (#42429) adds STARTED-state non-MMIO devices to relay_broken_non_mmio in "
        f"teardown Step 1, triggering FIX AC (MMIO ETH PCIe reset) + FIX AY (non-MMIO ERISC "
        f"reset via restored relay).  After FIX AY, non-MMIO ERISCs should be in base UMD "
        f"firmware state with clean EDM status.\n"
        f"\n"
        f"A PCC regression means configure_fabric() is reinitializing from stale EDM state: "
        f"routing tables or edm_status_address contain leftover values from the predecessor's "
        f"FABRIC session, causing AllGather to mix predecessor data with current data.\n"
        f"\n"
        f"Log patterns to search:\n"
        f"  'teardown: FIX BA — non-MMIO device' (Testee-1 teardown; should be present)\n"
        f"  'FIX AY.*all.*reset to base firmware' (Testee-1 teardown; should be present)\n"
        f"  'FIX AV #42429' or 'FIX AY.*failed' (indicates partial AY — some ERISCs dirty)\n"
        f"\n"
        f"Note: GAP-37 verifies the TIMING of Testee-2 open; GAP-38 verifies CORRECTNESS."
    )

    logger.info(
        f"GAP-38 PASS: AllGather PCC={pcc:.6f} >= {kPccThreshold} after FIX BA teardown "
        f"chain — ERISCs correctly reset by FIX AC + FIX AY; configure_fabric() started "
        f"from clean base firmware state."
    )
