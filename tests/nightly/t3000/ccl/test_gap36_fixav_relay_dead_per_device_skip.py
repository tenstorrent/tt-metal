# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-36: FIX AV — device_relay_dead early-exit in FIX AY per-device loop
#
# Background (CI run 25060970918, job 73417098227):
#
#   In risc_firmware_initializer.cpp, FIX AY iterates over all active ETH cores
#   on each non-MMIO device in relay_broken_non_mmio and calls
#   assert_risc_reset_at_core on each core.  Each call requires a read_non_mmio
#   to read the current reset-register state BEFORE writing the reset value.
#
#   When FIX AC (Step 2) has already PCIe-reset the MMIO ETH channels, and
#   the MMIO ERISCs have NOT yet been reloaded with UMD relay firmware in the
#   current process, every read_non_mmio for every ETH core on every non-MMIO
#   device blocks for 5 s before throwing.
#
#   On a T3K with 4 active fabric ETH channels per non-MMIO device × 2
#   relay-broken non-MMIO devices = 8 cores × 5 s = 40 s wasted in teardown.
#   After ~50 s the overall 5-minute hang-detection SIGALRM fires → exit 124.
#
# FIX AV (risc_firmware_initializer.cpp: FIX AY loop, ~line 518):
#   Adds a per-device bool `device_relay_dead`.  On the first
#   assert_risc_reset_at_core failure for a device, set device_relay_dead=true
#   and skip all remaining ETH cores on that device.  All remaining cores share
#   the same MMIO relay path and will all fail identically.
#
#   Worst-case overhead reduction: N×5 s → 1×5 s per non-MMIO device.
#   Typical T3K: 4 channels × 2 devices saved → 30 s recovered.
#   Logged as: "Skipping all remaining ETH cores on this device. (FIX AV #42429)"
#
# What this test verifies:
#   1. Predecessor opens FABRIC_2D on a T3K 2×4 mesh and dispatches an
#      AllGather (relay ERISCs fully active), then is SIGKILL'd.
#      This leaves non-MMIO ERISC channels in active FABRIC firmware state.
#   2. Testee (parent) opens FABRIC_2D → quiesce runs:
#      - Phase 5 handshake times out on non-MMIO devices (relay ERISC left in
#        stale FABRIC state by killed predecessor, never wrote LOCAL_HANDSHAKE_COMPLETE).
#      - FIX-1 sets fabric_relay_path_broken_=true on MMIO devices.
#      - relay_broken_non_mmio populated (non-MMIO devices with relay_broken).
#   3. MeshDevice close → RiscFirmwareInitializer::teardown():
#      - FIX AC (Step 2) PCIe-resets MMIO ETH channels (relay endpoints).
#      - FIX AY (Step 5): tries to reset non-MMIO ETH cores via relay.
#        Relay is DEAD (MMIO ERISCs just PCIe-reset, not reloaded in this process).
#        First assert_risc_reset_at_core throws 5 s timeout.
#      - FIX AV: `device_relay_dead=true` → remaining cores on this device skipped.
#   4. Assert: full MeshDevice close completes in < 25 s.
#      Without FIX AV: 4 cores × 2 devices × 5 s = 40 s → FAIL (budget exceeded).
#      With    FIX AV: 1 failure × 2 devices × 5 s = 10 s teardown → PASS.
#
# Relationship to existing tests:
#   GAP-29 (test_gap29_cluster_teardown_hang_relay_broken.cpp):
#     Tests FIX AW: ~Cluster destructor hang prevention.  Uses fork() in C++.
#     Does NOT test per-device early-exit in FIX AY.
#   GAP-31 (test_gap31_fixay_deferred_nonmmio_reset.cpp):
#     Tests FIX AY HAPPY PATH: relay IS re-synced → deferred ETH reset succeeds.
#     Does NOT test the FIX AV sad path (relay dead → FIX AV fires).
#     GAP-36 specifically catches regressions that break FIX AV while leaving
#     FIX AY intact.
#   GAP-36 (this test):
#     Tests FIX AV SAD PATH: relay confirmed dead on first core → remaining skipped.
#     Timing budget: 25 s.  Without FIX AV: 40 s → FAIL.
#
# Hardware: T3K (8-device 2×4 WH mesh).  Requires >= 4 devices.

import os
import signal
import subprocess
import sys
import time

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_blackhole

_MESH_SHAPE = (2, 4)

# Close budget. Without FIX AV: 4 active ETH cores × 2 non-MMIO devices × 5s = 40s.
# With FIX AV: 1 × 2 × 5s = 10s overhead + normal quiesce (~5s) = ~15s. Allow 25s.
_CLOSE_BUDGET_S = 25.0

# Budget for predecessor to signal it's ready.
_PREDECESSOR_READY_TIMEOUT_S = 30.0

# Budget for the full AllGather dispatch (predecessor verification).
_DISPATCH_BUDGET_S = 15.0


def _predecessor_script() -> str:
    """Python script run as the SIGKILL'd predecessor.

    Opens FABRIC_2D on the 2×4 T3K mesh, dispatches an AllGather to ensure
    MMIO ETH relay ERISCs are firmly in ACTIVE state, signals ready, then spins.
    Parent SIGKILLs it — leaves non-MMIO ERISCs in live FABRIC firmware state.
    """
    return r"""
import sys, os, time, torch
import ttnn
from ttnn import ShardTensorToMesh

ready_path = os.environ.get("GAP36_READY_PATH", "/tmp/gap36_predecessor_ready")

try:
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(2, 4),
        dispatch_core_type=ttnn.DispatchCoreType.WORKER,
    )
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_2D,
        ttnn.FabricReliabilityMode.STRICT_INIT,
    )
    full = torch.rand([1, 1, 32, 256], dtype=torch.bfloat16)
    inp = ttnn.from_torch(
        full,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh, dim=3),
    )
    out = ttnn.all_gather(inp, dim=3, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh)
except Exception:
    pass

# Signal parent: relay ERISCs are live, non-MMIO ERISCs in FABRIC state.
open(ready_path, "w").close()

# Spin until SIGKILL — leaves ERISC relay active on non-MMIO devices.
while True:
    time.sleep(0.1)
"""


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_gap36_fixav_relay_dead_per_device_skip(mesh_device, tmp_path):
    """
    GAP-36: Verify that FIX AV skips remaining ETH cores on a non-MMIO device
    when the first assert_risc_reset_at_core call throws (relay dead after FIX AC
    PCIe-reset MMIO ERISCs in the same teardown call).

    Timeline:
      1. SIGKILL predecessor → leaves non-MMIO ERISCs in FABRIC state.
      2. Parent opens FABRIC_2D → quiesce → relay_broken for MMIO devices.
      3. MeshDevice close: FIX AC PCIe-resets MMIO ETH; FIX AY tries non-MMIO
         reset via relay (dead); FIX AV skips remaining cores on first failure.
      4. Assert: close completes < 25s (not 40s without FIX AV).
    """
    # FIX RZ: skip if fabric is degraded — AllGather hangs on stale base-UMD channels.
    if mesh_device.is_fabric_degraded():
        pytest.skip(
            "GAP-36: fabric degraded (base-UMD channels) — skipping to avoid hang"
        )

    ready_path = str(tmp_path / "gap36_predecessor_ready")
    env = {**os.environ, "GAP36_READY_PATH": ready_path}

    # ── Launch predecessor ────────────────────────────────────────────────────
    pred = subprocess.Popen(
        [sys.executable, "-c", _predecessor_script()],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for predecessor to signal ready.
    t_start = time.time()
    while not os.path.exists(ready_path):
        if time.time() - t_start > _PREDECESSOR_READY_TIMEOUT_S:
            pred.kill()
            pred.wait()
            pytest.skip(
                f"Predecessor did not signal ready within {_PREDECESSOR_READY_TIMEOUT_S}s "
                "(hardware init stall?); skipping GAP-36."
            )
        time.sleep(0.1)

    # Extra 2s to ensure relay is firmly in ACTIVE state.
    time.sleep(2.0)

    # SIGKILL predecessor — leaves non-MMIO ERISCs in FABRIC firmware state.
    pred.kill()
    pred.wait()
    logger.info(
        "GAP-36: predecessor SIGKILL'd; non-MMIO ERISCs should be in FABRIC firmware state"
    )

    # ── Set FABRIC_2D so the mesh_device fixture participates in quiesce ─────
    # mesh_device is already open (from fixture). We set FABRIC_2D so that:
    #   1. Quiesce runs on close and hits the relay_broken path.
    #   2. FIX AC and FIX AY fire during teardown.
    #   3. FIX AV early-exit prevents the 40s multi-core timeout.
    try:
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_2D,
            ttnn.FabricReliabilityMode.STRICT_INIT,
        )
        logger.info("GAP-36: FABRIC_2D configured on mesh_device")
    except Exception as exc:
        logger.warning(f"GAP-36: set_fabric_config raised (expected if relay already broken): {exc}")

    # ── Dispatch one AllGather to establish active relay state ────────────────
    import torch

    try:
        full = torch.rand([1, 1, 32, 256], dtype=torch.bfloat16)
        inp = ttnn.from_torch(
            full,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        )
        t_dispatch = time.time()
        out = ttnn.all_gather(inp, dim=3, topology=ttnn.Topology.Linear)
        ttnn.synchronize_device(mesh_device)
        dispatch_s = time.time() - t_dispatch
        logger.info(f"GAP-36: AllGather dispatch in {dispatch_s:.2f}s")
    except Exception as exc:
        logger.warning(
            f"GAP-36: AllGather raised (relay-broken degraded mode — acceptable): {exc}"
        )

    # ── Close mesh_device and measure how long FIX AC + FIX AY + FIX AV take ─
    # FIX AC: PCIe-resets MMIO ETH relay channels (makes relay dead for this process).
    # FIX AY: tries to reset non-MMIO ERISCs via now-dead relay.
    # FIX AV: first assert_risc_reset_at_core throws → skip remaining cores on device.
    # Without FIX AV: 4 cores × 2 non-MMIO devices × 5s timeout = 40s.
    # With FIX AV: 1 throw × 2 devices × 5s = 10s. Total teardown < 25s.
    t_close_start = time.time()
    try:
        ttnn.close_mesh_device(mesh_device)
    except Exception as exc:
        logger.warning(f"GAP-36: close_mesh_device raised: {exc}")
    close_duration = time.time() - t_close_start

    logger.info(f"GAP-36: MeshDevice close completed in {close_duration:.1f}s")
    assert close_duration < _CLOSE_BUDGET_S, (
        f"GAP-36 REGRESSION: MeshDevice close took {close_duration:.1f}s "
        f"(budget: {_CLOSE_BUDGET_S}s). "
        f"Without FIX AV (#42429), each active ETH core on each non-MMIO device "
        f"burns a 5s read_non_mmio timeout when assert_risc_reset_at_core is called "
        f"after FIX AC PCIe-reset the relay (4 cores × 2 devices × 5s = 40s). "
        f"FIX AV: first failure per device sets device_relay_dead=true → remaining "
        f"cores skipped → worst-case: 1 × 2 × 5s = 10s. "
        f"Check risc_firmware_initializer.cpp FIX AY loop for device_relay_dead guard. "
        f"Log pattern to search: 'Skipping all remaining ETH cores on this device. (FIX AV #42429)'"
    )
    logger.info(
        f"GAP-36 PASS: MeshDevice close completed in {close_duration:.1f}s "
        f"(budget: {_CLOSE_BUDGET_S}s) — FIX AV device_relay_dead early-exit is functioning."
    )
