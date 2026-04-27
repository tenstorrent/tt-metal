# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP_12: Cross-device relay_broken race in Phase 5 concurrent quiesce
#
# Strategy: Run AllGather (activates all ERISC channels across the T3K mesh),
# then call quiesce (set_fabric_config DISABLED) repeatedly.  On T3K, devices
# 1-7 are non-MMIO and share device 0's relay ERISCs.  If Phase 3 on device 0
# overwrites the relay ERISC L1 while another non-MMIO device is reading through
# it in Phase 5, the per-device fabric_relay_path_broken_ flag (FIX AN) should
# prevent the reading device from spinning indefinitely.
#
# Pass  = every quiesce cycle completes within the 30 s timing bound.
# Fail  = quiesce hangs (relay_broken flag not consulted → Phase 5 spins forever
#         reading a relay ERISC whose L1 has already been overwritten by device 0).
#
# Background — FIX AN (relay_broken per-device flag):
#   Phase 3 of quiesce on the MMIO device (device 0) calls
#   assert_risc_reset_at_core() for the relay ERISCs, which overwrites the
#   relay ERISC's L1 region.  A non-MMIO device that is concurrently executing
#   Phase 5 (polling the relay ERISC's connection-teardown register via the
#   relay path) may receive stale or zeroed data indefinitely because the relay
#   is gone.  FIX AN adds a per-device fabric_relay_path_broken_ flag; once set,
#   Phase 5 treats the relay as unreachable and exits the polling loop rather
#   than hanging.
#
# Timing bound per quiesce cycle: 30 s (generous for this race — normal quiesce
# should complete in < 5 s on healthy hardware).

import time
import torch
import pytest
from loguru import logger
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor
from models.common.utility_functions import skip_for_blackhole

# T3K topology: 1×8 linear mesh (8 Wormhole devices).
_MESH_SHAPE = (1, 8)

# Minimum number of devices needed to exercise the relay path.
# With < 4 devices there are no non-MMIO devices sharing a relay that would
# expose the Phase 3 / Phase 5 race.
_MIN_DEVICES = 4

# AllGather output shape — large enough to ensure all ERISC channels are
# active when quiesce is called, small enough for tight loop performance.
# [1, 1, 32, 256]: each device holds [1, 1, 32, 32] sharded on dim=3.
_AG_OUTPUT_SHAPE = [1, 1, 32, 256]
_AG_DIM = 3

# Number of quiesce cycles to exercise.
_NUM_CYCLES = 5

# Maximum allowable wall-clock time per quiesce cycle (seconds).
_QUIESCE_TIMEOUT_S = 30


def _run_allgather(mesh_device, seed: int) -> None:
    """Launch a single AllGather across the full mesh and discard the result.

    The purpose is to drive all ERISC channels into an active state so that
    the subsequent quiesce hits the concurrent teardown race condition targeted
    by GAP_12.
    """
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(seed)

    # Full gathered shape; each device contributes 1 shard along _AG_DIM.
    per_device_width = _AG_OUTPUT_SHAPE[_AG_DIM] // num_devices
    full_shape = _AG_OUTPUT_SHAPE[:]

    torch_input_full = torch.rand(full_shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input_full,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=_AG_DIM),
    )

    tt_out = ttnn.all_gather(
        input_tensor,
        dim=_AG_DIM,
        topology=ttnn.Topology.Ring,
    )

    logger.info(f"[seed={seed}] AllGather complete, output shape: {list(tt_out.shape)}")


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_cross_device_relay_broken_race(mesh_device):
    """GAP_12: Validate that concurrent quiesce does not hang when device 0's
    Phase 3 kills the relay ERISC L1 while another device is in Phase 5.

    On T3K, non-MMIO devices (all except the MMIO device in the mesh) reach
    the relay ERISC through device 0.  The per-device fabric_relay_path_broken_
    flag (FIX AN) must prevent Phase 5 on those devices from spinning forever
    after the relay is reset by device 0's Phase 3.

    Each of the _NUM_CYCLES iterations:
      1. Run AllGather to activate all ERISC channels across the mesh.
      2. Call set_fabric_config(DISABLED) to trigger quiesce across all devices
         concurrently.  This must complete within _QUIESCE_TIMEOUT_S seconds.
      3. Re-enable fabric (FABRIC_1D_RING) ready for the next iteration.

    A hang in step 2 means FIX AN is absent or ineffective.
    """
    num_devices = mesh_device.get_num_devices()

    # Relay race only meaningful with enough non-MMIO devices to share a relay.
    # The mesh fixture already skips when fewer devices than _MESH_SHAPE are
    # available; this is an explicit guard for the test's logical requirement.
    if num_devices < _MIN_DEVICES:
        pytest.skip(
            f"GAP_12 requires >= {_MIN_DEVICES} devices to exercise the relay path "
            f"(found {num_devices})"
        )

    logger.info(
        f"GAP_12: starting {_NUM_CYCLES} quiesce cycles on {num_devices}-device mesh. "
        f"MMIO device is the mesh's first device; remaining {num_devices - 1} device(s) "
        f"use relay ERISCs through it."
    )

    try:
        for cycle in range(_NUM_CYCLES):
            logger.info(f"=== GAP_12 cycle {cycle + 1}/{_NUM_CYCLES} ===")

            # Step 1: AllGather — drive all ERISC channels into active state.
            logger.info(f"[cycle {cycle}] Running AllGather to activate ERISC channels")
            _run_allgather(mesh_device, seed=cycle)

            # Step 2: Quiesce with timing bound.
            # set_fabric_config(DISABLED) triggers the full multi-phase quiesce
            # sequence across all devices concurrently.  On T3K this means device 0
            # runs Phase 3 (assert reset on relay ERISCs) while devices 1-7 run
            # Phase 5 (poll relay teardown).  FIX AN's fabric_relay_path_broken_
            # flag must fire so Phase 5 on non-MMIO devices exits rather than
            # looping forever reading zeroed relay L1.
            logger.info(f"[cycle {cycle}] Calling set_fabric_config(DISABLED) — quiesce start")
            t_start = time.monotonic()

            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

            elapsed = time.monotonic() - t_start
            logger.info(f"[cycle {cycle}] Quiesce completed in {elapsed:.2f}s")

            assert elapsed < _QUIESCE_TIMEOUT_S, (
                f"[cycle {cycle}] Quiesce hung: took {elapsed:.2f}s (limit {_QUIESCE_TIMEOUT_S}s). "
                f"fabric_relay_path_broken_ flag (FIX AN) may not be preventing Phase 5 spin "
                f"on non-MMIO devices after relay ERISC L1 was overwritten by device 0 Phase 3."
            )

            # Step 3: Re-enable fabric for the next cycle.
            logger.info(f"[cycle {cycle}] Re-enabling fabric (FABRIC_1D_RING)")
            ttnn.set_fabric_config(
                ttnn.FabricConfig.FABRIC_1D_RING,
                ttnn.FabricReliabilityMode.STRICT_INIT,
            )

            logger.info(f"[cycle {cycle}] PASSED ({elapsed:.2f}s)")

        logger.info(f"GAP_12: all {_NUM_CYCLES} quiesce cycle(s) completed within timing bound.")

    finally:
        # Leave fabric in DISABLED state so the conftest fixture teardown
        # (which calls set_fabric_config(DISABLED)) does not error on re-entry.
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass
