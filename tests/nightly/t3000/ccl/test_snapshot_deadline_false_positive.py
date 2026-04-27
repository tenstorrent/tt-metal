# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP_15: ENTRY snapshot deadline false positive does not permanently degrade fabric
#
# Background:
#   During quiesce, each non-MMIO device performs an ENTRY snapshot — a diagnostic
#   read of each active ERISC channel's status register through the MMIO device's
#   relay ERISCs (device.cpp ~line 770).  A 6 s deadline (kSnapshotDeadlineMs=6000)
#   is applied to the total snapshot time.  If the MMIO device's relay ERISCs are
#   mid-reboot (e.g. in the middle of their own READY_FOR_TRAFFIC transition) when
#   a non-MMIO device starts the snapshot, individual reads can each stall for ~5 s
#   before UMD raises an exception.  Once the accumulated stall exceeds the 6 s
#   deadline the code sets fabric_relay_path_broken_=true as a side effect of a
#   diagnostic read — not because the relay is actually dead.
#
# The bug this test targets:
#   After the snapshot deadline fires and sets fabric_relay_path_broken_=true, the
#   flag must be cleared at the start of the next quiesce cycle (Phase 1 / re-init).
#   If it is NOT cleared, every subsequent ENTRY snapshot is entirely skipped
#   (because fabric_relay_path_broken_ remains true) and worse, Phase 2.5 and
#   Phase 3 also skip relay operations, leaving the fabric unable to fully
#   reinitialise.  This manifests as a cascading 5 s-per-channel timeout
#   accumulation across the 10 cycles, yielding total time >> 120 s, and the
#   final AllGather failing because the relay path was never recovered.
#
# Pass criteria:
#   - Every quiesce cycle completes within 30 s (no 5 s-per-channel accumulation).
#   - Total time for all 10 cycles < 120 s (no cascading degradation).
#   - A final AllGather after the stress loop succeeds and has the correct shape.
#
# Hardware requirement:
#   >= 4 devices (T3K).  With fewer devices there are no non-MMIO devices sharing a
#   relay ERISC and the snapshot-deadline race cannot be triggered.

import time
import torch
import pytest
from loguru import logger
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from models.common.utility_functions import skip_for_blackhole

# T3K topology: 1×8 linear mesh (8 Wormhole devices in a ring).
_MESH_SHAPE = (1, 8)

# Minimum number of devices to expose MMIO / non-MMIO relay interaction.
_MIN_DEVICES = 4

# AllGather tensor dimensions.
# [1, 1, 32, 256] gathered on dim=3 → each device holds [1, 1, 32, 32].
# Small enough for tight loops, large enough to activate all ERISC channels.
_AG_OUTPUT_SHAPE = [1, 1, 32, 256]
_AG_DIM = 3

# Number of quiesce cycles to run under stress.
_NUM_CYCLES = 10

# Per-cycle wall-clock deadline (seconds).
_MAX_CYCLE_SECS = 30.0

# Total wall-clock deadline for all cycles combined (seconds).
# 10 cycles × 3 s expected each = 30 s nominal; 120 s allows ~4× headroom.
# If the relay_broken flag leaks across cycles, each cycle accumulates 5 s per
# active channel (typically 8-16 channels → 40-80 s extra per cycle) → total
# easily exceeds 120 s.
_MAX_TOTAL_SECS = 120.0


def _run_allgather_and_verify(mesh_device, seed: int):
    """Run a single AllGather across the full mesh and verify output shape and data.

    Returns the output shape as a list for the caller to log / assert.
    """
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(seed)

    ag_out_shape = _AG_OUTPUT_SHAPE[:]
    torch_full = torch.rand(ag_out_shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_full,
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

    out_shape = list(tt_out.shape)
    assert out_shape == ag_out_shape, (
        f"AllGather output shape {out_shape} does not match expected {ag_out_shape}"
    )

    # Verify data correctness: ConcatMeshToTensor stacks per-device replicas on
    # dim=0; take the first replica and compare against the original input.
    torch_out = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=0))
    torch_out_single = torch_out[:1, :, :, : ag_out_shape[_AG_DIM]]

    eq, msg = comp_equal(torch_out_single, torch_full[:1])
    assert eq, f"AllGather data mismatch (seed={seed}): {msg}"

    logger.info(f"[seed={seed}] AllGather OK — shape={out_shape}")
    return out_shape


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_snapshot_deadline_false_positive(mesh_device):
    """GAP_15: ENTRY snapshot deadline firing must not permanently set
    fabric_relay_path_broken_ across subsequent quiesce cycles.

    The test runs _NUM_CYCLES of:
      1. AllGather — activates all ERISC channels so the ENTRY snapshot has
         something to read through the MMIO device's relay ERISCs.
      2. Quiesce (set_fabric_config DISABLED) — triggers the ENTRY snapshot on each
         non-MMIO device.  When the MMIO device's relay ERISCs are mid-reboot, reads
         stall and the 6 s deadline (kSnapshotDeadlineMs) may fire, setting
         fabric_relay_path_broken_=true as a false positive.
      3. Re-enable fabric (FABRIC_1D_RING) — Phase 1 / re-init must clear
         fabric_relay_path_broken_ so the next cycle can take the relay path again.

    After all cycles a final AllGather must succeed, proving the relay path was
    recovered and not permanently disabled by the false-positive snapshot deadline.

    Timing assertions:
      - Each quiesce <= _MAX_CYCLE_SECS to detect per-cycle 5 s-per-channel hangs.
      - Total time for all cycles <= _MAX_TOTAL_SECS to detect cascading degradation
        (the primary symptom when relay_broken leaks across cycles).
    """
    num_devices = mesh_device.get_num_devices()

    if num_devices < _MIN_DEVICES:
        pytest.skip(
            f"GAP_15 requires >= {_MIN_DEVICES} devices for MMIO/non-MMIO relay "
            f"interaction (found {num_devices})"
        )

    logger.info(
        f"GAP_15: starting {_NUM_CYCLES} quiesce cycles on {num_devices}-device mesh "
        f"(MMIO device = first mesh device; remaining {num_devices - 1} use relay ERISCs)."
    )

    total_start = time.monotonic()

    try:
        for cycle in range(_NUM_CYCLES):
            logger.info(f"=== GAP_15 cycle {cycle + 1}/{_NUM_CYCLES} ===")

            # Step 1: AllGather — drive all ERISC channels into READY_FOR_TRAFFIC so
            # the ENTRY snapshot has active channels to read through the relay.
            logger.info(f"[cycle {cycle}] Running AllGather to activate ERISC channels")
            _run_allgather_and_verify(mesh_device, seed=cycle)

            # Step 2: Quiesce — triggers the ENTRY snapshot on every non-MMIO device.
            # If the MMIO device's relay ERISCs are mid-reboot when a non-MMIO device
            # starts its snapshot reads, individual reads stall for ~5 s each.  Once
            # accumulated stall > kSnapshotDeadlineMs (6 s), fabric_relay_path_broken_
            # is set.  This is a *false positive* — the relay will be available again
            # after re-init.  If the flag is NOT cleared on re-init, subsequent cycles
            # skip the relay path entirely and fabric never recovers.
            logger.info(f"[cycle {cycle}] Quiescing fabric (DISABLED) — snapshot deadline under test")
            cycle_start = time.monotonic()

            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

            cycle_elapsed = time.monotonic() - cycle_start
            logger.info(f"[cycle {cycle}] Quiesce completed in {cycle_elapsed:.2f} s")

            assert cycle_elapsed < _MAX_CYCLE_SECS, (
                f"[cycle {cycle}] Quiesce took {cycle_elapsed:.2f} s "
                f"(limit {_MAX_CYCLE_SECS} s). "
                "Possible 5 s-per-channel timeout accumulation — "
                "fabric_relay_path_broken_ may not have been cleared on entry to this cycle, "
                "or the relay path is genuinely broken."
            )

            # Step 3: Re-enable fabric — Phase 1 must clear fabric_relay_path_broken_.
            # If it does, the next cycle's ENTRY snapshot can use the relay again.
            # If it does not, the next snapshot is skipped and Phase 2.5/3 relay
            # writes are also skipped, causing cascading degradation.
            if cycle < _NUM_CYCLES - 1:
                logger.info(f"[cycle {cycle}] Re-enabling fabric (FABRIC_1D_RING) for next cycle")
                ttnn.set_fabric_config(
                    ttnn.FabricConfig.FABRIC_1D_RING,
                    ttnn.FabricReliabilityMode.STRICT_INIT,
                )

            logger.info(f"[cycle {cycle}] PASSED ({cycle_elapsed:.2f} s)")

        total_elapsed = time.monotonic() - total_start
        logger.info(
            f"GAP_15: all {_NUM_CYCLES} cycles completed in {total_elapsed:.2f} s "
            f"(limit {_MAX_TOTAL_SECS} s)."
        )

        assert total_elapsed < _MAX_TOTAL_SECS, (
            f"Total time for {_NUM_CYCLES} quiesce cycles: {total_elapsed:.2f} s "
            f"(limit {_MAX_TOTAL_SECS} s). "
            "Cascading timeout accumulation detected — "
            "fabric_relay_path_broken_ false-positive flag likely leaking across cycles."
        )

        # Final AllGather: confirms the relay path was fully recovered after the
        # stress loop and the fabric is still functional for production workloads.
        logger.info("GAP_15: re-enabling fabric for final AllGather verification")
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D_RING,
            ttnn.FabricReliabilityMode.STRICT_INIT,
        )

        logger.info("GAP_15: running final AllGather to confirm fabric is still functional")
        final_shape = _run_allgather_and_verify(mesh_device, seed=_NUM_CYCLES * 100)

        assert final_shape == _AG_OUTPUT_SHAPE, (
            f"Final AllGather output shape {final_shape} != expected {_AG_OUTPUT_SHAPE}"
        )

        logger.info(
            f"GAP_15: PASSED — {_NUM_CYCLES} quiesce cycles completed in {total_elapsed:.2f} s, "
            f"final AllGather shape {final_shape} correct.  "
            "ENTRY snapshot deadline false-positive does not permanently degrade fabric."
        )

    finally:
        # Leave fabric in DISABLED so the conftest fixture teardown does not
        # error on re-entry to set_fabric_config(DISABLED).
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass
