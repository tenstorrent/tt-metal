# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP_14: Phase 4 MUX timeout force-reset + throw does not leave device in bad state
#
# Strategy: Run a tight AllGather + quiesce loop (5 iterations) with a very short
# MUX terminate timeout (TT_FABRIC_MUX_TERMINATE_TIMEOUT_MS=50 ms).  With this
# setting, Phase 4 quiesce is likely to time out and throw before the MUX handshake
# completes, leaving the handshake in an incomplete state.
#
# The test validates:
#   1. Each quiesce attempt (pass or throw) completes within 30 s — no infinite hang.
#   2. At least 3 of 5 quiesce cycles succeed without exception — fabric recovers.
#   3. A final AllGather after the loop produces correct output — fabric is still
#      functional after one or more partial Phase 4 / Phase 5b interactions.
#
# Background — Phase 4 / Phase 5b interaction (branch nsexton/0-racecondition-hunt):
#   Phase 4 sends a TERMINATE message to the MUX and then polls for ACK.  If the MUX
#   does not ACK within TT_FABRIC_MUX_TERMINATE_TIMEOUT_MS milliseconds, Phase 4
#   issues a force-reset on the MUX core and throws.  This aborts the quiesce path
#   before Phase 5b (socket teardown) has run, which can leave internal socket state
#   and ERISC registers in a partially-torn-down condition.  A subsequent quiesce
#   attempt then enters Phase 5b with stale state, which previously caused a hang or
#   a second unrecoverable throw.
#
# Pass criteria:
#   - No per-quiesce wall-clock time exceeds 30 s.
#   - >= 3/5 quiesce cycles complete without raising an exception.
#   - Final AllGather returns data that matches the expected gathered tensor.

import os
import time
import pytest
import torch
from loguru import logger
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from models.common.utility_functions import skip_for_blackhole

# T3K mesh: 1×8 linear (8 Wormhole devices in a ring)
_MESH_SHAPE = (1, 8)

# Minimum number of devices required for this test (needs at least 2 for AllGather)
_MIN_DEVICES = 2

# AllGather output shape — small enough for tight loops, large enough to stress fabric.
# Shape: [1, 1, 32, 256] gathered on dim=3 → each device holds [1, 1, 32, 32].
_AG_OUTPUT_SHAPE = [1, 1, 32, 256]
_AG_DIM = 3

# Phase 4 MUX terminate timeout — deliberately short to provoke timeout path.
_MUX_TIMEOUT_MS = "50"

# Loop parameters
_NUM_ITERS = 5
_MIN_SUCCESS_ITERS = 3

# Per-quiesce wall-clock deadline (seconds) — if exceeded, the test fails immediately.
_MAX_QUIESCE_SECS = 30.0


def _run_allgather_and_verify(mesh_device, seed):
    """
    Run a single AllGather on mesh_device and verify the output matches the expected
    gathered tensor.  Raises AssertionError on data corruption.
    """
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(seed)

    ag_out_shape = _AG_OUTPUT_SHAPE[:]
    # Full shape is already set; each device shard is size / num_devices along dim.
    per_device_size = ag_out_shape[_AG_DIM] // num_devices  # noqa: F841 (used implicitly by ShardTensorToMesh)

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

    # ConcatMeshToTensor stacks per-device replicas on dim=0; take the first one.
    torch_out = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=0))
    torch_out_single = torch_out[:1, :, :, : ag_out_shape[_AG_DIM]]

    eq, msg = comp_equal(torch_out_single, torch_full[:1])
    assert eq, f"AllGather output mismatch (seed={seed}): {msg}"

    logger.info(f"AllGather OK — seed={seed}, output shape: {list(tt_out.shape)}")
    return list(tt_out.shape)


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
def test_phase4_mux_timeout_recovery(mesh_device):
    """
    GAP_14: Validate that Phase 4 MUX timeout force-reset + throw does not leave
    the device in an unrecoverable state for subsequent quiesce cycles.

    Each iteration of the loop:
      1. Run AllGather to bring ERISC channels and the MUX into active state.
      2. Attempt quiesce (set_fabric_config DISABLED).  With a 50 ms MUX timeout,
         Phase 4 may throw before the handshake completes.  Exceptions are caught and
         counted; the per-quiesce wall-clock time is asserted to be < 30 s.
      3. Re-enable fabric for the next iteration (on success) or attempt recovery
         (on exception).

    After the loop a final AllGather confirms that the fabric is still functional.
    """
    if mesh_device.get_num_devices() < _MIN_DEVICES:
        pytest.skip(f"Requires at least {_MIN_DEVICES} devices; got {mesh_device.get_num_devices()}")

    os.environ["TT_FABRIC_MUX_TERMINATE_TIMEOUT_MS"] = _MUX_TIMEOUT_MS
    logger.info(f"GAP_14: TT_FABRIC_MUX_TERMINATE_TIMEOUT_MS={_MUX_TIMEOUT_MS} ms")

    success_count = 0
    quiesce_times = []

    try:
        for i in range(_NUM_ITERS):
            logger.info(f"=== GAP_14 iteration {i + 1}/{_NUM_ITERS} ===")

            # --- Step 1: AllGather — brings MUX and ERISC channels into active state ---
            logger.info(f"[iter {i}] Running AllGather")
            _run_allgather_and_verify(mesh_device, seed=i * 100)

            # --- Step 2: Attempt quiesce with short MUX timeout ---
            quiesce_start = time.monotonic()
            quiesce_exception = None
            try:
                logger.info(f"[iter {i}] Quiescing fabric (DISABLED) — MUX timeout={_MUX_TIMEOUT_MS} ms")
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                success_count += 1
                logger.info(f"[iter {i}] Quiesce succeeded ({success_count} successes so far)")
            except Exception as exc:
                quiesce_exception = exc
                logger.warning(f"[iter {i}] Quiesce threw (expected for Phase 4 timeout): {exc!r}")
            finally:
                quiesce_elapsed = time.monotonic() - quiesce_start
                quiesce_times.append(quiesce_elapsed)
                logger.info(f"[iter {i}] Quiesce wall-clock: {quiesce_elapsed:.2f} s")

            # Assert no infinite hang — whether the quiesce succeeded or threw.
            assert quiesce_elapsed < _MAX_QUIESCE_SECS, (
                f"[iter {i}] Quiesce took {quiesce_elapsed:.2f} s >= {_MAX_QUIESCE_SECS} s — "
                "possible infinite hang in Phase 4/5b after MUX timeout"
            )

            # --- Step 3: Re-enable fabric for next iteration ---
            # If quiesce threw, fabric may be in a partially-disabled state; attempt to
            # force-disable first so the subsequent re-enable starts from a clean slate.
            if quiesce_exception is not None:
                logger.info(f"[iter {i}] Quiesce threw — attempting force-disable before re-enable")
                try:
                    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
                except Exception as inner_exc:
                    logger.warning(f"[iter {i}] Force-disable after exception also threw: {inner_exc!r}")

            if i < _NUM_ITERS - 1:
                # Re-enable only if there are more iterations to run.
                logger.info(f"[iter {i}] Re-enabling fabric (FABRIC_1D_RING) for next iteration")
                ttnn.set_fabric_config(
                    ttnn.FabricConfig.FABRIC_1D_RING,
                    ttnn.FabricReliabilityMode.STRICT_INIT,
                )

        # --- Assert success rate ---
        logger.info(
            f"GAP_14: {success_count}/{_NUM_ITERS} quiesce cycles succeeded "
            f"(required >= {_MIN_SUCCESS_ITERS})"
        )
        assert success_count >= _MIN_SUCCESS_ITERS, (
            f"Only {success_count}/{_NUM_ITERS} quiesce cycles succeeded; "
            f"expected at least {_MIN_SUCCESS_ITERS}.  "
            "Phase 4 MUX timeout may be leaving device in unrecoverable state."
        )

        # --- Final AllGather: confirm fabric is still functional ---
        logger.info("GAP_14: Running final AllGather to verify fabric is still functional")
        # Re-enable fabric if it is not already enabled (last iteration left it disabled).
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D_RING,
            ttnn.FabricReliabilityMode.STRICT_INIT,
        )
        _run_allgather_and_verify(mesh_device, seed=9999)
        logger.info("GAP_14: Final AllGather passed — fabric functional after MUX timeout stress.")

    finally:
        os.environ.pop("TT_FABRIC_MUX_TERMINATE_TIMEOUT_MS", None)
        # Leave fabric in DISABLED so the conftest fixture can tear down cleanly.
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass
