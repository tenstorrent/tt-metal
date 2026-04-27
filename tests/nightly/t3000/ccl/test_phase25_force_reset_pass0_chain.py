# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP_11: Phase 2.5 force-reset + Pass-0 canary chain validation
#
# Strategy: Run tight AllGather + quiesce loops with short ERISC terminate timeout
# to force more channels into force-reset path.  After each quiesce, verify second
# AllGather succeeds (proves force-reset channels were properly re-launched via Pass-0).
#
# Pass = all iterations complete with AllGather producing correct output.
# Fail = quiesce hangs (Pass-0 deassert failed) or AllGather returns corrupt data
#        (channel was rejected by FIX-3 when canary wasn't present after 50 ms).
#
# Background — FIX AS (Pass-0 approach):
#   Phase 2.5 may issue assert_risc_reset_at_core(ALL) on ERISCs that do not self-
#   terminate within the polling budget (force-reset path).  FIX AS deasserts ALL
#   RISCs for those channels BEFORE writing the launch message, then sleeps 50 ms
#   so the base UMD firmware completes .bss init and enters its launch-message
#   polling loop (canary 0x49706550 visible after init).  FIX-3's pre-launch gate
#   was updated to accept the UMD relay canary as a valid quiesced state for
#   force-reset channels.  Without this ordering the ERISC ignores the launch
#   message (.bss init zeroes the mailbox area after deassert), stays in base
#   firmware, and the next AllGather sees an unresponsive channel.
#
# Note on TT_FABRIC_ERISC_TERMINATE_TIMEOUT_MS:
#   This env var is set here as a signal to the runtime (it may be honoured in
#   future builds to override the Phase 2.5 polling budget) and to document the
#   intent of stressing the force-reset path.  In current builds the timeout is
#   hardcoded at 2000 ms inside device.cpp; the force-reset path is therefore
#   exercised naturally whenever AllGather leaves ERISCs in mid-operation state at
#   quiesce time (which the tight loop reliably achieves).

import os
import torch
import pytest
from loguru import logger
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from models.common.utility_functions import skip_for_blackhole

# T3K: 1×8 linear mesh (8 Wormhole devices in a ring)
_MESH_SHAPE = (1, 8)

# AllGather output shape — small enough for tight loops, large enough to stress fabric.
# Shape: [1, 1, 32, 256] gathered on dim=3 → each device holds [1, 1, 32, 32].
_AG_OUTPUT_SHAPE = [1, 1, 32, 256]
_AG_DIM = 3


def _run_allgather_and_verify(mesh_device, iteration_label):
    """
    Run a single AllGather on `mesh_device`, verify the output shape, and return
    the output shape.  Correctness is validated with comp_equal so any corruption
    from a mis-launched ERISC channel causes an assertion failure here.
    """
    num_devices = mesh_device.get_num_devices()
    torch.manual_seed(iteration_label)

    # Input per device: [1, 1, 32, 32] (dim=3 sharded across devices)
    ag_out_shape = _AG_OUTPUT_SHAPE[:]
    ag_out_shape[_AG_DIM] = _AG_OUTPUT_SHAPE[_AG_DIM]  # full gathered shape
    per_device_size = ag_out_shape[_AG_DIM] // num_devices

    # Build the full reference tensor (what the gathered result should look like on
    # every device: the concatenation of all device shards along dim=3).
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

    # Bring output back to host and compare against the expected gathered tensor.
    # ConcatMeshToTensor along dim=0 stacks per-device replicas; we keep only the
    # first replica (all should be identical after a correct AllGather).
    torch_out = ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(mesh_device, dim=0))
    # Shape after concat on dim=0: [num_devices, 1, 32, 256]; take device 0's view.
    torch_out_single = torch_out[:1, :, :, : ag_out_shape[_AG_DIM]]

    eq, msg = comp_equal(torch_out_single, torch_full[:1])
    assert eq, f"[{iteration_label}] AllGather output mismatch — force-reset channel may not have been re-launched: {msg}"

    logger.info(f"[{iteration_label}] AllGather OK, output shape: {list(tt_out.shape)}")
    return list(tt_out.shape)


@skip_for_blackhole("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("num_iters", [3, 5], ids=["iters_3", "iters_5"])
def test_phase25_force_reset_pass0_chain(mesh_device, num_iters):
    """
    GAP_11: Validate that Phase 2.5 force-reset channels complete the
    Pass-0 deassert → UMD firmware boot → canary acceptance chain before
    the second AllGather is dispatched.

    Each iteration:
      1. Run AllGather (first — establishes active ERISC channels).
      2. Quiesce fabric (set_fabric_config DISABLED → triggers Phase 2.5 + Pass-0
         for any force-reset channels → re-initialises fabric workers).
      3. Re-enable fabric (set_fabric_config FABRIC_1D_RING).
      4. Run second AllGather and verify correct output shape and values.
         A corrupt result or hang here means a force-reset channel was not
         properly re-launched through the Pass-0 canary acceptance path.
    """
    # Signal intent to stress the force-reset path (see module-level docstring).
    os.environ["TT_FABRIC_ERISC_TERMINATE_TIMEOUT_MS"] = "100"

    try:
        for i in range(num_iters):
            logger.info(f"=== GAP_11 iteration {i + 1}/{num_iters} ===")

            # --- Step 1: first AllGather (brings ERISC channels into active state) ---
            logger.info(f"[iter {i}] Running first AllGather")
            shape1 = _run_allgather_and_verify(mesh_device, iteration_label=i * 10)

            # --- Step 2: quiesce (tears down ERISC channels, forces Phase 2.5) ---
            # set_fabric_config(DISABLED) calls quiesce_devices() internally, which
            # runs the Phase 2.5 TERMINATE poll.  With ERISCs left in mid-operation
            # state from the AllGather, the short termination window increases the
            # probability that channels hit the force-reset path (assert_risc_reset).
            logger.info(f"[iter {i}] Quiescing fabric (DISABLED)")
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

            # --- Step 3: re-enable fabric (triggers Phase-0 launch / Pass-0 sequence) ---
            # FIX AS ensures force-reset channels are deasserted BEFORE their launch
            # message is written, so the UMD canary (0x49706550) is present when
            # FIX-3's pre-launch gate checks channel readiness.
            logger.info(f"[iter {i}] Re-enabling fabric (FABRIC_1D_RING)")
            ttnn.set_fabric_config(
                ttnn.FabricConfig.FABRIC_1D_RING,
                ttnn.FabricReliabilityMode.STRICT_INIT,
            )

            # --- Step 4: second AllGather (validates that re-launched channels work) ---
            logger.info(f"[iter {i}] Running second AllGather (post-quiesce)")
            shape2 = _run_allgather_and_verify(mesh_device, iteration_label=i * 10 + 1)

            assert shape1 == shape2, (
                f"[iter {i}] Output shape changed between first ({shape1}) and "
                f"second ({shape2}) AllGather — ERISC channel state inconsistency"
            )
            logger.info(f"[iter {i}] PASSED — both AllGathers produced shape {shape2}")

        logger.info(f"GAP_11: all {num_iters} iteration(s) passed.")

    finally:
        # Clean up env var so subsequent tests in the same process are not affected.
        os.environ.pop("TT_FABRIC_ERISC_TERMINATE_TIMEOUT_MS", None)
        # Restore fabric to the state the fixture expects on teardown (DISABLED).
        # The conftest fixture's reset_fabric() will call set_fabric_config(DISABLED)
        # again, which is safe when already disabled.
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass
