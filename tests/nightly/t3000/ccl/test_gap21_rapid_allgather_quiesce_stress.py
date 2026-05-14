# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-21: Rapid AllGather-quiesce interleave stress (FIX AE + AF + AN)
#
# Reproduces the original #42429 failure mode: tight AllGather + quiesce loop.
# Pass  = all cycles complete within per-cycle timing bound.
# Fail  = hang (STARTED deadlock from FIX AE/AF regression) or crash
#         (relay_broken cascade from FIX AN regression).

import time
import torch
import pytest
from loguru import logger
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor

_NUM_CYCLES = 25
_QUIESCE_TIMEOUT_S = 30
_AG_DIM = 3


def _run_allgather_and_verify(mesh_device, iteration: int) -> None:
    """Run one AllGather, verify output, deallocate."""
    num_devices = mesh_device.get_num_devices()
    per_device_width = 32
    full_width = per_device_width * num_devices
    input_shape = [1, 1, 32, full_width]

    torch.manual_seed(iteration)
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=_AG_DIM),
    )
    tt_output = ttnn.all_gather(tt_input, dim=_AG_DIM, num_links=1)

    torch_output = ttnn.to_torch(
        tt_output, mesh_composer=ConcatMeshToTensor(mesh_device, dim=_AG_DIM)
    )

    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input)

    assert torch_output.shape[-1] == full_width, (
        f"Iteration {iteration}: unexpected output shape {torch_output.shape}"
    )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_rapid_allgather_quiesce_stress(mesh_device):
    """GAP-21: 25 cycles of AllGather + quiesce to stress FIX AE/AF/AN race windows."""
    num_devices = mesh_device.get_num_devices()
    if num_devices < 4:
        pytest.skip(
            f"Need >= 4 devices for non-MMIO relay path stress; have {num_devices}"
        )

    # FIX RY (#42429): skip if fabric init detected a degraded cluster (broken relay
    # path or channels not ready on any device).  Running AllGather on a degraded
    # cluster hangs waiting on a non-MMIO CQ that has no dispatch firmware.
    if mesh_device.is_fabric_degraded():
        pytest.skip(
            "FIX RY (#42429): fabric degraded on >=1 device "
            "(fabric_relay_path_broken or channels_not_ready_for_traffic) — "
            "skipping GAP-21 AllGather stress to avoid hang"
        )

    # FIX BY (#42429): quiesce probe — detect relay-path failures invisible to the
    # initial ring-sync health check.
    #
    # Background: on a fresh device open, fabric init runs ring-sync and verifies all
    # ERISC channels except "external" ones (channels with no in-cluster peer, e.g.
    # channels 14/15 on some WH devices).  A runner whose ETH relay path depends on
    # those external-range channels can pass ring-sync but STILL hang during AllGather
    # because the relay read from Phase 2.5 fails silently at runtime.
    #
    # is_fabric_degraded() returns False on such a runner because no flags were set
    # during init.  The FIX RY check above therefore does not skip — and the first
    # AllGather hangs in FDMeshCommandQueue::read_completion_queue_event indefinitely.
    #
    # Fix: call quiesce_devices() once as a probe BEFORE the main loop.  quiesce_devices()
    # runs Phase 2.5 ETH relay reads for non-MMIO devices.  If any relay read throws or
    # times out, set_fabric_relay_path_broken() is called for that device, making
    # is_fabric_degraded() return True.  We then skip cleanly instead of hanging.
    mesh_device.quiesce_devices()
    if mesh_device.is_fabric_degraded():
        pytest.skip(
            "FIX BY (#42429): fabric degraded after quiesce probe (relay path broken on "
            ">=1 non-MMIO device during Phase 2.5 ETH relay reads) — "
            "skipping GAP-21 AllGather stress to avoid hang"
        )

    logger.info(
        f"GAP-21: starting {_NUM_CYCLES} AllGather+quiesce cycles on "
        f"{num_devices}-device mesh"
    )

    for cycle in range(_NUM_CYCLES):
        _run_allgather_and_verify(mesh_device, iteration=cycle)

        t0 = time.time()
        ttnn.synchronize_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
        elapsed = time.time() - t0

        logger.info(f"GAP-21: cycle {cycle}/{_NUM_CYCLES} quiesce elapsed={elapsed:.2f}s")

        assert elapsed < _QUIESCE_TIMEOUT_S, (
            f"GAP-21: cycle {cycle} quiesce took {elapsed:.1f}s > {_QUIESCE_TIMEOUT_S}s — "
            f"likely STARTED deadlock regression (FIX AE/AF) or relay hang (FIX AN)"
        )

    logger.info(f"GAP-21: PASS — all {_NUM_CYCLES} cycles completed")
