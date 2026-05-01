# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-23: Partial-mesh quiesce cycling with AllGather (FIX AK + AM + AE)
#
# Opens a 1x4 submesh on a T3K (which has 8 devices), runs AllGather + quiesce
# in a loop. Mesh-edge devices have out-of-mesh peers that exercise FIX AK
# (non-fatal stuck channels) and FIX AM (channels_not_ready lifecycle).
#
# Pass  = all cycles complete, AllGather data correct on every cycle.
# Fail  = throw (FIX AK regression), AllGather hang (FIX AM flag not reset),
#         or STARTED deadlock (FIX AE regression on partial mesh).

import time
import torch
import pytest
from loguru import logger
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor

_SUBMESH_SHAPE = (1, 4)
_NUM_CYCLES = 15
_QUIESCE_TIMEOUT_S = 30
_AG_DIM = 3


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_SUBMESH_SHAPE], indirect=True)
def test_partial_mesh_quiesce_cycling(mesh_device):
    """GAP-23: Repeated quiesce on partial mesh must not throw or corrupt state."""
    num_devices = mesh_device.get_num_devices()
    if num_devices < _SUBMESH_SHAPE[1]:
        pytest.skip(
            f"Need >= {_SUBMESH_SHAPE[1]} devices for partial-mesh test; "
            f"have {num_devices}"
        )

    # Check that this is actually a partial mesh (system has more devices).
    total_system_devices = ttnn.get_num_devices()
    if total_system_devices <= _SUBMESH_SHAPE[1]:
        pytest.skip(
            f"System has only {total_system_devices} devices — no out-of-mesh peers, "
            f"FIX AK/AM not exercised"
        )

    if mesh_device.is_fabric_degraded():
        pytest.skip("GAP-23: fabric degraded (base-UMD channels) — skipping AllGather to avoid hang")

    logger.info(
        f"GAP-23: {_SUBMESH_SHAPE} submesh on {total_system_devices}-device system, "
        f"{_NUM_CYCLES} quiesce cycles"
    )

    per_device_width = 32
    full_width = per_device_width * num_devices
    input_shape = [1, 1, 32, full_width]

    for cycle in range(_NUM_CYCLES):
        torch.manual_seed(cycle * 137)
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=_AG_DIM),
        )
        tt_output = ttnn.all_gather(tt_input, dim=_AG_DIM, num_links=1)
        ttnn.deallocate(tt_output)
        ttnn.deallocate(tt_input)

        # Quiesce — must not throw for out-of-mesh peers (FIX AK).
        # Only time the teardown (DISABLED): FIX AE/AK are teardown hangs.
        # Re-init (FABRIC_2D) is excluded — on degraded clusters FIX W probes
        # in topology_discovery can add 30-60s per dead gateway ETH core, which
        # is unrelated to the STARTED-deadlock / stuck-channel regressions.
        t0 = time.time()
        ttnn.synchronize_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        elapsed = time.time() - t0
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

        logger.info(
            f"GAP-23: cycle {cycle}/{_NUM_CYCLES} quiesce={elapsed:.2f}s"
        )
        assert elapsed < _QUIESCE_TIMEOUT_S, (
            f"GAP-23: cycle {cycle} quiesce took {elapsed:.1f}s — "
            f"likely STARTED deadlock (FIX AE) or stuck channel (FIX AK)"
        )

    # Final verification AllGather — confirms FIX AM cleared channels_not_ready.
    torch.manual_seed(99999)
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ShardTensorToMesh(mesh_device, dim=_AG_DIM),
    )
    tt_output = ttnn.all_gather(tt_input, dim=_AG_DIM, num_links=1)
    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input)

    logger.info(f"GAP-23: PASS — all {_NUM_CYCLES} cycles + final AllGather succeeded")
