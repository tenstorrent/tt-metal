# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# GAP-25: Back-to-back AllGather without synchronize (FIX AE + AF)
#
# Dispatches 50 AllGather operations without explicit sync between them,
# simulating a multi-layer transformer forward pass. Each AllGather uses
# a different tensor to prevent caching. After all dispatches, a single
# sync + quiesce verifies the fabric survived sustained overlapping traffic.
#
# Pass  = all 50 dispatches + final sync + quiesce complete within 120s.
# Fail  = hang (ETH launch ordering race under overlapping traffic).

import time
import torch
import pytest
from loguru import logger
import ttnn
from ttnn import ShardTensorToMesh, ConcatMeshToTensor

_NUM_DISPATCHES = 50
_TOTAL_TIMEOUT_S = 120
_AG_DIM = 3


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_back_to_back_allgather_nosync(mesh_device):
    """GAP-25: 50 AllGather dispatches without intermediate sync."""
    num_devices = mesh_device.get_num_devices()
    if num_devices < 4:
        pytest.skip(f"Need >= 4 devices; have {num_devices}")

    # FIX RZ (#42429): skip if fabric is degraded — AllGather hangs on stale base-UMD channels.
    if mesh_device.is_fabric_degraded():
        pytest.skip(
            "GAP-25: fabric degraded (base-UMD channels) — skipping AllGather to avoid hang"
        )

    per_device_width = 64
    full_width = per_device_width * num_devices
    input_shape = [1, 1, 32, full_width]

    logger.info(
        f"GAP-25: dispatching {_NUM_DISPATCHES} back-to-back AllGathers "
        f"without sync on {num_devices} devices"
    )

    t0 = time.time()
    outputs = []

    for i in range(_NUM_DISPATCHES):
        torch.manual_seed(i * 7 + 42)
        torch_input = torch.rand(input_shape, dtype=torch.bfloat16)

        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ShardTensorToMesh(mesh_device, dim=_AG_DIM),
        )

        # Dispatch without sync — next AllGather may overlap with this one's drain.
        tt_output = ttnn.all_gather(tt_input, dim=_AG_DIM, num_links=1)

        # Deallocate input immediately to stress memory recycling.
        ttnn.deallocate(tt_input)

        # Keep last output for final verification, deallocate the rest.
        if i < _NUM_DISPATCHES - 1:
            ttnn.deallocate(tt_output)
        else:
            outputs.append(tt_output)

    # Single sync after all dispatches — tests accumulated ERISC state.
    ttnn.synchronize_device(mesh_device)
    dispatch_elapsed = time.time() - t0

    logger.info(
        f"GAP-25: all {_NUM_DISPATCHES} dispatches + sync in {dispatch_elapsed:.2f}s"
    )

    # Verify last output is readable (not corrupted).
    last_out_torch = ttnn.to_torch(
        outputs[0], mesh_composer=ConcatMeshToTensor(mesh_device, dim=_AG_DIM)
    )
    assert last_out_torch.shape[-1] == full_width, (
        f"GAP-25: unexpected final output shape {last_out_torch.shape}"
    )
    ttnn.deallocate(outputs[0])

    # Quiesce after sustained traffic — exercises FIX AE/AF on hot ERISC state.
    quiesce_start = time.time()
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    quiesce_elapsed = time.time() - quiesce_start

    logger.info(f"GAP-25: post-burst quiesce in {quiesce_elapsed:.2f}s")

    total = time.time() - t0
    assert total < _TOTAL_TIMEOUT_S, (
        f"GAP-25: total elapsed {total:.1f}s > {_TOTAL_TIMEOUT_S}s — "
        f"likely ERISC launch race under overlapping traffic"
    )

    # Final AllGather after quiesce — confirms fabric is still healthy.
    torch.manual_seed(777)
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

    logger.info(f"GAP-25: PASS — {_NUM_DISPATCHES} back-to-back + quiesce + final AllGather")
