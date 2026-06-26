# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Silent data-corruption detector for the multicore TILIZE compute path.

On a faulty Galaxy this reproduces silent, intermittent data corruption when
``ttnn.tilize`` (use_multicore=True) tilizes a wide bf16 tensor. The op is a
bitwise-exact layout change (ROW_MAJOR -> TILE), so the device result must equal
the host input exactly; any differing element is silent hardware corruption.

Scope of the fault (device-confirmed on the affected chip): the corruption is
localized to a *single* compute core. The tensor is sized so the multicore
tilize work-split hands a contiguous, known band of tile-rows to each core,
which lets a corrupt row be back-mapped to the exact logical / NoC core that
produced it. A healthy device reads back clean and the test passes.
"""

import pytest
import torch
import ttnn

TILE = 32
# Multicore tilize hands this many rows-of-tiles to each compute core per block.
ROWS_OF_TILES_PER_CORE = 5
# Wide tensor (a multiple of TILE) to exercise the full multicore tilize path.
WIDTH_IN_TILES = 90


def _noc_xy(mesh_device, x, y):
    noc = mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
    return noc.x, noc.y


def _deallocate(tensor):
    if tensor is not None:
        ttnn.deallocate(tensor, force=True)


@pytest.mark.parametrize(
    "device_params",
    # Use the default dispatch-core axis. The fault is on one specific compute core
    # (logical (7,1) on the affected chip); a COL dispatch axis can place that core
    # among the dispatch cores, removing it from the compute grid and hiding the
    # fault entirely. The default axis keeps it in the tilize compute grid.
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}],
    ids=["fabric_2d"],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((8, 4), (8, 4), id="8x4_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("num_iters", [10])
def test_tilize_silent_corruption(mesh_device, mesh_shape, num_iters):
    assert mesh_shape == (8, 4)

    grid = mesh_device.compute_with_storage_grid_size()
    nblocks = grid.x * grid.y * ROWS_OF_TILES_PER_CORE
    rows = nblocks * TILE
    cols = WIDTH_IN_TILES * TILE

    total_corrupt = 0
    for i in range(num_iters):
        ref = torch.rand((rows, cols), dtype=torch.bfloat16)

        tt = None
        out = None
        try:
            # Keep this bf16 -> bf16. The fault lives in the multicore TILIZE
            # compute; an fp32 -> bf16 path routes through a different op and
            # can miss it.
            tt = ttnn.from_torch(
                ref,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            out = ttnn.tilize(tt, use_multicore=True)

            for chip_idx, chip_tensor in enumerate(ttnn.get_device_tensors(out)):
                actual = ttnn.to_torch(chip_tensor)
                diff = actual != ref
                corrupt_count = int(diff.sum().item())
                if corrupt_count == 0:
                    continue

                total_corrupt += corrupt_count
                corrupt_rows = diff.nonzero(as_tuple=False)[:, 0]
                ordinals = torch.div(
                    corrupt_rows, TILE * ROWS_OF_TILES_PER_CORE, rounding_mode="floor"
                )
                for ordinal in ordinals.unique(sorted=True).tolist():
                    cnt = int((ordinals == ordinal).sum().item())
                    x = int(ordinal // grid.y)
                    y = int(ordinal % grid.y)
                    nx, ny = _noc_xy(mesh_device, x, y)
                    print(
                        f"CORRUPT chip={chip_idx} iter={i} "
                        f"core_logical=({x},{y}) noc=({nx},{ny}) corrupt_count={cnt}"
                    )
        finally:
            _deallocate(out)
            _deallocate(tt)

    assert total_corrupt == 0
