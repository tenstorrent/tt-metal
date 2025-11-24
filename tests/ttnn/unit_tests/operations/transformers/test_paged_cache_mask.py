# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from models.common.utility_functions import nearest_y


def get_random_devices(mesh_shape: list[int]) -> set[ttnn.MeshCoordinate]:
    """Get a set of random device coordinates based on the mesh shape."""
    assert len(mesh_shape) == 2, "Mesh shape must be a 2D shape."
    mesh_shape = torch.tensor(mesh_shape)

    num_devices = torch.randint(1, torch.prod(mesh_shape) + 1, (1,)).item()
    device_coords = set()
    while len(device_coords) < num_devices:
        coord = (torch.randint(0, mesh_shape[0], (1,)).item(), torch.randint(0, mesh_shape[1], (1,)).item())
        device_coords.add(coord)

    return {ttnn.MeshCoordinate(*coord) for coord in device_coords}


def run_test_update_cache(mesh_device: ttnn.MeshDevice, cache_shape: list[int], dtype: ttnn.DataType) -> None:
    """Test the paged cache update operation on a mesh device, with only using select devices.

    ┌─────┬─────┬─────┬─────┐
    │  x  │     │  x  │     │
    │(0,0)│(0,1)│(0,2)│(0,3)│
    ├─────┼─────┼─────┼─────┤
    │     │  x  │     │  x  │
    │(1,0)│(1,1)│(1,2)│(1,3)│
    └─────┴─────┴─────┴─────┘

    Legend:
    - Each square represents a device in the mesh
    - (row,col) shows the mesh coordinate
    - 'x' marks devices that perform the update operation
    - Empty squares are idle devices (no update performed)

    Args:
        mesh_device (ttnn.MeshDevice): The mesh device to run the test on.
            Must be a 2D mesh with shape accessible via mesh_device.shape.
        cache_shape (list[int]): Shape of the cache tensor as [bsz, nh, seq_len, dim].
            - bsz: Batch size, must be divisible by mesh_device.shape[1]
            - nh: Number of heads, must be ≤ 32 due to operation limitations
            - seq_len: Sequence length for the cache
            - dim: Head dimension
        dtype (ttnn.DataType): Data type for the tensors (e.g., ttnn.bfloat16).

    Returns:
        None.
    Raises:
        AssertionError: If batch size is not divisible by mesh width, if number
                       of heads exceeds 32, or if output PCC is below threshold.

    Note:
        - The cache is replicated across mesh dimension 0 and sharded across mesh dimension 1
        - Input tensor replicated across mesh dimension 0 and sharded across mesh dimension 1
        - Only devices specified in randomly generated mesh_coords perform updates
    """

    # Setup and validation
    mesh_shape = list(mesh_device.shape)
    bsz, nh, seq_len, dim = cache_shape
    assert bsz % mesh_shape[1] == 0, "Batch size must be divisible by mesh width"
    assert nh <= 32, "Number of heads must be ≤ 32"
    bsz_per_device = bsz // mesh_shape[1]

    # Create torch tensors
    cache = torch.zeros(cache_shape).bfloat16().float()
    inp = torch.ones((1, bsz, nh, dim)).bfloat16().float()
    update_idxs = torch.randint(0, seq_len, (bsz,), dtype=torch.int32)
    mesh_coords = get_random_devices(mesh_shape)

    # TTNN cache setup
    tt_cache = ttnn.from_torch(
        cache,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=mesh_shape),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # TTNN input setup with sharding config
    grid_size = mesh_device.compute_with_storage_grid_size()
    inp_mem_cfg = ttnn.create_sharded_memory_config(
        shape=(nearest_y(nh, ttnn.TILE_SIZE), dim),
        core_grid=ttnn.num_cores_to_corerangeset(bsz_per_device, grid_size, row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    tt_inp = ttnn.from_torch(
        inp,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=mesh_shape),
        memory_config=inp_mem_cfg,
    )

    tt_update_idxs = ttnn.from_torch(
        update_idxs,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=mesh_shape),
    )

    # TTNN operation
    ttnn.experimental.paged_update_cache(tt_cache, tt_inp, update_idxs_tensor=tt_update_idxs, mesh_coords=mesh_coords)

    # Convert back and reshape
    tt_out_torch = ttnn.to_torch(
        tt_cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    )  # Returned in row-major order
    tt_out_torch = tt_out_torch.unsqueeze(0).reshape(mesh_shape[0], *cache_shape)

    # Torch reference implementation
    cache = cache.unsqueeze(0).repeat(mesh_shape[0], 1, 1, 1, 1)
    inp = inp.repeat(mesh_shape[0], 1, 1, 1)

    for coord in mesh_coords:
        coord = list(coord)
        row, col = coord[0], coord[1]
        col_slice = slice(col * bsz_per_device, (col + 1) * bsz_per_device)
        inp_to_use = inp[row, col_slice]
        idxs_to_use = update_idxs[col_slice]

        for i in range(bsz_per_device):
            idx = idxs_to_use[i].item()
            b_idx = col * bsz_per_device + i
            cache[row, b_idx, :, idx] = inp_to_use[i]

    # Validation
    out_pass, out_pcc = comp_equal(tt_out_torch, cache)
    logger.info(f"Output PCC: {out_pcc}")
    assert out_pass, f"Output mismatch: PCC {out_pcc} < 1.0"


@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2_grid")], indirect=True)
@pytest.mark.parametrize("cache_shape", [(32, 1, 32, 128)])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
def test_update_cache(mesh_device, cache_shape, dtype, reset_seeds):
    run_test_update_cache(mesh_device, cache_shape, dtype)
