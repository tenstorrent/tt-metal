# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from dataclasses import dataclass, field
from typing import List
from collections import defaultdict

from ttnn import (
    ShardTensor2dMesh,
    ConcatMesh2dToTensor,
)
from models.utility_functions import nearest_32

# from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


@dataclass
class Location:
    row_t: int  # nth tile in the row
    col_t: int  # nth tile in the column
    row_d: int  # nth row of a tile
    col_d: int  # nth column of a tile


@dataclass
class DeviceFailure:
    device_loc: tuple
    num_failures: int = 0
    diffs: List[float] = field(default_factory=list)
    locs: List[Location] = field(default_factory=list)


def aggregate_failures(failures: List[dict]) -> dict:
    aggregate = defaultdict(lambda: DeviceFailure(device_loc=None))

    for failure_dict in failures:
        for device_loc, dev_failure in failure_dict.items():
            if dev_failure is None:
                continue

            if aggregate[device_loc].device_loc is None:
                aggregate[device_loc].device_loc = device_loc

            aggregate[device_loc].num_failures += dev_failure.num_failures
            aggregate[device_loc].diffs.extend(dev_failure.diffs)
            aggregate[device_loc].locs.extend(dev_failure.locs)

    return aggregate


@pytest.mark.parametrize(
    "mesh_shape, mesh_device", [pytest.param((8, 4), (8, 4), id="8x4_grid")], indirect=["mesh_device"]
)
@pytest.mark.parametrize(
    "M, K, N, weights_dtype",
    [
        pytest.param(32, 8192, 32768, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize(
    "num_iters",
    [
        10000,
        1000,
    ],
    ids=[
        "long",
        "smoke",
    ],
)
def test_galaxy_nd(M, K, N, weights_dtype, mesh_shape, mesh_device, num_iters):
    torch.manual_seed(1234)

    # Use a combination of ones and randn to get more readible output
    act_pt = torch.ones(1, 1, M, K)
    weights_pt = torch.randn(1, 1, K, N) * 32

    gt = act_pt @ weights_pt

    act_shard_dim = (3, None)
    weight_shard_dim = (2, 3)
    concat_dim = (0, 1)

    K = K // mesh_shape[0]
    N = N // mesh_shape[1]

    act_mem_config = ttnn.create_sharded_memory_config(
        shape=(M, K // 8),
        core_grid=ttnn.CoreGrid(y=1, x=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    act = ttnn.from_torch(
        act_pt,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=act_mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, dims=act_shard_dim, mesh_shape=mesh_shape),
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    weight_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(mesh_device.dram_grid_size().x - 1, mesh_device.dram_grid_size().y - 1),
            )
        }
    )
    shard_shape = (K, nearest_32(N // 12))  # padded cols to divide by 12
    shard_spec = ttnn.ShardSpec(weight_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    weight_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    weights = ttnn.from_torch(
        weights_pt,
        dtype=weights_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=weight_mem_config,
        mesh_mapper=ShardTensor2dMesh(mesh_device, dims=weight_shard_dim, mesh_shape=mesh_shape),
    )

    DRAM_SHARDED_PROGCFG = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=K // 8 // 32,  # K = 8192 / TILE_WIDTH=32 / Grid_Size is based on compute_with_storage_grid_size
        per_core_M=M // 32,  # M / TILE_HEIGHT = 32 / 32
        per_core_N=N // 8 // 32,  # N / TILE_WIDTH / Grid_Size is based on compute_with_storage_grid_size
        fused_activation=None,
    )

    outs = []
    for i in range(num_iters):
        logger.info(f"Running iteration {i}")
        out = ttnn.matmul(
            act,
            weights,
            program_config=DRAM_SHARDED_PROGCFG,
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        out = ttnn.to_torch(
            out, mesh_composer=ConcatMesh2dToTensor(mesh_device, dims=concat_dim, mesh_shape=mesh_shape)
        )
        outs.append(out)

        # PCC checking against torch is disabled for now
        # r, c, m, n = out.shape
        # out = out.permute(0, 2, 1, 3).contiguous()  # from [r, c, m, n] to [r, m, c, n]
        # out = out.view(r, 1, m, c * n)
        # out = torch.sum(out, dim=0, keepdim=True)

        # out_pass, out_pcc = comp_pcc(gt, out, pcc=0.99)
        # if not out_pass:
        #     logger.warning(f"PCC value: {out_pcc}")

    # assert out_pass, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"

    # Compare outputs to make sure they are same
    failures = []
    all_passing = True
    golden = outs[0]
    for i, out in enumerate(outs):
        logger.info(f"Checking output for iteration {i}")

        passing = torch.all(out == golden)

        if passing:
            logger.info(f"Output for iteration {i} is equal to golden")
        else:
            locs = torch.where(out != golden)
            diff = out[locs] - golden[locs]
            device_groups = {(y, x): None for y in range(mesh_shape[0]) for x in range(mesh_shape[1])}

            # Create all relevent DeviceFailure objects
            num_failures = len(locs[0])
            for i in range(num_failures):
                # Get the device location of the failure
                row_device = locs[0][i].item()
                col_device = locs[1][i].item()

                device_loc = (row_device, col_device)
                assert device_loc in device_groups, f"Device {device_loc} not in device_groups"

                # Update the DeviceFailure object
                if device_groups[device_loc] is None:
                    device_groups[device_loc] = DeviceFailure(device_loc)

                # Get the location of the failure
                row_t = locs[2][i].item() // 32
                col_t = locs[3][i].item() // 32
                row_d = locs[2][i].item() % 32
                col_d = locs[3][i].item() % 32

                device_groups[device_loc].num_failures += 1
                device_groups[device_loc].diffs.append(diff[i].item())
                device_groups[device_loc].locs.append(Location(row_t, col_t, row_d, col_d))

            failures.append(device_groups)

            logger.warning(f"Output for iteration {i} is NOT equal to golden")
        all_passing = all_passing and passing

    aggregated = aggregate_failures(failures)

    # Print report
    logger.info("\n===== ND Failure Report =====\n")
    if not aggregated:
        logger.info("No ND failures detected.")
    else:
        for device_loc, failure in sorted(aggregated.items()):
            mean_diff = sum(failure.diffs) / len(failure.diffs) if failure.diffs else 0
            max_diff = max(failure.diffs) if failure.diffs else 0
            logger.info(f"Device {device_loc}:")
            logger.info(f"  Total Failures: {failure.num_failures}")
            logger.info(f"  Mean Diff: {mean_diff:.6f}")
            logger.info(f"  Max Diff: {max_diff:.6f}")
            logger.info(f"  Failure Locations (tile coords):")
            for loc in failure.locs[:5]:  # limit to first 5 for brevity
                logger.info(f"    Tile ({loc.row_t}, {loc.col_t}), Pos in tile: ({loc.row_d}, {loc.col_d})")
            if len(failure.locs) > 5:
                logger.info(f"    ... and {len(failure.locs) - 5} more")

    logger.info("Saving report to nd_failure_summary.txt")
    with open("nd_failure_summary.txt", "w") as f:
        f.write("===== ND Failure Report =====\n\n")

        if not aggregated:
            f.write("No ND failures detected.\n")
        else:
            failed_devices = sorted(aggregated.keys())
            f.write(f"Total devices with failures: {len(failed_devices)}\n")
            f.write("Devices with failures:\n")
            for dev in failed_devices:
                f.write(f"  - {dev}\n")
            f.write("\n")

            for device_loc, failure in sorted(aggregated.items()):
                mean_diff = sum(failure.diffs) / len(failure.diffs) if failure.diffs else 0
                max_diff = max(failure.diffs) if failure.diffs else 0
                f.write(f"Device {device_loc}:\n")
                f.write(f"  Total Failures: {failure.num_failures}\n")
                f.write(f"  Mean Diff: {mean_diff:.6f}\n")
                f.write(f"  Max Diff: {max_diff:.6f}\n")
                f.write(f"  Failure Locations (tile coords):\n")
                for loc in failure.locs:
                    f.write(f"    Tile ({loc.row_t}, {loc.col_t}), Pos in tile: ({loc.row_d}, {loc.col_d})\n")
                f.write("\n")

    assert all_passing, f"ND behavior detected!"
