# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import ttnn
from models.utility_functions import is_wormhole_b0, is_grayskull, skip_for_wormhole_b0
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, pad_by_zero, roundup32
import torch
import itertools
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_equal,
    comp_pcc,
)
import random
import math
from models.utility_functions import is_wormhole_b0, is_grayskull, is_wormhole_b0, is_blackhole
from tracy import signpost

from models.demos.llama3_subdevices.tt.model_config import (
    PREFETCHER_NOC1_GRID,
)


random.seed(10)


def num_cores_to_rectangle_grid(num_cores, device):
    """
    Find a rectangular core grid size, given an number of cores.

    Return None if rectangle grid is not possible.
    """
    x = device.compute_with_storage_grid_size().x
    while x > 0 and num_cores % x != 0:
        x -= 1

    if x == 0:
        return None

    y = num_cores // x
    return (x, y)


def get_physical_to_logical_core_mapping(device):
    """
    Get a mapping from physical core coords to logical core coords

    Returns a dictionary.
    """
    mapping = {}
    grid = device.compute_with_storage_grid_size()
    for x in range(grid.x):
        for y in range(grid.y):
            physical_core = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
            mapping[(physical_core.x, physical_core.y)] = (x, y)
    return mapping


def round_up(a, b):
    """
    Round up a to the nearest multiple of b
    """
    return b * math.ceil(a / b)


# physical coords
PREFETCHER_GRID = [
    (8, 11),
    (8, 9),
    (8, 8),
    (8, 7),
    (8, 5),
    (8, 3),
    (8, 2),
    (8, 1),
    (7, 1),
    (7, 2),
    (7, 3),
    (7, 5),
    (7, 7),
    (7, 8),
    (7, 9),
    (7, 11),
    (3, 11),
    (3, 7),
    (3, 5),
    (3, 1),
    (2, 1),
    (2, 5),
    (2, 7),
    (2, 11),
]

# dram sharded MM output logical coords
PREFETCHER_NOC1_OUTPUT_GRID = [
    (1, 9),
    (2, 9),
    (1, 0),
    (2, 0),
    (1, 4),
    (2, 4),
    (1, 5),
    (2, 5),
    (5, 0),
    (6, 0),
    (5, 9),
    (6, 9),
    (5, 1),
    (6, 1),
    (5, 7),
    (6, 7),
    (5, 6),
    (6, 6),
    (5, 2),
    (6, 2),
    (5, 4),
    (6, 4),
    (5, 5),
    (6, 5),
]

LM_HEAD_32_GRID = list(
    itertools.chain(
        itertools.product([1, 2, 3], range(10)),  # Generates (1,0)-(1,9), (2,0)-(2,9), (3,0)-(3,9)
        itertools.product([5], range(2)),  # Generates (5,0), (5,1)
    )
)


def run_multi_core_matmul_1d(
    device,
    in0_dtype,
    in1_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    activation,
    grid,
    use_arbitrary_cores,
    num_iters,
    output_dtype=None,
    max_dst_tiles=8,
    pcc_threshold=0.98,
    use_physical_to_logical_mapping=True,
    hop_grid=None,
    in1_is_dram_interleaved=False,
    in1_is_in_dram=False,
    untilize_out=False,
):
    assert not has_bias, "Bias not supported for gather_in0 mode."
    if not isinstance(grid, tuple) and not use_arbitrary_cores:
        pytest.skip("Grid is not a tuple and not using arbitrary cores")

    if output_dtype is None:
        output_dtype = in0_dtype

    in0_shape = [1, B, M, K]
    in1_shape = [1, 1, K, N]
    num_cores = grid[0] * grid[1] if isinstance(grid, tuple) else len(grid)

    storage_grid = num_cores_to_rectangle_grid(num_cores, device)
    if storage_grid is None:
        pytest.skip(f"Could not find a rectangle grid for num_cores: {num_cores}")

    M *= B  # Fuse batch always enabled

    K_per_shard = round_up(math.ceil(K / num_cores), ttnn.TILE_SIZE)
    K_padded = K_per_shard * num_cores
    N_per_shard = round_up(math.ceil(N / num_cores), ttnn.TILE_SIZE)
    N_per_shard_in_dram = N_per_shard * 2
    N_padded = N_per_shard * num_cores

    logger.info(f"K_per_shard {K_per_shard}")
    logger.info(f"K_padded {K_padded}")
    logger.info(f"N_per_shard {N_per_shard}")
    logger.info(f"N_padded {N_padded}")

    in0_block_h = M // ttnn.TILE_SIZE
    in0_block_w = K // num_cores // ttnn.TILE_SIZE
    while (K / ttnn.TILE_SIZE) % in0_block_w != 0:
        in0_block_w -= 1

    out_block_h = M // ttnn.TILE_SIZE
    out_block_w = N_padded // num_cores // ttnn.TILE_SIZE

    num_blocks_y = (M // ttnn.TILE_SIZE - 1) // out_block_h + 1
    num_blocks_x = (N_padded // ttnn.TILE_SIZE - 1) // out_block_w + 1
    num_blocks_total = num_blocks_y * num_blocks_x

    if num_blocks_total != num_cores:
        pytest.skip(f"num_blocks_total {num_blocks_total} != num_cores {num_cores}")

    out_subblock_h = 1
    out_subblock_w = max_dst_tiles
    while out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1

    logger.debug("in0 block h w " + str(in0_block_h) + " " + str(in0_block_w))
    logger.debug("in1 block h w " + str(in0_block_w) + " " + str(out_block_w))
    logger.debug("out block h w " + str(out_block_h) + " " + str(out_block_w))
    logger.debug("out subblock h w " + str(out_subblock_h) + " " + str(out_subblock_w))

    if use_arbitrary_cores:
        # x, y
        if isinstance(grid, tuple):  # Generate random grid
            CORE_RANGE = [(x, y) for y in range(storage_grid[1]) for x in range(storage_grid[0])]
            random.shuffle(CORE_RANGE)
        else:  # Use custom grid
            if use_physical_to_logical_mapping:
                mapping = get_physical_to_logical_core_mapping(device)
                CORE_RANGE = [mapping[physical_coord] for physical_coord in grid]
            else:
                CORE_RANGE = grid

        core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in CORE_RANGE
            ]
        )
    else:
        core_range_set = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(storage_grid[0] - 1, storage_grid[1] - 1),
                ),
            }
        )

    hop_core_range_set = ttnn.CoreRangeSet([])
    if hop_grid is not None:
        hop_core_range_set = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(x, y),
                    ttnn.CoreCoord(x, y),
                )
                for x, y in hop_grid
            }
        )

    in0_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_set,
            [M, K_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    if in1_is_in_dram:
        if in1_is_dram_interleaved:
            in1_sharded_mem_config = ttnn.DRAM_MEMORY_CONFIG
        else:
            in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
            in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
            in1_sharded_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    in1_shard_grid,
                    [K, N_per_shard_in_dram],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
    else:
        in1_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                core_range_set,
                [K, N_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

    dram_sharded_output_core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(x, y),
                ttnn.CoreCoord(x, y),
            )
            for x, y in PREFETCHER_NOC1_OUTPUT_GRID
        ]
    )

    output_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            dram_sharded_output_core_range_set if (in1_is_in_dram and not in1_is_dram_interleaved) else core_range_set,
            [M, N_per_shard],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    in0 = torch.randn(in0_shape)
    in1 = torch.randn(in1_shape)

    in0_t = ttnn.from_torch(
        in0,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=in0_dtype,
        memory_config=in0_sharded_mem_config,
    )
    in1_t = ttnn.from_torch(
        in1,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=in1_dtype,
        memory_config=in1_sharded_mem_config,
    )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=storage_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=activation,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=hop_core_range_set,
        untilize_out=untilize_out,
    )

    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=True,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fidelity,
            math_approx_mode=True,
            fp32_dest_acc_en=fp32_acc_mode,
            packer_l1_acc=packer_l1_acc,
            dst_full_sync_en=True,
        )

    signpost("start")
    for _ in range(num_iters):
        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=output_sharded_mem_config,
            compute_kernel_config=compute_kernel_config,
            dtype=output_dtype,
        )
    signpost("stop")
    tt_out = ttnn.to_torch(output_t)
    pt_out = in0 @ in1

    if activation:
        act_fnc = torch.nn.functional.silu if activation == ttnn.UnaryOpType.SILU else torch.nn.functional.relu
        pt_out = act_fnc(pt_out)

    passing, output = comp_pcc(pt_out, tt_out, pcc_threshold)
    logger.info(output)

    assert passing

    # Check program cache
    assert device.num_program_cache_entries() == 1  # Only 1 op


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.skipif(is_blackhole(), reason="Test suite for GS only")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "B, M, K, N, in0_dtype, in1_dtype, fidelity, packer_l1_acc, fp32_acc_mode, grid",
    [
        (1, 32, 2048, 3584, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3)),
        (1, 32, 2048, 16 * 1024, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, False, False, (8, 4)),
        (1, 32, 7520, 8192, ttnn.bfloat8_b, ttnn.bfloat16, ttnn.MathFidelity.HiFi4, True, True, (6, 7)),
    ],
)
@pytest.mark.parametrize(
    "activation",
    [
        None,
    ],
)
@pytest.mark.parametrize(
    "use_arbitrary_cores, hop_grid",
    [
        (False, None),
        (False, [(3, 6)]),
        (True, None),
    ],
)
@pytest.mark.parametrize(
    "in1_is_dram_interleaved",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "num_iters",
    [
        3,
    ],
)
def test_multi_core_matmul_1d_in1_dram_wh(
    device,
    in0_dtype,
    in1_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    activation,
    grid,
    hop_grid,
    use_arbitrary_cores,
    in1_is_dram_interleaved,
    num_iters,
    function_level_defaults,
):
    run_multi_core_matmul_1d(
        device,
        in0_dtype,
        in1_dtype,
        fidelity,
        has_bias,
        fp32_acc_mode,
        packer_l1_acc,
        B,
        M,
        K,
        N,
        activation,
        grid,
        use_arbitrary_cores,
        num_iters,
        hop_grid=hop_grid,
        in1_is_dram_interleaved=in1_is_dram_interleaved,
        in1_is_in_dram=True,
    )


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.skipif(is_blackhole(), reason="Test suite for GS only")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "B, M, K, N, in0_dtype, in1_dtype, fidelity, packer_l1_acc, fp32_acc_mode, grid",
    [
        (1, 32, 2048, 1280, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, True, (8, 3)),
        (1, 32, 1280, 2048, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, True, (8, 3)),
        (1, 32, 2048, 3584, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, False, (8, 3)),
        (1, 32, 2048, 3584, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, False, (8, 3)),
        (1, 32, 3584, 2048, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, False, (8, 3)),
        (1, 32, 96, 64, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (2, 1)),
    ],
)
@pytest.mark.parametrize(
    "activation",
    [
        None,
    ],
)
@pytest.mark.parametrize(
    "use_arbitrary_cores, hop_grid",
    [
        (False, None),
        (False, [(3, 6)]),
    ],
)
@pytest.mark.parametrize(
    "num_iters",
    [
        1,
    ],
)
def test_multi_core_matmul_1d_pad_wh(
    device,
    in0_dtype,
    in1_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    activation,
    grid,
    hop_grid,
    use_arbitrary_cores,
    num_iters,
    function_level_defaults,
):
    run_multi_core_matmul_1d(
        device,
        in0_dtype,
        in1_dtype,
        fidelity,
        has_bias,
        fp32_acc_mode,
        packer_l1_acc,
        B,
        M,
        K,
        N,
        activation,
        grid,
        use_arbitrary_cores,
        num_iters,
        hop_grid=hop_grid,
    )


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.skipif(is_blackhole(), reason="Test suite for GS only")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "B, M, K, N, in0_dtype, in1_dtype, fidelity, packer_l1_acc, fp32_acc_mode, grid",
    [
        # # 32, 2304, 3840 (PREFETCHER), only works on TG
        # (1, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, PREFETCHER_GRID),
        # 32, 2304, 3840
        (1, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, False, False, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, False, True, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, True, False, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, False, False, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, True, False, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, False, True, (8, 3)),
        # 256, 1024, 8192
        (1, 256, 1024, 8192, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi4, True, True, (8, 4)),
        # 256, 1024, 8192
        (1, 256, 1024, 8192, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi4, True, True, (8, 4)),
        # # 128, 8192, 2048
        # (1, 128, 8192, 2048, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi2, True, True, (8, 8)),
        # # 128, 8192, 2048
        # (1, 128, 8192, 2048, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi2, True, False, (8, 8)),
        # # 128, 8192, 2048
        # (1, 128, 8192, 2048, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi2, False, True, (8, 8)), # Fails with 0.98 PCC
        # 32, 64, 64
        (1, 32, 64, 64, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, True, True, (2, 1)),
        # 32, 64, 64
        (11, 32, 64, 64, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, True, True, (2, 1)),
        (1, 32, 64, 64, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, True, True, (1, 1)),
        (1, 32, 64, 64, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, False, False, (1, 1)),
    ],
)
@pytest.mark.parametrize(
    "activation",
    [
        None,
        ttnn.UnaryOpType.SILU,
        ttnn.UnaryOpType.RELU,
    ],
)
@pytest.mark.parametrize(
    "use_arbitrary_cores",
    [False, True],
)
@pytest.mark.parametrize(
    "num_iters",
    [1, 3],
)
def test_multi_core_matmul_1d_wh(
    device,
    in0_dtype,
    in1_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    activation,
    grid,
    use_arbitrary_cores,
    num_iters,
    function_level_defaults,
):
    run_multi_core_matmul_1d(
        device,
        in0_dtype,
        in1_dtype,
        fidelity,
        has_bias,
        fp32_acc_mode,
        packer_l1_acc,
        B,
        M,
        K,
        N,
        activation,
        grid,
        use_arbitrary_cores,
        num_iters,
    )


@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32")
@pytest.mark.skipif(is_blackhole(), reason="Test suite for GS only")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "B, M, K, N, in0_dtype, in1_dtype, fidelity, packer_l1_acc, fp32_acc_mode, grid",
    [
        # Disabled for post-commit
        # (1, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, PREFETCHER_NOC1_GRID),
        (1, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, True, True, (8, 3)),
    ],
)
@pytest.mark.parametrize(
    "hop_grid",
    [
        # [
        #     (3, 6),
        # ],
        [
            (6, 3),
        ],
    ],
)
@pytest.mark.parametrize(
    "activation",
    [
        None,
        ttnn.UnaryOpType.SILU,
        ttnn.UnaryOpType.RELU,
    ],
)
@pytest.mark.parametrize(
    "use_arbitrary_cores",
    [False],
)
@pytest.mark.parametrize(
    "num_iters",
    [1, 3],
)
def test_multi_core_matmul_1d_ring_hop_wh(
    device,
    in0_dtype,
    in1_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    activation,
    grid,
    hop_grid,
    use_arbitrary_cores,
    num_iters,
    function_level_defaults,
):
    run_multi_core_matmul_1d(
        device,
        in0_dtype,
        in1_dtype,
        fidelity,
        has_bias,
        fp32_acc_mode,
        packer_l1_acc,
        B,
        M,
        K,
        N,
        activation,
        grid,
        use_arbitrary_cores,
        num_iters,
        use_physical_to_logical_mapping=False,
        hop_grid=hop_grid,
    )


@pytest.mark.skipif(is_wormhole_b0(), reason="Test suite for GS only")
@pytest.mark.skipif(is_blackhole(), reason="Test suite for GS only")
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "B, M, K, N, in0_dtype, in1_dtype, fidelity, packer_l1_acc, fp32_acc_mode, grid",
    [
        # 32, 2304, 3840
        (1, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.LoFi, False, False, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi, False, False, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2, False, False, (8, 3)),
        # 32, 2304, 3840
        (3, 32, 2304, 3840, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, False, False, (8, 3)),
        # 256, 1024, 8192
        (1, 256, 1024, 8192, ttnn.bfloat16, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi4, False, False, (8, 4)),
        # 128, 8192, 2048
        (1, 128, 4096, 2048, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.MathFidelity.HiFi2, False, False, (8, 8)),
        # 32, 64, 64
        (1, 32, 64, 64, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, False, False, (2, 1)),
        # 32, 64, 64
        (11, 32, 64, 64, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, False, False, (2, 1)),
        (1, 32, 64, 64, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi4, False, False, (1, 1)),
    ],
)
@pytest.mark.parametrize(
    "activation",
    [
        None,
        ttnn.UnaryOpType.SILU,
        ttnn.UnaryOpType.RELU,
    ],
)
@pytest.mark.parametrize(
    "use_arbitrary_cores",
    [False, True],
)
@pytest.mark.parametrize(
    "num_iters",
    [1, 3],
)
def test_multi_core_matmul_1d_gs(
    device,
    in0_dtype,
    in1_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    activation,
    grid,
    use_arbitrary_cores,
    num_iters,
    function_level_defaults,
):
    run_multi_core_matmul_1d(
        device,
        in0_dtype,
        in1_dtype,
        fidelity,
        has_bias,
        fp32_acc_mode,
        packer_l1_acc,
        B,
        M,
        K,
        N,
        activation,
        grid,
        use_arbitrary_cores,
        num_iters,
        pcc_threshold=0.96,
    )


@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "B, M, K, N, in0_dtype, in1_dtype, output_dtype, fidelity, packer_l1_acc, fp32_acc_mode, grid, in1_is_dram_interleaved, untilize_out",
    [
        (
            1,
            32,
            2048,
            1280,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.bfloat16,
            ttnn.MathFidelity.HiFi2,
            True,
            True,
            PREFETCHER_NOC1_GRID,
            False,
            True,
        ),
        (
            1,
            32,
            2048,
            1280,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
            True,
            True,
            PREFETCHER_NOC1_GRID,
            False,
            False,
        ),
        (
            1,
            32,
            1280,
            2048,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
            True,
            True,
            PREFETCHER_NOC1_GRID,
            False,
            False,
        ),
        (
            1,
            32,
            2048,
            3584,
            ttnn.bfloat16,
            ttnn.bfloat4_b,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.LoFi,
            True,
            False,
            PREFETCHER_NOC1_GRID,
            False,
            False,
        ),
        (
            1,
            32,
            3584,
            2048,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
            True,
            True,
            PREFETCHER_NOC1_GRID,
            False,
            False,
        ),
        (
            1,
            32,
            2048,
            16 * 1024,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
            True,
            True,
            LM_HEAD_32_GRID,
            True,
            False,
        ),
    ],
    ids=[
        "qkv_rm",
        "qkv",
        "do",
        "ff13",
        "ff2",
        "lm_head",
    ],
)
@pytest.mark.parametrize(
    "num_iters",
    [50],
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_matmul_1d_ring_llama_perf(
    device,
    in0_dtype,
    in1_dtype,
    output_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    grid,
    in1_is_dram_interleaved,
    untilize_out,
    num_iters,
    function_level_defaults,
):
    # Only run these tests on unharvested TG
    device_grid = (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Skipping test_run_prefetcher because it only works with a 7x10 grid")

    if in1_is_dram_interleaved:
        hop_grid = None
    else:
        hop_grid = [
            (3, 6),
        ]

    run_multi_core_matmul_1d(
        device,
        in0_dtype,
        in1_dtype,
        fidelity,
        has_bias,
        fp32_acc_mode,
        packer_l1_acc,
        B,
        M,
        K,
        N,
        None,  # activation,
        grid,
        True,
        num_iters,
        output_dtype=output_dtype,
        use_physical_to_logical_mapping=False,
        hop_grid=hop_grid,
        in1_is_dram_interleaved=in1_is_dram_interleaved,
        untilize_out=untilize_out,
    )


@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
@pytest.mark.parametrize(
    "B, M, K, N, in0_dtype, in1_dtype, output_dtype, fidelity, packer_l1_acc, fp32_acc_mode, grid, in1_is_dram_interleaved, in1_is_in_dram",
    [
        (
            1,
            32,
            2048,
            16 * 1024,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
            ttnn.MathFidelity.HiFi2,
            True,
            True,
            PREFETCHER_NOC1_GRID,
            False,
            True,
        ),
    ],
    ids=[
        "lm_head",
    ],
)
@pytest.mark.parametrize(
    "num_iters",
    [5],
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
def test_matmul_1d_ring_llama_lm_head(
    device,
    in0_dtype,
    in1_dtype,
    output_dtype,
    fidelity,
    has_bias,
    fp32_acc_mode,
    packer_l1_acc,
    B,
    M,
    K,
    N,
    grid,
    in1_is_dram_interleaved,
    in1_is_in_dram,
    num_iters,
    function_level_defaults,
):
    # Only run these tests on unharvested TG
    device_grid = (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y)
    if device_grid != (7, 10):
        pytest.skip("Skipping test_run_prefetcher because it only works with a 7x10 grid")

    if in1_is_dram_interleaved:
        hop_grid = None
    else:
        hop_grid = [
            (3, 6),
        ]

    run_multi_core_matmul_1d(
        device,
        in0_dtype,
        in1_dtype,
        fidelity,
        has_bias,
        fp32_acc_mode,
        packer_l1_acc,
        B,
        M,
        K,
        N,
        None,  # activation,
        grid,
        True,
        num_iters,
        output_dtype=output_dtype,
        use_physical_to_logical_mapping=False,
        hop_grid=hop_grid,
        in1_is_dram_interleaved=in1_is_dram_interleaved,
        in1_is_in_dram=in1_is_in_dram,
    )
