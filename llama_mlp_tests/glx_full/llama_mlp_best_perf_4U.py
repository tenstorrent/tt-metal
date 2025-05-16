import torch
import ttnn
from models.utility_functions import (
    comp_pcc,
)
import math
import pytest
import tracy

from loguru import logger

TILE_SIZE = 32
DRAM_WEIGHT_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 0))})

SEQ_LENS = [
    128,
    256,
    512,
    1024,
    2048,
    4096,
    6144,
    8192,
    10240,
    12288,
    14336,
    16384,
    24576,
    32768,
    51200,
    65536,
    86016,
    131072,
    "all",
]


def gen_w1_pcfg_new(height):
    if height < 4096:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 10),
            in0_block_w=8,
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max(
                1, 8 if height >= 2048 else height // TILE_SIZE // 8  # 8 rows
            ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=math.ceil(28672 / 8 / 32 / 7),  # N / TILE_WIDTH / grid width
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=height <= 2048,
        )

    if height == 4096:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 7),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=19,
            out_block_w=16,
            per_core_M=19,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    if height == 6144:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 7),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=14,
            out_block_w=16,
            per_core_M=28,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    if height == 8192:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 7),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=19,
            out_block_w=16,
            per_core_M=38,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

    if height == 10240:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 7),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=12,
            out_block_w=16,
            per_core_M=48,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    if height == 12288:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 7),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=14,
            out_block_w=16,
            per_core_M=56,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    if height == 14336:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 7),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=16,
            out_block_w=16,
            per_core_M=64,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    if height == 16384:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 7),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=15,
            out_block_w=16,
            per_core_M=75,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    if height == 24576:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 7),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=16,
            out_block_w=16,
            per_core_M=112,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    if height == 32768:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 7),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=15,
            out_block_w=16,
            per_core_M=150,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    if height == 51200:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 7),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=10,
            out_block_w=16,
            per_core_M=230,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    if height == 65536:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 7),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=15,
            out_block_w=16,
            per_core_M=300,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    if height == 86016:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 7),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=12,
            out_block_w=16,
            per_core_M=384,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
    if height == 131072:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 7),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=8,
            out_block_h=16,
            out_block_w=16,
            per_core_M=592,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )


def create_dram_sharded_mem_config(k, n):
    """Create DRAM-sharded memory config for width-sharded tensors"""
    dram_cores = 12
    padded_size = math.ceil(n / (TILE_SIZE * dram_cores)) * (TILE_SIZE * dram_cores)
    shard_spec = ttnn.ShardSpec(DRAM_WEIGHT_GRID, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


w1 = torch.randn((1, 1, 8192, 28672))


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    SEQ_LENS,
)
@pytest.mark.parametrize(
    "stress_test",
    [True, False],
)
###################################################################################
# ON 4U THE BEST PERF WE CAN GET (WITHOUT HANGING) IS ALREADY IMPLEMENTED IN MAIN #
###################################################################################
def test_w1(mesh_device, seq_len, stress_test):
    w1_tt_sharded = ttnn.from_torch(
        w1,
        device=mesh_device,
        dtype=ttnn.bfloat4_b,
        memory_config=create_dram_sharded_mem_config(2048, 3584),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, -2), mesh_shape=(8, 4)),
        layout=ttnn.TILE_LAYOUT,
    )
    w1_tt_interleaved = ttnn.from_torch(
        w1,
        device=mesh_device,
        dtype=ttnn.bfloat4_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, -2), mesh_shape=(8, 4)),
        layout=ttnn.TILE_LAYOUT,
    )

    def run_test(seq_len):
        activations = torch.randn((1, 1, seq_len, 2048))
        w1_tt = w1_tt_sharded if seq_len < 4096 else w1_tt_interleaved
        activations_tt = ttnn.from_torch(
            activations,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            layout=ttnn.TILE_LAYOUT,
        )
        iterations = 1000 if stress_test else 1
        logger.info("BEGINNING TEST")
        for i in range(iterations):
            logger.info(f"SEQ_LEN: {seq_len} | Running iteration {i}")
            out_tt = ttnn.linear(
                activations_tt,
                w1_tt,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                    dst_full_sync_en=True,
                ),
                program_config=gen_w1_pcfg_new(seq_len),
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    if seq_len == "all":
        for seq_len in SEQ_LENS[:-1]:
            logger.info(f"Testing W1: seq_len = {seq_len}")
            run_test(seq_len)
    else:
        logger.info(f"Testing W1: seq_len = {seq_len}")
        run_test(seq_len)


def gen_w2_pcfg_new(height):
    if height < 4096:
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(7, 10),
            in0_block_w=8,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max(1, 8 if height >= 2048 else height // TILE_SIZE // 8),  # 8~10 rows
            per_core_N=math.ceil(2048 / 32 / 7),  # N / TILE_WIDTH / grid width
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=height <= 2048,
        )

    # For very large activation heights (arbitrarily chosen to be > 320) we want the per_core_M to have many divisors
    # so that there are many options for out_block_h and out_block_w. Padding to the next multiple of 8 ensures that
    # per_core_M can at least be divisible by 2, 4, and 8 in addition to 1 and itself.
    #
    # If the number is less than or equal to 320 we still wouldn't want it to be prime so we'll add one if thats the case.
    next_multiple_of_8 = lambda x: int(x + (8 - x % 8) % 8)
    add_one_if_prime = lambda n: n + 1 if n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1)) else n
    total_per_core_out_M = add_one_if_prime(math.ceil(height / (7 * TILE_SIZE)))
    per_core_M = next_multiple_of_8(total_per_core_out_M) if total_per_core_out_M > 320 else total_per_core_out_M
    per_core_N = 10

    # Want out_block_h and out_block_w such that:
    # out_block_h * out block_w <= 320
    # out_block_h % per_core_M == 0
    # out_block_w % per_core_N == 0
    # Since we're fixing per_core_N = 10, out_block_w can only be 5 or 10

    def find_out_block_h(out_block_w):
        max_out_block_h = -1
        for i in range(1, per_core_M + 1):
            if i * out_block_w > 320:
                break
            if per_core_M % i == 0:
                if i > max_out_block_h:
                    max_out_block_h = i
        if max_out_block_h == -1:
            return None
        return max_out_block_h

    out_block_h_if_w_5 = find_out_block_h(5)
    out_block_h_if_w_10 = find_out_block_h(10)

    if out_block_h_if_w_5 is None and out_block_h_if_w_10 is None:
        assert False, "This should never happen"

    # Pick the configuration that exists if one of them does not
    if out_block_h_if_w_5 is None:
        out_block_w = 10
        out_block_h = out_block_h_if_w_10
    elif out_block_h_if_w_10 is None:
        out_block_w = 5
        out_block_h = out_block_h_if_w_5
    # If both exist, pick the one that is larger in volume
    elif out_block_h_if_w_5 * 5 > out_block_h_if_w_10 * 10:
        out_block_h = out_block_h_if_w_5
        out_block_w = 5
    elif out_block_h_if_w_10 * 10 > out_block_h_if_w_5 * 5:
        out_block_h = out_block_h_if_w_10
        out_block_w = 10
    # If neither exist pick the one which is more "square"
    else:
        # Want to use the out_block_h/w combination which is the most "square"
        # This calculates the height/width ratio of the blocks and then gets their
        # distance from 1 (1 is the ideal ratio) to determine which is more square
        squareness_5 = abs(1 - (max(out_block_h_if_w_5, 5) / min(out_block_h_if_w_5, 5)))
        squareness_10 = abs(1 - (max(out_block_h_if_w_10, 10) / min(out_block_h_if_w_10, 10)))

        if squareness_5 < squareness_10:
            out_block_w = 5
            out_block_h = out_block_h_if_w_5
        else:
            out_block_w = 10
            out_block_h = out_block_h_if_w_10

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(7, 7),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=5,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


w2 = torch.randn(28672, 8192)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    SEQ_LENS,
)
@pytest.mark.parametrize(
    "stress_test",
    [True],
)
def test_w2(mesh_device, seq_len, stress_test):
    w2_tt = ttnn.from_torch(
        w2,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=(8, 4)),
        layout=ttnn.TILE_LAYOUT,
    )

    def run_test(seq_len):
        activations = torch.randn(1, 1, seq_len, 3584)

        activations_tt = ttnn.from_torch(
            activations,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            layout=ttnn.TILE_LAYOUT,
        )

        logger.info(f"BEGINNING TEST")
        iterations = 1000 if stress_test else 1
        for i in range(iterations):
            logger.info(f"SEQ_LEN: {seq_len} | Running iteration {i+1}")
            out_tt = ttnn.linear(
                activations_tt,
                w2_tt,
                compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                    dst_full_sync_en=True,
                ),
                program_config=gen_w2_pcfg_new(seq_len),
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    if seq_len == "all":
        for seq_len in SEQ_LENS[:-1]:
            logger.info(f"Testing W2: seq_len = {seq_len}")
            run_test(seq_len)
    else:
        logger.info(f"Testing W2: seq_len = {seq_len}")
        run_test(seq_len)


# W1 STRESS TEST RESULTS:
# 128 - PASS
# 256 - PASS
# 512 - PASS
# 1024 - PASS
# 2048 - PASS
# 4096 - PASS
# 6144 - PASS
# 8192 - PASS
# 10240 - PASS
# 12288 - PASS
# 14336 - PASS
# 16384 - PASS
# 24576 - PASS
# 32768 - PASS
# 51200 - PASS
# 65536 - PASS
# 86016 - PASS
# 131072 - PASS

# W2 STRESS TEST RESULTS:
# 128 - PASS
# 256 - PASS
# 512 - PASS
# 1024 - PASS
# 2048 - PASS
# 4096 - PASS
# 6144 - PASS
# 8192 - PASS
# 10240 - PASS
# 12288 - PASS
# 14336 - PASS
# 16384 - PASS
# 24576 - PASS
# 32768 - PASS
# 51200 - PASS
# 65536 - PASS
# 86016 - PASS
# 131072 - PASS
