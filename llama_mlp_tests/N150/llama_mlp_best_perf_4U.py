import torch
import ttnn
from models.utility_functions import (
    comp_pcc,
)
import math
import pytest
import tracy

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
]


def generate_w1_w3_program_config(seq_len):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(7, 10),
        in0_block_w=8,
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max(
            1, 8 if seq_len >= 2048 else seq_len // TILE_SIZE // 8  # 8 rows
        ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=math.ceil(28672 / 8 / 32 / 7),  # N / TILE_WIDTH / grid width
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=seq_len <= 2048,
    )


def create_dram_sharded_mem_config(k, n):
    """Create DRAM-sharded memory config for width-sharded tensors"""
    dram_cores = 12
    padded_size = math.ceil(n / (TILE_SIZE * dram_cores)) * (TILE_SIZE * dram_cores)
    shard_spec = ttnn.ShardSpec(DRAM_WEIGHT_GRID, (k, padded_size // dram_cores), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    SEQ_LENS,
)
###################################################################################
# ON 4U THE BEST PERF WE CAN GET (WITHOUT HANGING) IS ALREADY IMPLEMENTED IN MAIN #
###################################################################################
def test_w1(device, seq_len):
    activations = (
        torch.randn((1, seq_len // 1024, 1024, 2048)) if (seq_len >= 1024) else torch.randn((1, 1, seq_len, 2048))
    )
    w1 = torch.randn((1, 1, 2048, 3584))

    golden = activations @ w1.squeeze()

    activations_tt = ttnn.from_torch(
        activations,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    w1_tt = ttnn.from_torch(
        w1,
        device=device,
        dtype=ttnn.bfloat4_b,
        memory_config=create_dram_sharded_mem_config(2048, 3584),
        layout=ttnn.TILE_LAYOUT,
    )

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
        program_config=generate_w1_w3_program_config(seq_len),
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_torch = ttnn.to_torch(out_tt)
    passed, msg = comp_pcc(golden, out_torch, 0.99)
    assert passed, msg


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

    per_core_M = math.ceil(height / 224)
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


def largest_divisor_less_than_y(x, y):
    """
    Find the largest divisor of X that is less than or equal to Y using a more efficient algorithm.

    Args:
        x (int): The number to find divisors for
        y (int): The upper bound for the divisor

    Returns:
        int: The largest divisor of X that is <= Y
    """
    # Handle edge cases
    if x <= 0 or y <= 0:
        raise ValueError("Both X and Y must be positive integers")

    # Special cases
    if y >= x:
        return x  # x is a divisor of itself and is <= y

    # Collect all divisors of x
    divisors = []
    i = 1

    # Only need to check up to square root of x
    while i * i <= x:
        if x % i == 0:
            divisors.append(i)
            # Don't add the same divisor twice
            if i != x // i:
                divisors.append(x // i)
        i += 1

    # Sort divisors in descending order
    divisors.sort(reverse=True)

    # Find the largest divisor that is <= y
    for d in divisors:
        if d <= y:
            return d

    # This should never be reached if inputs are positive
    return 1


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    SEQ_LENS,
)
def test_w2(device, seq_len):
    activations = torch.randn(
        (
            1,
            1,
            seq_len // largest_divisor_less_than_y(seq_len, 16384),
            largest_divisor_less_than_y(seq_len, 16384),
            3584,
        )
    )

    w2 = torch.randn((1, 1, 3584, 2048))

    golden = activations @ w2.squeeze()

    activations_tt = ttnn.from_torch(
        activations,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    w2_tt = ttnn.from_torch(
        w2,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

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
        program_config=gen_w2_pcfg_new(activations_tt.shape[-2]),
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_torch = ttnn.to_torch(out_tt)
    passed, msg = comp_pcc(golden, out_torch, 0.99)
    assert passed, msg
