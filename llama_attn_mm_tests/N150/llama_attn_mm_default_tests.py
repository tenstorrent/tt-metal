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


def generate_kqv_projection_program_config(seq_len):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(7, 10),
        in0_block_w=8,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max(
            1, 8 if seq_len >= 2048 else seq_len // TILE_SIZE // 8  # 8 rows
        ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=math.ceil(1280 / 32 / 7),  # N / TILE_WIDTH / grid width
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=seq_len <= 2048,
    )


def generate_wo_program_config(seq_len):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(7, 10),
        in0_block_w=8,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max(1, 4 if seq_len >= 1024 else seq_len // TILE_SIZE // 8),  # 8~10 rows
        per_core_N=math.ceil(2048 / 32 / 7),  # N / TILE_WIDTH / grid width
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=seq_len <= 1024,
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
def test_kqv_projection(device, seq_len):
    activations = (
        torch.randn((1, seq_len // 2048, 2048, 2048)) if (seq_len >= 2048) else torch.randn((1, 1, seq_len, 2048))
    )
    wkqv = torch.randn((1, 1, 2048, 1280))

    golden = activations @ wkqv.squeeze()

    activations_tt = ttnn.from_torch(
        activations,
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    wkqv_tt = ttnn.from_torch(
        wkqv,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=create_dram_sharded_mem_config(8192 // 4, 12288 // 8),
        layout=ttnn.TILE_LAYOUT,
    )

    out_tt = ttnn.linear(
        activations_tt,
        wkqv_tt,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        ),
        program_config=generate_kqv_projection_program_config(seq_len),
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_torch = ttnn.to_torch(out_tt)
    passed, msg = comp_pcc(golden, out_torch, 0.99)
    assert passed, msg


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    SEQ_LENS,
)
def test_wo(device, seq_len):
    activations = (
        torch.randn((1, 1, seq_len // 1024, 1024)) if (seq_len >= 1024) else torch.randn((1, 1, seq_len, 1024))
    )
    wo = torch.randn((1, 1, 1024, 2048))

    golden = activations @ wo.squeeze()

    activations_tt = ttnn.from_torch(
        activations,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    wo_tt = ttnn.from_torch(
        wo,
        device=device,
        dtype=ttnn.bfloat8_b,
        memory_config=create_dram_sharded_mem_config(8192 // 8, 9216 // 4),
        layout=ttnn.TILE_LAYOUT,
    )

    out_tt = ttnn.linear(
        activations_tt,
        wo_tt,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            dst_full_sync_en=True,
        ),
        program_config=generate_wo_program_config(seq_len),
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_torch = ttnn.to_torch(out_tt)
    passed, msg = comp_pcc(golden, out_torch, 0.99)
    assert passed, msg
