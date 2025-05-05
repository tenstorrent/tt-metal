import torch
import ttnn
from models.utility_functions import (
    comp_pcc,
)
import math
import pytest
import tracy

from tracy import signpost

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


def generate_w2_program_config(seq_len):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(7, 10),
        in0_block_w=8,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max(1, 8 if seq_len >= 2048 else seq_len // TILE_SIZE // 8),  # 8~10 rows
        per_core_N=math.ceil(2048 / 32 / 7),  # N / TILE_WIDTH / grid width
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=seq_len <= 2048,
    )


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
def test_w1_interleaved_current_pcfg(mesh_device):
    w1 = torch.randn(8192, 28672)

    w1_tt = ttnn.as_tensor(
        w1,
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, -2), mesh_shape=(8, 4)),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for seq_len in SEQ_LENS:
        activations = (
            torch.randn((1, seq_len // 1024, 1024, 2048)) if (seq_len >= 1024) else torch.randn((1, 1, seq_len, 2048))
        )
        signpost(f"Testing W1: seq_len = {seq_len}")
        activations_tt = ttnn.from_torch(
            activations,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
def test_w2_interleaved_current_pcfg(mesh_device):
    w2_torch = torch.randn(28672, 8192)
    w2_tt = ttnn.as_tensor(
        w2_torch,
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=(8, 4)),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for seq_len in SEQ_LENS:
        activations = (
            torch.randn((1, seq_len // 1024, 1024, 3584)) if (seq_len >= 1024) else torch.randn((1, 1, seq_len, 3584))
        )
        signpost(f"Testing W2: seq_len = {seq_len}")
        activations_tt = ttnn.from_torch(
            activations,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
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
            program_config=generate_w2_program_config(seq_len),
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


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
def test_w1_interleaved_auto_pcfg(mesh_device):
    w1 = torch.randn(8192, 28672)

    w1_tt = ttnn.as_tensor(
        w1,
        dtype=ttnn.bfloat4_b,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, -2), mesh_shape=(8, 4)),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for seq_len in SEQ_LENS:
        activations = (
            torch.randn((1, seq_len // 1024, 1024, 2048)) if (seq_len >= 1024) else torch.randn((1, 1, seq_len, 2048))
        )
        signpost(f"Testing W1: seq_len = {seq_len}")
        activations_tt = ttnn.from_torch(
            activations,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
            core_grid=ttnn.CoreGrid(x=7, y=7),
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


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
def test_w2_interleaved_auto_pcfg(mesh_device):
    w2_torch = torch.randn(28672, 8192)
    w2_tt = ttnn.as_tensor(
        w2_torch,
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=(8, 4)),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    for seq_len in SEQ_LENS:
        activations = (
            torch.randn((1, seq_len // 1024, 1024, 3584)) if (seq_len >= 1024) else torch.randn((1, 1, seq_len, 3584))
        )
        signpost(f"Testing W2: seq_len = {seq_len}")
        activations_tt = ttnn.from_torch(
            activations,
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
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
            core_grid=ttnn.CoreGrid(x=7, y=7),
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
