import pytest
import math
import torch
import ttnn
from models.utility_functions import skip_for_grayskull


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_w1_mm(mesh_device):
    seq_len = 2048

    dram_cores = 12
    padded_size = math.ceil(3840 / (32 * dram_cores)) * (32 * dram_cores)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(mesh_device.dram_grid_size().x - 1, mesh_device.dram_grid_size().y - 1),
                )
            }
        ),
        (1280, 3840 // dram_cores),
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    pc_1 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(7, 10),
        in0_block_w=1,
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
        per_core_M=max(
            1, 8 if seq_len >= 2048 else seq_len // 32 // 8  # 8 rows
        ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
        per_core_N=math.ceil(25600 / 8 / 32 / 7),  # N / TILE_WIDTH / grid width
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=seq_len <= 2048,
    )

    w1_torch = (2 * torch.randn(1, 1, 5120, 25600)) + 1
    w1 = ttnn.from_torch(
        w1_torch,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, -2), mesh_shape=(8, 4)),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec),
        dtype=ttnn.bfloat8_b,
    )

    torch_input = torch.randn(1, 1, seq_len, 5120)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 3),
            mesh_shape=(8, 4),
        ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_input = ttnn.reshape(tt_input, (1, seq_len // 1024, 1024, -1))

    w1_out = ttnn.linear(
        tt_input,
        w1,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        ),
        dtype=ttnn.bfloat8_b,
        program_config=pc_1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ref_torch = torch_input @ w1_torch
    tt_out_torch = ttnn.to_torch(w1_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device=mesh_device, dim=3))
    breakpoint()
