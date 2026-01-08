import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal

from models.common.utility_functions import comp_ulp, comp_pcc


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_mm1(device):
    torch.manual_seed(0)
    # H = 512  # passes when no program configs are specified (and things are not the same)
    H = 2048  # fails when no program configs are specified (and things are not the same)
    W = 1280
    input_shape = [128, H]
    input_shape2 = [H, W]
    N = 32  # concat 32 times
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_input_tensor2 = torch.rand(input_shape2, dtype=torch.bfloat16)
    concat_input = [torch_input_tensor for i in range(N)]
    torch_batch_input_tensor = torch.stack(concat_input)
    # dtype = ttnn.bfloat16
    dtype = ttnn.bfloat8_b

    # MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(x=4;y=10);in0_block_w=16;out_subblock_h=4;out_subblock_w=1;out_block_h=4;out_block_w=1;per_core_M=4;per_core_N=1;fuse_batch=1;fused_activation=std::nullopt;mcast_in0=1;gather_in0=0;hop_cores={};num_global_cb_receivers=1;untilize_out=0)
    # fill in the values for the program config
    non_batched_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 10),
        in0_block_w=16,
        out_subblock_h=4,
        out_subblock_w=1,
        out_block_h=4,
        out_block_w=1,
        per_core_M=4,
        per_core_N=1,
        fuse_batch=True,
        mcast_in0=True,
    )

    # MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(x=7;y=10);in0_block_w=8;out_subblock_h=1;out_subblock_w=2;out_block_h=8;out_block_w=6;per_core_M=8;per_core_N=6;transpose_mcast=0;fused_activation=std::nullopt;fuse_batch=0)
    # fill in the values for the program config
    batched_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(7, 10),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=2,
        out_block_h=4,
        out_block_w=6,
        per_core_M=4,
        per_core_N=6,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    batch_input_tensor = ttnn.from_torch(torch_batch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.matmul(input_tensor, input_tensor2, program_config=non_batched_program_config)
    batch_output_tensor = ttnn.matmul(batch_input_tensor, input_tensor2, program_config=batched_program_config)
    logger.info(f"OT {output_tensor}\nBOT {batch_output_tensor}")
    tt_output_tensor = ttnn.to_torch(output_tensor)
    tt_batch_output_tensor = ttnn.to_torch(batch_output_tensor)
    logger.info(f"TOT {tt_output_tensor}\nTBOT {tt_batch_output_tensor}")
    for i in range(N):
        chunked_tensor = tt_batch_output_tensor[i]
        print(f"I {i} TOT {tt_output_tensor}\nCT {chunked_tensor}")
        print(comp_ulp(tt_output_tensor, chunked_tensor, 0))
        assert_equal(tt_output_tensor, chunked_tensor)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_mm2(device):
    torch.manual_seed(0)
    # H = 512  # passes when no program configs are specified (and things are not the same)
    H = 1024  # fails when no program configs are specified (and things are not the same)
    W = 2048
    input_shape = [128, H]
    input_shape2 = [H, W]
    N = 32  # concat 32 times
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_input_tensor2 = torch.rand(input_shape2, dtype=torch.bfloat16)
    concat_input = [torch_input_tensor for i in range(N)]
    torch_batch_input_tensor = torch.stack(concat_input)
    # dtype = ttnn.bfloat16
    dtype = ttnn.bfloat8_b

    # MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(x=7;y=9);in0_block_w=16;out_subblock_h=4;out_subblock_w=2;out_block_h=4;out_block_w=2;per_core_M=4;per_core_N=2;fuse_batch=1;fused_activation=std::nullopt;mcast_in0=1;gather_in0=0;hop_cores={};num_global_cb_receivers=1;untilize_out=0)
    # fill in the values for the program config
    non_batched_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(7, 9),
        in0_block_w=8,
        out_subblock_h=4,
        out_subblock_w=2,
        out_block_h=4,
        out_block_w=2,
        per_core_M=4,
        per_core_N=2,
        fuse_batch=True,
        mcast_in0=True,
    )

    # MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(x=7;y=10);in0_block_w=8;out_subblock_h=1;out_subblock_w=2;out_block_h=4;out_block_w=10;per_core_M=4;per_core_N=10;transpose_mcast=0;fused_activation=std::nullopt;fuse_batch=0)
    # fill in the values for the program config
    batched_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(7, 10),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=2,
        out_block_h=4,
        out_block_w=10,
        per_core_M=4,
        per_core_N=10,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    batch_input_tensor = ttnn.from_torch(torch_batch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.matmul(input_tensor, input_tensor2, program_config=non_batched_program_config)
    logger.info(f"non batched output tensor shape: {output_tensor.shape}")
    batch_output_tensor = ttnn.matmul(batch_input_tensor, input_tensor2, program_config=batched_program_config)
    # print(f"OT {output_tensor}\nBOT {batch_output_tensor}")
    logger.info(f"batched output tensor shape: {batch_output_tensor.shape}")
    tt_output_tensor = ttnn.to_torch(output_tensor)
    tt_batch_output_tensor = ttnn.to_torch(batch_output_tensor)
    # print(f"TOT {tt_output_tensor}\nTBOT {tt_batch_output_tensor}")
    for i in range(N):
        chunked_tensor = tt_batch_output_tensor[i]
        print(f"I {i} TOT {tt_output_tensor}\nCT {chunked_tensor}")
        print(comp_ulp(tt_output_tensor, chunked_tensor, 0))
        assert_equal(tt_output_tensor, chunked_tensor)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_mm3(device):
    torch.manual_seed(0)
    # H = 512  # passes when no program configs are specified (and things are not the same)
    H = 2048  # fails when no program configs are specified (and things are not the same)
    W = 3584
    input_shape = [128, H]
    input_shape2 = [H, W]
    N = 32  # concat 32 times
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_input_tensor2 = torch.rand(input_shape2, dtype=torch.bfloat16)
    concat_input = [torch_input_tensor for i in range(N)]
    torch_batch_input_tensor = torch.stack(concat_input)
    # dtype = ttnn.bfloat16
    input_dtype = ttnn.bfloat8_b
    weight_dtype = ttnn.bfloat4_b

    # MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=(x=7;y=4);in0_block_w=16;out_subblock_h=2;out_subblock_w=4;out_block_h=4;out_block_w=4;per_core_M=4;per_core_N=4;fuse_batch=1;fused_activation=std::nullopt;mcast_in0=1;gather_in0=0;hop_cores={};num_global_cb_receivers=1;untilize_out=0)
    # fill in the values for the program config
    non_batched_program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(7, 4),
        in0_block_w=16,
        out_subblock_h=2,
        out_subblock_w=4,
        out_block_h=4,
        out_block_w=4,
        per_core_M=4,
        per_core_N=4,
        fuse_batch=True,
        mcast_in0=True,
    )

    # MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(x=7;y=7);in0_block_w=4;out_subblock_h=1;out_subblock_w=8;out_block_h=10;out_block_w=16;per_core_M=20;per_core_N=16;transpose_mcast=0;fused_activation=std::nullopt;fuse_batch=0)
    # fill in the values for the program config
    batched_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(7, 7),
        in0_block_w=16,
        out_subblock_h=1,
        out_subblock_w=8,
        out_block_h=10,
        out_block_w=16,
        per_core_M=20,
        per_core_N=16,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    batch_input_tensor = ttnn.from_torch(
        torch_batch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.matmul(input_tensor, input_tensor2, program_config=non_batched_program_config)
    logger.info(f"non batched output tensor shape: {output_tensor.shape}")
    batch_output_tensor = ttnn.matmul(batch_input_tensor, input_tensor2, program_config=batched_program_config)
    # print(f"OT {output_tensor}\nBOT {batch_output_tensor}")
    logger.info(f"batched output tensor shape: {batch_output_tensor.shape}")
    tt_output_tensor = ttnn.to_torch(output_tensor)
    tt_batch_output_tensor = ttnn.to_torch(batch_output_tensor)
    # print(f"TOT {tt_output_tensor}\nTBOT {tt_batch_output_tensor}")
    for i in range(N):
        chunked_tensor = tt_batch_output_tensor[i]
        print(f"I {i} TOT {tt_output_tensor}\nCT {chunked_tensor}")
        print(comp_ulp(tt_output_tensor, chunked_tensor, 0))
        assert_equal(tt_output_tensor, chunked_tensor)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_mm4(device):
    torch.manual_seed(0)
    # H = 512  # passes when no program configs are specified (and things are not the same)
    H = 3584  # fails when no program configs are specified (and things are not the same)
    W = 2048
    input_shape = [128, H]
    input_shape2 = [H, W]
    N = 32  # concat 32 times
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_input_tensor2 = torch.rand(input_shape2, dtype=torch.bfloat16)
    concat_input = [torch_input_tensor for i in range(N)]
    torch_batch_input_tensor = torch.cat(concat_input)
    # dtype = ttnn.bfloat16
    input_dtype = ttnn.bfloat8_b
    weight_dtype = ttnn.bfloat8_b

    # MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(x=7;y=10);in0_block_w=8;out_subblock_h=1;out_subblock_w=2;out_block_h=1;out_block_w=10;per_core_M=1;per_core_N=10;transpose_mcast=0;fused_activation=std::nullopt;fuse_batch=1)
    # Fill in the values for the program config
    non_batched_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(7, 10),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=2,
        out_block_h=1,
        out_block_w=10,
        per_core_M=1,
        per_core_N=10,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )

    # MatmulMultiCoreReuseMultiCastProgramConfig(compute_with_storage_grid_size=(x=7;y=10);in0_block_w=8;out_subblock_h=1;out_subblock_w=2;out_block_h=16;out_block_w=10;per_core_M=16;per_core_N=10;transpose_mcast=0;fused_activation=std::nullopt;fuse_batch=0)
    # Fill in the values for the program config
    batched_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(7, 10),
        in0_block_w=8,
        out_subblock_h=1,
        out_subblock_w=2,
        out_block_h=16,
        out_block_w=10,
        per_core_M=16,
        per_core_N=10,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    batch_input_tensor = ttnn.from_torch(
        torch_batch_input_tensor, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.matmul(input_tensor, input_tensor2, program_config=non_batched_program_config)
    logger.info(f"non batched output tensor shape: {output_tensor.shape}")
    batch_output_tensor = ttnn.matmul(batch_input_tensor, input_tensor2, program_config=batched_program_config)
    # print(f"OT {output_tensor}\nBOT {batch_output_tensor}")
    logger.info(f"batched output tensor shape: {batch_output_tensor.shape}")
    tt_output_tensor = ttnn.to_torch(output_tensor)
    tt_batch_output_tensor = ttnn.to_torch(batch_output_tensor)
    chunked_tensors = tt_batch_output_tensor.chunk(N)
    # print(f"TOT {tt_output_tensor}\nTBOT {tt_batch_output_tensor}")
    # print(f"TOT {tt_output_tensor}\nTBOT {tt_batch_output_tensor}")
    for i in range(N):
        chunked_tensor = chunked_tensors[i]
        print(f"I {i} TOT {tt_output_tensor}\nCT {chunked_tensor}")
        print(comp_ulp(tt_output_tensor, chunked_tensor, 0))
        assert_equal(tt_output_tensor, chunked_tensor)
