import torch
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc, _normalize_tensor
from models.common.utility_functions import comp_pcc


def create_tiled_tensor_vectorized(shape, tile_size=32, dtype=torch.bfloat16):
    B, C, H, W = shape
    tiles_h = H // tile_size
    tiles_w = W // tile_size
    tile_indices = torch.arange(1, tiles_h * tiles_w + 1, dtype=dtype).view(tiles_h, tiles_w)
    tensor = tile_indices.repeat_interleave(tile_size, dim=0).repeat_interleave(tile_size, dim=1)
    tensor = tensor.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1).contiguous()
    return tensor * 0.1


def create_batched_tiled_tensor_column_first_vectorized(
    shape, tile_size=32, start_val=0.1, increment=0.1, dtype=torch.bfloat16
):
    """
    Vectorized version with column-first tile ordering
    """
    B, C, H, W = shape
    tiles_h = H // tile_size  # 16 tiles
    tiles_w = W // tile_size  # 4 tiles
    tiles_per_batch = tiles_h * tiles_w  # 64 tiles per batch

    tile_row_coords = torch.arange(tiles_h, dtype=dtype).unsqueeze(1).expand(tiles_h, tiles_w)
    tile_col_coords = torch.arange(tiles_w, dtype=dtype).unsqueeze(0).expand(tiles_h, tiles_w)

    base_indices = tile_col_coords * tiles_h + tile_row_coords
    base_pattern = base_indices.repeat_interleave(tile_size, dim=0).repeat_interleave(tile_size, dim=1)
    batch_offsets = torch.arange(C, dtype=dtype) * tiles_per_batch
    tile_indices = base_pattern.unsqueeze(0).unsqueeze(0).expand(1, C, -1, -1) + batch_offsets.view(1, C, 1, 1)
    tensor = start_val + tile_indices * increment
    tensor = tensor.expand(B, -1, -1, -1).contiguous()

    return tensor


def test_matmul_bfp8_blackhole():
    SEQ_LEN = 3200
    num_heads = 32
    hidden_dim = 512
    output_dim = 128

    input_shape_a = [1, 1, SEQ_LEN, hidden_dim]  # [1, 1, 3200, 512]
    input_shape_b = [1, num_heads, hidden_dim, output_dim]  # [1, 32, 512, 128]

    logger.info(f"Testing matmul with shapes: {input_shape_a} @ {input_shape_b}")
    logger.info(f"SEQ_LEN = {SEQ_LEN}")

    device = ttnn.open_device(device_id=0)
    actual_grid = device.compute_with_storage_grid_size()

    prog_config_mm4_bh = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(actual_grid.x, actual_grid.y),
        in0_block_w=8,
        out_subblock_h=2,
        out_subblock_w=4,
        per_core_M=2,
        per_core_N=4,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=False,
    )

    input_mem_config = ttnn.DRAM_MEMORY_CONFIG
    output_mem_config = ttnn.DRAM_MEMORY_CONFIG

    actual_grid = device.compute_with_storage_grid_size()
    logger.info(f"Device compute grid: {actual_grid.x}x{actual_grid.y}")

    try:
        # torch_input_a = torch.full(input_shape_a, 2.0, dtype=torch.bfloat16)
        # torch_input_a = create_tiled_tensor_vectorized(input_shape_a, tile_size=32, dtype=torch.bfloat16)
        torch_input_a = torch.randn(input_shape_a, dtype=torch.bfloat16)

        # torch_input_b = create_batched_tiled_tensor_column_first_vectorized(input_shape_b, 32, dtype=torch.bfloat16)
        torch_input_b = torch.randn(input_shape_b, dtype=torch.bfloat16)
        # torch_input_b = torch.full(input_shape_b, 3.0, dtype=torch.bfloat16)

        # print("------------------------------------------------------------------------------------")
        corex = 0
        corey = 0
        core_index = corey * actual_grid.x + corex

        # print(f"first 5 numbers of first 2 tiles for core {corex},{corey}")
        # print(f"in0[0,0,{2*32*core_index},0:5] = {torch_input_a[0,0,2*32*core_index,0:5]}")
        # print(f"in0[0,0,{2*32*core_index},32:37] = {torch_input_a[0,0,2*32*core_index,32:37]}")
        # print("in0[0,0,0,0:5] = ", torch_input_a[0,0,0,0:5])
        # print("in0[0,0,0,32:37] = ", torch_input_a[0,0,0,32:37])
        # print("in0[0,0,32,0:5] = ", torch_input_a[0,0,32,0:5])
        # print("in0[0,0,32,32:37] = ", torch_input_a[0,0,32,32:37])
        # print("in0[0,0,64,0:5] = ", torch_input_a[0,0,64,0:5])
        # print("in0[0,0,64,32:37] = ", torch_input_a[0,0,64,32:37])
        # print("")

        # for i in range(0,32):
        #     print(f"first 5 numbers of first 2 tiles for batch {i} for core {corex},{corey}")
        #     print(f"in1[0,{i},0,0:5] = {torch_input_b[0,i,0,0:5]}")
        #     print(f"in1[0,{i},0,32:37] = {torch_input_b[0,i,0,32:37]}")
        # print("in1[0,0,0,0:5] = ", torch_input_b[0,0,0,0:5])
        # print("in1[0,0,0,32:37] = ", torch_input_b[0,0,0,32:37])
        # print("in1[0,0,32,0:5] = ", torch_input_b[0,0,32,0:5])
        # print("in1[0,0,32,32:37] = ", torch_input_b[0,0,32,32:37])
        # print("------------------------------------------------------------------------------------")

        torch_output = torch.matmul(torch_input_a.repeat(1, 32, 1, 1), torch_input_b)

        tt_input_a = ttnn.from_torch(
            torch_input_a,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=input_mem_config,
        )

        tt_input_b = ttnn.from_torch(
            torch_input_b,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=input_mem_config,
        )

        logger.info("Performing matmul operation...")

        tt_output = ttnn.matmul(
            tt_input_a,
            tt_input_b,
            memory_config=output_mem_config,
            dtype=ttnn.bfloat8_b,
            program_config=prog_config_mm4_bh,
        )

        tt_output_torch = ttnn.to_torch(tt_output)
        print("------------------------------------------------------------------------------------")
        print(f"TENSTORRENT OUTPUT.shape = {tt_output_torch.shape}")
        print(f"TORCH       OUTPUT.shape = {torch_output.shape}")
        print("")

        print(f"TENSTORRENT OUTPUT BATCH 0[0:5] = {tt_output_torch[0,0,0,0:5]}")
        print(f"TORCH       OUTPUT BATCH 0[0:5] = {torch_output[0,0,0,0:5]}")
        print("")

        print(f"TENSTORRENT OUTPUT BATCH 1[0:5] = {tt_output_torch[0,1,0,0:5]}")
        print(f"TORCH       OUTPUT BATCH 1[0:5] = {torch_output[0,1,0,0:5]}")
        print("")

        print(f"TENSTORRENT OUTPUT BATCH 2[0:5] = {tt_output_torch[0,2,0,0:5]}")
        print(f"TORCH       OUTPUT BATCH 2[0:5] = {torch_output[0,2,0,0:5]}")
        print("")

        print(f"TENSTORRENT OUTPUT BATCH 31[0:5] = {tt_output_torch[0,31,0,0:5]}")
        print(f"TORCH       OUTPUT BATCH 31[0:5] = {torch_output[0,31,0,0:5]}")
        print("")
        print("------------------------------------------------------------------------------------")

        logger.info("starting per batch PCC check...")
        for b in range(0, 31):
            pcc = comp_pcc(_normalize_tensor(tt_output_torch[0, b, :, :]), _normalize_tensor(torch_output[0, b, :, :]))
            logger.info(f"PCC for batch {b}: {pcc}")

        logger.info("Verifying correctness with PCC...")
        passed, pcc = assert_with_pcc(tt_output_torch, torch_output, pcc=0.95)  # Lower PCC due to bfp8

        logger.info(f"PCC: {pcc}")
        if passed:
            logger.info("✓ Test PASSED!")
        else:
            logger.error("✗ Test FAILED!")

        logger.info("Program config applied successfully")
        logger.info("Kernels selected based on the configuration:")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_matmul_bfp8_blackhole()
