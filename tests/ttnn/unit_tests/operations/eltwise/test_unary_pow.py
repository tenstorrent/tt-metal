import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_allclose
from models.common.utility_functions import comp_ulp_check

from tests.ttnn.utils_for_testing import assert_with_pcc


def get_memory_config(memory_config_type, shape, device):
    """Get memory configuration based on type and shape"""
    if memory_config_type == "L1_interleaved":
        return ttnn.L1_MEMORY_CONFIG
    elif memory_config_type == "L1_block_sharded":
        # For (512, 512) - block sharding
        return ttnn.create_sharded_memory_config(
            shape,
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.BLOCK,
        )
    elif memory_config_type == "L1_height_sharded":
        # For (25600, 128) - height sharding
        # Use 50 cores so shard height = 25600/50 = 512 (tile-aligned)
        core_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),  # 8x6 = 48 cores
                ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(1, 6)),  # 2x1 = 2 cores
            }
        )  # Total: 50 cores
        return ttnn.create_sharded_memory_config(
            (512, 128),  # Explicit shard shape: 512 is tile-aligned (512 % 32 = 0)
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
    elif memory_config_type == "height_sharded":
        # For YOLOv4 shape [1, 16, 320, 320] -> 5120 x 320
        # YOLOv4 uses ROW_MAJOR_LAYOUT, but we use TILE_LAYOUT, so need tile alignment
        # Use 40 cores so shard height = 5120/40 = 128 (tile-aligned)
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4))})  # 8x5 = 40 cores
        return ttnn.create_sharded_memory_config(
            (128, 320),  # Explicit shard shape: 128 is tile-aligned (128 % 32 = 0)
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
    else:  # DRAM_interleaved
        return ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "input_shapes, memory_config_type",
    [
        (torch.Size([32, 32]), "L1_interleaved"),  # Single tile - L1 Interleaved
        (torch.Size([64, 128]), "L1_interleaved"),  # 8 tile - L1 Interleaved
        (torch.Size([512, 512]), "L1_block_sharded"),  # L1 Block sharded
        (torch.Size([25600, 128]), "L1_height_sharded"),  # L1 Height sharded
        (torch.Size([1, 16, 320, 320]), "height_sharded"),  # Original YOLOv4 shape
    ],
)
@pytest.mark.parametrize(
    "activation_func, func_name, params, has_approx",
    [
        # ReLU family - NO approx mode
        (ttnn.pow, "pow", {}, True),
        # (ttnn.prelu, "prelu", {"weight": 0.25}, False),
    ],
)
@pytest.mark.parametrize("exponent", [0.0, 1.0, 2.0, 3.56, 3.0, 0.65])
def test_activation_functions(
    input_shapes, exponent, memory_config_type, activation_func, func_name, params, has_approx, device
):
    """Test activation functions"""
    torch.manual_seed(0)

    in_data = torch.empty(input_shapes, dtype=torch.bfloat16).uniform_(-10, 10)

    # Get appropriate memory configuration
    memory_config = get_memory_config(memory_config_type, list(input_shapes), device)

    input_tensor = ttnn.from_torch(
        in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )

    output_tensor = ttnn.pow(input_tensor, exponent)
    # Lower PCC threshold for approximate modes
    pcc_threshold = 0.95

    # Verify correctness
    golden_function = ttnn.get_golden_function(activation_func)
    golden_tensor = golden_function(in_data, exponent, device=device)
    assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=pcc_threshold)

    # print(f"{func_name}: FAILED - {e}")
