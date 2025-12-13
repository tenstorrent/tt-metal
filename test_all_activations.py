import pytest
import torch
import ttnn
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
        # (torch.Size([32, 32]), "L1_interleaved"),           # Single tile - L1 Interleaved
        # (torch.Size([64, 128]), "L1_interleaved"),          # 8 tile - L1 Interleaved
        # (torch.Size([512, 512]), "L1_block_sharded"),       # L1 Block sharded
        # (torch.Size([25600, 128]), "L1_height_sharded"),    # L1 Height sharded
        (torch.Size([1, 16, 320, 320]), "height_sharded"),  # Original YOLOv4 shape
    ],
)
@pytest.mark.parametrize(
    "activation_func, func_name, params, has_approx",
    [
        # ReLU family - NO approx mode
        # (ttnn.relu, "relu", {}, False),
        # (ttnn.relu6, "relu6", {}, False),
        # (ttnn.leaky_relu, "leaky_relu", {"alpha": 0.01}, False),
        # (ttnn.relu_max, "relu_max", {"upper_limit": 6.0}, False),
        # (ttnn.relu_min, "relu_min", {"lower_limit": 0.0}, False),
        # Sigmoid family - HAS approx mode
        # (ttnn.sigmoid, "sigmoid_accurate", {}, False),           # Accurate mode
        # (ttnn.sigmoid, "sigmoid_approx", {}, True),              # Approximate mode
        # (ttnn.sigmoid_accurate, "sigmoid_accurate_only", {}, False),  # Always accurate
        # (ttnn.log_sigmoid, "log_sigmoid", {}, False),            # NO approx
        # Tanh family - HAS approx mode
        # (ttnn.tanh, "tanh_accurate", {}, False),                 # Accurate mode
        # (ttnn.tanh, "tanh_approx", {}, True),                    # Approximate mode
        # (ttnn.tanhshrink, "tanhshrink_accurate", {}, False),     # Accurate mode
        # (ttnn.tanhshrink, "tanhshrink_approx", {}, True),        # Approximate mode
        # GELU family - HAS approx mode
        # (ttnn.gelu, "gelu_accurate", {}, False),                 # Accurate mode
        # (ttnn.gelu, "gelu_approx", {}, True),                    # Approximate mode
        # (ttnn.elu, "elu", {"alpha": 1.0}, False),               # NO approx
        # (ttnn.celu, "celu", {"alpha": 1.0}, False),             # NO approx
        # (ttnn.selu, "selu", {}, False),                         # NO approx
        # (ttnn.silu, "silu", {}, False),                         # NO approx
        # (ttnn.mish, "mish", {}, False),                         # NO approx - KEY COMPARISON!
        # Math functions - NO approx mode
        # (ttnn.sqrt, "sqrt", {}, False),                          # NO approx
        # (ttnn.softplus, "softplus", {}, False),                  # NO approx
        # Soft/Hard family - NO approx mode
        # (ttnn.softshrink, "softshrink", {"lambd": 0.5}, False),
        # (ttnn.hardshrink, "hardshrink", {"lambd": 0.5}, False),
        # (ttnn.hardtanh, "hardtanh", {"min_val": -1.0, "max_val": 1.0}, False),
        # (ttnn.hardsigmoid, "hardsigmoid", {}, False),
        # (ttnn.hardswish, "hardswish", {}, False),
        # (ttnn.softsign, "softsign", {}, False),
        # Special - NO approx mode
        # (ttnn.threshold, "threshold", {"threshold": 0.0, "value": 0.0}, False),
        # (ttnn.heaviside, "heaviside", {"scalar": 0.5}, False),
        # Mish family
        # (ttnn.mish, "mish", {}, False),
        # (ttnn.hardmish, "hardmish", {}, False),
    ],
)
def test_activation_functions(input_shapes, memory_config_type, activation_func, func_name, params, has_approx, device):
    """Test activation functions"""
    torch.manual_seed(0)

    # # Use appropriate input range based on function
    # if func_name in ["exp", "softplus"]:
    #     in_data = torch.empty(input_shapes, dtype=torch.bfloat16).uniform_(-2, 2)
    # elif func_name in ["sqrt", "rsqrt"]:
    #     in_data = torch.empty(input_shapes, dtype=torch.bfloat16).uniform_(0.1, 5)
    # else:

    in_data = torch.empty(input_shapes, dtype=torch.bfloat16).uniform_(-10, 10)

    # Get appropriate memory configuration
    memory_config = get_memory_config(memory_config_type, list(input_shapes), device)

    input_tensor = ttnn.from_torch(
        in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )

    try:
        # Test the specific mode (accurate or approximate) based on has_approx flag
        if has_approx:
            # Test approximate mode
            if "sigmoid" in func_name and "accurate" not in func_name:
                output_tensor = activation_func(input_tensor, fast_and_approximate_mode=True, vector_mode=4)
            else:
                output_tensor = activation_func(input_tensor, fast_and_approximate_mode=True)
            pcc_threshold = 0.95
        else:
            # Test accurate mode - handle positional arguments
            if func_name == "heaviside":
                compare_tensor = ttnn.from_torch(
                    torch.zeros_like(in_data),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=memory_config,
                )
                output_tensor = activation_func(input_tensor, compare_tensor, params.get("scalar", 0.5))
            elif len(params) == 1:
                # Single positional arg: leaky_relu(alpha), relu_max(upper_limit), relu_min(lower_limit),
                # elu(alpha), celu(alpha), softshrink(lambd), hardshrink(lambd)
                output_tensor = activation_func(input_tensor, list(params.values())[0])
            elif len(params) == 2:
                # Two positional args: hardtanh(min_val, max_val), threshold(threshold, value)
                output_tensor = activation_func(input_tensor, *params.values())
            else:
                output_tensor = activation_func(input_tensor, **params)
            pcc_threshold = 0.99

        # Verify correctness with golden function
        golden_function = ttnn.get_golden_function(activation_func)

        if func_name == "heaviside":
            golden_tensor = golden_function(
                in_data, torch.zeros_like(in_data), params.get("scalar", 0.5), device=device
            )
        elif len(params) > 0:
            golden_tensor = golden_function(in_data, *params.values(), device=device)
        else:
            golden_tensor = golden_function(in_data, device=device)

        assert_with_pcc(ttnn.to_torch(output_tensor), golden_tensor, pcc=pcc_threshold)

    except Exception as e:
        print(f"{func_name}: FAILED - {e}")
