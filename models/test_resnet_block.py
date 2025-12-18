import pytest
import torch
from loguru import logger

import ttnn


def resnet_block(x, w1, w2):
    identity = x
    x = ttnn.linear(
        input_tensor_a=x,
        input_tensor_b=w1,
        bias=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation="relu",
        core_grid=ttnn.CoreGrid(x=11, y=10),
    )
    x = ttnn.linear(
        input_tensor_a=x,
        input_tensor_b=w2,
        bias=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation=None,
        core_grid=ttnn.CoreGrid(x=11, y=10),
    )
    return x + identity


def resnet_model(x, depth, w1, w2):
    for _ in range(depth):
        x = resnet_block(x, w1, w2)
    return x


LAYERS_PER_BLOCK = 2


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32, 64, 128, 1024])
@pytest.mark.parametrize("depth", [1, 16, 32])
@pytest.mark.parametrize("layer_size", [256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
def test_resnet_block(batch, depth, layer_size, dtype, device):
    model_size = layer_size * layer_size * (2 if dtype == ttnn.bfloat16 else 1) * (depth * LAYERS_PER_BLOCK)
    logger.info(
        f"Running RN50 model with {model_size * 1e-6 : 0.1f}M parameters batch={batch}, depth={depth}, layer_size={layer_size}x{layer_size}"
    )

    torch_input_tensor = torch.randn(batch, layer_size)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    torch_weight_a = torch.randn(layer_size, layer_size)
    ttnn_weight_a = ttnn.from_torch(
        torch_weight_a, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    torch_weight_b = torch.randn(layer_size, layer_size)
    ttnn_weight_b = ttnn.from_torch(
        torch_weight_b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output_tensor = resnet_model(ttnn_input_tensor, depth, ttnn_weight_a, ttnn_weight_b)
    print("output_shape ", output_tensor.shape)
