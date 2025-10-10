import math
import ttnn
import torch
from loguru import logger


def _nearest_k(x, k):
    return math.ceil(x / k) * k


def test(device):
    compute_grid = device.compute_with_storage_grid_size()
    grid_size = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)

    batch, height, width, channels = 1, 159, 159, 256
    nhw = batch * height * width
    padded_nhw = 25312  # _nearest_k(nhw, 32 * grid_size.x * grid_size.y)

    # create an input tensor 1,1,nhw,channels
    x_torch = torch.randn(1, 1, nhw, channels)
    # x = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.ROW_MAJOR_LAYOUT) # reshape expands padded shape to target value
    x = ttnn.from_torch(
        x_torch, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
    )  # reshape expands padded shape to 32-aligned value
    logger.info(f"{x.shape=}, {x.padded_shape=} {x.layout=} {x.memory_config()=}")

    # try expanding the padded shape to padded_nhw using reshape
    x = ttnn.reshape(
        x,
        ttnn.Shape([1, 1, nhw, channels]),  # logical shape (unchanged)
        ttnn.Shape([1, 1, padded_nhw, channels]),  # padded shape (expanded)
    )
    logger.info(f"{x.shape=}, {x.padded_shape=} {x.layout=} {x.memory_config()=}")
    assert x.padded_shape[2] == padded_nhw

    # create sharded memory config, and call to memory_config
    sharded_mem_config = ttnn.create_sharded_memory_config_(
        shape=(1, 1, _nearest_k(nhw, 32 * grid_size.x * grid_size.y), channels),
        core_grid=grid_size,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        tile_layout=True,
    )
    x = ttnn.to_memory_config(x, memory_config=sharded_mem_config)
    logger.info(f"{x.shape=}, {x.padded_shape=} {x.layout=} {x.memory_config()=}")

    # try expanding the padded shape to padded_nhw using reshape
    x = ttnn.reshape(
        x,
        ttnn.Shape([1, 1, nhw, channels]),  # logical shape (unchanged)
        ttnn.Shape([1, 1, padded_nhw, channels]),  # padded shape (expanded)
    )
    logger.info(f"{x.shape=}, {x.padded_shape=} {x.layout=} {x.memory_config()=}")
    assert x.padded_shape[2] == padded_nhw

    # Assuming you have a tensor with logical shape [1, 159, 159, 256]
    # and padded shape [1, 159, 159, 256]
    # input_torch = torch.randn(1, 159, 159, 256)
    # input_tensor = ttnn.from_torch(input_torch, dtype=ttnn.float32, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.ROW_MAJOR_LAYOUT)
    # logger.info(f"{input_tensor.shape=}, {input_tensor.padded_shape=}")
    # output_tensor = ttnn.reshape(
    #     input_tensor,
    #     ttnn.Shape([1, 159, 159, 256]),  # logical shape (unchanged)
    #     ttnn.Shape([1, 160, 160, 256])   # padded shape (expanded)
    # )
    # logger.info(f"{output_tensor.shape=}, {output_tensor.padded_shape=}")
