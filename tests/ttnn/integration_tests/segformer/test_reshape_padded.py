import pytest
import torch
import ttnn

from models.utility_functions import skip_for_grayskull


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "shape",
    [
        # nchw
        (1, 3, 512, 512),
    ],
)
@pytest.mark.parametrize(
    "channel_padding",
    [
        32,
    ],
)
@pytest.mark.parametrize(
    "is_sharded",
    [
        True,
        False,
    ],
)
def test_reshape_padded(device, shape, channel_padding, is_sharded):
    # Create and reorder torch tensor
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)

    # Calculate shard spec and create memory config
    n, h, w, c = torch_input_tensor.shape
    core_grid = ttnn.CoreGrid(y=8, x=8)
    num_cores = core_grid.x * core_grid.y
    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))}
    )
    shard_spec = ttnn.ShardSpec(
        shard_grid, ((n * w * h + num_cores - 1) // num_cores, channel_padding), ttnn.ShardOrientation.ROW_MAJOR
    )

    input_mem_config = None
    if is_sharded:
        # sharded case fails
        input_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
    else:
        # interleaved works as expected
        input_mem_config = ttnn.L1_MEMORY_CONFIG

    # Pad tensor and send to device
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    input_padded = ttnn.pad(input_tensor, [n, h, w, channel_padding], [0, 0, 0, 0], 0)

    input_device = ttnn.to_device(input_padded, device, input_mem_config)

    # flatten
    shape_flat = (1, 1, n * h * w, c)
    shape_flat_padded = (1, 1, n * h * w, channel_padding)

    input_flat_padded = ttnn.reshape(input_device, shape_flat, shape_flat_padded)

    # Ensure correct padding
    assert input_flat_padded.padded_shape[3] == input_padded.padded_shape[3], "No padding after reshape"
