# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import math
from models.common.utility_functions import skip_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_lib.utils import _nearest_32
import ttnn

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shape",
    (([1, 2048, 7, 7], ([1, 64, 1, 32]))),
    ids=["resnet50_unpadded", "tile_divisible"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
def test_run_average_pool2d(
    input_shape,
    dtype,
    device,
):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(input_shape)
    torch_output_tensor = torch.nn.functional.adaptive_avg_pool2d(torch_input_tensor, (1, 1))

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))  # ttnn operates on channels-last tensors
    input_tensor = ttnn.from_torch(input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.global_avg_pool2d(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "act_shape",
    (
        pytest.param(
            [1, 7, 7, 2048],
            marks=pytest.mark.xfail(
                strict=True,
                raises=RuntimeError,
                reason="padded RM: NC folds to one slice but 49 logical rows sit in 224 padded ones, "
                "which reduce refuses - see test_global_avg_pool2d_legacy_pad_to_tile_rejected",
            ),
        ),
        [1, 1, 32, 64],
        [1, 1, 16, 128],
    ),
    ids=["resnet50_unpadded", "rm_dense_alligned", "rm_dense_unaligned"],
)
def test_global_avg_pool2d_legacy_pad_to_tile(act_shape, device):
    """Legacy pad_to_tile ROW_MAJOR inputs. Dense RM mean/sum allows suffix H padding when NC=1."""
    torch.manual_seed(0)
    act = torch.randn(act_shape, dtype=torch.bfloat16).float()
    ttact = ttnn.Tensor(act, ttnn.bfloat16)
    padded = [act_shape[0], act_shape[1], _nearest_32(act_shape[2]), _nearest_32(act_shape[3])]
    if act_shape != padded:
        ttact = ttact.pad_to_tile(0.0)
    ttact = ttact.to(device)

    out = ttnn.to_torch(ttnn.global_avg_pool2d(ttact))
    out = torch.permute(out, (0, 3, 1, 2))
    golden = torch.nn.AdaptiveAvgPool2d((1, 1))(torch.permute(act, (0, 3, 1, 2)))
    assert_with_pcc(golden, out)


@pytest.mark.parametrize("act_shape", ([[1, 7, 7, 2048]]), ids=["resnet50_unpadded"])
def test_global_avg_pool2d_legacy_pad_to_tile_rejected(act_shape, device, expect_error):
    """pad_to_tile pads H per NC slice, so the pool's canonical (N, 1, H*W, C) view carries 224
    padded rows over 49 logical ones. The dense RM reader steps by the logical row count and so does
    the tilize a fallback would run, so reduce refuses instead of silently converting the layout."""
    torch.manual_seed(0)
    act = torch.randn(act_shape, dtype=torch.bfloat16).float()
    ttact = ttnn.Tensor(act, ttnn.bfloat16).pad_to_tile(0.0).to(device)

    with expect_error(RuntimeError, "ROW_MAJOR input padded on H is only supported"):
        ttnn.global_avg_pool2d(ttact)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["BFLOAT16"])
@pytest.mark.parametrize(
    "batch, nhw, channels, num_cores, shard_height",
    [
        # resnet50 final feature map: 7x7 = 49 rows over 2 shards of 32 rows.
        (1, 49, 2048, 2, 32),
    ],
    ids=["resnet50_49rows_2cores"],
)
@pytest.mark.parametrize("fill", ["ones", "random"])
def test_global_avg_pool2d_height_sharded_row_major_padding(
    device, dtype, batch, nhw, channels, num_cores, shard_height, fill
):
    """Average pool over a row-major height-sharded tensor whose shard spec over-covers the logical
    rows. The extra rows are L1 capacity in the last shard, not tensor-spec padding — `padded_shape`
    stays at the logical row count — so the guard is that the divisor and the reads stay logical."""
    logical_shape = [batch, 1, nhw, channels]
    padded_rows = num_cores * shard_height
    assert padded_rows > batch * nhw, "shard spec must over-cover the logical rows"

    torch.manual_seed(0)

    if fill == "ones":
        torch_input = torch.ones(logical_shape, dtype=torch.bfloat16)
    else:
        torch_input = torch.randn(logical_shape, dtype=torch.bfloat16)

    shard_mem_config = ttnn.create_sharded_memory_config(
        shape=(shard_height, channels),
        core_grid=ttnn.CoreGrid(y=1, x=num_cores),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=shard_mem_config,
    )
    assert list(tt_input.shape) == logical_shape
    assert tt_input.memory_config().shard_spec.shape == [shard_height, channels]

    tt_output = ttnn.to_torch(ttnn.global_avg_pool2d(tt_input)).float()
    assert list(tt_output.shape) == [batch, 1, 1, channels]

    if fill == "ones":
        # Dividing by the shard footprint instead of nhw would give 49/64 = 0.765625, and a read
        # past the logical rows something arbitrary.
        torch.testing.assert_close(tt_output, torch.ones_like(tt_output), atol=1e-2, rtol=0)
    else:
        assert_with_pcc(torch.mean(torch_input.float(), dim=2, keepdim=True), tt_output, 0.999)


@pytest.mark.parametrize(
    "input_shape",
    (
        [1, 144, 7, 7],  # EfficientNet case: 144 channels (not tile-aligned)
        [12, 144, 56, 56],  # Larger batch with non-aligned channels
        [1, 48, 14, 14],  # 48 channels (padded to 64)
        [1, 80, 14, 14],  # 80 channels (padded to 96)
        [1, 112, 14, 14],  # 112 channels (padded to 128)
    ),
    ids=["efficientnet_144", "efficientnet_144_batch12", "mobilenet_48", "mobilenet_80", "mobilenet_112"],
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat16,),
)
def test_global_avg_pool2d_non_tile_aligned(
    input_shape,
    dtype,
    device,
):
    """
    Regression test for non-tile-aligned channel dimensions.

    This test ensures that global_avg_pool2d correctly handles tensors where
    the channel dimension is not a multiple of 32 (tile size). Previously,
    the operation would return the padded channel count in the output shape
    instead of the logical (unpadded) channel count.

    Example: Input with 144 channels would incorrectly output 160 channels
    (144 padded to next multiple of 32) instead of preserving 144 channels.
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(input_shape)
    torch_output_tensor = torch.nn.functional.adaptive_avg_pool2d(torch_input_tensor, (1, 1))

    input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))  # ttnn operates on channels-last tensors
    input_tensor = ttnn.from_torch(input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.global_avg_pool2d(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "h_w, channels, grid_end",
    [
        (49, 1280, (7, 0)),  # 1280 ch (EfficientNet head), 8-way ND shard; h_w=49 is not tile-aligned
        (25, 320, (4, 0)),
        (20, 64, (0, 0)),
    ],
)
def test_global_avg_pool2d_nd_sharded_row_major_non_tile_aligned_h(device, h_w, channels, grid_end):
    torch.manual_seed(0)

    end_x, end_y = grid_end
    grid_size = device.compute_with_storage_grid_size()
    if end_x >= grid_size.x or end_y >= grid_size.y:
        pytest.skip(f"Device grid {grid_size.x}x{grid_size.y} is smaller than required {end_x + 1}x{end_y + 1}")

    num_cores = (end_x + 1) * (end_y + 1)
    assert channels % num_cores == 0, "channels must split evenly across cores"
    assert h_w % 32 != 0, "test targets non-tile-aligned H*W"

    torch_input = torch.randn(1, 1, h_w, channels, dtype=torch.bfloat16)
    torch_output = torch_input.mean(dim=2, keepdim=True)

    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))])
    shard_shape = ttnn.Shape([1, 1, h_w, channels // num_cores])
    memory_config = ttnn.MemoryConfig(ttnn.BufferType.L1, ttnn.NdShardSpec(shard_shape, grid))

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
        device=device,
    )

    output_tensor = ttnn.global_avg_pool2d(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output, output_tensor, 0.99)
