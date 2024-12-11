# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
import math

from models.utility_functions import is_wormhole_b0, is_grayskull, is_x2_harvested
from tests.ttnn.utils_for_testing import assert_with_pcc

import ttnn


def run_max_pool(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    dtype,
    memory_config=None,
    shard_scheme=None,
):
    in_n, in_c, in_h, in_w = act_shape
    kernel_h, kernel_w = kernel_size
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    if shard_scheme != ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        if 2 * pad_h > kernel_h or 2 * pad_w > kernel_w:
            pytest.skip("Invalid case")

    if (
        (kernel_h == 13 and pad_h != 6)
        or (kernel_h == 9 and pad_h != 4)
        or (kernel_h == 5 and pad_h != 2)
        or (kernel_h == 3 and pad_h != 1)
        or (kernel_h == 2 and pad_h != 0)
    ):
        pytest.skip("kernel size and padding combination not supported")

    out_h = math.floor((in_h + 2 * pad_h - (dilation_h * kernel_h - 1) - 1) / stride_h) + 1
    out_w = math.floor((in_w + 2 * pad_w - (dilation_w * kernel_w - 1) - 1) / stride_w) + 1
    cores_x = device.core_grid.x
    cores_y = device.core_grid.y
    max_cores = cores_x * cores_y

    if shard_scheme == ttnn.TensorMemoryLayout.HEIGHT_SHARDED or shard_scheme is None:
        if in_c % 16 != 0:
            pytest.skip("Current maxpool writer needs nchannels to be multiple of 16!")
        if in_c == 16 and dtype == ttnn.bfloat8_b and in_n * in_h * in_w > 600000:
            pytest.skip("This case runs out of memory")
        if in_n > 16 and in_c > 64 and dtype == ttnn.bfloat8_b and is_wormhole_b0():
            pytest.skip("This case runs out of memory on Wormhole b0")
        if (
            stride == (1, 1)
            and (act_shape == [16, 64, 112, 112] or act_shape == [4, 16, 1056, 160] or act_shape == [16, 16, 528, 80])
            and is_wormhole_b0()
        ):
            pytest.skip("This case runs out of memory on Wormhole b0")
        if stride == (1, 1) and act_shape == [8, 16, 528, 80] and is_grayskull():
            pytest.skip("This case runs out of memory on Grayskull")
        if kernel_h > 3 and kernel_w > 3 and act_shape == [16, 64, 112, 112] and is_grayskull():
            pytest.skip("This case runs out of memory on Grayskull")
        if kernel_size == (13, 13) and act_shape == [128, 32, 132, 20] and is_grayskull():
            pytest.skip("This case runs out of memory on Grayskull")
        if kernel_h > 5 and kernel_w > 5 and act_shape == [16, 64, 112, 112] and is_x2_harvested(device):
            pytest.skip("This case runs out of memory on Wormhole X2")
        if stride == (1, 1) and act_shape == [128, 32, 132, 20] and is_x2_harvested(device):
            pytest.skip("This case runs out of memory on Wormhole X2")
        if stride == (1, 1) and kernel_size == (13, 13) and act_shape == [32, 32, 264, 40] and is_x2_harvested(device):
            pytest.skip("This case runs out of memory on Wormhole X2")
        if (
            dtype == ttnn.bfloat8_b
            and (act_shape == [4, 16, 1056, 160] or act_shape == [16, 16, 528, 80])
            and is_x2_harvested(device)
        ):
            pytest.skip("This case runs out of memory on Wormhole X2")

    if shard_scheme == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        if in_c < max_cores:
            pytest.skip("Width sharding requires channles >= cores")
        if in_c / max_cores < 16:
            pytest.skip("Width sharding requires large enough channels to shard (at least 16 per core)")
        if (
            kernel_size == (13, 13)
            and (act_shape == [8, 4096, 10, 16] or act_shape == [1, 32768, 10, 10])
            and is_grayskull()
        ):
            pytest.skip("This case runs out of memory on Grayskull")
        if (
            stride == (1, 1)
            and kernel_h > 5
            and kernel_w > 5
            and (act_shape == [4, 1024, 40, 40] or act_shape == [2, 2048, 40, 40] or act_shape == [8, 4096, 10, 16])
            and is_x2_harvested(device)
        ):
            pytest.skip("This case runs out of memory on Wormhole X2")
        if kernel_h > 5 and kernel_w > 5 and act_shape == [8, 4096, 10, 16] and is_x2_harvested(device):
            pytest.skip("This case runs out of memory on Wormhole X2")
        if kernel_size == (13, 13) and act_shape == [1, 32768, 10, 10] and is_x2_harvested(device):
            pytest.skip("This case runs out of memory on Wormhole X2")

    if shard_scheme == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        if in_c < cores_x:
            pytest.skip("Block sharding requires channles >= cores")
        if in_c / cores_x < 16:
            pytest.skip("Block sharding requires large enough channels to shard (at least 16 per core)")

    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=500, threshold=10000, edgeitems=32)

    ## construct the tensor in NCHW shape
    # act = torch.randn(act_shape, dtype=torch.bfloat16)
    act = torch.empty(act_shape, dtype=torch.bfloat16)
    for n in range(act_shape[0]):
        for c in range(act_shape[1]):
            for h in range(act_shape[2]):
                for w in range(act_shape[3]):
                    act[n, c, h, w] = 4  # h * in_w + w
    act_shape = (1, 1, in_n * in_h * in_w, in_c)
    act_permuted = torch.permute(act, (0, 2, 3, 1))
    act_reshaped = act_permuted.reshape(act_shape)

    if dtype == ttnn.bfloat8_b:
        if shard_scheme == ttnn.TensorMemoryLayout.WIDTH_SHARDED and (in_c / max_cores) % 32 != 0:
            pytest.skip("For BFP8_B datatype, input channels / max_cores should be multiple of 32")
        if shard_scheme == ttnn.TensorMemoryLayout.BLOCK_SHARDED and (in_c / cores_x) % 32 != 0:
            pytest.skip("For BFP8_B datatype, input channels / cores_x should be multiple of 32")
        ttact = ttnn.from_torch(act_reshaped, dtype, layout=ttnn.TILE_LAYOUT)
    else:
        ttact = ttnn.from_torch(act_reshaped, dtype)

    pre_shard = shard_scheme == None

    ttact_device = ttnn.to_device(ttact, device)
    if pre_shard:
        parallel_config = ttnn._ttnn.operations.conv.determine_parallel_config(
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            batch_size=in_n,
            input_channels=in_c,
            output_height=out_h,
            output_width=out_w,
            output_channels=in_c,
            compute_grid_size=(1, 1),
            block_shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
            enable_channels_padding=False,
            is_out_tiled=False,
        )
        sharded_memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
            tensor_shape=ttact_device.shape,
            parallel_config=parallel_config,
            tile_size=32 if dtype == ttnn.bfloat8_b else 1,
        )
        ttact_device = ttnn.to_memory_config(ttact_device, sharded_memory_config)
    output = ttnn.max_pool2d(
        input_tensor=ttact_device,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=[kernel_h, kernel_w],
        stride=[stride_h, stride_w],
        padding=[pad_h, pad_w],
        dilation=[dilation_h, dilation_w],
        memory_config=memory_config,
        applied_shard_scheme=shard_scheme,
    )

    output_host = output.cpu()
    output_pytorch_padded = torch.Tensor(ttnn.to_torch(output_host))
    output_pytorch = output_pytorch_padded[:, :, :, :in_c]

    ## reference
    golden_pytorch = torch.nn.MaxPool2d(
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=False,
        ceil_mode=False,
    )(act)

    ## test for equivalance
    golden_shape = golden_pytorch.shape
    output_pytorch = output_pytorch.reshape(golden_shape[0], golden_shape[2], golden_shape[3], golden_shape[1])

    output_pytorch = torch.permute(output_pytorch, (0, 3, 1, 2))  ## N, C, H, W

    pcc_thresh = 1.0
    if dtype == ttnn.bfloat8_b:
        pcc_thresh = 0.9997

    passing, pcc = assert_with_pcc(output_pytorch, golden_pytorch, pcc_thresh)

    logger.debug(f"Passing: {passing}, PCC: {pcc}")

    ## do more rigorous comparision for each element
    atol, rtol = torch.testing._comparison.default_tolerances(torch.bfloat16)
    if dtype == ttnn.bfloat8_b:
        atol = 0.35

    allclose = torch.allclose(output_pytorch, golden_pytorch, atol=atol)
    isclose = torch.all(torch.isclose(output_pytorch, golden_pytorch, atol=atol))
    isequal = torch.equal(output_pytorch, golden_pytorch)

    assert allclose
    assert isclose
    if dtype == ttnn.bfloat16:
        assert isequal

    if memory_config:
        logger.debug(f"Output memory config: {memory_config}")
        assert ttnn.get_memory_config(output) == memory_config


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "act_shape",  ## NCHW
    (
        (  ## resnet shapes
            [1, 384, 2, 2],  # fails
            # [1, 512, 2, 2],  # passes
        )
    ),
)
@pytest.mark.parametrize(
    "kernel_size",
    (
        # (2, 2),
        (3, 3),
        # (5, 5),
        # (9, 9),
        # (13, 13),
    ),
)
@pytest.mark.parametrize(
    "padding",
    (
        # (0, 0),
        (1, 1),
        # (2, 2),
        # (4, 4),
        # (6, 6),
    ),
)
@pytest.mark.parametrize(
    "stride",
    (
        (1, 1),
        # (2, 2),
    ),
)
@pytest.mark.parametrize("dilation", ((1, 1),))  ## default
@pytest.mark.parametrize(
    "dtype",
    [
        ttnn.bfloat16,
        # ttnn.bfloat8_b,
    ],
)
def test_run_max_pool(
    act_shape,
    kernel_size,
    padding,
    stride,
    dilation,
    device,
    dtype,
    use_program_cache,
):
    run_max_pool(act_shape, kernel_size, padding, stride, dilation, device, dtype)
