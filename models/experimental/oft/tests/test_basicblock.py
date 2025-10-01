# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.experimental.oft.reference.resnet import BasicBlock
from models.experimental.oft.tt.tt_resnet import TTBasicBlock
from tests.ttnn.utils_for_testing import assert_with_pcc

# from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters_resnet
from tests.ttnn.unit_tests.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores
from loguru import logger


@pytest.mark.parametrize(
    "n, in_ch, out_ch, h, w, stride, sharding, is_sliced",
    [(1, 256, 256, 159, 159, 1, "HS", True)],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8 * 1024}], indirect=True)
def test_tt_topdownblock_with_8_basicblocks(device, n, in_ch, out_ch, h, w, stride, sharding, is_sliced):
    skip_if_not_blackhole_20_cores(device)
    device.disable_and_clear_program_cache()  # test hangs without this line on P150
    torch.manual_seed(42)
    input_tensor = torch.randn(n, in_ch, h, w)
    # Create 8 BasicBlock modules from oft
    blocks = []
    params_list = []
    for i in range(8):
        block = BasicBlock(inplanes=in_ch, planes=out_ch, stride=stride)
        blocks.append(block)
        params = create_OFT_model_parameters_resnet(block, input_tensor, device)
        params_list.append(params)
    # Reference output using PyTorch blocks sequentially
    out_ref = input_tensor
    for block in blocks:
        out_ref = block(out_ref)
    # Create 8 TTBasicBlock modules
    tt_blocks = [
        TTBasicBlock(
            device,
            params_list[i],
            params_list[i].conv_args,
            inplanes=in_ch,
            planes=out_ch,
            stride=stride,
            is_sliced=is_sliced,
        )
        for i in range(8)
    ]
    # Prepare TTNN input
    n, c, h, w = input_tensor.shape
    x_for_ttnn = input_tensor.permute(0, 2, 3, 1).view(1, 1, n * h * w, c)
    ttnn_x = ttnn.from_torch(x_for_ttnn, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    # Forward through TTBasicBlocks sequentially
    ttnn_out = ttnn_x
    for block in tt_blocks:
        ttnn_out = block.forward(device, ttnn_out, gn_shard=sharding, num_splits=2)
    ttnn_out = ttnn.to_torch(ttnn_out)
    # Compare output
    B, C, H, W = out_ref.shape
    out_ref = out_ref.permute(0, 2, 3, 1).reshape(1, 1, B * H * W, C)
    pcc, message = assert_with_pcc(ttnn_out, out_ref, 0.99)
    logger.info(f"PCC for topdown block with 8 BasicBlocks: {pcc}, Message: {message}")


@pytest.mark.parametrize(
    "n, in_ch, out_ch, h, w, stride, sharding, is_sliced",
    [
        (1, 128, 128, 48, 160, 1, "HS", True),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8 * 1024}], indirect=True)
def test_tt_basicblock(device, n, in_ch, out_ch, h, w, stride, sharding, is_sliced):
    skip_if_not_blackhole_20_cores(device)
    torch.manual_seed(42)
    input_tensor = torch.randn(n, in_ch, h, w)
    torch_model = BasicBlock(inplanes=in_ch, planes=out_ch, stride=stride)

    out = torch_model.forward(input_tensor)
    params = create_OFT_model_parameters_resnet(torch_model, input_tensor, device)
    block = TTBasicBlock(
        device, params, params.conv_args, inplanes=in_ch, planes=out_ch, stride=stride, is_sliced=is_sliced
    )

    n, c, h, w = input_tensor.shape
    x_for_ttnn = input_tensor.permute(0, 2, 3, 1).view(1, 1, n * h * w, c)
    ttnn_x = ttnn.from_torch(x_for_ttnn, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ttnn_out = block.forward(device, ttnn_x, gn_shard=sharding)

    logger.info(f"Output shape: {ttnn_out.shape}, torch out {out.shape}")
    B, C, H, W = out.shape
    ttnn_out = ttnn.to_torch(ttnn_out)
    out = out.permute(0, 2, 3, 1).reshape(1, 1, B * H * W, C)
    pcc, message = assert_with_pcc(ttnn_out, out, 0.99)
    logger.info(f"PCC: {pcc}, Message: {message}")
