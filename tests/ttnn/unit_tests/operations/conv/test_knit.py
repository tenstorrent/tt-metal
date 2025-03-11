# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from models.utility_functions import (
    is_wormhole_b0,
    skip_for_grayskull,
)
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
import ttnn

torch.set_printoptions(linewidth=400, profile="full", sci_mode=False)


@pytest.mark.parametrize(
    "input_shape_nhwc, num_output_channels_after_knit",
    (
        ((1, 9, 65, 4), 1),
        ((1, 9, 65, 8), 2),
    ),
    ids=("1_out_channels_after_knit", "2_out_channels_after_knit"),
)
def test_conv_split_knit(device, input_shape_nhwc, num_output_channels_after_knit):
    torch.manual_seed(0)

    B = input_shape_nhwc[0]
    H = input_shape_nhwc[1]
    W = input_shape_nhwc[2]
    C = input_shape_nhwc[3]

    torch_input_tensor = torch.randn([1, 1, B * H * W, C], dtype=torch.bfloat16).float()

    # tensor_permuted = torch_input_tensor.reshape([B, H, W, C])
    # tensor_permuted = tensor_permuted.permute(0, 3, 1, 2)

    # new
    # tt_split_knit_out_ = torch.empty(tt_output_tensor.shape[0], out_h*2, out_w*2, tt_output_tensor.shape[3]//4, dtype=tt_output_tensor.dtype)

    ref_knit_tensor_out = torch.empty(
        B,
        H * 2,
        W * 2,
        C // 4,
        dtype=torch.bfloat16,
    )

    for h in range(H):
        for w in range(W):
            ref_knit_tensor_out[0, h * 2, w * 2, :] = torch_input_tensor[
                0,
                0,
                h * W + w,
                num_output_channels_after_knit * 0 : num_output_channels_after_knit * 0
                + num_output_channels_after_knit,
            ]
            ref_knit_tensor_out[0, h * 2, w * 2 + 1, :] = torch_input_tensor[
                0,
                0,
                h * W + w,
                num_output_channels_after_knit * 1 : num_output_channels_after_knit * 1
                + num_output_channels_after_knit,
            ]
            # odd rows
            ref_knit_tensor_out[0, h * 2 + 1, w * 2, :] = torch_input_tensor[
                0,
                0,
                h * W + w,
                num_output_channels_after_knit * 2 : num_output_channels_after_knit * 2
                + num_output_channels_after_knit,
            ]
            ref_knit_tensor_out[0, h * 2 + 1, w * 2 + 1, :] = torch_input_tensor[
                0,
                0,
                h * W + w,
                num_output_channels_after_knit * 3 : num_output_channels_after_knit * 3
                + num_output_channels_after_knit,
            ]

    # pad torch_input_tensor channels to 32

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    tt_input_tensor = ttnn.pad(
        tt_input_tensor,
        [tt_input_tensor.shape[0], tt_input_tensor.shape[1], tt_input_tensor.shape[2], 32],
        [0, 0, 0, 0],
        0,
    )

    core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_shape = (
        tt_input_tensor.shape[2] // core_range_set.num_cores(),
        tt_input_tensor.shape[3],
    )
    tt_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_set,
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, tt_mem_config)

    # print(
    #     f"Conv knit call: kernel height: {2}, num_output_channels: {C // 4}, input_width: {W}, num_input_channels: {C}"
    # )
    tt_knited_tensor = ttnn.conv_knit(tt_input_tensor, 2, C // 4, W, C)
    ttnn.synchronize_device(device)
    # print(f"Shape check: ref_knit_tensor_out {ref_knit_tensor_out.shape} tt_knited_tensor {tt_knited_tensor.shape}")

    tt_knited_tensor_out = ttnn.to_torch(tt_knited_tensor, mesh_composer=None)
    tt_knited_tensor_out = tt_knited_tensor_out.reshape(ref_knit_tensor_out.shape)
    # print("Out shape is: ", tt_knited_tensor_out.shape)

    # row_id = 0
    # print("TT output  is:", tt_knited_tensor_out[:, row_id, :, :])
    # print("Ref is:", ref_knit_tensor_out[:, row_id, :, :])

    pcc = 0.999
    passing, pcc_msg = check_with_pcc_without_tensor_printout(tt_knited_tensor_out, ref_knit_tensor_out, pcc=pcc)

    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing


# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import torch
import pytest
from models.utility_functions import (
    is_wormhole_b0,
    skip_for_grayskull,
)
from tests.ttnn.utils_for_testing import check_with_pcc_without_tensor_printout
import ttnn

torch.set_printoptions(linewidth=400, profile="full", sci_mode=False)


@pytest.mark.parametrize(
    "input_shape_nhwc, num_output_channels_after_knit, crop_h, crop_w",
    (
        ((1, 9, 65, 4), 1, 1, 1),
        ((1, 9, 65, 8), 2, 1, 1),
    ),
    ids=("1_out_channels_after_knit", "2_out_channels_after_knit"),
)
def test_conv_knit_and_crop(device, input_shape_nhwc, num_output_channels_after_knit, crop_h, crop_w):
    torch.manual_seed(0)

    B = input_shape_nhwc[0]
    H = input_shape_nhwc[1]
    W = input_shape_nhwc[2]
    C = input_shape_nhwc[3]

    torch_input_tensor = torch.randn([1, 1, B * H * W, C], dtype=torch.bfloat16).float()

    # tensor_permuted = torch_input_tensor.reshape([B, H, W, C])
    # tensor_permuted = tensor_permuted.permute(0, 3, 1, 2)

    # new
    # tt_split_knit_out_ = torch.empty(tt_output_tensor.shape[0], out_h*2, out_w*2, tt_output_tensor.shape[3]//4, dtype=tt_output_tensor.dtype)

    ref_knit_tensor_out = torch.empty(
        B,
        H * 2,
        W * 2,
        C // 4,
        dtype=torch.bfloat16,
    )

    for h in range(H):
        for w in range(W):
            ref_knit_tensor_out[0, h * 2, w * 2, :] = torch_input_tensor[
                0,
                0,
                h * W + w,
                num_output_channels_after_knit * 0 : num_output_channels_after_knit * 0
                + num_output_channels_after_knit,
            ]
            ref_knit_tensor_out[0, h * 2, w * 2 + 1, :] = torch_input_tensor[
                0,
                0,
                h * W + w,
                num_output_channels_after_knit * 1 : num_output_channels_after_knit * 1
                + num_output_channels_after_knit,
            ]
            # odd rows
            ref_knit_tensor_out[0, h * 2 + 1, w * 2, :] = torch_input_tensor[
                0,
                0,
                h * W + w,
                num_output_channels_after_knit * 2 : num_output_channels_after_knit * 2
                + num_output_channels_after_knit,
            ]
            ref_knit_tensor_out[0, h * 2 + 1, w * 2 + 1, :] = torch_input_tensor[
                0,
                0,
                h * W + w,
                num_output_channels_after_knit * 3 : num_output_channels_after_knit * 3
                + num_output_channels_after_knit,
            ]

    ref_crop_tensor_out = ref_knit_tensor_out

    # print(f"Ref pre crop tensor shape: {ref_crop_tensor_out.shape}")
    ref_crop_tensor_out = ref_crop_tensor_out[:, crop_h:-1, crop_w:-1, :]
    print(f"Ref post crop tensor shape: {ref_crop_tensor_out.shape}")

    # pad torch_input_tensor channels to 32

    tt_input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    tt_input_tensor = ttnn.pad(
        tt_input_tensor,
        [tt_input_tensor.shape[0], tt_input_tensor.shape[1], tt_input_tensor.shape[2], 32],
        [0, 0, 0, 0],
        0,
    )

    core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_shape = (
        tt_input_tensor.shape[2] // core_range_set.num_cores(),
        tt_input_tensor.shape[3],
    )
    tt_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_set,
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, tt_mem_config)

    # print(
    #     f"Conv knit call: kernel height: {2}, num_output_channels: {C // 4}, input_width: {W}, num_input_channels: {C}"
    # )
    tt_knited_tensor = ttnn.conv_knit(tt_input_tensor, 2, C // 4, W, C)
    ttnn.synchronize_device(device)
    # print(f"Shape check: ref_knit_tensor_out {ref_knit_tensor_out.shape} tt_knited_tensor {tt_knited_tensor.shape}")

    post_crop_tensor_h = ref_crop_tensor_out.shape[1]
    post_crop_tensor_w = ref_crop_tensor_out.shape[2]
    pre_crop_tensor_h = ref_knit_tensor_out.shape[1]
    pre_crop_tensor_w = ref_knit_tensor_out.shape[2]
    post_crop_tensor_hw = post_crop_tensor_h * post_crop_tensor_w
    out_core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))})
    print("Setting up core range for num_cores: ", out_core_range_set.num_cores())
    print(
        f"Setting up ttnn:conv_crop pre_crop_tensor_h: {pre_crop_tensor_h}, pre_crop_tensor_w: {pre_crop_tensor_w}, post_crop_tensor_h: {post_crop_tensor_h}, post_crop_tensor_w: {post_crop_tensor_w} post_crop_tensor_hw: {post_crop_tensor_hw}"
    )
    out_shard_shape = (
        post_crop_tensor_hw // out_core_range_set.num_cores(),
        tt_knited_tensor.padded_shape[3],
    )
    print(f"Setting up shard shape: {out_shard_shape}")
    post_crop_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            out_core_range_set,
            out_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # static ttnn::Tensor invoke(
    #     QueueId queue_id,
    #     const ttnn::Tensor& input_tensor,
    #     const MemoryConfig& memory_config,
    #     const int crop_height,
    #     const int crop_width,
    #     const int pre_crop_height,
    #     const int pre_crop_width);

    print("ttnn::conv_crop call")
    tt_cropped_tensor = ttnn.conv_crop(
        tt_knited_tensor, post_crop_mem_config, crop_h, crop_w, pre_crop_tensor_h, pre_crop_tensor_w
    )
    ttnn.synchronize_device(device)

    tt_out_cropped_tensor = ttnn.to_torch(tt_cropped_tensor, mesh_composer=None)
    tt_out_cropped_tensor = tt_out_cropped_tensor.reshape(ref_crop_tensor_out.shape)
    print("tt_out_cropped_tensor shape: ", tt_out_cropped_tensor.shape)
    print(f"Ref post crop tensor shape: {ref_crop_tensor_out.shape}")

    # tt_knited_tensor_out = ttnn.to_torch(tt_knited_tensor, mesh_composer=None)
    # tt_knited_tensor_out = tt_knited_tensor_out.reshape(ref_knit_tensor_out.shape)
    # print("Out shape is: ", tt_knited_tensor_out.shape)

    # row_id = 1
    # print("TT output  is:", tt_out_cropped_tensor[:, row_id, :, :])
    # print("Ref is:", ref_crop_tensor_out[:, row_id, :, :])

    pcc = 0.999
    passing, pcc_msg = check_with_pcc_without_tensor_printout(tt_out_cropped_tensor, ref_crop_tensor_out, pcc=pcc)

    logger.info(f"PCC = {pcc_msg}. Threshold = {pcc}")
    assert passing
