# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import gc
from loguru import logger
import torch
import pytest
import ttnn
from models.experimental.stable_diffusion_xl_base.tt.tt_resnetblock2d import TtResnetBlock2D
from models.experimental.stable_diffusion_xl_base.tt.model_configs import ModelOptimisations
from diffusers import UNet2DConditionModel
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import torch_random
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "input_shape, temb_shape, down_block_id, resnet_id, conv_shortcut, split_in, block, pcc",
    [
        # conv1 and conv2 are both on 40 cores, using same config ABH_1024_ADB_WDB_BS
        # conv1 & conv2 regular perf = 690k
        # conv1 & conv2 throttle perf = 1.395 mil
        # conv1 & conv2 subblock 1x1 perf = 890k
        # does not need throttle or subblock forcing as there is 40 cores
        ((1, 320, 128, 128), (1, 1280), 0, 0, False, 1, "down_blocks", 0.999),
        # conv1 and conv2 are both on 56 cores, using same config ABH_0_ADB_WDB_BS
        # regular perf: conv1 = 246k conv2 = 475k
        # throttle perf: conv1 = 500k conv2 = 1.04mil
        # subblock 1x1 perf: conv1 = 312k conv2 = 707k
        # needs subblock forcing as there is 56 cores doing compute
        ((1, 320, 64, 64), (1, 1280), 1, 0, True, 1, "down_blocks", 0.999),
        # conv1 and conv2 are both on 56 cores, using same config ABH_0_ADB_WDB_BS
        # regular perf: conv1 = 475k conv2 = 475k
        # throttle perf: conv1 = 1.04mil conv2 = 1.04mil
        # subblock 1x1 perf: conv1 = 707k conv2 = 707k
        # needs subblock forcing as there is 56 cores doing compute
        ((1, 640, 64, 64), (1, 1280), 1, 1, False, 1, "down_blocks", 0.999),
        # conv1 and conv2 are both on 64 cores, using same config ABH_0_ADB_WDB_BS
        # regular perf: conv1 = 163k conv2 = 290k
        # throttle perf: conv1 = 400k conv2 = 735k
        # subblock 1x1 perf: conv1 = 244k conv2 = 440k
        # needs subblock forcing as there is 64 cores doing compute
        ((1, 640, 32, 32), (1, 1280), 2, 0, True, 1, "down_blocks", 0.999),
        # conv1 and conv2 are both on 64 cores, using same config ABH_0_ADB_WDB_BS
        # regular perf: conv1 = 290k conv2 = 290k
        # throttle perf: conv1 = 735k conv2 = 735k
        # subblock 1x1 perf: conv1 = 440k conv2 = 440k
        # needs subblock forcing as there is 64 cores doing compute
        ((1, 1280, 32, 32), (1, 1280), 2, 1, False, 1, "down_blocks", 0.999),
        # conv1 and conv2 are both on 40 cores, conv1 is using ABH_128_ADB_WDB_MOVE_BS and conv2 is using ABH_1024_ADB_WDB_BS
        # regular perf: conv1 = 2.170mil conv2 = 858k
        # throttle perf: conv1 = 4.17mil conv2 = 1.54mil
        # subblock 1x1 perf: conv1 = 2.8mil  conv2 = 1.2mil
        # does not need throttle or subblock forcing as there is 40 cores
        ((1, 960, 128, 128), (1, 1280), 2, 0, True, 1, "up_blocks", 0.998),
        # conv1 and conv2 are both on 40 cores, conv1 is using ABH_1024_ADB_WDB_BS and conv2 is using ABH_256_ADB_WDB_BS
        # regular perf: conv1 = 1.50mil conv2 = 850k
        # throttle perf: conv1 = 2.8mil conv2 = 1.5mil
        # subblock 1x1 perf: conv1 = 2mil   conv2 =  1.2mil
        # does not need throttle or subblock forcing as there is 40 cores
        ((1, 640, 128, 128), (1, 1280), 2, 1, True, 1, "up_blocks", 0.998),
        # conv1 and conv2 are both on 64 cores, both are using ABH_0_ADB_WDB_BS
        # regular perf: conv1 = 560k conv2 = 290k
        # throttle perf: conv1 = 1.4mil conv2 = 747k
        # subblock 1x1 perf: conv1 = 858k   conv2 =  440k
        # needs subblock forcing as there is 64 cores doing compute
        ((1, 2560, 32, 32), (1, 1280), 0, 0, True, 1, "up_blocks", 0.999),
        # conv1 and conv2 are both on 64 cores, both are using ABH_0_ADB_WDB_BS
        # regular perf: conv1 = 430k conv2 = 290k
        # throttle perf: conv1 = 1.1mil conv2 = 737k
        # subblock 1x1 perf: conv1 = 662k  conv2 =  440k
        # needs subblock forcing as there is 64 cores doing compute
        ((1, 1920, 32, 32), (1, 1280), 0, 2, True, 1, "up_blocks", 0.999),
        # conv1 and conv2 are both on 56 cores, conv1 is using ABH_128_ADB_WDB_BS and conv2 is using ABH_0_ADB_WDB_BS
        # regular perf: conv1 = 1.18mil conv2 = 476k
        # throttle perf: conv1 = 2.8mil conv2 = 1.04mil
        # subblock 1x1 perf: conv1 = 1.78mil  conv2 =  708k
        # needs subblock forcing as there is 56 cores doing compute
        ((1, 1920, 64, 64), (1, 1280), 1, 0, True, 1, "up_blocks", 0.999),
        # conv1 and conv2 are both on 56 cores, conv1 is using ABH_256_ADB_WDB_BS_SUBBLOCK_1x1 and conv2 is using ABH_0_ADB_WDB_BS
        # regular perf: conv1 = 802k conv2 = 476k
        # throttle perf: conv1 = 1.8mil conv2 = 1.04mil
        # subblock 1x1 perf: conv1 = 1.2mil conv2 = 708k
        # needs subblock forcing as there is 56 cores doing compute
        ((1, 1280, 64, 64), (1, 1280), 1, 1, True, 1, "up_blocks", 0.999),
        # conv1 and conv2 are both on 56 cores, conv1 is using ABH_256_ADB_WDB_BS_SUBBLOCK_1x1 and conv2 is using ABH_0_ADB_WDB_BS
        # regular perf: conv1 = 660k conv2 = 478k
        # throttle perf: conv1 = 1.5mil conv2 = 1.04mil
        # subblock 1x1 perf: conv1 = 990k conv2 = 710k
        # needs subblock forcing as there is 56 cores doing compute
        ((1, 960, 64, 64), (1, 1280), 1, 2, True, 1, "up_blocks", 0.999),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_resnetblock2d(
    device,
    temb_shape,
    input_shape,
    down_block_id,
    resnet_id,
    conv_shortcut,
    split_in,
    block,
    pcc,
    debug_mode,
    is_ci_env,
    reset_seeds,
):
    import os

    os.environ["TT_MM_THROTTLE_PERF"] = "5"
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        subfolder="unet",
        local_files_only=is_ci_env,
    )
    unet.eval()
    state_dict = unet.state_dict()

    if block == "down_blocks":
        torch_resnet = unet.down_blocks[down_block_id].resnets[resnet_id]
    elif block == "up_blocks":
        torch_resnet = unet.up_blocks[down_block_id].resnets[resnet_id]
    else:
        assert "Incorrect block name"

    model_config = ModelOptimisations()
    tt_resnet = TtResnetBlock2D(
        device,
        state_dict,
        f"{block}.{down_block_id}.resnets.{resnet_id}",
        model_config,
        conv_shortcut,
        split_in,
        debug_mode=debug_mode,
        use_negative_mask=block == "up_blocks" and down_block_id == 2,
    )

    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_temb_tensor = torch_random(temb_shape, -0.1, 0.1, dtype=torch.float32)
    torch_output_tensor = torch_resnet(torch_input_tensor, torch_temb_tensor)

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    B, C, H, W = list(ttnn_input_tensor.shape)

    ttnn_input_tensor = ttnn.permute(ttnn_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.reshape(ttnn_input_tensor, (B, 1, H * W, C))

    ttnn_temb_tensor = ttnn.from_torch(
        torch_temb_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_output_tensor, output_shape = tt_resnet.forward(ttnn_input_tensor, ttnn_temb_tensor, [B, C, H, W])

    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(input_shape[0], output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del unet
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is {pcc_message}")
