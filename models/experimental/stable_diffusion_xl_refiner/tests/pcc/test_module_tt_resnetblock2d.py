import gc
from loguru import logger
import torch
import pytest
import ttnn
from diffusers import UNet2DConditionModel
from models.utility_functions import torch_random
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.stable_diffusion_xl_refiner.tt.tt_resnetblock2d import TtResnetBlock2D
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE


@pytest.mark.parametrize(
    "input_shape, temb_shape, down_block_id, resnet_id, conv_shortcut, block, pcc",
    [
        # Down blocks
        # DownBlock2D - 2 x ResnetBlock2D
        # [(1, 384, 128, 128), (1, 1536)]	[(1, 384, 128, 128)]
        ((1, 384, 128, 128), (1, 1536), 0, 0, False, "down_blocks", 0.998),
        # DownBlock2D - Downsample2D
        # CrossAttnDownBlock2D - 2 x Transformer2DModel
        # CrossAttnDownBlock2D - ResnetBlock2D
        # [(1, 384, 64, 64), (1, 1536)]	[(1, 768, 64, 64)]
        ((1, 384, 64, 64), (1, 1536), 1, 0, True, "down_blocks", 0.998),
        # CrossAttnDownBlock2D - ResnetBlock2D
        # [(1, 768, 64, 64), (1, 1536)]	[(1, 768, 64, 64)]
        ((1, 768, 64, 64), (1, 1536), 1, 1, False, "down_blocks", 0.997),
        # CrossAttnDownBlock2D - Downsample2D
        # CrossAttnDownBlock2D - 2 x Transformer2DModel
        # CrossAttnDownBlock2D - ResnetBlock2D
        # [(1, 768, 32, 32), (1, 1536)]	[(1, 1536, 32, 32)]
        ((1, 768, 32, 32), (1, 1536), 2, 0, True, "down_blocks", 0.997),
        # CrossAttnDownBlock2D - ResnetBlock2D
        # [(1, 1536, 32, 32), (1, 1536)]	[(1, 1536, 32, 32)]
        ((1, 1536, 32, 32), (1, 1536), 2, 1, False, "down_blocks", 0.997),
        # CrossAttnDownBlock2D - Downsample2D
        # DownBlock2D - 2 x ResnetBlock2D
        # [(1, 1536, 16, 16), (1, 1536)]	[(1, 1536, 16, 16)]
        ((1, 1536, 16, 16), (1, 1536), 3, 0, False, "down_blocks", 0.997),
        # Up blocks
        # UpBlock2D - 3 x ResnetBlock2D
        # [(1, 3072, 16, 16), (1, 1536)]	[(1, 1536, 16, 16)]
        ((1, 3072, 16, 16), (1, 1536), 0, 0, True, "up_blocks", 0.998),
        # UpBlock2D - Upsample2D
        # CrossAttnUpBlock2D - 3 x Transformer2DModel
        # CrossAttnUpBlock2D - 2 x ResnetBlock2D
        # [(1, 3072, 32, 32), (1, 1536)]	[(1, 1536, 32, 32)]
        ((1, 3072, 32, 32), (1, 1536), 1, 0, True, "up_blocks", 0.998),
        # CrossAttnUpBlock2D - ResnetBlock2D
        # [(1, 2304, 32, 32), (1, 1536)]	[(1, 1536, 32, 32)]
        ((1, 2304, 32, 32), (1, 1536), 1, 2, True, "up_blocks", 0.998),
        # CrossAttnUpBlock2D - Upsample2D
        # CrossAttnUpBlock2D - 3 x Transformer2DModel
        # CrossAttnUpBlock2D - ResnetBlock2D
        # [(1, 2304, 64, 64), (1, 1536)]	[(1, 768, 64, 64)]
        ((1, 2304, 64, 64), (1, 1536), 2, 0, True, "up_blocks", 0.998),
        # CrossAttnUpBlock2D - ResnetBlock2D
        # [(1, 1536, 64, 64), (1, 1536)]	[(1, 768, 64, 64)]
        ((1, 1536, 64, 64), (1, 1536), 2, 1, True, "up_blocks", 0.998),
        # CrossAttnUpBlock2D - ResnetBlock2D
        # [(1, 1152, 64, 64), (1, 1536)]	[(1, 768, 64, 64)]
        ((1, 1152, 64, 64), (1, 1536), 2, 2, True, "up_blocks", 0.998),
        # CrossAttnUpBlock2D - Upsample2D
        # UpBlock2D - ResnetBlock2D
        # [(1, 1152, 128, 128), (1, 1536)]	[(1, 384, 128, 128)]
        ((1, 1152, 128, 128), (1, 1536), 3, 0, True, "up_blocks", 0.998),
        # UpBlock2D - 2 x ResnetBlock2D
        # [(1, 768, 128, 128), (1, 1536)]	[(1, 384, 128, 128)]
        ((1, 768, 128, 128), (1, 1536), 3, 1, True, "up_blocks", 0.998),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
def test_resnetblock2d_refiner(
    device,
    input_shape,
    temb_shape,
    down_block_id,
    resnet_id,
    conv_shortcut,
    block,
    pcc,
    is_ci_env,
):
    if is_ci_env:
        pytest.skip("Skipping test in CI environment")

    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
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

    module_path = f"{block}.{down_block_id}.resnets.{resnet_id}"

    tt_resnet = TtResnetBlock2D(
        device=device,
        state_dict=state_dict,
        module_path=module_path,
        use_conv_shortcut=conv_shortcut,
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
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output_tensor, output_shape = tt_resnet.forward(ttnn_input_tensor, ttnn_temb_tensor, [B, C, H, W])

    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(input_shape[0], output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del unet
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC is {pcc_message}")
