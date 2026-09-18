# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
import gc
import os

import pytest
import torch
from diffusers import UNet2DConditionModel
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, torch_random
from models.demos.stable_diffusion_xl_base.tt.model_configs import load_model_optimisations
from models.demos.stable_diffusion_xl_base.tt.tt_crossattnupblock2d import TtCrossAttnUpBlock2D
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "image_resolution, input_shape, temb_shape, residuals, encoder_shape, query_dim, num_attn_heads, out_dim, block_id, pcc",
    [
        # 1024x1024 image resolution
        (
            (1024, 1024),
            (1, 1280, 32, 32),
            (1, 1280),
            ((1, 640, 32, 32), (1, 1280, 32, 32), (1, 1280, 32, 32)),
            (1, 77, 2048),
            1280,
            20,
            1280,
            0,
            0.970 if not is_blackhole() else 0.968,
        ),
        (
            (1024, 1024),
            (1, 1280, 64, 64),
            (1, 1280),
            ((1, 320, 64, 64), (1, 640, 64, 64), (1, 640, 64, 64)),
            (1, 77, 2048),
            640,
            10,
            640,
            1,
            0.993,
        ),
        # 512x512 image resolution
        (
            (512, 512),
            (1, 1280, 16, 16),
            (1, 1280),
            ((1, 640, 16, 16), (1, 1280, 16, 16), (1, 1280, 16, 16)),
            (1, 77, 2048),
            1280,
            20,
            1280,
            0,
            0.985,
        ),
        (
            (512, 512),
            (1, 1280, 32, 32),
            (1, 1280),
            ((1, 320, 32, 32), (1, 640, 32, 32), (1, 640, 32, 32)),
            (1, 77, 2048),
            640,
            10,
            640,
            1,
            0.992,
        ),
    ],
)
def test_crossattnup(
    device,
    image_resolution,
    input_shape,
    temb_shape,
    residuals,
    encoder_shape,
    query_dim,
    num_attn_heads,
    out_dim,
    block_id,
    pcc,
    debug_mode,
    is_ci_env,
    is_ci_v2_env,
    sdxl_base_unet_location,
    reset_seeds,
):
    if image_resolution == (512, 512) and is_blackhole():
        pytest.skip("512x512 not supported on Blackhole")
    unet = UNet2DConditionModel.from_pretrained(
        sdxl_base_unet_location,
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_v2_env or is_ci_env,
        subfolder=None if is_ci_v2_env else "unet",
    )
    unet.eval()
    state_dict = unet.state_dict()

    torch_crosattn = unet.up_blocks[block_id]

    model_config = load_model_optimisations(image_resolution)
    tt_crosattn = TtCrossAttnUpBlock2D(
        device,
        state_dict,
        f"up_blocks.{block_id}",
        model_config,
        query_dim,
        num_attn_heads,
        out_dim,
        True,
        debug_mode=debug_mode,
    )
    torch_input_tensor = torch_random(input_shape, -0.1, 0.1, dtype=torch.float32)
    torch_temb_tensor = torch_random(temb_shape, -0.1, 0.1, dtype=torch.float32)
    torch_encoder_tensor = torch_random(encoder_shape, -0.1, 0.1, dtype=torch.float32)

    torch_residual_tensors = ()
    for r in residuals:
        residual = torch_random(r, -0.1, 0.1, dtype=torch.float32)
        torch_residual_tensors = torch_residual_tensors + (residual,)

    _dump = os.environ.get("SDXL_UPBLOCK_DUMP")  # investigation aid: torch per-stage intermediates

    if _dump:

        def _mk(name):
            def _h(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out

                o = o.sample if hasattr(o, "sample") else o

                torch.save(o.detach().float(), f"{_dump}/torch_{name}.pt")

            return _h

        for _i, _m in enumerate(torch_crosattn.resnets):
            _m.register_forward_hook(_mk(f"r{_i}_resnet"))

        for _i, _m in enumerate(torch_crosattn.attentions):
            _m.register_forward_hook(_mk(f"r{_i}_attn"))

            _m.norm.register_forward_hook(_mk(f"r{_i}_attn_gn"))

            _m.proj_in.register_forward_hook(_mk(f"r{_i}_attn_projin"))
        _tb0 = torch_crosattn.attentions[0].transformer_blocks[0]
        for _nm in ("norm1", "attn1", "norm2", "attn2", "norm3", "ff"):
            getattr(_tb0, _nm).register_forward_hook(_mk(f"tb_{_nm}"))

        if torch_crosattn.upsamplers is not None:
            torch_crosattn.upsamplers[0].register_forward_hook(_mk("r9_upsampler"))

    torch_output_tensor = torch_crosattn(
        torch_input_tensor, torch_residual_tensors, temb=torch_temb_tensor, encoder_hidden_states=torch_encoder_tensor
    )

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    B, C, H, W = list(ttnn_input_tensor.shape)

    ttnn_input_tensor = ttnn.permute(ttnn_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn.reshape(ttnn_input_tensor, (B, 1, H * W, C))

    ttnn_residual_tensors = ()
    for torch_residual in torch_residual_tensors:
        ttnn_residual = ttnn.from_torch(torch_residual, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        Br, Cr, Hr, Wr = list(ttnn_residual.shape)
        ttnn_residual = ttnn.permute(ttnn_residual, (0, 2, 3, 1))
        ttnn_residual = ttnn.reshape(ttnn_residual, (Br, 1, Hr * Wr, Cr))
        ttnn_residual_tensors = ttnn_residual_tensors + (ttnn_residual,)

    ttnn_temb_tensor = ttnn.from_torch(torch_temb_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_temb_tensor = ttnn.silu(ttnn_temb_tensor)
    ttnn_encoder_tensor = ttnn.from_torch(
        torch_encoder_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
    )
    ttnn_output_tensor, output_shape = tt_crosattn.forward(
        ttnn_input_tensor,
        ttnn_residual_tensors,
        [B, C, H, W],
        temb=ttnn_temb_tensor,
        encoder_hidden_states=ttnn_encoder_tensor,
    )

    output_tensor = ttnn.to_torch(ttnn_output_tensor)
    output_tensor = output_tensor.reshape(B, output_shape[1], output_shape[2], output_shape[0])
    output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))

    del unet, tt_crosattn
    gc.collect()

    _, pcc_message = assert_with_pcc(torch_output_tensor, output_tensor, pcc)

    _e = output_tensor.float() - torch_output_tensor.float()

    logger.info(
        f"BIAS bias={_e.mean().item():+.5f} rms={_e.pow(2).mean().sqrt().item():.5f} std_ratio={(output_tensor.float().std() / torch_output_tensor.float().std()).item():.5f} torch_std={torch_output_tensor.float().std().item():.4f}"
    )
    logger.info(f"PCC is: {pcc_message}")
