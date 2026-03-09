# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import DiffusionPipeline
from transformers import CLIPTextModel, CLIPTextModelWithProjection

import ttnn
from conftest import is_galaxy
from models.demos.stable_diffusion_xl_base.tests.test_common import (
    CONCATENATED_TEXT_EMBEDINGS_SIZE,
    MAX_SEQUENCE_LENGTH,
    SDXL_L1_SMALL_SIZE,
    SDXL_TRACE_REGION_SIZE,
    TEXT_ENCODER_2_PROJECTION_DIM,
    determinate_min_batch_size,
    prepare_device,
)
from models.demos.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig
from tests.ttnn.utils_for_testing import assert_with_pcc


def _run_forward_pass(tt_sdxl, pipeline, prompt, negative_prompt, batch_size):
    prompts = [prompt] + [""] * (batch_size - 1)
    negative_prompts = [negative_prompt] + [""] * (batch_size - 1)
    all_prompt_embeds_torch, torch_add_text_embeds = tt_sdxl.encode_prompts(prompts, negative_prompts)
    tt_latents, tt_prompt_embeds, tt_add_text_embeds = tt_sdxl.generate_input_tensors(
        all_prompt_embeds_torch, torch_add_text_embeds, start_latent_seed=0
    )
    tt_sdxl.prepare_input_tensors([tt_latents, tt_prompt_embeds[0], tt_add_text_embeds[0]])
    imgs = tt_sdxl.generate_images()
    img = imgs[0].unsqueeze(0)
    out = pipeline.image_processor.postprocess(img, output_type="pt")
    return out[0]


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": SDXL_L1_SMALL_SIZE, "trace_region_size": SDXL_TRACE_REGION_SIZE}],
    indirect=["device_params"],
)
@pytest.mark.parametrize(
    "prompt, negative_prompt, lora_prompt",
    [
        (
            "An astronaut riding a green horse",
            "disturbing",
            "A Coloring Book of an astronaut riding a green horse",
        )
    ],
)
@torch.no_grad()
def test_lora_rollback(mesh_device, is_ci_env, lora_path, prompt, negative_prompt, lora_prompt):
    prepare_device(mesh_device, use_cfg_parallel=False)
    batch_size = determinate_min_batch_size(mesh_device, use_cfg_parallel=False)

    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env,
    )
    assert isinstance(pipeline.text_encoder, CLIPTextModel)
    assert isinstance(pipeline.text_encoder_2, CLIPTextModelWithProjection)

    tt_sdxl = TtSDXLPipeline(
        ttnn_device=mesh_device,
        torch_pipeline=pipeline,
        pipeline_config=TtSDXLPipelineConfig(
            capture_trace=False,
            vae_on_device=True,
            encoders_on_device=True,
            num_inference_steps=50,
            guidance_scale=5.0,
            is_galaxy=is_galaxy(),
            use_cfg_parallel=False,
            crop_coords_top_left=(0, 0),
            guidance_rescale=0.0,
        ),
    )

    tt_sdxl.compile_text_encoding()
    tt_sdxl.generate_input_tensors(
        all_prompt_embeds_torch=torch.randn(batch_size, 2, MAX_SEQUENCE_LENGTH, CONCATENATED_TEXT_EMBEDINGS_SIZE),
        torch_add_text_embeds=torch.randn(batch_size, 2, TEXT_ENCODER_2_PROJECTION_DIM),
        timesteps=None,
        sigmas=None,
    )
    tt_sdxl.compile_image_processing()

    img_base = _run_forward_pass(tt_sdxl, pipeline, prompt, negative_prompt, batch_size)

    tt_sdxl.load_lora_weights(lora_path)
    tt_sdxl.fuse_lora()
    _run_forward_pass(tt_sdxl, pipeline, lora_prompt, negative_prompt, batch_size)

    tt_sdxl.unload_lora_weights()
    img_rollback = _run_forward_pass(tt_sdxl, pipeline, prompt, negative_prompt, batch_size)

    ttnn.synchronize_device(mesh_device)

    assert_with_pcc(img_base, img_rollback, pcc=1.0)
