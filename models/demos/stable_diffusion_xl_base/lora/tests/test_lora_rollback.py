# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from diffusers import DiffusionPipeline
from transformers import CLIPTextModel, CLIPTextModelWithProjection

import ttnn
from models.demos.stable_diffusion_xl_base.conftest import is_galaxy
from models.demos.stable_diffusion_xl_base.lora.tt_te_lora_weights_manager import _lora_capable_modules
from models.demos.stable_diffusion_xl_base.tests.test_common import (
    CONCATENATED_TEXT_EMBEDINGS_SIZE,
    MAX_SEQUENCE_LENGTH,
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


@pytest.mark.parametrize(
    "prompt, negative_prompt, lora_prompt",
    [
        (
            "An astronaut riding a green horse",
            "disturbing",
            "An alienzkin astronaut riding a green horse",
        )
    ],
)
@torch.no_grad()
def test_text_encoder_lora_rollback(mesh_device, is_ci_env, te_lora_path, prompt, negative_prompt, lora_prompt):
    """Rollback for a text-encoder-impacting LoRA: after unload, the base image must
    return bit-for-bit (pcc=1.0), which requires the CLIP encoders to be restored on
    device (`TtTextEncoderLoRAWeightsManager.unload`), not just the UNet. The default rollback test
    uses a UNet-only adapter and does not cover this.
    """
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
            encoders_on_device=True,  # required for the TE LoRA path
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

    tt_sdxl.load_lora_weights(te_lora_path)
    assert tt_sdxl._te_lora_weights_manager.state()["components"], "chosen LoRA does not impact the text encoders"
    tt_sdxl.fuse_lora()
    assert tt_sdxl.get_lora_status()["text_encoder"] is True
    _run_forward_pass(tt_sdxl, pipeline, lora_prompt, negative_prompt, batch_size)

    tt_sdxl.unload_lora_weights()
    assert tt_sdxl.get_lora_status()["text_encoder"] is False
    img_rollback = _run_forward_pass(tt_sdxl, pipeline, prompt, negative_prompt, batch_size)

    ttnn.synchronize_device(mesh_device)

    assert_with_pcc(img_base, img_rollback, pcc=1.0)


@pytest.mark.parametrize(
    "device_params",
    [
        {},
    ],
    indirect=True,
)
@torch.no_grad()
def test_text_encoder_lora_bind_unbind_round_trip(mesh_device, is_ci_env, te_lora_path):
    """Fuse then roll back a text-encoder LoRA and check the device weights come back
    bit-for-bit.

    `test_text_encoder_lora_rollback` asserts the same property through a generated image
    at pcc=1.0, which is the behaviour users see but a slow and indirect way to find out
    what moved. This checks the weights directly, because the exactness is not free: the
    on-device bind adds a delta into a bfloat16 weight and unbinding subtracts it again,
    which does not round-trip, so the manager restores from a host copy instead.
    """
    prepare_device(mesh_device, use_cfg_parallel=False)

    pipeline = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True,
        local_files_only=is_ci_env,
    )
    tt_sdxl = TtSDXLPipeline(
        ttnn_device=mesh_device,
        torch_pipeline=pipeline,
        pipeline_config=TtSDXLPipelineConfig(
            num_inference_steps=1,
            guidance_scale=5.0,
            is_galaxy=is_galaxy(),
            capture_trace=False,
            encoders_on_device=True,  # required for the TE LoRA path
        ),
    )

    encoders = {
        "text_encoder": tt_sdxl.tt_text_encoder,
        "text_encoder_2": tt_sdxl.tt_text_encoder_2,
    }

    def device_weights():
        snap = {}
        for component, encoder in encoders.items():
            if encoder is None:
                continue
            for path, module in _lora_capable_modules(encoder).items():
                snap[f"{component}.{path}"] = ttnn.to_torch(module.weight.data).clone()
        return snap

    base = device_weights()
    assert base, "no LoRA-capable device modules found; were the encoders built with lora_enabled?"

    tt_sdxl.load_lora_weights(te_lora_path)
    components = tt_sdxl._te_lora_weights_manager.state()["components"]
    assert components, "chosen LoRA does not impact the text encoders"

    tt_sdxl.fuse_lora(lora_scale=1.0, clip_scale=1.0)
    assert tt_sdxl.get_lora_status()["text_encoder"] is True, "text-encoder LoRA was not marked fused"
    fused = device_weights()
    changed = [k for k in base if not torch.equal(base[k], fused[k])]
    assert changed, "fusing a text-encoder LoRA changed no device weight"

    tt_sdxl.unload_lora_weights()
    assert tt_sdxl.get_lora_status()["text_encoder"] is False, "text-encoder LoRA still reported fused after unload"
    restored = device_weights()
    drifted = {k: (base[k] - restored[k]).abs().max().item() for k in base if not torch.equal(base[k], restored[k])}
    assert not drifted, f"{len(drifted)} weights did not return to base, worst drift {max(drifted.values()):.3e}"
