# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Reusable in-process SDXL text-to-image generator.

Opens the mesh once, keeps weights and the captured device trace warm across calls,
generates images, and releases cleanly. Drives ``TtSDXLPipeline`` directly — no
FastAPI / tt-inference-server in the loop. The explicit open/generate/close lifecycle
lets a host (e.g. a ComfyUI node) hold the device warm and, via ``close()``, release
the trace + mesh so a different model can take the device next.

Mirrors the device-side ``Generator`` pattern the LLM demos expose to vLLM; here the
generate entry point is text prompt -> image. Factors the one-shot flow in
``demo/demo.py::run_demo_inference`` into a warm-once/generate-many object.
"""

import torch
from diffusers import DiffusionPipeline
from loguru import logger

import ttnn
from models.demos.stable_diffusion_xl_base.tests.test_common import (
    SDXL_FABRIC_CONFIG,
    SDXL_L1_SMALL_SIZE,
    prepare_device,
)
from models.demos.stable_diffusion_xl_base.tt.tt_sdxl_pipeline import TtSDXLPipeline, TtSDXLPipelineConfig

DEFAULT_MODEL_LOCATION = "stabilityai/stable-diffusion-xl-base-1.0"

# tensor_parallel -> (mesh_shape, use_cfg_parallel). TP1 = a single chip (no CFG
# parallel); TP2 = two chips running the CFG passes in parallel.
_TP_TOPOLOGY = {
    1: ((1, 1), False),
    2: ((2, 1), True),
}


class SDXLGenerator:
    def __init__(
        self,
        tensor_parallel=2,
        *,
        image_resolution=(1024, 1024),
        num_inference_steps=20,
        guidance_scale=5.0,
        model_location=DEFAULT_MODEL_LOCATION,
    ):
        if tensor_parallel not in _TP_TOPOLOGY:
            raise ValueError(f"unsupported tensor_parallel {tensor_parallel!r}; expected one of {sorted(_TP_TOPOLOGY)}")
        self.tensor_parallel = tensor_parallel
        self.mesh_shape, self.use_cfg_parallel = _TP_TOPOLOGY[tensor_parallel]
        self.image_resolution = tuple(image_resolution)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.model_location = model_location

        self.mesh_device = None
        self.pipeline = None
        self.tt_sdxl = None
        # Set from the pipeline in open(): TtSDXLPipeline fixes its own batch_size from
        # the mesh, and our prompt batching must match it.
        self.batch_size = 1

    def open(self):
        """Open the mesh, load weights, build the pipeline, and warm the trace."""
        if self.mesh_device is not None:
            return self

        if self.use_cfg_parallel:
            # cfg-parallel's all_gather_async needs fabric set before the mesh opens.
            ttnn.set_fabric_config(SDXL_FABRIC_CONFIG)
        self.mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(*self.mesh_shape), l1_small_size=SDXL_L1_SMALL_SIZE)
        prepare_device(self.mesh_device, self.use_cfg_parallel)

        logger.info(f"SDXLGenerator: loading {self.model_location}")
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_location, torch_dtype=torch.float32, use_safetensors=True
        )
        self.tt_sdxl = TtSDXLPipeline(
            ttnn_device=self.mesh_device,
            torch_pipeline=self.pipeline,
            pipeline_config=TtSDXLPipelineConfig(
                image_resolution=self.image_resolution,
                capture_trace=True,
                vae_on_device=True,
                encoders_on_device=True,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                use_cfg_parallel=self.use_cfg_parallel,
                is_galaxy=False,
            ),
        )
        self.batch_size = self.tt_sdxl.batch_size

        # Warm with one real generate (text encode -> encode -> image-processing
        # compile -> trace capture). It must be a real generate, not the demo's
        # random-tensor compile path, which crashes the UNet time embedding on this
        # build.
        self.generate("warmup", negative_prompt="", seed=0)
        logger.info("SDXLGenerator: ready")
        return self

    def generate(
        self, prompt, negative_prompt=None, *, seed=0, num_images=1, num_inference_steps=None, guidance_scale=None
    ):
        """Generate images for a single prompt. Returns a list of PIL images.

        num_inference_steps and guidance_scale are runtime-adjustable against the
        captured trace and default to the construction values: the trace is one
        denoise iteration replayed per step, and guidance is an in-place device
        scalar, so neither needs a re-open."""
        if self.tt_sdxl is None:
            raise RuntimeError("SDXLGenerator.generate() called before open()")

        steps = self.num_inference_steps if num_inference_steps is None else int(num_inference_steps)
        scale = self.guidance_scale if guidance_scale is None else float(guidance_scale)
        # generate_input_tensors() below rebuilds the scheduler timesteps from this,
        # and generate_images() replays the single-iteration trace `steps` times.
        self.tt_sdxl.set_num_inference_steps(steps)
        self.tt_sdxl.set_guidance_scale(scale)

        prompts = [prompt] * num_images
        negatives = [negative_prompt or ""] * num_images
        needed_padding = (self.batch_size - len(prompts) % self.batch_size) % self.batch_size
        prompts += [""] * needed_padding
        negatives += [""] * needed_padding

        self.tt_sdxl.compile_text_encoding()

        images = []
        iters = len(prompts) // self.batch_size
        for it in range(iters):
            lo, hi = it * self.batch_size, (it + 1) * self.batch_size
            all_prompt_embeds_torch, torch_add_text_embeds = self.tt_sdxl.encode_prompts(
                prompts[lo:hi], negatives[lo:hi], None, None
            )
            tt_latents, tt_prompt_embeds, tt_add_text_embeds = self.tt_sdxl.generate_input_tensors(
                all_prompt_embeds_torch,
                torch_add_text_embeds,
                start_latent_seed=seed,
            )
            self.tt_sdxl.prepare_input_tensors([tt_latents, tt_prompt_embeds[0], tt_add_text_embeds[0]])
            self.tt_sdxl.compile_image_processing()
            imgs = self.tt_sdxl.generate_images()
            for idx, img in enumerate(imgs):
                if it == iters - 1 and idx >= self.batch_size - needed_padding:
                    break
                images.append(self.pipeline.image_processor.postprocess(img.unsqueeze(0), output_type="pil")[0])
        return images

    def close(self):
        """Release the captured trace and close the mesh so the device is free for
        the next model. Safe to call more than once."""
        if self.tt_sdxl is not None:
            # __release_trace is the pipeline's only trace-teardown path (name-mangled,
            # also run from its __del__). Calling it explicitly while the mesh is still
            # open avoids leaking the trace region into the next model's session.
            release = getattr(self.tt_sdxl, "_TtSDXLPipeline__release_trace", None)
            if release is not None:
                try:
                    release()
                except Exception as e:
                    logger.warning(f"SDXLGenerator: trace release failed: {e}")
            self.tt_sdxl = None
        self.pipeline = None
        if self.mesh_device is not None:
            ttnn.close_mesh_device(self.mesh_device)
            self.mesh_device = None
