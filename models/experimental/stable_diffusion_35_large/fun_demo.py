# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import os

import pytest

import ttnn

from .tt.fun_pipeline import TtStableDiffusion3Pipeline
from .tt.parallel_config import StableDiffusionParallelManager, EncoderParallelManager, create_vae_parallel_config


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "model_name, image_w, image_h, guidance_scale, num_inference_steps",  # "prompt_sequence_length", "spatial_sequence_length",
    [
        #        ("medium", 512, 512, 4.5, 40, 333, 1024),
        #        ("medium", 1024, 1024, 4.5, 40, 333, 4096),
        #        ("large", 512, 512, 3.5, 28, 333, 1024),
        ("large", 1024, 1024, 3.5, 28),  # , 333, 4096),
    ],
)
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, topology, num_links",
    [
        [(2, 4), (2, 1), (2, 0), (2, 1), ttnn.Topology.Linear, 1],
        [(4, 8), (2, 1), (4, 0), (4, 1), ttnn.Topology.Linear, 4],
    ],
    ids=[
        "t3k_cfg2_sp2_tp2",
        "tg_cfg2_sp4_tp4",
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 25000000}],
    indirect=True,
)
@pytest.mark.parametrize("traced", [True, False], ids=["yes_traced", "no_traced"])
def test_sd3(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name,
    image_w,
    image_h,
    guidance_scale,
    num_inference_steps,
    cfg,
    sp,
    tp,
    topology,
    num_links,
    no_prompt,
    model_location_generator,
    traced,
    galaxy_type,
) -> None:
    if galaxy_type == "4U":
        pytest.skip("4U is not supported for this test")

    cfg_factor, cfg_axis = cfg
    sp_factor, sp_axis = sp
    tp_factor, tp_axis = tp
    parallel_manager = StableDiffusionParallelManager(
        mesh_device,
        cfg_factor,
        sp_factor,
        tp_factor,
        sp_factor,
        tp_factor,
        topology,
        cfg_axis=cfg_axis,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
    )

    # HACK: reshape submesh device 0 from 2D to 1D
    encoder_device = parallel_manager.submesh_devices[0]

    if parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape[1] != 4:
        # If reshaping, vae_device must be on submesh 0. That means T5 can't fit, so disable it.
        vae_device = parallel_manager.submesh_devices[0]
        enable_t5_text_encoder = False

        cfg_shape = parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape
        assert cfg_shape[0] * cfg_shape[1] == 4, f"Cannot reshape {cfg_shape} to a 1x4 mesh"
        print(f"Reshaping submesh device 0 from {cfg_shape} to (1, 4) for CLIP")
        encoder_device.reshape(ttnn.MeshShape(1, 4))
    else:
        # vae_device can only be on submesh 1 if submesh is not getting reshaped.
        vae_device = parallel_manager.submesh_devices[1]
        enable_t5_text_encoder = True

    encoder_parallel_manager = EncoderParallelManager(
        encoder_device,
        topology,
        mesh_axis=1,  # 1x4 submesh, parallel on axis 1
        num_links=num_links,
    )
    vae_parallel_manager = create_vae_parallel_config(vae_device, parallel_manager)
    # HACK: reshape submesh device 0 from 1D to 2D
    parallel_manager.submesh_devices[0].reshape(
        ttnn.MeshShape(*parallel_manager.dit_parallel_config.cfg_parallel.mesh_shape)
    )

    if guidance_scale > 1 and cfg_factor == 1:
        guidance_cond = 2
    else:
        guidance_cond = 1

    print(f"T5 enabled: {enable_t5_text_encoder}")

    pipeline = TtStableDiffusion3Pipeline(
        checkpoint_name=f"stabilityai/stable-diffusion-3.5-{model_name}",
        mesh_device=mesh_device,
        enable_t5_text_encoder=enable_t5_text_encoder,
        guidance_cond=guidance_cond,
        parallel_manager=parallel_manager,
        encoder_parallel_manager=encoder_parallel_manager,
        vae_parallel_manager=vae_parallel_manager,
        height=image_h,
        width=image_w,
        model_location_generator=model_location_generator,
    )

    pipeline.prepare(
        batch_size=1,
        width=image_w,
        height=image_h,
        guidance_scale=guidance_scale,
        prompt_sequence_length=333,
        spatial_sequence_length=4096,
    )

    prompt = (
        "An epic, high-definition cinematic shot of a rustic snowy cabin glowing "
        "warmly at dusk, nestled in a serene winter landscape. Surrounded by gentle "
        "snow-covered pines and delicate falling snowflakes - captured in a rich, "
        "atmospheric, wide-angle scene with deep cinematic depth and warmth."
    )

    if no_prompt:
        negative_prompt = ""
        images = pipeline(
            prompt_1=[prompt],
            prompt_2=[prompt],
            prompt_3=[prompt],
            negative_prompt_1=[negative_prompt],
            negative_prompt_2=[negative_prompt],
            negative_prompt_3=[negative_prompt],
            num_inference_steps=num_inference_steps,
            seed=0,
            traced=traced,
        )
        images[0].save(f"sd35_{image_w}_{image_h}.png")

    else:
        ## interactive demo
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit:")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break

            negative_prompt = ""

            images = pipeline(
                prompt_1=[prompt],
                prompt_2=[prompt],
                prompt_3=[prompt],
                negative_prompt_1=[negative_prompt],
                negative_prompt_2=[negative_prompt],
                negative_prompt_3=[negative_prompt],
                num_inference_steps=num_inference_steps,
                seed=0,
                traced=traced,
            )

            images[0].save(f"sd35_{image_w}_{image_h}.png")

        for submesh_device in parallel_manager.submesh_devices:
            ttnn.synchronize_device(submesh_device)
