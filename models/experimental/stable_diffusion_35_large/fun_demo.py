# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest

# if TYPE_CHECKING:
import ttnn

from .tt.fun_pipeline import TtStableDiffusion3Pipeline
from .tt.utils import create_global_semaphores, initialize_sd_parallel_config


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (2, 4), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_name, image_w, image_h, guidance_scale, num_inference_steps, cfg_factor, sp_factor, tp_factor, topology",  # "prompt_sequence_length", "spatial_sequence_length",
    [
        #        ("medium", 512, 512, 4.5, 40, 333, 1024),
        #        ("medium", 1024, 1024, 4.5, 40, 333, 4096),
        #        ("large", 512, 512, 3.5, 28, 333, 1024),
        ("large", 1024, 1024, 3.5, 28, 2, 2, 2, ttnn.Topology.Linear),  # , 333, 4096),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 8192, "trace_region_size": 15210496}],
    indirect=True,
)
@pytest.mark.usefixtures("use_program_cache")
def test_sd3(
    *,
    mesh_device: ttnn.MeshDevice,
    model_name,
    image_w,
    image_h,
    guidance_scale,
    num_inference_steps,
    cfg_factor,
    sp_factor,
    tp_factor,
    topology,
) -> None:  # , prompt_sequence_length, spatial_sequence_length,) -> None:
    mesh_shape = tuple(mesh_device.shape)
    dit_parallel_config = initialize_sd_parallel_config(mesh_shape, cfg_factor, sp_factor, tp_factor, topology)

    # create submeshes and update mesh_shape before passing to parallel_configs
    num_devices = mesh_device.get_num_devices() if isinstance(mesh_device, ttnn.MeshDevice) else 1
    submesh_devices = (
        mesh_device.create_submeshes(ttnn.MeshShape(mesh_shape[0], mesh_shape[1] // cfg_factor))
        if isinstance(mesh_device, ttnn.MeshDevice) and cfg_factor > 1
        else [mesh_device]
    )

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )

    # create global semaphore handles
    ag_ccl_semaphore_handles = [
        create_global_semaphores(submesh_devices[i], submesh_devices[i].get_num_devices(), ccl_sub_device_crs, 0)
        for i in range(cfg_factor)
    ]
    rs_from_ccl_semaphore_handles = [
        create_global_semaphores(submesh_devices[i], submesh_devices[i].get_num_devices(), ccl_sub_device_crs, 0)
        for i in range(cfg_factor)
    ]
    rs_to_ccl_semaphore_handles = [
        create_global_semaphores(submesh_devices[i], submesh_devices[i].get_num_devices(), ccl_sub_device_crs, 0)
        for i in range(cfg_factor)
    ]

    if guidance_scale > 1 and cfg_factor == 1:
        guidance_cond = 2
    else:
        guidance_cond = 1

    pipeline = TtStableDiffusion3Pipeline(
        checkpoint=f"stabilityai/stable-diffusion-3.5-{model_name}",
        mesh_device=mesh_device,
        submesh_devices=submesh_devices,
        enable_t5_text_encoder=False,  # submesh_devices[0].get_num_devices() >= 4,
        guidance_cond=guidance_cond,
        parallel_config=dit_parallel_config,
        ag_ccl_semaphore_handles=ag_ccl_semaphore_handles,
        rs_from_ccl_semaphore_handles=rs_from_ccl_semaphore_handles,
        rs_to_ccl_semaphore_handles=rs_to_ccl_semaphore_handles,
        height=image_h,
        width=image_w,
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

    while True:
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
        )

        images[0].save(f"sd35_{image_w}_{image_h}.png")

        for submesh_device in submesh_devices:
            ttnn.synchronize_device(submesh_device)
