# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os
from collections.abc import Callable

import numpy as np
import pytest
from loguru import logger
from PIL import Image

import ttnn
from models.common.utility_functions import is_blackhole

from ....pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from ....utils.test import skip_if_unsupported_num_links
from .device_params import line_params_flux2, line_params_flux2_transformer


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "device_params",
    [line_params_flux2],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height"), [(1024, 1024)])
@pytest.mark.parametrize(
    "variant, num_inference_steps, mesh_device, sp_axis, tp_axis, encoder_tp_axis, vae_tp_axis, topology, "
    "num_links, is_fsdp, dynamic_load, traced",
    [
        # is_fsdp and dynamic_load are both mandatory for 2x2 -- the 32B transformer, the 24B encoder and the VAE
        # cannot be co-resident on 4 chips.
        ["dev", 12, (2, 2), 0, 1, 1, 1, ttnn.Topology.Linear, 2, True, True, False],
        ["dev", 12, (1, 8), 0, 1, 1, 1, ttnn.Topology.Linear, 1, False, True, False],
        ["dev", 12, (4, 8), 0, 1, 1, 0, ttnn.Topology.Linear, 4, True, False, True],
        ["dev", 12, (4, 8), 0, 1, 1, 0, ttnn.Topology.Linear, 2, False, False, True],
        ["klein-4B", 4, (2, 2), 0, 1, 1, 1, ttnn.Topology.Linear, 2, False, False, True],
        ["klein-4B", 4, (4, 8), 0, 1, 1, 0, ttnn.Topology.Linear, 2, False, False, True],
    ],
    ids=[
        "dev-2x2sp0tp1vaetp1nl2_linear_is_fsdp1",
        "dev-1x8sp0tp1vaetp1nl1_linear_is_fsdp0",
        "dev-4x8sp0tp1vaetp0nl4_linear_is_fsdp1",
        "dev-4x8sp0tp1vaetp0nl2_linear_is_fsdp0",
        "klein-4B-2x2sp0tp1vaetp1nl2_linear_is_fsdp0",
        "klein-4B-4x8sp0tp1vaetp0nl2_linear_is_fsdp0",
    ],
    indirect=["mesh_device"],
)
def test_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
    variant: str,
    width: int,
    height: int,
    num_inference_steps: int,
    sp_axis: int,
    tp_axis: int,
    encoder_tp_axis: int,
    vae_tp_axis: int,
    topology: ttnn.Topology,
    num_links: int,
    no_prompt: bool,
    traced: bool,
    is_fsdp: bool,
    dynamic_load: bool,
    model_location_generator,
    is_ci_env: bool,
) -> None:
    skip_if_unsupported_num_links(mesh_device, num_links)

    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=model_location_generator(f"black-forest-labs/FLUX.2-{variant}"),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        encoder_tp_axis=encoder_tp_axis,
        vae_tp_axis=vae_tp_axis,
        vae_h_axis=1 - vae_tp_axis,
        vae_w_axis=None,
        num_links=num_links,
        topology=topology,
        width=width,
        height=height,
        resolutions=[(width, height)],
        is_fsdp=is_fsdp,
        dynamic_load=dynamic_load,
        trace_warmup=traced,
        shard_prompt=True,
    )

    prompts = [
        "Neon-lit cyberpunk alley, rain-soaked, cinematic wide shot",
        # "Golden retriever astronaut drifting in sunlit space",
        # "Minimalist Scandinavian kitchen, morning light, ultra clean",
        # "Ancient desert temple at dawn, soft fog, wide angle",
        # "Steampunk airship over Victorian city, dramatic clouds",
        # "Macro dewdrops on fern, shallow depth of field",
        # "Luxury wristwatch on marble, studio lighting, hyper-detail",
        # "Stormy coastline lighthouse, crashing waves, long exposure",
        # "Futuristic Tokyo street market, vibrant signage, motion blur",
    ]

    arch = "bh" if is_blackhole() else "wh"
    mesh_tag = "x".join(str(s) for s in mesh_device.shape)
    filename_prefix = f"flux2_{variant}_{arch}_{width}_{height}_{mesh_tag}"
    if not traced:
        filename_prefix += "_untraced"

    def run(*, prompt: str, number: int, seed: int) -> None:
        images = pipeline(
            prompts=[prompt],
            num_inference_steps=num_inference_steps,
            seed=seed,
            traced=traced,
        )

        output_filename = f"{filename_prefix}_{number}.png"
        images[0].save(output_filename)
        logger.info(f"Image saved as {output_filename}")

    if no_prompt:
        for i, prompt in enumerate(prompts):
            run(prompt=prompt, number=i, seed=0)
    else:
        prompt = prompts[0]
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, number=i, seed=i)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        [(2, 4), line_params_flux2_transformer],
        [(2, 2), line_params_flux2_transformer],
        [(4, 8), line_params_flux2_transformer],
    ],
    ids=["2x4", "2x2", "4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_sizes_and_editing(
    *, mesh_device: ttnn.MeshDevice, model_location_generator: Callable[[str], str]
) -> None:
    """Klein 4B with its preset layout and default resolutions, with and without a reference image.

    Every case runs untraced first. The traced runs then alternate between the cases twice, so each
    trace replays after the others were captured, and each must reproduce its untraced image.
    """
    mesh_shape = tuple(mesh_device.shape)
    if not is_blackhole() and mesh_shape != (2, 4):
        pytest.skip(f"no FLUX.2 preset for WH {mesh_shape}")

    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-klein-4B"),
        trace_warmup=False,
    )
    reference = Image.open("models/sample_data/demo.jpeg")

    cases = [(width, height, edit) for width, height in pipeline.resolutions for edit in (False, True)]

    def run(width: int, height: int, *, edit: bool, traced: bool) -> np.ndarray:
        images = pipeline(
            prompts=["turn it into a watercolor painting" if edit else "a lighthouse on a cliff, golden hour"],
            images=[reference] if edit else None,
            num_inference_steps=4,
            seed=0,
            width=width,
            height=height,
            traced=traced,
        )
        assert images[0].size == (width, height)
        name = f"flux2_klein-4B_{width}x{height}_{'edit' if edit else 't2i'}_{'traced' if traced else 'untraced'}"
        images[0].save(f"{name}.png")
        return np.asarray(images[0].convert("RGB")).astype(np.float64)

    untraced = {case: run(*case[:2], edit=case[2], traced=False) for case in cases}

    for _ in range(2):
        for case in cases:
            traced = run(*case[:2], edit=case[2], traced=True)
            mse = np.mean((traced - untraced[case]) ** 2)
            psnr = 10 * np.log10(255**2 / mse) if mse > 0 else np.inf
            logger.info(f"{case}: traced against untraced {psnr:.1f} dB")
            assert psnr > 40, f"{case}: the traced image differs from the untraced one ({psnr:.1f} dB)"
