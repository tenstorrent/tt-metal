# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
from loguru import logger

import ttnn

from ....pipelines.flux2.pipeline_flux2 import Flux2Pipeline


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 37000000}],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height", "num_inference_steps"), [(1024, 1024, 40)])
@pytest.mark.parametrize(
    (
        "mesh_device",
        "cfg",
        "sp",
        "tp",
        "encoder_tp",
        "vae_tp",
        "topology",
        "num_links",
        "mesh_test_id",
        "use_torch_prompt_encoder",
        "use_torch_vae_decoder",
    ),
    [
        pytest.param(
            (1, 8),  # mesh_device
            (1, 0),  # cfg
            (1, 0),  # sp
            (8, 1),  # tp
            (8, 1),  # encoder_tp
            (8, 1),  # vae_tp
            ttnn.Topology.Linear,
            1,  # num_links
            "1x8tp1",
            True,  # use_torch_prompt_encoder
            True,  # use_torch_vae_decoder
            id="1x8tp1",
        ),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "traced",
    [
        pytest.param(True, id="traced"),
        # pytest.param(False, id="not_traced"),
    ],
)
def test_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
    width: int,
    height: int,
    num_inference_steps: int,
    cfg: tuple[int, int],
    sp: tuple[int, int],
    tp: tuple[int, int],
    encoder_tp: tuple[int, int],
    vae_tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
    no_prompt: bool,
    use_torch_prompt_encoder: bool,
    use_torch_vae_decoder: bool,
    traced: bool,
    mesh_test_id: str,
) -> None:
    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        dit_cfg=cfg,
        dit_sp=sp,
        dit_tp=tp,
        encoder_tp=encoder_tp,
        vae_tp=vae_tp,
        use_torch_prompt_encoder=use_torch_prompt_encoder,
        use_torch_vae_decoder=use_torch_vae_decoder,
        num_links=num_links,
        topology=topology,
        width=width,
        height=height,
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

    filename_prefix = f"flux2_{width}_{height}_{mesh_test_id}"
    if use_torch_prompt_encoder:
        filename_prefix += "_encodercpu"
    if use_torch_vae_decoder:
        filename_prefix += "_vaecpu"
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
