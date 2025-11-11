# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
import ttnn
from loguru import logger

from ....pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from ....pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    TimingCollector,
)


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 37000000}],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height", "num_inference_steps"), [(1024, 1024, 50)])
@pytest.mark.parametrize(
    ("mesh_device", "cfg", "sp", "tp", "encoder_tp", "vae_tp", "topology", "num_links", "mesh_test_id"),
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
            id="1x8tp1",
        ),
        # pytest.param(
        #     (2, 4),  # mesh_device
        #     (2, 0),  # cfg
        #     (1, 0),  # sp
        #     (4, 1),  # tp
        #     (4, 1),  # encoder_tp
        #     (4, 1),  # vae_tp
        #     ttnn.Topology.Linear,
        #     1,  # num_links
        #     "2x4cfg0sp0tp1",
        #     id="2x4cfg0sp0tp1",
        # ),
        # pytest.param(
        #     (2, 4),  # mesh_device
        #     (2, 1),  # cfg
        #     (2, 0),  # sp
        #     (2, 1),  # tp
        #     (4, 1),  # encoder_tp
        #     (4, 1),  # vae_tp
        #     ttnn.Topology.Linear,
        #     1,  # num_links
        #     "2x4cfg1sp0tp1",
        #     id="2x4cfg1sp0tp1",
        # ),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "use_torch_text_encoder",
    [
        pytest.param(True, id="encoder_cpu"),
        # pytest.param(False, id="encoder_device"),
    ],
)
@pytest.mark.parametrize(
    "traced",
    [
        pytest.param(True, id="traced"),
        # pytest.param(False, id="not_traced"),
    ],
)
def test_qwenimage_pipeline(
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
    use_torch_text_encoder: bool,
    traced: bool,
    mesh_test_id: str,
) -> None:
    pipeline = QwenImagePipeline.create_pipeline(
        mesh_device=mesh_device,
        dit_cfg=cfg,
        dit_sp=sp,
        dit_tp=tp,
        encoder_tp=encoder_tp,
        vae_tp=vae_tp,
        use_torch_text_encoder=use_torch_text_encoder,
        use_torch_vae_decoder=False,
        num_links=num_links,
        topology=topology,
        width=width,
        height=height,
    )
    pipeline.timing_collector = TimingCollector()

    prompts = [
        'A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light '
        'beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the '
        'poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'
        ", Ultra HD, 4K, cinematic composition."
    ]

    filename_prefix = f"qwenimage_{width}_{height}_{mesh_test_id}"
    if use_torch_text_encoder:
        filename_prefix += "_encodercpu"
    if not traced:
        filename_prefix += "_untraced"

    def run(*, prompt: str, number: int, seed: int) -> None:
        images = pipeline(
            prompts=[prompt],
            negative_prompts=[None],
            num_inference_steps=num_inference_steps,
            cfg_scale=4.0,
            seed=seed,
            traced=traced,
        )

        output_filename = f"{filename_prefix}_{number}.png"
        images[0].save(output_filename)
        logger.info(f"Image saved as {output_filename}")

        timing_data = pipeline.timing_collector.get_timing_data()
        logger.info(f"CLIP encoding time: {timing_data.clip_encoding_time:.2f}s")
        logger.info(f"T5 encoding time: {timing_data.t5_encoding_time:.2f}s")
        logger.info(f"Total encoding time: {timing_data.total_encoding_time:.2f}s")
        logger.info(f"VAE decoding time: {timing_data.vae_decoding_time:.2f}s")
        logger.info(f"Total pipeline time: {timing_data.total_time:.2f}s")
        if timing_data.denoising_step_times:
            avg_step_time = sum(timing_data.denoising_step_times) / len(timing_data.denoising_step_times)
            logger.info(f"Average denoising step time: {avg_step_time:.2f}s")

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
