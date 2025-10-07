# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
import ttnn
from loguru import logger

from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....pipelines.motif.pipeline_motif import MotifPipeline
from ....pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    TimingCollector,
)


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 31000000}],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height", "num_inference_steps"), [(1024, 1024, 50)])
@pytest.mark.parametrize(
    ("mesh_device", "cfg", "sp", "tp", "topology", "num_links", "mesh_test_id"),
    [
        pytest.param((1, 4), (1, 0), (1, 0), (4, 1), ttnn.Topology.Linear, 1, "1x4sp0tp1", id="1x4sp0tp1"),
        pytest.param((2, 4), (1, 0), (2, 0), (4, 1), ttnn.Topology.Linear, 1, "2x4sp0tp1", id="2x4sp0tp1"),
        pytest.param((4, 8), (1, 0), (8, 1), (4, 0), ttnn.Topology.Linear, 4, "4x8sp1tp0", id="4x8sp1tp0"),
        pytest.param((2, 4), (2, 0), (1, 0), (4, 1), ttnn.Topology.Linear, 1, "2x4cfg0sp0tp1", id="2x4cfg0sp0tp1"),
        pytest.param((4, 8), (2, 0), (4, 1), (4, 0), ttnn.Topology.Linear, 4, "4x8cfg0sp1tp0", id="4x8cfg0sp1tp0"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    ("enable_t5_text_encoder", "use_torch_t5_text_encoder", "use_torch_clip_text_encoder"),
    [
        # pytest.param(True, True, True, id="encoder_cpu"),
        pytest.param(True, False, False, id="encoder_device"),
    ],
)
@pytest.mark.parametrize(
    "traced",
    [
        pytest.param(True, id="traced"),
        pytest.param(False, id="not_traced"),
    ],
)
def test_motif_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
    width: int,
    height: int,
    num_inference_steps: int,
    cfg: tuple[int, int],
    sp: tuple[int, int],
    tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
    no_prompt: bool,
    enable_t5_text_encoder: bool,
    use_torch_t5_text_encoder: bool,
    use_torch_clip_text_encoder: bool,
    traced: bool,
    mesh_test_id: str,
) -> None:
    cfg_factor, cfg_axis = cfg
    sp_factor, sp_axis = sp
    tp_factor, tp_axis = tp

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=cfg_factor, mesh_axis=cfg_axis),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )

    logger.info(f"Mesh device shape: {mesh_device.shape}")
    logger.info(f"Parallel config: {parallel_config}")
    logger.info(f"T5 enabled: {enable_t5_text_encoder}")

    timing_collector = TimingCollector()

    pipeline = MotifPipeline(
        mesh_device=mesh_device,
        enable_t5_text_encoder=enable_t5_text_encoder,
        use_torch_t5_text_encoder=use_torch_t5_text_encoder,
        use_torch_clip_text_encoder=use_torch_clip_text_encoder,
        parallel_config=parallel_config,
        topology=topology,
        num_links=num_links,
        width=width,
        height=height,
        use_cache=True,
    )

    pipeline.timing_collector = timing_collector

    prompts = [
        "cinematic film still of Kodak Motion Picture Film (Sharp Detailed Image) An Oscar winning movie for Best "
        "Cinematography a woman in a kimono standing on a subway train in Japan Kodak Motion Picture Film Style, "
        "shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, "
        "film grain, grainy"
    ]

    filename_prefix = f"motif_{width}_{height}_{mesh_test_id}"
    if enable_t5_text_encoder:
        if use_torch_t5_text_encoder:
            filename_prefix += "_t5cpu"
    else:
        filename_prefix += "_t5off"
    if use_torch_clip_text_encoder:
        filename_prefix += "_clipcpu"
    if not traced:
        filename_prefix += "_untraced"

    def run(*, prompt: str, number: int, seed: int) -> None:
        images = pipeline(
            prompt_1=[prompt],
            prompt_2=[prompt],
            prompt_3=[prompt],
            negative_prompt_1=[None],
            negative_prompt_2=[None],
            negative_prompt_3=[None],
            num_inference_steps=num_inference_steps,
            cfg_scale=5.0,
            seed=seed,
            traced=traced,
        )

        output_filename = f"{filename_prefix}_{number}.png"
        images[0].save(output_filename)
        logger.info(f"Image saved as {output_filename}")

        timing_data = timing_collector.get_timing_data()
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
