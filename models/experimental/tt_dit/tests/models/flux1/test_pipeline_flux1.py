# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
import ttnn
from loguru import logger

from ....parallel.config import DiTParallelConfig, ParallelFactor, EncoderParallelConfig, VAEParallelConfig
from ....pipelines.flux1.pipeline_flux1 import Flux1Pipeline
from ....pipelines.stable_diffusion_35_large.pipeline_stable_diffusion_35_large import (
    TimingCollector,
)


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 34000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("model_variant", "width", "height", "num_inference_steps"),
    [
        ("schnell", 1024, 1024, 4),
        ("dev", 1024, 1024, 28),
    ],
)
@pytest.mark.parametrize(
    ("mesh_device", "sp", "tp", "encoder_tp", "vae_tp", "topology", "num_links", "mesh_test_id"),
    [
        pytest.param((1, 4), (1, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1, "1x4sp0tp1", id="1x4sp0tp1"),
        pytest.param((2, 4), (2, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1, "2x4sp0tp1", id="2x4sp0tp1"),
        pytest.param((4, 8), (4, 0), (8, 1), (4, 0), (4, 0), ttnn.Topology.Linear, 4, "4x8sp0tp1", id="4x8sp0tp1"),
        pytest.param((4, 8), (8, 1), (4, 0), (4, 0), (4, 0), ttnn.Topology.Linear, 4, "4x8sp1tp0", id="4x8sp1tp0"),
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
@pytest.mark.parametrize(
    "use_cache",
    [
        pytest.param(True, id="yes_use_cache"),
        pytest.param(False, id="no_use_cache"),
    ],
)
def test_flux1_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
    model_variant: str,
    width: int,
    height: int,
    num_inference_steps: int,
    sp: tuple[int, int],
    tp: tuple[int, int],
    encoder_tp: tuple[int, int],
    vae_tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
    no_prompt: bool,
    enable_t5_text_encoder: bool,
    use_torch_t5_text_encoder: bool,
    use_torch_clip_text_encoder: bool,
    model_location_generator,
    traced: bool,
    mesh_test_id: str,
    use_cache: bool,
    is_ci_env: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Setup CI environment
    if is_ci_env:
        if use_cache:
            monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")
        else:
            pytest.skip("Skipping. No use cache is implicitly tested with the configured non persistent cache path.")
        if traced:
            pytest.skip("Skipping traced test in CI environment. Use Performance test for detailed timing analysis.")

    sp_factor, sp_axis = sp
    tp_factor, tp_axis = tp

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    encoder_parallel_config = EncoderParallelConfig(
        tensor_parallel=ParallelFactor(factor=encoder_tp[0], mesh_axis=encoder_tp[1])
    )
    vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=vae_tp[0], mesh_axis=vae_tp[1]))

    logger.info(f"Mesh device shape: {mesh_device.shape}")
    logger.info(f"Parallel config: {parallel_config}")
    logger.info(f"Encoder parallel config: {encoder_parallel_config}")
    logger.info(f"VAE parallel config: {vae_parallel_config}")
    logger.info(f"T5 enabled: {enable_t5_text_encoder}")

    timing_collector = TimingCollector()

    pipeline = Flux1Pipeline.create_pipeline(
        checkpoint_name=model_location_generator(f"black-forest-labs/FLUX.1-{model_variant}"),
        mesh_device=mesh_device,
        dit_sp=sp,
        dit_tp=tp,
        encoder_tp=encoder_tp,
        vae_tp=vae_tp,
        enable_t5_text_encoder=enable_t5_text_encoder,
        use_torch_t5_text_encoder=use_torch_t5_text_encoder,
        use_torch_clip_text_encoder=use_torch_clip_text_encoder,
        num_links=num_links,
        topology=topology,
    )

    pipeline.timing_collector = timing_collector

    prompts = [
        "A luxury sports car.",
        # "Neon-lit cyberpunk alley, rain-soaked, cinematic wide shot",
        # "Golden retriever astronaut drifting in sunlit space",
        # "Minimalist Scandinavian kitchen, morning light, ultra clean",
        # "Ancient desert temple at dawn, soft fog, wide angle",
        # "Steampunk airship over Victorian city, dramatic clouds",
        # "Macro dewdrops on fern, shallow depth of field",
        # "Luxury wristwatch on marble, studio lighting, hyper-detail",
        # "Stormy coastline lighthouse, crashing waves, long exposure",
        # "Futuristic Tokyo street market, vibrant signage, motion blur",
    ]

    filename_prefix = f"flux_{model_variant}_{width}_{height}_{mesh_test_id}"
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
        images = pipeline.run_single_prompt(
            width=width, height=height, prompt=prompt, num_inference_steps=num_inference_steps, seed=seed, traced=traced
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
