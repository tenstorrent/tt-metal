# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

from ....pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from ....utils.check import assert_quality


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("width", "height", "num_inference_steps"),
    [
        (1024, 1024, 28),
    ],
)
@pytest.mark.parametrize(
    ("mesh_device", "sp", "tp", "vae_tp", "topology", "num_links", "mesh_test_id"),
    [
        pytest.param((2, 4), (2, 0), (4, 1), (4, 1), ttnn.Topology.Linear, 1, "2x4sp0tp1", id="2x4sp0tp1"),
    ],
    indirect=["mesh_device"],
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
def test_flux2_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
    no_prompt: bool,
    width: int,
    height: int,
    num_inference_steps: int,
    sp: tuple[int, int],
    tp: tuple[int, int],
    vae_tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
    model_location_generator,
    traced: bool,
    mesh_test_id: str,
    use_cache: bool,
    is_ci_env: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if is_ci_env:
        if not use_cache:
            pytest.skip("Skipping no-cache variant in CI")
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    if is_ci_env and traced:
        pytest.skip("Skipping traced test in CI (use performance test instead)")

    logger.info(f"Mesh device shape: {mesh_device.shape}")

    pipeline = Flux2Pipeline.create_pipeline(
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-dev"),
        mesh_device=mesh_device,
        dit_sp=sp,
        dit_tp=tp,
        vae_tp=vae_tp,
        num_links=num_links,
        topology=topology,
    )

    prompts = [
        "A luxury sports car.",
    ]

    filename_prefix = f"flux2_{width}_{height}_{mesh_test_id}"
    if not traced:
        filename_prefix += "_untraced"

    def run(*, prompt: str, number: int, seed: int) -> None:
        benchmark_profiler = BenchmarkProfiler()
        with benchmark_profiler("run", iteration=0):
            images = pipeline.run_single_prompt(
                width=width,
                height=height,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                seed=seed,
                traced=traced,
                profiler=benchmark_profiler,
                profiler_iteration=0,
            )

        output_filename = f"{filename_prefix}_{number}.png"
        images[0].save(output_filename)
        logger.info(f"Image saved as {output_filename}")

        logger.info(f"Total encoding time: {benchmark_profiler.get_duration('encoder', 0):.2f}s")
        logger.info(f"VAE decoding time: {benchmark_profiler.get_duration('vae', 0):.2f}s")
        logger.info(f"Total pipeline time: {benchmark_profiler.get_duration('total', 0):.2f}s")
        avg_step_time = benchmark_profiler.get_duration("denoising", 0) / num_inference_steps
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


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("mesh_device", "sp", "tp", "vae_tp", "topology", "num_links", "mesh_test_id"),
    [
        pytest.param((2, 4), (2, 0), (4, 1), (4, 1), ttnn.Topology.Linear, 1, "2x4sp0tp1", id="2x4sp0tp1"),
    ],
    indirect=["mesh_device"],
)
def test_flux2_pipeline_pcc(
    *,
    mesh_device: ttnn.MeshDevice,
    sp: tuple[int, int],
    tp: tuple[int, int],
    vae_tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
    model_location_generator,
    mesh_test_id: str,
    is_ci_env: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    prompt = "A luxury sports car."
    num_inference_steps = 4
    width = 1024
    height = 1024
    seed = 42
    guidance_scale = 4.0

    checkpoint_name = model_location_generator("black-forest-labs/FLUX.2-dev")

    logger.info("creating TT pipeline...")
    tt_pipeline = Flux2Pipeline.create_pipeline(
        checkpoint_name=checkpoint_name,
        mesh_device=mesh_device,
        dit_sp=sp,
        dit_tp=tp,
        vae_tp=vae_tp,
        num_links=num_links,
        topology=topology,
    )

    logger.info("running TT pipeline (untraced)...")
    tt_images = tt_pipeline.run_single_prompt(
        width=width,
        height=height,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        seed=seed,
        guidance_scale=guidance_scale,
        traced=False,
    )

    from torchvision import transforms

    tt_image_tensor = transforms.ToTensor()(tt_images[0]).unsqueeze(0)

    logger.info("running diffusers reference pipeline...")
    from diffusers import Flux2Pipeline as DiffusersFlux2Pipeline

    ref_pipeline = DiffusersFlux2Pipeline.from_pretrained(
        checkpoint_name,
        torch_dtype=torch.bfloat16,
    )

    generator = torch.Generator("cpu").manual_seed(seed)
    ref_output = ref_pipeline(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pt",
    )
    ref_image_tensor = ref_output.images[0].unsqueeze(0).float()

    del ref_pipeline
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    tt_image_tensor = tt_image_tensor.float()
    ref_image_tensor = ref_image_tensor.float()

    logger.info(f"TT image tensor shape: {tt_image_tensor.shape}")
    logger.info(f"Reference image tensor shape: {ref_image_tensor.shape}")

    tt_images[0].save(f"flux2_pcc_tt_{mesh_test_id}.png")
    import numpy as np
    from PIL import Image

    ref_np = ref_output.images[0].permute(1, 2, 0).cpu().numpy()
    ref_np = (ref_np * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(ref_np).save(f"flux2_pcc_ref_{mesh_test_id}.png")

    assert_quality(ref_image_tensor, tt_image_tensor, pcc=0.95, relative_rmse=15.0)
