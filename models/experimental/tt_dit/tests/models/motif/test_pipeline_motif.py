# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
import ttnn
from loguru import logger

from models.perf.benchmarking_utils import BenchmarkProfiler
from ....pipelines.motif.pipeline_motif import MotifPipeline


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 31000000}],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height", "num_inference_steps"), [(1024, 1024, 20)])
@pytest.mark.parametrize(
    ("mesh_device", "cfg", "sp", "tp", "encoder_tp", "vae_tp", "topology", "num_links", "mesh_test_id"),
    [
        [(2, 4), (2, 0), (1, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1, "2x4cfg0sp0tp1"],
        [(2, 4), (2, 1), (2, 0), (2, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1, "2x4cfg1sp0tp1"],
        [(4, 8), (2, 1), (4, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 4, "4x8cfg1sp0tp1"],
    ],
    ids=[
        "2x4cfg0sp0tp1",
        "2x4cfg1sp0tp1",
        "4x8cfg1sp0tp1",
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
def test_motif_pipeline(
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
    enable_t5_text_encoder: bool,
    use_torch_t5_text_encoder: bool,
    use_torch_clip_text_encoder: bool,
    traced: bool,
    mesh_test_id: str,
    use_cache: bool,
    is_ci_env: bool,
    model_location_generator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = MotifPipeline.create_pipeline(
        mesh_device=mesh_device,
        dit_cfg=cfg,
        dit_sp=sp,
        dit_tp=tp,
        encoder_tp=encoder_tp,
        vae_tp=vae_tp,
        enable_t5_text_encoder=True,
        use_torch_t5_text_encoder=False,
        use_torch_clip_text_encoder=False,
        num_links=num_links,
        topology=topology,
        width=width,
        height=height,
        model_checkpoint_path=model_location_generator("Motif-Technologies/Motif-Image-6B-Preview"),
    )

    # Setup CI environment
    if is_ci_env:
        if use_cache:
            monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")
        else:
            pytest.skip("Skipping. No use cache is implicitly tested with the configured non persistent cache path.")
        if traced:
            pytest.skip("Skipping traced test in CI environment. Use Performance test for detailed timing analysis.")

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
        benchmark_profiler = BenchmarkProfiler()
        with benchmark_profiler("run", iteration=0):
            images = pipeline.run_single_prompt(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                cfg_scale=5.0,
                seed=seed,
                traced=traced,
                timer=benchmark_profiler,
                timer_iteration=0,
            )

        output_filename = f"{filename_prefix}_{number}.png"
        images[0].save(output_filename)
        logger.info(f"Image saved as {output_filename}")

        logger.info(f"CLIP encoding time: {benchmark_profiler.get_duration('clip_encoding', 0):.2f}s")
        logger.info(f"T5 encoding time: {benchmark_profiler.get_duration('t5_encoding', 0):.2f}s")
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
