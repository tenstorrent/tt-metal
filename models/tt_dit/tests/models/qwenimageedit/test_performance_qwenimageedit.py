# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import pytest
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

from ....pipelines.qwenimageedit.pipeline_qwenimageedit import QwenImageEditPipeline


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 47000000}],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height", "num_inference_steps"), [(1024, 1024, 50)])
@pytest.mark.parametrize(
    "mesh_device, cfg, sp, tp, encoder_tp, vae_tp, topology, num_links",
    [
        [(2, 4), (2, 0), (1, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1],
        [(4, 8), (2, 1), (4, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 4],
    ],
    ids=[
        "2x4cfg2sp1tp4",
        "4x8cfg2sp4tp4",
    ],
    indirect=["mesh_device"],
)
def test_qwenimageedit_performance(
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
    is_ci_env: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if is_ci_env:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", "/tmp/TT_DIT_CACHE")

    pipeline = QwenImageEditPipeline.create_pipeline(
        mesh_device=mesh_device,
        dit_cfg=cfg,
        dit_sp=sp,
        dit_tp=tp,
        encoder_tp=encoder_tp,
        vae_tp=vae_tp,
        use_torch_text_encoder=False,
        use_torch_vae_decoder=False,
        num_links=num_links,
        topology=topology,
        width=width,
        height=height,
    )

    profiler = BenchmarkProfiler()

    edit_prompt = (
        "Transform the building facade into a cyberpunk neon-lit structure " "while preserving the architectural shape."
    )

    num_warmup = 1
    num_iterations = 3

    for i in range(num_warmup):
        logger.info(f"Warmup iteration {i + 1}/{num_warmup}")
        pipeline.run_single_edit(
            prompt=edit_prompt,
            image=None,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=4.0,
            seed=42,
            traced=True,
            profiler=None,
        )

    for i in range(num_iterations):
        logger.info(f"Benchmark iteration {i + 1}/{num_iterations}")
        pipeline.run_single_edit(
            prompt=edit_prompt,
            image=None,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=4.0,
            seed=42 + i,
            traced=True,
            profiler=profiler,
            profiler_iteration=i,
        )

    total_times = [profiler.get_duration("total", i) for i in range(num_iterations)]
    encoder_times = [profiler.get_duration("encoder", i) for i in range(num_iterations)]
    denoising_times = [profiler.get_duration("denoising", i) for i in range(num_iterations)]
    vae_times = [profiler.get_duration("vae", i) for i in range(num_iterations)]

    avg_total = sum(total_times) / len(total_times)
    avg_encoder = sum(encoder_times) / len(encoder_times)
    avg_denoising = sum(denoising_times) / len(denoising_times)
    avg_vae = sum(vae_times) / len(vae_times)

    logger.info(f"Average total time: {avg_total:.2f}s")
    logger.info(f"Average encoder time: {avg_encoder:.2f}s")
    logger.info(f"Average denoising time ({num_inference_steps} steps): {avg_denoising:.2f}s")
    logger.info(f"Average VAE decode time: {avg_vae:.2f}s")
