# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the FLUX.1-Kontext-dev editing pipeline.

Mirrors ``test_pipeline_flux1.py`` but supplies a reference image and a
single-line edit instruction. Start from the ``1x2sp0tp1`` / ``2x4sp0tp1``
configs (sp reduces to a plain concat/slice; see pipeline_flux1_kontext §4).

Provide a reference image via the ``KONTEXT_INPUT_IMAGE`` env var; otherwise a
synthetic gradient image is generated so the test is self-contained.
"""

import os

import pytest
from loguru import logger
from PIL import Image

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

from ....parallel.config import DiTParallelConfig, EncoderParallelConfig, VAEParallelConfig
from ....pipelines.events import profiler_event_callback
from ....pipelines.flux1.pipeline_flux1_kontext import Flux1KontextPipeline, Flux1KontextPipelineConfig


def _load_reference_image(width: int, height: int) -> Image.Image:
    path = os.environ.get("KONTEXT_INPUT_IMAGE")
    if path and os.path.isfile(path):
        return Image.open(path).convert("RGB").resize((width, height))
    # Synthetic fallback: horizontal RGB gradient.
    img = Image.new("RGB", (width, height))
    px = img.load()
    for x in range(width):
        for y in range(height):
            px[x, y] = (int(255 * x / width), int(255 * y / height), 128)
    return img


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "l1_small_size": 32768,
            "trace_region_size": 50000000,
            "require_exact_physical_num_devices": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    ("width", "height", "num_inference_steps"),
    [(1024, 1024, 28)],
    ids=["kontext_dev"],
)
@pytest.mark.parametrize(
    ("mesh_device", "sp", "tp", "encoder_tp", "vae_tp", "topology", "num_links", "mesh_test_id"),
    [
        # Start with sp=1 (tp only): concat/slice reduces to the simple path.
        pytest.param((1, 2), (1, 0), (2, 1), (2, 1), (2, 1), ttnn.Topology.Linear, 2, "1x2sp0tp1", id="1x2sp0tp1"),
        pytest.param((2, 4), (2, 0), (4, 1), (4, 1), (4, 1), ttnn.Topology.Linear, 1, "2x4sp0tp1", id="2x4sp0tp1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("traced", [False], ids=["untraced"])
def test_flux1_kontext_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
    width: int,
    height: int,
    num_inference_steps: int,
    sp: tuple[int, int],
    tp: tuple[int, int],
    encoder_tp: tuple[int, int],
    vae_tp: tuple[int, int],
    topology: ttnn.Topology,
    num_links: int,
    mesh_test_id: str,
    traced: bool,
    model_location_generator,
    is_ci_env: bool,
) -> None:
    if is_ci_env and traced:
        pytest.skip("Skipping traced test in CI environment.")

    parallel_config = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=sp, tp=tp)
    encoder_parallel_config = EncoderParallelConfig.from_tuple(encoder_tp)
    vae_parallel_config = VAEParallelConfig.from_tuple(vae_tp)

    logger.info(f"Mesh device shape: {mesh_device.shape}")

    pipeline = Flux1KontextPipeline(
        device=mesh_device,
        config=Flux1KontextPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            dit_parallel_config=parallel_config,
            encoder_parallel_config=encoder_parallel_config,
            vae_parallel_config=vae_parallel_config,
            num_links=num_links,
            topology=topology,
            width=width,
            height=height,
            checkpoint_name=model_location_generator("black-forest-labs/FLUX.1-Kontext-dev"),
        ),
    )

    reference_image = _load_reference_image(width, height)
    prompt = "Add a small red hat to the subject"

    benchmark_profiler = BenchmarkProfiler()
    with benchmark_profiler("run", iteration=0):
        images = pipeline(
            image=reference_image,
            prompts=[prompt],
            num_inference_steps=num_inference_steps,
            guidance_scale=3.5,
            seed=0,
            traced=traced,
            vae_traced=False,
            encoder_traced=False,
            on_event=profiler_event_callback(benchmark_profiler, 0),
        )

    assert len(images) == 1
    out = f"flux_kontext_{width}_{height}_{mesh_test_id}.png"
    images[0].save(out)
    logger.info(f"Edited image saved as {out}")
    logger.info(f"Total pipeline time: {benchmark_profiler.get_duration('total', 0):.2f}s")
