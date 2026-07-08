# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

import pytest
from loguru import logger
from PIL import Image

import ttnn
from models.common.utility_functions import is_blackhole
from models.perf.benchmarking_utils import BenchmarkProfiler

from ....pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from ....utils.test import line_params, line_params_8k, ring_params, ring_params_8k

# Flux2 VAE uses conv2d which needs L1_SMALL buffers.
line_params_flux2 = {**line_params, "l1_small_size": 65536}
ring_params_flux2 = {**ring_params, "l1_small_size": 65536}
ring_params_8k_flux2 = {**ring_params_8k, "l1_small_size": 65536}
line_params_8k_flux2 = {**line_params_8k, "l1_small_size": 65536}


def _load_init_image(*, width: int, height: int, mesh_device: ttnn.MeshDevice) -> Image.Image:
    """Load a real reference image for img2img (Flux2 conditions on encoded reference latents)."""
    arch = "bh" if is_blackhole() else "wh"
    mesh_tag = "x".join(str(s) for s in mesh_device.shape)
    ref_path = f"flux2_{arch}_{width}_{height}_{mesh_tag}_0.png"
    if os.path.isfile(ref_path):
        logger.info(f"Using reference image: {ref_path}")
        return Image.open(ref_path).convert("RGB")

    logger.warning(f"Reference image not found at {ref_path}; using synthetic gradient")
    gradient = Image.new("RGB", (width, height))
    pixels = gradient.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (x * 255 // width, y * 255 // height, 128)
    return gradient


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), False)],
)
@pytest.mark.parametrize(
    "device_params",
    [line_params_flux2],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height", "num_inference_steps"), [(1024, 1024, 12)])
@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, encoder_tp_axis, vae_tp_axis, topology, num_links, is_fsdp, dynamic_load, traced",
    [
        [(1, 8), 0, 1, 1, 1, ttnn.Topology.Linear, 1, False, True, False],
        [(4, 8), 0, 1, 1, 1, ttnn.Topology.Linear, 4, True, False, True],
        [(4, 8), 0, 1, 1, 1, ttnn.Topology.Linear, 2, False, False, True],
    ],
    ids=[
        "1x8tp1",
        "wh_4x8",
        "bh_4x8",
    ],
    indirect=["mesh_device"],
)
def test_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
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
    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-dev"),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        encoder_tp_axis=encoder_tp_axis,
        vae_tp_axis=vae_tp_axis,
        vae_h_axis=1 - vae_tp_axis,
        vae_w_axis=None,  # 1-vae_tp_axis,
        num_links=num_links,
        topology=topology,
        width=width,
        height=height,
        is_fsdp=is_fsdp,
        dynamic_load=dynamic_load,
        trace_warmup=traced,
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
    filename_prefix = f"flux2_{arch}_{width}_{height}_{mesh_tag}"
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
    "device_params",
    [line_params_flux2],
    indirect=True,
)
@pytest.mark.parametrize(("width", "height", "num_inference_steps"), [(1024, 1024, 12)])
@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, encoder_tp_axis, vae_tp_axis, topology, num_links, is_fsdp, dynamic_load",
    [
        [(4, 8), 0, 1, 1, 1, ttnn.Topology.Linear, 4, True, False],
    ],
    ids=["wh_4x8"],
    indirect=["mesh_device"],
)
def test_pipeline_img2img(
    tt_dit_cache_dir,
    *,
    mesh_device: ttnn.MeshDevice,
    width: int,
    height: int,
    num_inference_steps: int,
    sp_axis: int,
    tp_axis: int,
    encoder_tp_axis: int,
    vae_tp_axis: int,
    topology: ttnn.Topology,
    num_links: int,
    is_fsdp: bool,
    dynamic_load: bool,
    model_location_generator,
) -> None:
    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-dev"),
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
        is_fsdp=is_fsdp,
        dynamic_load=dynamic_load,
        trace_warmup=False,
    )

    init_image = _load_init_image(width=width, height=height, mesh_device=mesh_device)
    prompt = "winter night"

    benchmark_profiler = BenchmarkProfiler()
    ttnn.synchronize_device(mesh_device)
    with benchmark_profiler("run", iteration=0):
        images = pipeline(
            prompts=[prompt],
            image=init_image,
            num_inference_steps=num_inference_steps,
            seed=0,
            traced=False,
            profiler=benchmark_profiler,
            profiler_iteration=0,
        )
        ttnn.synchronize_device(mesh_device)

    def _dur(step: str) -> float:
        try:
            return benchmark_profiler.get_duration(step, 0)
        except Exception:
            return float("nan")

    total = _dur("run")
    print("\n" + "=" * 72)
    print(f"FLUX2 IMG2IMG PERF (4x8 WH galaxy, {width}x{height}, {num_inference_steps} steps)")
    print("-" * 72)
    print(f"  {'prompt+encoder':24} | {_dur('encoder') * 1e3:10.3f} ms")
    print(f"  {'VAE image encode':24} | {_dur('vae_encode') * 1e3:10.3f} ms")
    print(f"  {'denoising (total)':24} | {_dur('denoising') * 1e3:10.3f} ms")
    print(f"  {'VAE decode':24} | {_dur('vae') * 1e3:10.3f} ms")
    print("-" * 72)
    print(f"  {'total image-to-image':24} | {total * 1e3:10.3f} ms")
    print("=" * 72)
    logger.info(f"img2img total: {total:.3f}s (vae_encode {_dur('vae_encode'):.3f}s)")

    arch = "bh" if is_blackhole() else "wh"
    mesh_tag = "x".join(str(s) for s in mesh_device.shape)
    output_filename = f"flux2_{arch}_{width}_{height}_{mesh_tag}_img2img_0.png"
    images[0].save(output_filename)
    logger.info(f"Image saved as {output_filename}")
