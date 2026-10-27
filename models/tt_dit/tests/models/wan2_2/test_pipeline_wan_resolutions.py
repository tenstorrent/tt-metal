# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""One Wan 2.2 T2V or I2V pipeline serving several resolutions.

The pipeline is built with the mesh preset's resolutions. Each is generated once, then the first
again: the repeat must reproduce the first video, which fails if switching sizes overwrote a
persistent buffer or a captured trace input. I2V conditions every video on the official example
image, checks that its first frame resembles it, and checks the size picked for an image when the
call gives none.

Environment overrides:

- ``WAN_RES_STEPS`` (default 40).
- ``WAN_RES_SIZES``: comma separated ``WxH`` to generate instead of all configured ones.
- ``WAN_RES_EXACT=WxH``: configure only that size, so it runs unpadded; with ``WAN_RES_SAVE`` the
  output can be compared against the bucketed run of the same size.
- ``WAN_RES_NO_DBCACHE=1``: generate without DBCache.
- ``WAN_RES_NO_MASK=1``: do not mask the sequence padding out of DBCache's residual diff, to
  measure what the mask costs.
- ``WAN_RES_SAVE=dir``: save each video's frames as ``<dir>/<W>x<H>_<n>.npy``.
- ``WAN_RES_VIDEO=dir``: save each video as ``<dir>/<model>_<W>x<H>_<n>.mp4``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.tt_dit.parallel.config import DiTParallelConfig, EncoderParallelConfig, VaeHWParallelConfig
from models.tt_dit.pipelines.events import SectionEnd, SectionStart
from models.tt_dit.pipelines.wan.dbcache import ExpertCacheRunner
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline, WanPipelineConfig
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V
from models.tt_dit.utils.video import export_to_video

from ....utils.test import line_params_req_exact_devices, ring_params_req_exact_devices, skip_if_unsupported_num_links
from .common import check_first_frame_matches_seed, check_output_sanity
from .test_pipeline_wan_i2v import OFFICIAL_PROMPT, official_image

# Per spatial bucket, each expert has one trace without DBCache and three with it. The DBCache test
# fits both sets of one bucket in 256 MB.
_TRACE_REGION_SIZE_PER_BUCKET = 256 * 1024 * 1024


def _parse_size(text: str) -> tuple[int, int]:
    width, height = text.lower().split("x")
    return int(width), int(height)


@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp",
    [
        [
            (2, 2),
            0,
            1,
            2,
            # One bucket, 480p.
            {**line_params_req_exact_devices, "trace_region_size": _TRACE_REGION_SIZE_PER_BUCKET},
            ttnn.Topology.Linear,
            True,
        ],
        [
            (4, 8),
            1,
            0,
            2,
            # Two buckets, 480p and 720p.
            {**ring_params_req_exact_devices, "trace_region_size": 2 * _TRACE_REGION_SIZE_PER_BUCKET},
            ttnn.Topology.Ring,
            False,
        ],
    ],
    ids=["bh_2x2sp0tp1nl2_linear", "bh_4x8sp1tp0nl2_ring"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("model", ["t2v", "i2v"])
@pytest.mark.timeout(6 * 3600)
def test_wan_resolutions(
    *,
    model: str,
    monkeypatch: pytest.MonkeyPatch,
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
) -> None:
    skip_if_unsupported_num_links(mesh_device, num_links)
    if not ttnn.device.is_blackhole():
        pytest.skip("the resolution presets are for Blackhole meshes")

    num_frames = 81
    num_inference_steps = int(os.environ.get("WAN_RES_STEPS", "40"))
    exact = os.environ.get("WAN_RES_EXACT")
    save_dir = os.environ.get("WAN_RES_SAVE")
    video_dir = os.environ.get("WAN_RES_VIDEO")

    h_factor = tuple(mesh_device.shape)[tp_axis]
    w_factor = tuple(mesh_device.shape)[sp_axis]
    config_kwargs: dict[str, Any] = {}
    if exact:
        width, height = _parse_size(exact)
        config_kwargs = {"width": width, "height": height}
    if os.environ.get("WAN_RES_NO_DBCACHE") == "1":
        config_kwargs["cache_config"] = None
    if os.environ.get("WAN_RES_NO_MASK") == "1":
        run_step = ExpertCacheRunner.run_step
        monkeypatch.setattr(
            ExpertCacheRunner, "run_step", lambda self, **kwargs: run_step(self, **{**kwargs, "valid_11N1": None})
        )

    i2v = model == "i2v"
    pipeline_class = WanPipelineI2V if i2v else WanPipeline
    pipeline = pipeline_class(
        device=mesh_device,
        config=WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            dit_parallel_config=DiTParallelConfig.from_tuples(
                cfg=(1, 0), sp=(w_factor, sp_axis), tp=(h_factor, tp_axis)
            ),
            vae_parallel_config=VaeHWParallelConfig.from_tuples(height=(h_factor, tp_axis), width=(w_factor, sp_axis)),
            encoder_parallel_config=EncoderParallelConfig.from_tuple((h_factor, tp_axis)),
            num_links=num_links,
            dynamic_load=False,
            topology=topology,
            is_fsdp=is_fsdp,
            model_type=model,
            checkpoint_name=f"Wan-AI/Wan2.2-{model.upper()}-A14B-Diffusers",
            num_frames=num_frames,
            **config_kwargs,
        ),
    )

    sizes_env = os.environ.get("WAN_RES_SIZES")
    sizes = [_parse_size(s) for s in sizes_env.split(",")] if sizes_env else list(pipeline.resolutions)
    # The first size again, after the others.
    schedule = [*sizes, sizes[0]] if len(sizes) > 1 else sizes

    if i2v:
        # Without a size, an image of a configured size's aspect ratio gets that size, among the
        # sizes of about the default's area.
        default_width, default_height = pipeline.resolutions[0]
        for width, height in pipeline.resolutions:
            if abs(width * height / (default_width * default_height) - 1) < 0.25:
                picked = pipeline._resolve_size(None, None, Image.new("RGB", (width, height)))
                assert picked == (width, height), f"a {width}x{height} image got size {picked}"
        image = official_image()
        prompt = OFFICIAL_PROMPT
    else:
        image = None
        prompt = (
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
        )

    outputs: list[tuple[tuple[int, int], np.ndarray]] = []
    report = [f"{'size':>10}{'padded N':>10}{'denoise(s)':>12}{'vae(s)':>8}{'cached':>8}"]
    for n, (width, height) in enumerate(schedule):
        durations: dict[str, float] = {}
        starts: dict[str, float] = {}

        def on_event(event) -> None:
            if isinstance(event, SectionStart):
                starts[event.name] = time.perf_counter()
            elif isinstance(event, SectionEnd) and event.name in starts:
                durations[event.name] = time.perf_counter() - starts.pop(event.name)

        logger.info(f"=== {width}x{height} ({n + 1}/{len(schedule)}) ===")
        with torch.no_grad():
            frames = pipeline(
                prompts=[prompt],
                num_inference_steps=num_inference_steps,
                seed=0,
                output_type="uint8",
                traced=True,
                on_event=on_event,
                width=width,
                height=height,
                image_prompt=image,
            )
        frames = np.asarray(frames[0])
        check_output_sanity(frames, num_frames=num_frames, height=height, width=width)
        if image is not None:
            check_first_frame_matches_seed(frames, seed_image=image, width=width, height=height)
        outputs.append(((width, height), frames))

        cached = sum(len(b) for s in pipeline.cache_summary() for b in s["cached_steps"])
        for s in pipeline.cache_summary():
            logger.info(f"  [{s['name']}] cached_steps={s['cached_steps']}")
            for branch, diffs in enumerate(s["residual_diffs"]):
                logger.info(f"    branch {branch} diffs: " + ", ".join(f"{k}:{v:.3f}" for k, v in diffs.items()))
            if s["profile_ms"]:
                logger.info(
                    f"  [{s['name']}] profile (ms, mean per branch-call): "
                    + ", ".join(f"{k}={v:.2f}" for k, v in s["profile_ms"].items())
                )
        report.append(
            f"{f'{width}x{height}':>10}{pipeline._padded_sequence_lengths[width, height]:>10}"
            f"{durations.get('denoising', float('nan')):>12.1f}{durations.get('vae', float('nan')):>8.1f}{cached:>8}"
        )

        if save_dir and int(ttnn.distributed_context_get_rank()) == 0:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            np.save(Path(save_dir) / f"{width}x{height}_{n}.npy", frames)
        if video_dir and int(ttnn.distributed_context_get_rank()) == 0:
            Path(video_dir).mkdir(parents=True, exist_ok=True)
            export_to_video(frames, str(Path(video_dir) / f"{model}_{width}x{height}_{n}.mp4"), fps=16)

    logger.info("\n".join(report))
    pipeline.release_traces()

    if len(schedule) > 1:
        (_, first), (_, again) = outputs[0], outputs[-1]
        assert np.array_equal(first, again), (
            f"regenerating {schedule[0][0]}x{schedule[0][1]} after the other sizes changed the video "
            f"(max abs diff {np.abs(first.astype(int) - again.astype(int)).max()})"
        )
