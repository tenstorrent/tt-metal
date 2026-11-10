# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Sample videos from a FastVideo PDD export on the H3 benchmark's reference cases (REF2VA-L/M/H, SIZE-V).

`MINIMAX_H3_PDD_EXPORT` is the export, `MINIMAX_H3_MODEL_PATH` the base model, `H3BENCH_ASSETS` the benchmark's
asset pack; `H3BENCH_CASES` (comma separated) narrows the cases and `PDD_ATTENTION` is `vsa` (default) or `dense`.
Each case runs a compile pass and then a timed one; a failing case is recorded and the rest still run. Stages are
not co-resident, so timings include per-request component reloads.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from loguru import logger
from PIL import Image

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

from ....pipelines.events import profiler_event_callback
from ....pipelines.minimax_h3.packing_ref2va import (
    MiniMaxH3Reference,
    decode_reference_audio,
    reference_from_video_file,
)
from ....pipelines.minimax_h3.pipeline_minimax_h3_pdd import MiniMaxH3PDDPipeline
from .common_av import artifact_dir, log_pipeline_perf, to_uint8_frames, weights_dir, write_artifacts
from .test_pipeline_ref2va_minimax_h3 import MESHES

WIDTH, HEIGHT, FPS, SEED = 1344, 768, 24, 42
# 17n + 5 frame counts nearest the benchmark's 5 / 10 / 15 s.
FRAMES = {5: 124, 10: 243, 15: 362}

# The benchmark's cases.json, reference order images -> videos -> audio.
CASES = {
    "REF2VA-L": dict(
        seconds=5,
        prompt="prompt_min.txt",
        images=["img_min_256px_scientist.jpg"],
        videos=["vid_min_2s_city_skyline.mp4"],
        audios=["aud_min_2s_score.wav"],
    ),
    "REF2VA-M": dict(
        seconds=10,
        prompt="prompt_std.txt",
        images=["img_std_city_skyline.jpg", "img_std_old_man_portrait.jpg", "img_std_three_people.jpg"],
        videos=["vid_min_2s_city_skyline.mp4", "vid_std_5s_man_talking.mp4"],
        audios=["aud_min_2s_score.wav", "aud_std_5s_score.wav"],
    ),
    "REF2VA-H": dict(
        seconds=15,
        prompt="prompt_max_7000chars.txt",
        images=[
            "img_std_city_skyline.jpg",
            "img_std_old_man_portrait.jpg",
            "img_std_three_people.jpg",
            "img_std_robot_street.jpg",
            "img_std_young_man_canal.jpg",
            "img_std_earth_from_iss.jpg",
        ],
        videos=["vid_ns2_city.mp4", "vid_ns5_man.mp4", "vid_ns7_robot.mp4"],
        audios=["aud_min_2s_score.wav", "aud_std_5s_score.wav", "aud_max7_score.wav"],
    ),
    "SIZE-V": dict(
        seconds=5, prompt="prompt_std.txt", images=[], videos=["vid_max_8s_47mb_robot_street.mp4"], audios=[]
    ),
}


def _references(assets: Path, case: dict) -> list[MiniMaxH3Reference]:
    references = [MiniMaxH3Reference(image=Image.open(assets / name).convert("RGB")) for name in case["images"]]
    references += [reference_from_video_file(assets / name) for name in case["videos"]]
    for name in case["audios"]:
        waveform, sample_rate = decode_reference_audio(assets / name)
        references.append(MiniMaxH3Reference(audio=waveform, sample_rate=sample_rate))
    return references


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESHES, indirect=["mesh_device", "device_params"])
def test_pdd_bench_samples(mesh_device, reset_seeds):
    export = Path(os.environ.get("MINIMAX_H3_PDD_EXPORT", ""))
    assets = Path(os.environ.get("H3BENCH_ASSETS", ""))
    if not export.is_dir() or not assets.is_dir():
        pytest.skip("set MINIMAX_H3_PDD_EXPORT and H3BENCH_ASSETS")
    attention = os.environ.get("PDD_ATTENTION", "vsa")
    selected = [name for name in os.environ.get("H3BENCH_CASES", ",".join(CASES)).split(",") if name]

    # Reference videos and audio bring their encoders onto a 4x8 already full with the DiT and the text encoder;
    # evicting per stage fits them, at a reload per request.
    pipeline = MiniMaxH3PDDPipeline.create_pipeline(
        warmup=False,
        coresident=False,
        vae_output_type="float",
        mesh_device=mesh_device,
        weights_dir=weights_dir("text_encoder", "vae", "audio_vae"),
        task="ref2va",
        pdd_export_dir=export,
        vsa_config=MiniMaxH3PDDPipeline.export_vsa_config(export) if attention == "vsa" else None,
    )
    steps = pipeline.pdd.num_forwards + 1
    directory = artifact_dir("h3_pdd_bench")
    results = []
    for name in selected:
        case = CASES[name]
        num_frames = FRAMES[case["seconds"]]
        prompt = (assets / case["prompt"]).read_text().strip()
        kwargs = dict(
            references=_references(assets, case), num_frames=num_frames, height=HEIGHT, width=WIDTH, seed=SEED
        )
        stem = f"{name}_{attention}_{num_frames}f"
        try:
            t0 = time.time()
            with pipeline.quiet():
                pipeline(prompt, num_inference_steps=steps, **kwargs)
            cold = time.time() - t0
            profiler = BenchmarkProfiler()
            with profiler("run", iteration=0):
                output = pipeline(
                    prompt, num_inference_steps=steps, on_event=profiler_event_callback(profiler, 0), **kwargs
                )
                ttnn.synchronize_device(pipeline.mesh_device)
            frames = to_uint8_frames(output)
            paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, directory, stem=stem)
            log_pipeline_perf(
                profiler,
                label=f"pdd {name} {attention}",
                pipeline=pipeline,
                num_forwards=pipeline.pdd.num_forwards,
                width=WIDTH,
                height=HEIGHT,
                num_frames=num_frames,
                fps=FPS,
                num_inference_steps=steps,
            )
            results.append(
                dict(
                    case=name,
                    ok=True,
                    num_frames=num_frames,
                    references=len(kwargs["references"]),
                    cold_s=round(cold, 2),
                    warm_s=round(profiler.get_duration("run", 0), 2),
                    denoise_s=round(profiler.get_duration("denoising", 0), 2),
                    files=sorted(str(p) for p in paths.values()) if isinstance(paths, dict) else str(paths),
                )
            )
        except Exception as error:
            logger.exception(f"{name} failed")
            results.append(dict(case=name, ok=False, error=f"{type(error).__name__}: {error}"[:2000]))
        (directory / f"summary_{attention}.json").write_text(json.dumps(results, indent=2))
        logger.info(f"BENCH_RESULT {json.dumps(results[-1])}")

    assert all(result["ok"] for result in results), [r for r in results if not r["ok"]]
