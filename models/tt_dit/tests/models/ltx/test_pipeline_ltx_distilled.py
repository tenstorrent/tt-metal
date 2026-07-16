# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import functools
import itertools
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import (
    TEMPORAL_COMPRESSION,
    LTXDistilledPipeline,
    pixel_to_latent_frame,
)
from models.tt_dit.utils.ltx import (
    DEFAULT_LTX_PROMPT,
    default_ltx_checkpoint,
    default_ltx_gemma,
    print_ltx_timing_table,
)
from models.tt_dit.utils.patchifiers import AudioLatentShape, VideoPixelShape
from models.tt_dit.utils.test import line_params, ring_params
from models.tt_dit.utils.vbench import assert_vbench_quality


def _apply_local_iter_env() -> None:
    # setdefault from a local dev env file so pipeline iteration toggles (resident DiT,
    # warmup/gate toggles) apply to a bare local run without the broker's -e handoff. Kernel
    # prewarm needs nothing here: the framework runs it by default and self-bootstraps its manifest
    # under the JIT cache root, for the normal and profiler build_keys alike. Explicit env always
    # wins and an absent file is a no-op, so CI (this test is checkpoint-gated) and the committed
    # VBench/CLIP gate defaults are untouched. Flat "KEY: VALUE" lines; override path via LTX_ITER_ENV.
    path = os.environ.get("LTX_ITER_ENV") or os.path.join(
        os.environ.get("TT_METAL_HOME", ""), "tmp", "ltx_env_prewarm.yaml"
    )
    if not os.path.isfile(path):
        return
    applied = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = val.strip().strip('"').strip("'")
                applied.append(key)
    if applied:
        logger.info(f"LTX iter env: {len(applied)} defaults from {path}: {', '.join(applied)}")


_apply_local_iter_env()


# Trace region for LTX_TRACED=1. Holds both stage traces' command streams (s1 + larger-seq
# s2); measured need is ~236 MB at 1080p (get_trace_buffers_size), so 300 MB gives headroom.
# l1_small_size: native ttnn.conv1d (the depthwise audio taps) runs an UntilizeWithHalo gather
# whose sharding/config tensors allocate from the dedicated L1_SMALL pool; it defaults to 0, which
# OOMs the vocoder. 32 KB matches the audio component tests; the audio submesh (create_submesh)
# inherits it from the parent mesh.
# trace_region_size holds both stage traces' command streams (~236 MB measured at 1080p/145f;
# 500 MB gives headroom). Env-tunable (default unchanged) so a memory-tight long clip (15s/20s,
# where the stage-2 activation footprint fills DRAM) can drop it to 0 for an UNTRACED pass,
# reclaiming ~500 MB of general DRAM. Only meaningful with LTX_TRACED=0 (no trace is captured).
_LTX_TRACE_REGION = int(os.environ.get("LTX_TRACE_REGION", "500000000"))
ring_trace_params = {**ring_params, "trace_region_size": _LTX_TRACE_REGION, "l1_small_size": 32768}
line_trace_params = {**line_params, "trace_region_size": _LTX_TRACE_REGION, "l1_small_size": 32768}


# Default-off: full AV gen needs the real LTX checkpoint + Gemma, so it skips in the default suite
# (no checkpoint present). Runs the same prompt as the girl audio fixture (DEFAULT_LTX_PROMPT — the
# "young woman with a guitar sings Doo-be-doo" clip the audio tests use), so e2e and audio stay aligned.
@pytest.mark.skipif(
    not os.path.exists(default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")),
    reason="needs the LTX checkpoint (set LTX_CHECKPOINT to a local .safetensors)",
)
@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), True)],
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 2), (2, 2), 0, 1, 2, False, line_params, ttnn.Topology.Linear, True],
        [(2, 4), (2, 4), 0, 1, 1, True, line_params, ttnn.Topology.Linear, True],
        # BH on 2x4. trace_params (not bare line_params): LTX_TRACED needs the trace region, and the
        # native conv1d audio taps need the L1_SMALL pool (bare line_params leaves it 0 -> vocoder OOM).
        [(2, 4), (2, 4), 1, 0, 2, True, line_trace_params, ttnn.Topology.Linear, False],
        # WH (ring) on 4x8. Requires increased worker_l1_size to avoid code-size error in RingAttention.
        [(4, 8), (4, 8), 1, 0, 4, True, {"worker_l1_size": 1344544, **ring_params}, ttnn.Topology.Ring, True],
        # BH (linear) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, line_params, ttnn.Topology.Linear, False],
        # BH (ring) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, ring_trace_params, ttnn.Topology.Ring, False],
        [(4, 32), (4, 32), 1, 0, 2, False, ring_params, ttnn.Topology.Ring, False],
    ],
    ids=[
        "2x2sp0tp1",
        "2x4sp0tp1",
        "bh_2x4sp1tp0",
        "wh_4x8sp1tp0",
        "bh_4x8sp1tp0_linear",
        "bh_4x8sp1tp0_ring",
        "bh_4x32sp1tp0",
    ],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_distilled(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    is_fsdp,
    no_prompt,
):
    """LTX-2.3 distilled 2-stage AV pipeline."""
    ckpt = default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")
    gemma = default_ltx_gemma()

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    # Stage-2 output target: 1080p @ 24fps, ~6s. Constraints from the pipeline:
    #   - height/width must be divisible by 64 (asserted at the top of generate(),
    #     so half-res stage-1 is divisible by 32 for the VAE).
    #     1920 is already % 64; 1080 is not → round up to 1088 (next multiple of 64).
    #     If you need exactly 1080 lines, post-crop the 8 extra rows.
    #   - num_frames must satisfy (num_frames - 1) % 8 == 0 so the VAE decoder maps
    #     latent_frames → num_frames frames exactly. 144 (== 6s × 24fps) doesn't
    #     satisfy this; 145 does (19 latent frames → 145 decoded frames ≈ 6.04s).
    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    height = int(os.environ.get("HEIGHT", "1088"))
    width = int(os.environ.get("WIDTH", "1920"))

    run_warmup = os.environ.get("RUN_WARMUP", "0") in ("1", "true", "True")
    traced = os.environ.get("LTX_TRACED", "0") in ("1", "true", "True")

    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=ckpt,
        gemma_path=gemma,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=run_warmup,
        traced=traced,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    prompt = os.environ.get("PROMPT", DEFAULT_LTX_PROMPT)

    image_path = os.environ.get("LTX_I2V_IMAGE")
    images = None
    if image_path:
        if not os.path.exists(image_path):
            pytest.skip(f"LTX_I2V_IMAGE set but file not found: {image_path}")
        strength = float(os.environ.get("LTX_I2V_STRENGTH", "1.0"))
        images = [(image_path, 0, strength)]

    def run(*, prompt, number, seed):
        output_filename = os.environ.get("OUTPUT_PATH", f"ltx_av_fast_{width}x{height}_{number}.mp4")
        logger.info(f"Running LTX AV Fast: '{prompt[:80]}...'")
        logger.info(f"Config: {height}x{width}, {num_frames} frames")
        if images:
            logger.info(f"I2V: conditioning image {images[0][0]} (strength={images[0][2]})")

        if int(ttnn.distributed_context_get_rank()) != 0:
            logger.info(f"Skipping generation on rank {ttnn.distributed_context_get_rank()}")
            return

        pipeline.generate(
            prompt,
            output_path=output_filename,
            images=images,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
        )
        logger.info(f"Saved video to: {output_filename}")
        print_ltx_timing_table(
            pipeline,
            label="LTX DISTILLED",
            num_frames=num_frames,
            height=height,
            width=width,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            topology=topology,
            output_path=output_filename,
            prompt=prompt,
        )

    # Output resolutions the served pipeline supports (its quality/resolution dropdown), each
    # rounded up to the generate() %64 constraint. Every supported height must have a calibrated
    # vbench entry below; the coverage assert ties the two so adding a resolution here without
    # thresholds fails loud instead of KeyError-ing the gate at that height.
    SUPPORTED_RESOLUTIONS = {(1280, 704), (1920, 1088)}  # (width, height)

    # VBench floors per output height, each a small headroom below that resolution's measured
    # scores. 704 (720p) is a coarser render than 1088 (1080p): its imaging_quality measures far
    # lower, so its floor sits accordingly — that is the render, not a defect.
    vbench_thresholds_by_height = {
        # Measured 720p (fast, traced replay, girl-singing prompt): subject 0.948, background
        # 0.957, motion 0.995, dynamic 1.0, imaging 0.332. subject/background/motion floors keep
        # the headroom the 1088 entry uses below its own fast-tier measured (~0.03-0.06). imaging
        # sits ~0.03 below the measured 720p value: 720p renders coarser, and 1088's 0.645 imaging
        # floor reflects a higher-quality tier (fast-tier 1088 imaging measures ~0.46) so it does
        # not transfer to this fast-tier 720p entry.
        704: {
            "subject_consistency": 0.89,
            "background_consistency": 0.92,
            "motion_smoothness": 0.95,
            "dynamic_degree": 1.0,
            "imaging_quality": 0.30,
        },
        1088: {
            "subject_consistency": 0.92,
            "background_consistency": 0.93,
            "motion_smoothness": 0.955,
            "dynamic_degree": 1.0,
            "imaging_quality": 0.645,
        },
    }

    _uncovered = {h for _w, h in SUPPORTED_RESOLUTIONS} - vbench_thresholds_by_height.keys()
    assert not _uncovered, (
        f"no vbench thresholds calibrated for supported height(s) {sorted(_uncovered)}; add an "
        f"entry to vbench_thresholds_by_height for every resolution in {sorted(SUPPORTED_RESOLUTIONS)}"
    )

    # RUN_VBENCH=0 skips the quality gate (e.g. perf-only iteration); defaults on so CI gates.
    run_vbench = os.environ.get("RUN_VBENCH", "1") in ("1", "true", "True")
    # RUN_CLIP=0 skips the CLIP prompt-alignment gate; defaults on (mirrors the wan2.2 test).
    run_clip = os.environ.get("RUN_CLIP", "1") in ("1", "true", "True")

    def check_output_with_vbench(prompt, number):
        if not run_vbench:
            logger.info("RUN_VBENCH=0, skipping VBench quality gate")
            return
        if int(ttnn.distributed_context_get_rank()) == 0:
            output_filename = os.environ.get("OUTPUT_PATH", f"ltx_av_fast_{width}x{height}_{number}.mp4")
            if height not in vbench_thresholds_by_height:
                raise AssertionError(
                    f"no vbench thresholds calibrated for height {height}; calibrated heights are "
                    f"{sorted(vbench_thresholds_by_height)}. Add an entry to vbench_thresholds_by_height."
                )
            thresholds = vbench_thresholds_by_height[height]
            assert_vbench_quality(output_filename, prompt=prompt, thresholds=thresholds)

    def check_output_with_clip(prompt, number, clip_threshold=None):
        # Mirrors wan2.2's check_output_with_clip: sample ~8 evenly-spaced frames, score each
        # against the prompt with CLIP, assert the mean clears a floor. LTX writes the video to
        # disk (generate() returns only the path), so frames are read back from the mp4.
        if not run_clip:
            logger.info("RUN_CLIP=0, skipping CLIP score check")
            return
        if int(ttnn.distributed_context_get_rank()) != 0:
            return
        try:
            from decord import VideoReader
            from PIL import Image

            from models.tt_dit.tests.dataset_eval.clip_encoder import CLIPEncoder
        except ImportError as e:
            # Never silently pass (mirrors assert_vbench_quality): this gate is the only thing that
            # checks the render matches its prompt, and returning here reports success for a check
            # that never ran. Set RUN_CLIP=0 to skip it deliberately.
            raise RuntimeError(f"CLIP prompt-alignment gate requested but its deps are missing ({e})") from e

        # LTX baseline mean ~31.3 (vs wan2.2's ~37): the LTX prompt exceeds CLIP's 77-token limit
        # so only its head is scored, and frames are read back re-encoded (CRF=25). 28.0 leaves
        # ~3pt headroom for run-to-run variance. Override with CLIP_THRESHOLD.
        threshold = float(os.environ.get("CLIP_THRESHOLD", "28.0")) if clip_threshold is None else clip_threshold
        output_filename = os.environ.get("OUTPUT_PATH", f"ltx_av_fast_{width}x{height}_{number}.mp4")

        vr = VideoReader(output_filename)
        total_frames = len(vr)
        indices = np.linspace(0, total_frames - 1, min(8, total_frames), dtype=int)

        clip_encoder = CLIPEncoder()
        scores = []
        for i in indices:
            frame = vr[int(i)].asnumpy()  # (H, W, 3) uint8 RGB
            pil_img = Image.fromarray(frame.astype(np.uint8))
            score = clip_encoder.get_clip_score(prompt, pil_img).item() * 100.0
            scores.append(score)

        clip_min = min(scores)
        clip_max = max(scores)
        clip_mean = sum(scores) / len(scores)
        logger.info(f"CLIP scores: min={clip_min:.2f}, max={clip_max:.2f}, mean={clip_mean:.2f}")

        assert clip_mean >= threshold, (
            f"Mean CLIP score {clip_mean:.2f} is below threshold {threshold:.2f}. "
            f"Per-frame scores: {[f'{s:.2f}' for s in scores]}"
        )

    if no_prompt:
        seed = int(os.environ.get("SEED", "10"))
        run(prompt=prompt, number=0, seed=seed)
        # Traced: gen #0 captures (lazily, on first step of each stage); gen #1 is pure
        # replay — its Stage 1/2 denoise times are the steady-state measurement.
        if traced:
            logger.info("=== traced steady-state pass (gen #1, pure replay) ===")
            run(prompt=prompt, number=1, seed=seed)
            check_output_with_clip(prompt, 1)
            check_output_with_vbench(prompt, 1)
        else:
            check_output_with_clip(prompt, 0)
            check_output_with_vbench(prompt, 0)
    else:
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, number=i, seed=i)
            check_output_with_clip(prompt, i)
            check_output_with_vbench(prompt, i)

    if traced:
        pipeline.release_traces()


@functools.lru_cache(maxsize=1)
def _ffmpeg():
    """An ffmpeg executable. Prefer the system binary; fall back to imageio-ffmpeg's
    bundled static build (pip-installed on demand) so a host without ffmpeg still runs
    the frame-decode checks. The static build is real ffmpeg, so decoded pixels — and
    thus the calibrated seam thresholds — are unchanged."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError:
        pip_install = [sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg"]
        if subprocess.run(pip_install).returncode != 0:
            # uv-managed venvs ship without pip; bootstrap it from the stdlib, then retry.
            subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
            subprocess.run(pip_install, check=True)
        import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ffmpeg_frames(path, n):
    """Sample ``n`` luma frames evenly across ``path`` (via ffmpeg) as float ndarrays."""
    import io

    from PIL import Image

    frames = []
    for t in np.linspace(0.1, 5.5, n):
        raw = subprocess.run(
            [
                _ffmpeg(),
                "-v",
                "error",
                "-ss",
                f"{t}",
                "-i",
                path,
                "-vframes",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ],
            capture_output=True,
        ).stdout
        if raw:
            frames.append(np.asarray(Image.open(io.BytesIO(raw)).convert("L")).astype(float))
    return frames


def _image_seam_score(path):
    """Grid-seam strength (same V/H metric as _temporal_seam_score) on a SINGLE still image,
    so an output seam can be compared against the conditioning image's OWN inherent seam — the
    metric is content-sensitive, and a busy still can score >1.3 with no device seam present."""
    from PIL import Image

    f = np.asarray(Image.open(path).convert("L")).astype(float)
    h, w = f.shape
    gx = np.abs(np.diff(f, axis=1)).mean(0)
    gy = np.abs(np.diff(f, axis=0)).mean(1)
    v = float(np.mean([gx[round(w * i / 4)] for i in (1, 2, 3)]) / np.median(gx))
    hh = float(gy[round(h / 2)] / np.median(gy))
    return v, hh


def _temporal_seam_score(path):
    """Grid-seam strength at the 2x4 mesh boundaries (W/4,W/2,3W/4 vertical; H/2 horizontal),
    isolated from moving content via the per-column/row TEMPORAL MEDIAN of the gradient (the
    seam is static at fixed lines; single-frame metrics are content-confounded). Clean ~<=1;
    a baked seam pushes the boundary ratio well above (gridded measured V=1.5, H=2.4)."""
    fs = _ffmpeg_frames(path, 16)
    assert fs, f"no frames decoded from {path}"
    h, w = fs[0].shape
    gx = np.median(np.stack([np.abs(np.diff(f, axis=1)).mean(0) for f in fs]), 0)
    gy = np.median(np.stack([np.abs(np.diff(f, axis=0)).mean(1) for f in fs]), 0)
    v = float(np.mean([gx[round(w * i / 4)] for i in (1, 2, 3)]) / np.median(gx))
    hh = float(gy[round(h / 2)] / np.median(gy))
    return v, hh


# Conditioning frames of a served production generation, which stage 2 of the i2v e2e replays: three
# frames at full strength (0, an interior keyframe, and the last), 1080p, seed 11. Held outside the
# repo — they are ~5MB each against a 500KB file gate, and they depict an identifiable person, which
# does not belong in public git history. See <assets>/README.md for the gen they came from.
_KF_ASSETS = os.environ.get("LTX_I2V_KF_ASSETS", "/home/sulphur/ltx-test-assets")
_KF_SEED = int(os.environ.get("LTX_I2V_KF_SEED", "11"))
# Frame 72 of 145 is interior: it has generated neighbours on BOTH sides to reconcile against, which
# is the case few-step schedules used to scramble. "last" is the served alias for num_frames-1.
_KF_PINS = (("gen433_frame0.png", 0), ("gen433_last.png", "last"), ("gen433_kf72.png", 72))


def _kf_gen_images(num_frames):
    """The served gen's conditioning list: [(png, pixel_frame, s1, s2), ...] at full strength, in the
    order the server builds it (the frame-0 upload, then each keyframe). None if the assets are
    absent, so the caller can skip rather than silently test something else."""
    out = []
    for name, frame in _KF_PINS:
        p = os.path.join(_KF_ASSETS, name)
        if not os.path.exists(p):
            return None
        out.append((p, num_frames - 1 if frame == "last" else frame, 1.0, 1.0))
    return out


@pytest.mark.skipif(
    not os.path.exists(default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")),
    reason="needs the LTX checkpoint (set LTX_CHECKPOINT to a local .safetensors)",
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_trace_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_distilled_i2v(
    mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp, tmp_path
):
    """E2E I2V in two stages.

    Stage 1 renders the t2v e2e clip (same DEFAULT_LTX_PROMPT). Stage 2 replays a served production
    generation verbatim — its three conditioning frames (0, interior keyframe 72, last), full
    strength, seed 11, at 1088x1920. Replaying real served input is the point: the conditioning
    defects that reached users were all shaped by what users actually submit — several pins at once,
    an interior one with generated neighbours on both sides, and one on the last frame (the tail-pad
    path) — none of which a single frame-0 pin exercises.

    Asserts every pin took, that no pin decoded to high-frequency garbage, and that the output is
    free of the VAE 2x4 grid seam (guards the non-mesh-aligned i2v fix at 1088x1920, whose s1 cond
    latent is the uneven 17x30 that used to seam)."""
    import subprocess

    from PIL import Image

    ckpt = default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    height = int(os.environ.get("HEIGHT", "1088"))  # 1088x1920 -> s1 latent 17x30 (uneven, the fixed case)
    width = int(os.environ.get("WIDTH", "1920"))
    traced = os.environ.get("LTX_TRACED", "0") in ("1", "true", "True")
    seed = int(os.environ.get("SEED", "10"))
    prompt = os.environ.get("PROMPT", DEFAULT_LTX_PROMPT)

    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=ckpt,
        gemma_path=default_ltx_gemma(),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=traced,  # trace capture needs warmup; eager doesn't
        traced=traced,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    if int(ttnn.distributed_context_get_rank()) != 0:
        return

    def _gen(out, images, gen_seed=seed):
        pipeline.generate(
            prompt,
            output_path=str(out),
            images=images,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=gen_seed,
        )

    # 1) t2v e2e clip
    t2v = tmp_path / "t2v.mp4"
    _gen(t2v, None)

    # 2) the served gen, replayed verbatim
    kf_images = _kf_gen_images(num_frames)
    if kf_images is None:
        pytest.skip(f"served-gen conditioning frames not found under {_KF_ASSETS} (set LTX_I2V_KF_ASSETS)")
    i2v = tmp_path / "i2v.mp4"
    _gen(i2v, kf_images, gen_seed=_KF_SEED)

    def _luma(path, frame=0):
        raw = subprocess.run(
            # fmt: off
            [_ffmpeg(), "-v", "error", "-i", path, "-vf", f"select=eq(n\\,{frame})",
             "-vframes", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            # fmt: on
            capture_output=True,
        ).stdout
        return torch.from_numpy(
            np.asarray(Image.open(__import__("io").BytesIO(raw)).convert("L")).astype("float32")
        ).flatten()

    def _pcc(a, b):
        n = min(a.numel(), b.numel())
        return torch.corrcoef(torch.stack([a[:n], b[:n]]))[0, 1].item()

    # (a) every pin took. A pin is a VAE roundtrip + CRF away from its reference, so never identity;
    # what marks one that took is correlating with ITS OWN reference far above an unconditioned gen.
    # Score each separately — a frame-0-only check stayed green right through the interior-keyframe
    # scramble that reached users.
    refs = {idx: _luma(png) for png, idx, _, _ in kf_images}
    pccs = {idx: _pcc(ref, _luma(str(i2v), idx)) for idx, ref in refs.items()}
    for idx, p in sorted(pccs.items()):
        print(f"\nI2V_E2E pin f{idx}: PCC-vs-own-reference={p:.4f}", flush=True)

    # (b) no pin decoded to garbage. A pin whose neighbours went off-distribution comes back as a
    # high-frequency checkerboard, which spikes the Laplacian at the pinned frames while chromatic
    # metrics sit at ~1.0x — fringe alone once scored a destroyed clip clean. Ratio against the
    # clip's own clean frames: healthy ~1.0-1.15x, checkerboard ~2.9x.
    def _sharpness(frame):
        f = _luma(str(i2v), frame).reshape(height, width)
        return float(np.abs(4.0 * f[1:-1, 1:-1] - f[:-2, 1:-1] - f[2:, 1:-1] - f[1:-1, :-2] - f[1:-1, 2:]).mean())

    clean = [i for i in (20, 40, 100, 120) if i < num_frames and i not in refs]
    base_sharp = float(np.mean([_sharpness(i) for i in clean]))
    pin_sharp = {idx: _sharpness(idx) / base_sharp for idx in refs}
    for idx, r in sorted(pin_sharp.items()):
        print(f"I2V_E2E pin f{idx}: structure={r:.2f}x clean-frame sharpness (checkerboard ~2.9x)", flush=True)

    # (c) seam-free at the uneven 1088x1920 (the i2v grid-seam fix). Thresholds bracket the
    # measured clean range (V,H ~<=1.0) below the gridded baseline (V=1.5, H=2.4).
    v, hh = _temporal_seam_score(str(i2v))
    print(f"I2V_E2E seam V={v:.2f} H={hh:.2f} (clean<=~1.0, gridded V=1.5/H=2.4)", flush=True)

    if traced:
        pipeline.release_traces()

    for idx, p in sorted(pccs.items()):
        assert p > 0.85, f"pin f{idx} does not reproduce its conditioning frame (PCC={p:.4f}) — conditioning broken"
    for idx, r in sorted(pin_sharp.items()):
        assert r < 1.6, (
            f"pinned frame f{idx} is {r:.2f}x the clean-frame sharpness — it decoded to high-frequency "
            f"garbage, not a real image"
        )
    assert v < 1.3 and hh < 1.5, f"i2v grid seam present (V={v:.2f}, H={hh:.2f}) at 1088x1920"


# Default conditioning frame for the standalone i2v gate: a real ~1088x1920 frame stored on disk.
# _load_conditioning_image resizes/crops if dims differ. Override with LTX_I2V_COND_IMAGE.
_STANDALONE_I2V_COND_IMAGE = os.environ.get(
    "LTX_I2V_COND_IMAGE", "/home/smarton/tt-metal/tmp/ltx-rt-opt/frames/f_001.png"
)


@pytest.mark.skipif(
    not os.path.exists(default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")),
    reason="needs the LTX checkpoint (set LTX_CHECKPOINT to a local .safetensors)",
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_trace_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_distilled_i2v_standalone(
    mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp, tmp_path
):
    """STANDALONE E2E I2V: a single distilled i2v gen conditioned on a STORED image on disk,
    with NO preceding t2v generate() in the same process (and no model switch).

    This is the safe path for repeated / same-image i2v: the conditioning image encode runs
    exactly ONCE, in the post-warmup device state that ``_warmup_encode`` already exercised (and
    which is proven to work), instead of after a full t2v trace-replay cycle. Asserts the same
    quality bars as ``test_pipeline_distilled_i2v``: frame-0 reproduces the conditioning frame
    (PCC) and the output is free of the VAE 2x4 grid seam."""
    import subprocess

    from PIL import Image

    cond = _STANDALONE_I2V_COND_IMAGE
    if not os.path.exists(cond):
        pytest.skip(f"conditioning image not found: {cond} (set LTX_I2V_COND_IMAGE)")

    ckpt = default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    height = int(os.environ.get("HEIGHT", "1088"))  # 1088x1920 -> s1 latent 17x30 (uneven, the fixed case)
    width = int(os.environ.get("WIDTH", "1920"))
    traced = os.environ.get("LTX_TRACED", "0") in ("1", "true", "True")
    seed = int(os.environ.get("SEED", "10"))
    prompt = os.environ.get("PROMPT", DEFAULT_LTX_PROMPT)

    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=ckpt,
        gemma_path=default_ltx_gemma(),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=traced,  # trace capture needs warmup; eager doesn't
        traced=traced,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    if int(ttnn.distributed_context_get_rank()) != 0:
        return

    # SINGLE i2v gen conditioned on the stored image — no preceding t2v generate().
    i2v = tmp_path / "i2v.mp4"
    pipeline.generate(
        prompt,
        output_path=str(i2v),
        images=[(cond, 0, 1.0)],
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
    )

    # (a) conditioning works: i2v frame-0 reproduces the conditioning frame (VAE roundtrip + CRF,
    # so not identity — but a far tighter correlation than an unconditioned gen of the same prompt).
    def _luma0(path):
        raw = subprocess.run(
            [_ffmpeg(), "-v", "error", "-i", str(path), "-vframes", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True,
        ).stdout
        return torch.from_numpy(
            np.asarray(Image.open(__import__("io").BytesIO(raw)).convert("L")).astype("float32")
        ).flatten()

    c, f0 = _luma0(cond), _luma0(str(i2v))
    pcc = torch.corrcoef(torch.stack([c, f0]))[0, 1].item()
    print(f"\nI2V_E2E frame0-vs-cond PCC={pcc:.4f}", flush=True)

    # (b) seam-free at the uneven 1088x1920 (the i2v grid-seam fix). Thresholds bracket the
    # measured clean range (V,H ~<=1.0) below the gridded baseline (V=1.5, H=2.4).
    v, hh = _temporal_seam_score(str(i2v))
    print(f"I2V_E2E seam V={v:.2f} H={hh:.2f} (clean<=~1.0, gridded V=1.5/H=2.4)", flush=True)

    if traced:
        pipeline.release_traces()

    assert pcc > 0.85, f"i2v frame-0 does not reproduce the conditioning frame (PCC={pcc:.4f}) — conditioning broken"
    assert v < 1.3 and hh < 1.5, f"i2v grid seam present (V={v:.2f}, H={hh:.2f}) at 1088x1920"


@pytest.mark.skipif(
    not os.path.exists(default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")),
    reason="needs the LTX checkpoint (set LTX_CHECKPOINT to a local .safetensors)",
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_trace_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_distilled_i2v_two_images(
    mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp, tmp_path
):
    """JOB-QUEUE STEADY STATE: two i2v gens with DIFFERENT conditioning images back-to-back in
    one process. Gen B's conditioning encode runs AFTER gen A's traced denoise replay — the exact
    encode-after-replay case that used to hang the device (a different image each gen defeats the
    ``_i2v_cond_cache`` memoize, forcing a real eager encode every gen, as a real queue would).
    Both gens must complete WITHOUT hanging and reproduce their own conditioning frame (PCC).

    The grid-seam metric is content-sensitive, so the seam gate here is RELATIVE: the output seam
    must not materially exceed the conditioning image's OWN inherent seam (that would indicate a
    device-induced VAE grid seam). The absolute-seam gate on a clean cond frame lives in
    ``test_pipeline_distilled_i2v``."""
    import subprocess

    from PIL import Image

    cond_a = os.environ.get("LTX_I2V_COND_IMAGE_A", "/home/smarton/tt-metal/tmp/ltx-rt-opt/frames/f_001.png")
    cond_b = os.environ.get("LTX_I2V_COND_IMAGE_B", "/home/smarton/tt-metal/tmp/ltx-rt-opt/frames/f_002.png")
    for p in (cond_a, cond_b):
        if not os.path.exists(p):
            pytest.skip(f"conditioning image not found: {p}")

    ckpt = default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    height = int(os.environ.get("HEIGHT", "1088"))
    width = int(os.environ.get("WIDTH", "1920"))
    traced = os.environ.get("LTX_TRACED", "0") in ("1", "true", "True")
    seed = int(os.environ.get("SEED", "10"))
    prompt = os.environ.get("PROMPT", DEFAULT_LTX_PROMPT)

    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=ckpt,
        gemma_path=default_ltx_gemma(),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=traced,
        traced=traced,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    if int(ttnn.distributed_context_get_rank()) != 0:
        return

    def _luma0(path):
        raw = subprocess.run(
            [_ffmpeg(), "-v", "error", "-i", str(path), "-vframes", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True,
        ).stdout
        return torch.from_numpy(
            np.asarray(Image.open(__import__("io").BytesIO(raw)).convert("L")).astype("float32")
        ).flatten()

    results = []
    for tag, cond in (("A", cond_a), ("B", cond_b)):
        out = tmp_path / f"i2v_{tag}.mp4"
        # Different image each gen -> _i2v_cond_cache miss -> a real eager encode; for gen B this
        # encode runs after gen A's traced replay (the former hang case).
        pipeline.generate(
            prompt,
            output_path=str(out),
            images=[(cond, 0, 1.0)],
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
        )
        pcc = torch.corrcoef(torch.stack([_luma0(cond), _luma0(str(out))]))[0, 1].item()
        out_v, out_h = _temporal_seam_score(str(out))
        in_v, in_h = _image_seam_score(cond)
        print(
            f"\nI2V_TWO gen{tag} frame0-vs-cond PCC={pcc:.4f} | out seam V={out_v:.2f} H={out_h:.2f} "
            f"| cond-image seam V={in_v:.2f} H={in_h:.2f}",
            flush=True,
        )
        results.append((tag, pcc, out_v, out_h, in_v, in_h))

    if traced:
        pipeline.release_traces()

    # Primary (blocking bug): both gens completed without hang and reproduce their cond frame.
    for tag, pcc, *_ in results:
        assert pcc > 0.85, f"gen{tag}: i2v frame-0 does not reproduce the conditioning frame (PCC={pcc:.4f})"
    # Relative seam: a real device seam pushes the output boundary well past the cond image's own.
    for tag, _pcc, out_v, out_h, in_v, in_h in results:
        assert out_v < max(1.3, in_v + 0.4), f"gen{tag}: device V-seam (out={out_v:.2f} vs cond-image={in_v:.2f})"
        assert out_h < max(1.5, in_h + 0.4), f"gen{tag}: device H-seam (out={out_h:.2f} vs cond-image={in_h:.2f})"


@pytest.mark.skipif(
    not os.path.exists(default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")),
    reason="needs the LTX checkpoint (set LTX_CHECKPOINT to a local .safetensors)",
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_trace_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_distilled_i2v_arbitrary_frame(
    mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp, tmp_path
):
    """ARBITRARY-FRAME I2V: four consecutive gens in ONE process exercising the generalized
    conditioning — first frame, last frame, last frame again (traced-replay steady state), and a
    two-keyframe [first, last] gen. Each conditioning image pins its owning latent frame; the pin
    is position-agnostic so the same trace serves every case. Gates: no hang across all gens, the
    conditioned frame reproduces its image (PCC), and the last-again gen stays clean (the static /
    speckle replay gate — three consecutive conditioned gens with zero collapse). NUM_FRAMES sets
    the clip length so this doubles as the 10s (241f) i2v check; the last latent index is logged."""
    import subprocess

    from PIL import Image

    cond_a = os.environ.get("LTX_I2V_COND_IMAGE_A", "/home/smarton/tt-metal/tmp/ltx-rt-opt/frames/f_001.png")
    cond_b = os.environ.get("LTX_I2V_COND_IMAGE_B", "/home/smarton/tt-metal/tmp/ltx-rt-opt/frames/f_002.png")
    for p in (cond_a, cond_b):
        if not os.path.exists(p):
            pytest.skip(f"conditioning image not found: {p}")

    ckpt = default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    height = int(os.environ.get("HEIGHT", "1088"))
    width = int(os.environ.get("WIDTH", "1920"))
    traced = os.environ.get("LTX_TRACED", "0") in ("1", "true", "True")
    seed = int(os.environ.get("SEED", "10"))
    prompt = os.environ.get("PROMPT", DEFAULT_LTX_PROMPT)
    last_frame = num_frames - 1
    last_lat = pixel_to_latent_frame(last_frame, num_frames)
    print(
        f"\nI2V_ARB config: {num_frames}f {height}x{width} last_pixel={last_frame} -> last_latent={last_lat}",
        flush=True,
    )

    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=ckpt,
        gemma_path=default_ltx_gemma(),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=traced,
        traced=traced,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    if int(ttnn.distributed_context_get_rank()) != 0:
        return

    def _luma_png(path):
        return torch.from_numpy(np.asarray(Image.open(path).convert("L")).astype("float32")).flatten()

    def _luma_first(path):
        raw = subprocess.run(
            [_ffmpeg(), "-v", "error", "-i", str(path), "-vframes", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True,
        ).stdout
        return torch.from_numpy(
            np.asarray(Image.open(__import__("io").BytesIO(raw)).convert("L")).astype("float32")
        ).flatten()

    def _luma_last(path):
        # -update 1 overwrites per decoded frame, so the tail seek leaves the true final frame.
        dst = str(path) + ".last.png"
        subprocess.run(
            [_ffmpeg(), "-v", "error", "-sseof", "-0.5", "-i", str(path), "-update", "1", "-y", dst], check=True
        )
        return _luma_png(dst)

    def _pcc(a, b):
        n = min(a.numel(), b.numel())
        return torch.corrcoef(torch.stack([a[:n], b[:n]]))[0, 1].item()

    la, lb = _luma_png(cond_a), _luma_png(cond_b)

    def _gen(images):
        out = tmp_path / "arb.mp4"
        pipeline.generate(
            prompt, output_path=str(out), images=images, num_frames=num_frames, height=height, width=width, seed=seed
        )
        return out

    # 1) first frame, 2) last frame, 3) last frame again (replay/static gate), 4) two keyframes.
    o0 = _gen([(cond_a, 0, 1.0)])
    p0 = _pcc(la, _luma_first(str(o0)))
    print(f"I2V_ARB gen0 FIRST(A@0): first-vs-A PCC={p0:.4f}", flush=True)

    o1 = _gen([(cond_a, last_frame, 1.0)])
    p1 = _pcc(la, _luma_last(str(o1)))
    print(f"I2V_ARB gen1 LAST(A@{last_frame}->lat{last_lat}): last-vs-A PCC={p1:.4f}", flush=True)

    o2 = _gen([(cond_a, last_frame, 1.0)])
    p2 = _pcc(la, _luma_last(str(o2)))
    print(f"I2V_ARB gen2 LAST-AGAIN(A@{last_frame}): last-vs-A PCC={p2:.4f} (static/replay gate)", flush=True)

    o3 = _gen([(cond_a, 0, 1.0), (cond_b, last_frame, 1.0)])
    p3f = _pcc(la, _luma_first(str(o3)))
    p3l = _pcc(lb, _luma_last(str(o3)))
    print(f"I2V_ARB gen3 MULTI(A@0,B@{last_frame}): first-vs-A PCC={p3f:.4f} last-vs-B PCC={p3l:.4f}", flush=True)

    if traced:
        pipeline.release_traces()

    # Frame-0 pin reproduces its image tightly; last/keyframe pins are looser (temporal drift +
    # CRF) but must clearly track their conditioning — an unpinned frame would not correlate.
    assert p0 > 0.85, f"gen0 first-frame does not reproduce cond A (PCC={p0:.4f})"
    assert p1 > 0.5, f"gen1 last-frame does not track cond A (PCC={p1:.4f}) — last-frame pin broken"
    assert p2 > 0.5, f"gen2 last-frame-again collapsed (PCC={p2:.4f}) — static/replay regression"
    assert p3f > 0.7, f"gen3 keyframe first-frame lost cond A (PCC={p3f:.4f})"
    assert p3l > 0.5, f"gen3 keyframe last-frame lost cond B (PCC={p3l:.4f})"


@pytest.mark.skipif(
    not os.path.exists(default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")),
    reason="needs the LTX checkpoint (set LTX_CHECKPOINT to a local .safetensors)",
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_trace_params, ttnn.Topology.Linear, False],
        [(4, 8), (4, 8), 1, 0, 2, False, line_trace_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0", "bh_4x8sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_audio_decode_girl(mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp):
    """Just the audio section of the AV pipeline: real checkpoint weights, real girl-clip
    latent shape (num_frames -> als.frames), no transformer/gemma (decode_audio never
    encodes prompts). Profiles cold (weight load + compile) vs warm (steady-state per-gen)
    decode wall. LTX_TRACED=1 traces the main vocoder (cold/warm timing only — the eager
    torch-oracle quality block is skipped under trace; run LTX_TRACED=0 for the stats)."""
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "145"))  # ~6.04s @ 24fps
    height = int(os.environ.get("HEIGHT", "1088"))
    width = int(os.environ.get("WIDTH", "1920"))
    traced = os.environ.get("LTX_TRACED", "0") in ("1", "true", "True")

    # AUDIO_DEPTHWISE=mac reproduces the pre-conv1d (main) audio path for A/B profiling.
    import models.tt_dit.layers.audio_ops as _ao

    _ao._USE_CONV1D_DEPTHWISE = os.environ.get("AUDIO_DEPTHWISE", "conv1d") != "mac"

    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors"),
        gemma_path=default_ltx_gemma(),  # built as a lazy shell only; never loaded here
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=False,
        traced=traced,
        audio_only=True,  # skip the ~10-min video warmup; decode_audio captures its own trace
        num_frames=num_frames,
        height=height,
        width=width,
    )

    vps = VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=24)
    als = AudioLatentShape.from_video_pixel_shape(vps)
    # Default to the committed real girl-clip latent (39 KB fp16 fixture, dumped from a real gen
    # via LTX_DUMP_AUDIO_LATENT) so the decode runs on actual content with no transformer/gemma.
    # AUDIO_LATENT overrides (.npy or .pt); a seeded-random latent is the fallback if neither.
    _lat = os.environ.get("AUDIO_LATENT") or os.path.join(
        os.path.dirname(__file__), "fixtures", "girl_audio_latent.npy"
    )
    if os.path.exists(_lat):
        latent = (torch.from_numpy(np.load(_lat)) if _lat.endswith(".npy") else torch.load(_lat)).float()
    else:
        torch.manual_seed(0)
        latent = torch.randn(1, als.frames, pipeline.in_channels, dtype=torch.float32) * 0.5

    t0 = time.perf_counter()
    audio = pipeline.decode_audio(latent, num_frames, fps=24.0)  # cold: weight load + compile (+ capture)
    cold_ms = (time.perf_counter() - t0) * 1000

    # One untimed decode to absorb a late first-replay cost (the BWE trace captures on the
    # first post-cold call, not during the cold decode), so warm_ms is true steady state.
    pipeline.decode_audio(latent, num_frames, fps=24.0)
    # WARM_REPS shrinks the steady-state loop for fast dev iteration (the PCC oracle below is
    # unaffected); default 5 keeps warm_ms a stable average for reported timings.
    N = int(os.environ.get("WARM_REPS", "5"))
    t0 = time.perf_counter()
    for _ in range(N):
        audio = pipeline.decode_audio(latent, num_frames, fps=24.0)
    warm_ms = (time.perf_counter() - t0) * 1000 / N
    wav = audio.waveform
    sr = audio.sampling_rate
    out_dir = os.environ.get("AUDIO_OUT")

    def _save(w, name):
        if not out_dir:
            return
        import soundfile as sf

        os.makedirs(out_dir, exist_ok=True)
        a = w[0] if w.dim() == 3 else w  # (2, T) -> (T, 2)
        path = os.path.join(out_dir, name)
        sf.write(path, a.transpose(0, 1).cpu().numpy(), int(sr))
        logger.info(f"wrote {path}")

    _save(wav, f"girl_audio_{'conv1d' if _ao._USE_CONV1D_DEPTHWISE else 'mac'}{'_traced' if traced else ''}.wav")

    # Torch-oracle quality gate vs the diffusers vocoder+BWE (real checkpoint, CPU) — the ground
    # truth. The conv1d path runs under BOTH eager and traced (gating the traced vocoder/BWE
    # output, which is otherwise unverified — exactly the gap that let a broken trace ship). The
    # MAC comparison runs eager-only: running both depthwise toggles on one traced pipeline aliases
    # the shape-keyed trace cache. Per-1s-interval error-to-signal (rmse/σ, dB) + overall PCC; the
    # absolute level (~−18 dB conv1d) is the fp32/bf16 floor, gated only against a gross spike.
    from models.tt_dit.tests.models.ltx.test_audio_ltx import _build_torch_stage_c_real

    z = pipeline.tt_mel_decoder.z_channels
    audio_spatial = latent.reshape(1, latent.shape[1], z, latent.shape[2] // z).permute(0, 2, 1, 3).float()
    mel = pipeline.tt_mel_decoder(audio_spatial)  # TT mel, fed to both TT and torch
    _ao._USE_CONV1D_DEPTHWISE = True
    w_conv = pipeline.tt_vocoder_with_bwe(mel).squeeze(0).float()
    with torch.no_grad():
        w_torch = (
            _build_torch_stage_c_real(default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors"))(mel.float())
            .squeeze(0)
            .float()
        )
    _save(w_conv, f"girl_audio_conv1d{'_traced' if traced else ''}.wav")
    _save(w_torch, "girl_audio_torch.wav")

    nn = min(w_torch.shape[-1], w_conv.shape[-1])

    def _vs_torch(test, name):
        rows = []
        for start in range(0, nn, sr):
            r, t = w_torch[..., start : start + sr], test[..., start : start + sr]
            sig = r.pow(2).mean().sqrt().item() + 1e-9
            rr = ((r - t) ** 2).mean().sqrt().item() / sig
            rows.append(rr)
            logger.info(
                f"[oracle] t={start//sr}s {name}-vs-torch rmse/σ={rr:.3e} ({20*math.log10(rr+1e-12):+.1f} dB) "
                f"max|Δ|={(r - t).abs().max().item():.3e}"
            )
        pcc = torch.corrcoef(torch.stack([w_torch[..., :nn].flatten(), test[..., :nn].flatten()]))[0, 1].item()
        worst_, med_ = max(rows), sorted(rows)[len(rows) // 2]
        print(
            f"\nAUDIO_GIRL {name}-vs-torch traced={traced} worst_rmse/σ={worst_:.3e} "
            f"({20*math.log10(worst_+1e-12):+.1f} dB) median={med_:.3e} ratio={worst_/(med_+1e-9):.2f} PCC={pcc:.5f}",
            flush=True,
        )
        return worst_, med_, pcc

    cw, cm, c_pcc = _vs_torch(w_conv, "conv1d")
    if not traced:
        _ao._USE_CONV1D_DEPTHWISE = False
        w_mac = pipeline.tt_vocoder_with_bwe(mel).squeeze(0).float()
        _save(w_mac, "girl_audio_mac.wav")
        _vs_torch(w_mac, "mac")
    # Gross localized spike only: an interval >4x the median AND past a clearly-audible floor
    # (rmse/σ 0.5 = −6 dB). Also gate overall PCC vs torch — a broken (e.g. mis-traced) vocoder
    # drops PCC far below the ~0.99 fp32/bf16 floor.
    assert cw < 0.5 and cw < 4 * cm + 1e-6, f"conv1d-vs-torch localized spike: worst {cw:.3e} vs median {cm:.3e}"
    assert c_pcc > 0.95, f"conv1d-vs-torch PCC {c_pcc:.4f} below floor (traced={traced}) — vocoder output is wrong"

    pipeline.release_traces()
    pipeline.release_audio_submesh()
    dur = wav.shape[-1] / sr
    print(
        f"\nAUDIO_GIRL depthwise={'conv1d' if _ao._USE_CONV1D_DEPTHWISE else 'mac'} traced={traced} "
        f"latent_frames={als.frames} out={tuple(wav.shape)} {dur:.2f}s@{sr}Hz cold={cold_ms:.1f}ms warm={warm_ms:.1f}ms",
        flush=True,
    )
    assert torch.isfinite(wav).all(), "decoded waveform has non-finite samples"
    assert abs(dur - num_frames / 24.0) < 0.2, f"duration {dur:.2f}s != ~{num_frames/24.0:.2f}s"


@pytest.mark.skipif(
    not os.path.exists(default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")),
    reason="needs the LTX checkpoint (set LTX_CHECKPOINT to a local .safetensors)",
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_trace_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_distilled_i2v_middle_keyframe(
    mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp, tmp_path
):
    """MIDDLE-KEYFRAME I2V render check. The arbitrary-frame test only pins the first/last EDGES; a
    mid-sequence pin is the two-sided case a full-strength pin can't reconcile on the few-step
    fast/medium schedules — it pins the frame but chromatically scrambles the neighbors. The pipeline
    fixes this by aligning the S1 nodes to the distilled trajectory and softening the interior S1 pin
    (resolution-aware: 0.25 at the coarse 720p S1, 0.5 at 1080p; s2 re-locks at full strength). Pin
    one image at a middle latent frame and check the pinned frame reproduces its image (PCC).
    Neighbor-roughness is logged as a diagnostic only — a localized chromatic smear is hard to
    threshold from luma, so the real gate is eyeballing the post-pin frames."""
    import glob
    import subprocess

    from PIL import Image

    cond = os.environ.get("LTX_I2V_COND_IMAGE_A", "/home/smarton/tt-metal/tmp/ltx-rt-opt/frames/f_001.png")
    if not os.path.exists(cond):
        pytest.skip(f"conditioning image not found: {cond}")

    ckpt = default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    height = int(os.environ.get("HEIGHT", "1088"))
    width = int(os.environ.get("WIDTH", "1920"))
    traced = os.environ.get("LTX_TRACED", "0") in ("1", "true", "True")
    seed = int(os.environ.get("SEED", "11"))
    prompt = os.environ.get("PROMPT", DEFAULT_LTX_PROMPT)

    latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
    mid_lat = int(os.environ.get("MID_LATENT", str(latent_frames // 2)))
    pin_pixel = mid_lat * TEMPORAL_COMPRESSION  # inverse of pixel_to_latent_frame: pins the middle latent
    print(
        f"\nI2V_MID config: {num_frames}f {height}x{width} latent_frames={latent_frames} "
        f"mid_lat={mid_lat} pin_pixel={pin_pixel} seed={seed}",
        flush=True,
    )

    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=ckpt,
        gemma_path=default_ltx_gemma(),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=traced,
        traced=traced,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    if int(ttnn.distributed_context_get_rank()) != 0:
        return

    out = tmp_path / "mid.mp4"
    pipeline.generate(
        prompt,
        output_path=str(out),
        images=[(cond, pin_pixel, 1.0)],
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
    )
    if traced:
        pipeline.release_traces()

    keep = os.environ.get("OUTPUT_PATH")  # copy the clip out for eyeballing the A/B
    if keep:
        subprocess.run([_ffmpeg(), "-v", "error", "-i", str(out), "-y", keep], check=True)

    # Frame-accurate decode so the pinned index and its neighbors line up with pixel frames.
    fdir = tmp_path / "frames"
    fdir.mkdir()
    subprocess.run([_ffmpeg(), "-v", "error", "-i", str(out), "-vsync", "0", str(fdir / "f_%04d.png")], check=True)
    paths = sorted(glob.glob(str(fdir / "f_*.png")))
    assert paths, "no frames decoded"

    def _luma(p):
        return np.asarray(Image.open(p).convert("L")).astype("float32")

    def _rough(f):
        # mean |discrete Laplacian| over the interior: high-frequency energy. A scrambled frame is
        # noisy/blocky and spikes here versus a coherent one.
        lap = 4.0 * f[1:-1, 1:-1] - f[:-2, 1:-1] - f[2:, 1:-1] - f[1:-1, :-2] - f[1:-1, 2:]
        return float(np.abs(lap).mean())

    rough = np.array([_rough(_luma(p)) for p in paths])
    nfr = len(rough)
    med = float(np.median(rough))
    lo = max(0, pin_pixel - 14)
    hi = min(nfr, pin_pixel + 15)
    win_max = float(rough[lo:hi].max())
    ratio = win_max / med if med > 0 else float("inf")

    pin_idx = min(pin_pixel, nfr - 1)
    # Resize the native-res cond to the decoded frame before correlating — a raw native-vs-720p
    # flatten miscorrelates to ~0 even for a faithful pin (the rows no longer line up).
    pin_img = Image.open(paths[pin_idx]).convert("L")
    cond_l = np.asarray(Image.open(cond).convert("L").resize(pin_img.size)).astype("float32")
    cl = torch.from_numpy(cond_l).flatten()
    pf = torch.from_numpy(np.asarray(pin_img).astype("float32")).flatten()
    pcc = torch.corrcoef(torch.stack([cl, pf]))[0, 1].item()

    print(
        f"I2V_MID pinned-vs-cond PCC={pcc:.4f} | neighbor roughness "
        f"win_max/med={ratio:.2f} (med={med:.2f} win_max={win_max:.2f}) frames={nfr}",
        flush=True,
    )
    print("I2V_MID rough[pin±14]=" + ",".join(f"{r:.1f}" for r in rough[lo:hi]), flush=True)

    assert pcc > 0.5, f"middle keyframe does not track its image (PCC={pcc:.4f}) — conditioning broken"
    thr = float(os.environ.get("MID_ROUGH_RATIO_MAX", "999"))
    assert ratio < thr, f"middle-keyframe neighbor scramble: roughness ratio {ratio:.2f} >= {thr}"


@pytest.mark.skipif(
    not os.path.exists(default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")),
    reason="needs the LTX checkpoint (set LTX_CHECKPOINT to a local .safetensors)",
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_trace_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_kf_fringe_repro(
    mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp, tmp_path
):
    """Repro harness for the keyframe-neighbor CHROMATIC FRINGING (green tint on the edges of frames
    around a pin, worst just after the middle pin). Pins 3 images at [0, mid, last] and prints a
    per-frame chromatic-fringe profile (mean |per-channel laplacian - luma laplacian|) so a fix's
    effect on the post-pin fringe peak is measurable. LTX_KF_INTERIOR_S1 softens the interior pins'
    coarse hold. Saves to OUTPUT_PATH for eyeballing. Not a gate."""
    import glob
    import subprocess

    from PIL import Image

    base = os.environ.get("LTX_KF_DIR", "")
    kf0 = os.environ.get("LTX_KF0") or os.path.join(base, "kf0.png")
    kfmid = os.environ.get("LTX_KF72") or os.path.join(base, "kf72.png")
    kflast = os.environ.get("LTX_KFLAST") or os.path.join(base, "kf144.png")
    for p in (kf0, kfmid, kflast):
        if not os.path.exists(p):
            pytest.skip(f"conditioning image not found: {p} (set LTX_KF_DIR or LTX_KF0/72/LAST)")

    ckpt = default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    height = int(os.environ.get("HEIGHT", "1088"))
    width = int(os.environ.get("WIDTH", "1920"))
    traced = os.environ.get("LTX_TRACED", "0") in ("1", "true", "True")
    seed = int(os.environ.get("SEED", "10"))
    pf = os.environ.get("PROMPT_FILE")
    prompt = open(pf).read().strip() if pf and os.path.exists(pf) else os.environ.get("PROMPT", DEFAULT_LTX_PROMPT)
    s_int = float(os.environ.get("LTX_KF_INTERIOR_S1", "1.0"))  # interior-pin coarse (s1) hold
    s_int2 = float(os.environ.get("LTX_KF_INTERIOR_S2", "1.0"))  # interior-pin refine (s2) hold

    latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
    mid_pixel = (latent_frames // 2) * TEMPORAL_COMPRESSION
    last_pixel = num_frames - 1
    print(
        f"\nKF_FRINGE config: {num_frames}f {height}x{width} pins=[0,{mid_pixel},{last_pixel}] "
        f"interior_s1={s_int} seed={seed}",
        flush=True,
    )

    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=ckpt,
        gemma_path=default_ltx_gemma(),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=traced,
        traced=traced,
        num_frames=num_frames,
        height=height,
        width=width,
    )
    if int(ttnn.distributed_context_get_rank()) != 0:
        return

    out = tmp_path / "fringe.mp4"
    pipeline.generate(
        prompt,
        output_path=str(out),
        images=[(kf0, 0, 1.0, 1.0), (kfmid, mid_pixel, s_int, s_int2), (kflast, last_pixel, s_int, s_int2)],
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
    )
    if traced:
        pipeline.release_traces()

    keep = os.environ.get("OUTPUT_PATH")
    if keep:
        subprocess.run([_ffmpeg(), "-v", "error", "-i", str(out), "-y", keep], check=True)

    fdir = tmp_path / "frames"
    fdir.mkdir()
    subprocess.run([_ffmpeg(), "-v", "error", "-i", str(out), "-vsync", "0", str(fdir / "f_%04d.png")], check=True)
    paths = sorted(glob.glob(str(fdir / "f_*.png")))
    assert paths, "no frames decoded"

    def _fringe(p):
        im = np.asarray(Image.open(p).convert("RGB")).astype("float32")
        lum = im.mean(axis=2)

        def lap(f):
            return 4.0 * f[1:-1, 1:-1] - f[:-2, 1:-1] - f[2:, 1:-1] - f[1:-1, :-2] - f[1:-1, 2:]

        lL = lap(lum)
        return float(
            (np.abs(lap(im[..., 0]) - lL) + np.abs(lap(im[..., 1]) - lL) + np.abs(lap(im[..., 2]) - lL)).mean()
        )

    fr = np.array([_fringe(p) for p in paths])
    med = float(np.median(fr))
    nfr = len(fr)
    mid_idx = min(mid_pixel + 4, nfr - 1)  # fringe peaks ~4 frames after the pin
    peak_mid = float(fr[max(0, mid_pixel - 2) : min(nfr, mid_pixel + 10)].max()) / med
    peak_last = float(fr[max(0, last_pixel - 8) : nfr].max()) / med
    print(
        f"KF_FRINGE prof: median={med:.2f} peak@mid={peak_mid:.2f}x peak@last={peak_last:.2f}x "
        f"global={fr.max() / med:.2f}x@f{int(fr.argmax())}",
        flush=True,
    )
    print(
        "KF_FRINGE mid[pin..+12]=" + ",".join(f"{fr[i] / med:.2f}" for i in range(mid_pixel, min(nfr, mid_pixel + 13))),
        flush=True,
    )

    # Temporal COHERENCE — the real defect: jerk = |f[t+1] - 2 f[t] + f[t-1]| zeros out smooth motion
    # and spikes on frame-to-frame flicker/jumps. A hard interior pin makes its neighbors flicker (luma
    # + colour jumping). RATIO near 1.0 = as coherent as the rest of the clip.
    rgb = np.stack([np.asarray(Image.open(p).convert("RGB")).astype("float32") for p in paths])
    lum = rgb.mean(3)
    jerk = np.abs(lum[2:] - 2 * lum[1:-1] + lum[:-2]).mean((1, 2))
    nj = len(jerk)
    near = jerk[max(0, mid_pixel - 6) : min(nj, mid_pixel + 10)]
    far = np.concatenate([jerk[18:58], jerk[93:133]]) if nj > 133 else jerk
    fmean = float(far.mean()) or 1.0
    print(
        f"KF_JERK (temporal coherence): near-pin-max={near.max():.2f}@f{max(0, mid_pixel - 6) + int(near.argmax()) + 1} "
        f"far-mean={fmean:.2f} -> RATIO={near.max() / fmean:.2f}x (1.0=coherent, baseline~2.7x)",
        flush=True,
    )
    print(
        "KF_JERK near[pin-2..+10]="
        + ",".join(f"{jerk[i]:.1f}" for i in range(max(0, mid_pixel - 2), min(nj, mid_pixel + 11))),
        flush=True,
    )

    # HARD GATE — pinned-frame degeneracy. A softened append-token anchor leaves the held reference
    # mostly noise, and the frames that converge on it decode to a high-frequency checkerboard. That
    # blows up the Laplacian at the pinned frames while KF_FRINGE stays ~1.0x (it measures chromatic
    # fringing, not structure) — so fringe alone scored a destroyed 720p clip as clean and shipped it.
    # Ratio vs the clip's own clean frames: healthy ~1.0-1.15x, checkerboard ~2.9x.
    def _sharpness(i):
        f = lum[i]
        return float(np.abs(4.0 * f[1:-1, 1:-1] - f[:-2, 1:-1] - f[2:, 1:-1] - f[1:-1, :-2] - f[1:-1, 2:]).mean())

    clean = [i for i in (20, 40, 60, 90, 120) if i < nfr]
    base_sharp = float(np.mean([_sharpness(i) for i in clean]))
    for pin in (mid_pixel, last_pixel):
        if pin >= nfr:
            continue
        ratio = _sharpness(pin) / base_sharp
        print(f"KF_PIN_STRUCTURE f{pin}: {ratio:.2f}x clean-frame sharpness (checkerboard ~2.9x)", flush=True)
        assert ratio < 1.6, (
            f"pinned frame f{pin} is {ratio:.2f}x the clean-frame sharpness — the pinned frame decoded to "
            f"high-frequency garbage (anchor reference degenerate), not a real image"
        )


@pytest.mark.parametrize(
    "tier, latent_frames, expected",
    [
        # fast (S1=3): below the 4-step aligned tail, so it pays a step to stay on the trajectory.
        ("fast", 19, [1.0, 0.909375, 0.725, 0.421875, 0.0]),
        # medium (S1=6): spends its surplus on head nodes instead of collapsing onto fast's schedule.
        ("medium", 19, [1.0, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]),
        # A long clip's interior pin takes a head node even when the tier wouldn't.
        ("fast", 31, [1.0, 0.975, 0.909375, 0.725, 0.421875, 0.0]),
    ],
)
def test_keyframe_s1_sigmas_scale_with_the_tier(tier, latent_frames, expected, monkeypatch):
    """A keyframe gen must render differently on medium than on fast. A fixed aligned node list made
    the two bit-identical, so the tier was a no-op for every keyframe user. Every node must still come
    from the high schedule -- an off-trajectory node is what scrambles the pin's neighbors."""
    import importlib

    import models.tt_dit.pipelines.ltx.pipeline_ltx_distilled as mod
    from models.tt_dit.utils.ltx import apply_quality_env

    monkeypatch.setenv("LTX_QUALITY", tier)
    for v in ("LTX_S1_SIGMAS", "LTX_S2_SIGMAS", "LTX_KEYFRAME_S1_SIGMAS"):
        monkeypatch.delenv(v, raising=False)
    apply_quality_env()
    mod = importlib.reload(mod)

    got = mod._keyframe_s1_sigmas(latent_frames)
    assert got == expected
    assert set(got) <= set(mod._DEFAULT_S1_SIGMAS), "aligned nodes must be a subset of the high schedule"
    assert got[-len(mod._KEYFRAME_S1_TAIL) :] == mod._KEYFRAME_S1_TAIL, "the denoise tail is mandatory"


def test_keyframe_s1_sigmas_differ_between_tiers(monkeypatch):
    """The regression itself: medium and fast must not produce the same schedule."""
    import importlib

    import models.tt_dit.pipelines.ltx.pipeline_ltx_distilled as mod
    from models.tt_dit.utils.ltx import apply_quality_env

    out = {}
    for tier in ("fast", "medium"):
        monkeypatch.setenv("LTX_QUALITY", tier)
        for v in ("LTX_S1_SIGMAS", "LTX_S2_SIGMAS", "LTX_KEYFRAME_S1_SIGMAS"):
            monkeypatch.delenv(v, raising=False)
        apply_quality_env()
        mod = importlib.reload(mod)
        out[tier] = mod._keyframe_s1_sigmas(19)
    assert out["fast"] != out["medium"]
    assert len(out["medium"]) > len(out["fast"])


# --- full coverage matrix: every model x tier x resolution, t2v AND i2v ---------------------------
# Deselected by default (LTX_MATRIX=1 to arm): one cell loads 22B weights and renders two clips, so
# the full sweep is hours of device time — a manual gate, not a per-PR one.
#
# One cell per PROCESS, and that is a constraint, not a preference: conftest calls apply_quality_env()
# at import because the pipeline reads its sigma schedule into module constants at import time, so a
# process can only ever be one tier. The checkpoint and resolution ride the same per-process env.
# tools/ltx_matrix.py drives the sweep by running this test once per cell, which is also what keeps
# each device job inside the broker's window.
_MATRIX_MODELS = {
    "ltx": "ltx-2.3-22b-distilled-1.1.safetensors",
    "sulphur": "/home/sulphur/models/sulphur_distil_bf16.safetensors",
    "sulphur-lora": "/home/sulphur/models/sulphur_lora_fused_distil.safetensors",
    "10eros-lora": "/home/sulphur/models/10eros_distil_fused.safetensors",
    "lora1.1-cond72": "/home/sulphur/models/ltx11_cond72_fused_distil.safetensors",
    "lora1.1-cond32": "/home/sulphur/models/ltx11_cond32_fused_distil.safetensors",
}
_MATRIX_RESOLUTIONS = {"1080p": (1088, 1920), "720p": (704, 1280)}


def _matrix_checkpoint(model):
    spec = _MATRIX_MODELS[model]
    return spec if os.path.isabs(spec) else default_ltx_checkpoint(spec)


@pytest.mark.skipif(
    os.environ.get("LTX_MATRIX", "0") not in ("1", "true", "True"), reason="manual sweep: set LTX_MATRIX=1"
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_trace_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_ltx_matrix_cell(
    mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp, tmp_path
):
    """One matrix cell: render t2v AND i2v on one (model, tier, resolution) and gate both.

    The cell is the env the server would serve with — LTX_MATRIX_MODEL picks the checkpoint,
    LTX_QUALITY the tier (conftest has already expanded it into quant + sigmas), LTX_MATRIX_RES the
    resolution — so a green cell means that served config renders, not that some neighbouring
    config does.

    t2v gates on the clip being real (a traced replay of a never-captured trace decodes to a uniform
    grey field that every correlation metric happily calls fine, so per-frame variance is the check).
    i2v replays the served 3-pin conditioning and gates that every pin took and none decoded to
    high-frequency garbage.
    """
    import subprocess

    from PIL import Image

    model = os.environ.get("LTX_MATRIX_MODEL", "ltx")
    res = os.environ.get("LTX_MATRIX_RES", "1080p")
    tier = os.environ.get("LTX_QUALITY", "high").strip().lower() or "high"
    assert model in _MATRIX_MODELS, f"LTX_MATRIX_MODEL={model!r}; choose {sorted(_MATRIX_MODELS)}"
    assert res in _MATRIX_RESOLUTIONS, f"LTX_MATRIX_RES={res!r}; choose {sorted(_MATRIX_RESOLUTIONS)}"
    ckpt = _matrix_checkpoint(model)
    if not os.path.exists(ckpt):
        pytest.skip(f"{model}: checkpoint absent ({ckpt})")

    height, width = _MATRIX_RESOLUTIONS[res]
    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    traced = os.environ.get("LTX_TRACED", "0") in ("1", "true", "True")
    seed = int(os.environ.get("SEED", "11"))
    cell = f"{model}/{tier}/{res}"

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=ckpt,
        gemma_path=default_ltx_gemma(),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=traced,  # a traced replay of a trace that was never captured renders grey
        traced=traced,
        num_frames=num_frames,
        height=height,
        width=width,
    )
    if int(ttnn.distributed_context_get_rank()) != 0:
        return

    def _frames(path, count=8):
        raw = subprocess.run(
            # fmt: off
            [_ffmpeg(), "-v", "error", "-i", path, "-vf", f"select='not(mod(n\\,{max(1, num_frames // count)}))'",
             "-vsync", "0", "-f", "image2pipe", "-vcodec", "png", "-"],
            # fmt: on
            capture_output=True,
        ).stdout
        return raw

    def _luma(path, frame):
        raw = subprocess.run(
            # fmt: off
            [_ffmpeg(), "-v", "error", "-i", path, "-vf", f"select=eq(n\\,{frame})",
             "-vframes", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            # fmt: on
            capture_output=True,
        ).stdout
        return np.asarray(Image.open(__import__("io").BytesIO(raw)).convert("L")).astype("float32")

    # --- t2v ---
    t2v = tmp_path / "t2v.mp4"
    pipeline.generate(
        DEFAULT_LTX_PROMPT,
        output_path=str(t2v),
        images=None,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
    )
    assert t2v.exists() and t2v.stat().st_size > 0, f"{cell}: t2v produced no file"
    stds = [float(_luma(str(t2v), f).std()) for f in (0, num_frames // 2, num_frames - 1)]
    print(f"\nMATRIX {cell} t2v: per-frame std={['%.1f' % s for s in stds]}", flush=True)

    # --- i2v: the served 3-pin conditioning ---
    kf_images = _kf_gen_images(num_frames)
    if kf_images is None:
        pytest.skip(f"served-gen conditioning frames not found under {_KF_ASSETS} (set LTX_I2V_KF_ASSETS)")
    i2v = tmp_path / "i2v.mp4"
    pipeline.generate(
        DEFAULT_LTX_PROMPT,
        output_path=str(i2v),
        images=kf_images,
        num_frames=num_frames,
        height=height,
        width=width,
        seed=seed,
    )
    assert i2v.exists() and i2v.stat().st_size > 0, f"{cell}: i2v produced no file"

    def _pcc(a, b):
        a, b = a.flatten(), b.flatten()
        n = min(a.size, b.size)
        return float(np.corrcoef(a[:n], b[:n])[0, 1])

    refs = {idx: _luma(png, 0) for png, idx, _, _ in kf_images}
    pccs = {idx: _pcc(ref, _luma(str(i2v), idx)) for idx, ref in refs.items()}

    def _sharpness(frame):
        f = _luma(str(i2v), frame)
        return float(np.abs(4.0 * f[1:-1, 1:-1] - f[:-2, 1:-1] - f[2:, 1:-1] - f[1:-1, :-2] - f[1:-1, 2:]).mean())

    clean = [i for i in (20, 40, 100, 120) if i < num_frames and i not in refs]
    base_sharp = float(np.mean([_sharpness(i) for i in clean]))
    pin_sharp = {idx: _sharpness(idx) / base_sharp for idx in refs}
    for idx in sorted(refs):
        print(f"MATRIX {cell} i2v pin f{idx}: PCC={pccs[idx]:.4f} structure={pin_sharp[idx]:.2f}x", flush=True)

    if traced:
        pipeline.release_traces()

    # A grey clip is the signature of a replayed-but-never-captured trace; it correlates fine with
    # anything, so variance is what catches it.
    assert min(stds) > 5.0, f"{cell}: t2v is a flat/blank field (per-frame std={stds}) — nothing was rendered"
    for idx in sorted(pccs):
        assert pccs[idx] > 0.85, f"{cell}: i2v pin f{idx} did not take (PCC={pccs[idx]:.4f})"
    for idx in sorted(pin_sharp):
        assert pin_sharp[idx] < 1.6, (
            f"{cell}: i2v pin f{idx} is {pin_sharp[idx]:.2f}x the clean-frame sharpness — "
            f"it decoded to high-frequency garbage"
        )
