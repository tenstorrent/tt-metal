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
from models.tt_dit.models.audio_vae.audio_decoder_ltx import LTXAudioDecoderAdapter
from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
from models.tt_dit.utils.ltx import (
    DEFAULT_LTX_PROMPT,
    default_ltx_checkpoint,
    default_ltx_gemma,
    print_ltx_timing_table,
)
from models.tt_dit.utils.patchifiers import AudioLatentShape, VideoPixelShape
from models.tt_dit.utils.test import line_params, ring_params
from models.tt_dit.utils.vbench import assert_vbench_quality

# Trace region for LTX_TRACED=1. Holds both stage traces' command streams (s1 + larger-seq
# s2); measured need is ~236 MB at 1080p (get_trace_buffers_size), so 300 MB gives headroom.
# l1_small_size: native ttnn.conv1d (the depthwise audio taps) runs an UntilizeWithHalo gather
# whose sharding/config tensors allocate from the dedicated L1_SMALL pool; it defaults to 0, which
# OOMs the vocoder. 32 KB matches the audio component tests.
ring_trace_params = {**ring_params, "trace_region_size": 500_000_000, "l1_small_size": 32768}
line_trace_params = {**line_params, "trace_region_size": 500_000_000, "l1_small_size": 32768}


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
        # BH on 2x4. l1_small_size: the audio vocoder's conv2d needs an L1_SMALL scratch pool
        # (default 0 → OOM in decode).
        [(2, 4), (2, 4), 1, 0, 2, True, {**line_params, "l1_small_size": 32768}, ttnn.Topology.Linear, False],
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

    # Conditioning image (I2V). Its mere presence drives image_conditioning: with a path the
    # transformer builds the per-token video-timestep (I2V) modulation; without one pure T2V keeps
    # the fast scalar-AdaLN path. Resolved before create_pipeline so the bool can gate the build.
    image_path = os.environ.get("LTX_I2V_IMAGE")
    images = None
    if image_path:
        if not os.path.exists(image_path):
            pytest.skip(f"LTX_I2V_IMAGE set but file not found: {image_path}")
        strength = float(os.environ.get("LTX_I2V_STRENGTH", "1.0"))
        images = [(image_path, 0, strength)]

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
        image_conditioning=bool(image_path),
    )

    prompt = os.environ.get("PROMPT", DEFAULT_LTX_PROMPT)

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

    vbench_thresholds_by_height = {
        1088: {
            "subject_consistency": 0.92,
            "background_consistency": 0.93,
            "motion_smoothness": 0.955,
            "dynamic_degree": 1.0,
            "imaging_quality": 0.645,
        },
    }

    # RUN_VBENCH=0 skips the quality gate (e.g. perf-only iteration); defaults on so CI gates.
    run_vbench = os.environ.get("RUN_VBENCH", "1") in ("1", "true", "True")
    # RUN_CLIP=0 skips the CLIP prompt-alignment gate; defaults on (mirrors the wan2.2 test).
    run_clip = os.environ.get("RUN_CLIP", "1") in ("1", "true", "True")

    # An enabled quality gate whose deps are absent must report SKIPPED (visible), never a silent
    # green pass. Unguarded by rank on purpose: dep availability is per-process (same venv on every
    # rank), and a rank-divergent skip would hang the collective generate().
    if run_vbench:
        pytest.importorskip("vbench", reason="RUN_VBENCH=1 but vbench not installed (set RUN_VBENCH=0)")
    if run_clip:
        pytest.importorskip("decord", reason="RUN_CLIP=1 but decord not installed (set RUN_CLIP=0)")

    def check_output_with_vbench(prompt, number):
        if not run_vbench:
            logger.info("RUN_VBENCH=0, skipping VBench quality gate")
            return
        if int(ttnn.distributed_context_get_rank()) == 0:
            thresholds = vbench_thresholds_by_height.get(height)
            if thresholds is None:
                pytest.skip(f"no VBench thresholds calibrated for height {height}")
            output_filename = os.environ.get("OUTPUT_PATH", f"ltx_av_fast_{width}x{height}_{number}.mp4")
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
            pytest.skip(f"CLIP deps unavailable ({e})")

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


def _temporal_seam_score(path):
    """Grid-seam strength at the 2x4 mesh boundaries (W/4,W/2,3W/4 vertical; H/2 horizontal).
    Per-column/row TEMPORAL MEDIAN of the gradient isolates a static seam from moving content.
    Each boundary is scored against its LOCAL neighbour gradient (a ring +/-12 px, skipping
    +/-1 px for seam width) — not the global median, which a centred subject over a dark,
    low-gradient background inflates (the subject straddles W/2, not a mesh line). A baked seam
    is a thin gradient spike pinned to the boundary, so it stands well above its flat local
    ring; clean content scores ~1."""
    fs = _ffmpeg_frames(path, 16)
    assert fs, f"no frames decoded from {path}"
    h, w = fs[0].shape
    gx = np.median(np.stack([np.abs(np.diff(f, axis=1)).mean(0) for f in fs]), 0)
    gy = np.median(np.stack([np.abs(np.diff(f, axis=0)).mean(1) for f in fs]), 0)

    def _ring_ratio(g, idx, skip=1, span=12):
        lo, hi = max(0, idx - span), min(len(g), idx + span + 1)
        ring = np.concatenate([g[lo : idx - skip], g[idx + skip + 1 : hi]])
        return float(g[idx] / (np.median(ring) + 1e-6))

    v = float(np.mean([_ring_ratio(gx, round(w * i / 4)) for i in (1, 2, 3)]))
    hh = _ring_ratio(gy, round(h / 2))
    return v, hh


@pytest.mark.skipif(
    not os.path.exists(default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")),
    reason="needs the LTX checkpoint (set LTX_CHECKPOINT to a local .safetensors)",
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_trace_params, ttnn.Topology.Linear, False],
        # 4x8 Galaxy (ring): the full-res 1088x1920 latent shards unevenly on the 4x8 mesh
        # (s1 cond latent 17x30, full 34x60), so the VAE encoder fold + even-shard padding must
        # handle non-mesh-aligned dims here. The 2x4 loudbox shards evenly and never hits this.
        # The id stays out of the bh_*/wh_* namespace so a bare `-k bh_4x8sp1tp0_ring` (run_ltx's
        # canonical t2v/i2v launcher) never collides into this gated test.
        [(4, 8), (4, 8), 1, 0, 2, False, ring_trace_params, ttnn.Topology.Ring, False],
    ],
    ids=["bh_2x4sp1tp0", "i2v_4x8sp1tp0_ring"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_distilled_i2v(
    mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp, tmp_path
):
    """E2E chained t2v->i2v: generate a t2v clip, then i2v continuing from its LAST frame
    (same DEFAULT_LTX_PROMPT), and splice both into one continuous ~12s clip. Asserts the
    I2V output (a) reproduces the conditioning frame at frame-0 (conditioning works) and
    (b) is free of the VAE 2x4 grid seam (guards the non-mesh-aligned i2v fix at 1088x1920,
    whose s1 cond latent is the uneven 17x30 that used to seam). An external LTX_I2V_IMAGE
    overrides the t2v pre-gen (single i2v, no splice)."""
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
        # This test conditions on a frame produced in-process (t2v last frame), so there's no
        # conditioning-image path in hand at construction — force the I2V path on explicitly.
        image_conditioning=True,
    )

    if int(ttnn.distributed_context_get_rank()) != 0:
        return

    def _gen(out, images):
        pipeline.generate(
            prompt, output_path=str(out), images=images, num_frames=num_frames, height=height, width=width, seed=seed
        )

    # Conditioning image: an explicit LTX_I2V_IMAGE conditions directly on it and SKIPS the t2v
    # pre-gen (an external/user image, or a t2v last frame produced by a prior process). Else this
    # generates a t2v clip and conditions on its LAST FRAME so the i2v continues the shot.
    cond_override = os.environ.get("LTX_I2V_IMAGE")
    strength = float(os.environ.get("LTX_I2V_STRENGTH", "1.0"))
    t2v = None
    if cond_override:
        cond = cond_override
    else:
        t2v = tmp_path / "t2v.mp4"
        _gen(t2v, None)
        # -sseof seeks near the end and -update rewrites the same file per decoded frame, so it
        # lands on the true final frame without buffering the whole clip into memory.
        cond = tmp_path / "cond_lastframe.png"
        subprocess.run(
            [_ffmpeg(), "-v", "error", "-sseof", "-3", "-i", str(t2v), "-update", "1", "-y", str(cond)],
            check=True,
        )
        if os.environ.get("LTX_T2V_OUT"):
            shutil.copy(str(t2v), os.environ["LTX_T2V_OUT"])
        if os.environ.get("LTX_COND_OUT"):
            shutil.copy(str(cond), os.environ["LTX_COND_OUT"])
        # On a 4x8 mesh the image encoder and DiT coresident-evict, so a t2v->i2v chain in one
        # process reloads the encoder over t2v's captured traces and wedges. LTX_T2V_ONLY stops
        # here so a fresh process runs i2v on LTX_COND_OUT (via LTX_I2V_IMAGE) and host-splices.
        if os.environ.get("LTX_T2V_ONLY"):
            print(
                f"I2V_E2E t2v-only -> clip={os.environ.get('LTX_T2V_OUT')} cond={os.environ.get('LTX_COND_OUT')}",
                flush=True,
            )
            if traced:
                pipeline.release_traces()
            return

    # i2v conditioned on that frame
    i2v = tmp_path / "i2v.mp4"
    _gen(i2v, [(str(cond), 0, strength)])
    if os.environ.get("LTX_I2V_OUT"):
        shutil.copy(str(i2v), os.environ["LTX_I2V_OUT"])

    # In-process splice of t2v + i2v into one continuous clip (each gen is 145f@24fps -> ~12s).
    # Only reached where encoder+DiT+traces fit one process (not the 4x8 mesh — use LTX_T2V_ONLY
    # plus a second i2v process there). An external LTX_I2V_IMAGE has no preceding t2v to splice.
    if t2v is not None:
        chained = os.environ.get("LTX_CHAINED_OUT") or str(tmp_path / "chained.mp4")
        subprocess.run(
            [
                _ffmpeg(),
                "-v",
                "error",
                "-i",
                str(t2v),
                "-i",
                str(i2v),
                "-filter_complex",
                "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[v][a]",
                "-map",
                "[v]",
                "-map",
                "[a]",
                "-y",
                chained,
            ],
            check=True,
        )
        print(f"I2V_E2E chained t2v+i2v -> {chained}", flush=True)

    # (a) conditioning works: i2v frame-0 reproduces the conditioning frame (VAE roundtrip + CRF,
    # so not identity — but a far tighter correlation than an unconditioned gen of the same prompt).
    def _luma0(path):
        raw = subprocess.run(
            [_ffmpeg(), "-v", "error", "-i", path, "-vframes", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            capture_output=True,
        ).stdout
        return Image.open(__import__("io").BytesIO(raw)).convert("L")

    def _vec(im):
        return torch.from_numpy(np.asarray(im).astype("float32")).flatten()

    f0_img = _luma0(str(i2v))
    cond_img = _luma0(str(cond))
    # Compare a matched field of view: an external LTX_I2V_IMAGE can differ in resolution/aspect
    # from the generated frame (the pipeline center-crops the conditioning image to the target
    # aspect before encoding). Mirror that — center-crop cond to the frame's aspect, then resize.
    # No-op when cond is a same-size/same-aspect frame (the self-conditioned t2v-last-frame case).
    fw, fh = f0_img.size
    cw, ch = cond_img.size
    aspect = fw / fh
    if cw / ch > aspect:
        nw = round(ch * aspect)
        x0 = (cw - nw) // 2
        cond_img = cond_img.crop((x0, 0, x0 + nw, ch))
    elif cw / ch < aspect:
        nh = round(cw / aspect)
        y0 = (ch - nh) // 2
        cond_img = cond_img.crop((0, y0, cw, y0 + nh))
    if cond_img.size != f0_img.size:
        cond_img = cond_img.resize(f0_img.size)

    c, f0 = _vec(cond_img), _vec(f0_img)
    pcc = torch.corrcoef(torch.stack([c, f0]))[0, 1].item()
    print(f"\nI2V_E2E frame0-vs-cond PCC={pcc:.4f}", flush=True)

    # (b) seam-free at the uneven 1088x1920 (the i2v grid-seam fix). Local-ring ratios sit at
    # ~1.0 on clean content (measured 0.85-1.16 across all boundaries); a baked seam spikes
    # well above its flat neighbourhood. Threshold 1.5 clears the clean band with margin.
    v, hh = _temporal_seam_score(str(i2v))
    print(f"I2V_E2E seam V={v:.2f} H={hh:.2f} (clean~1.0, seam>>1)", flush=True)

    if traced:
        pipeline.release_traces()

    assert pcc > 0.85, f"i2v frame-0 does not reproduce the conditioning frame (PCC={pcc:.4f}) — conditioning broken"
    assert v < 1.5 and hh < 1.5, f"i2v grid seam present (V={v:.2f}, H={hh:.2f}) at 1088x1920"


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

    ckpt = default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")
    # Audio-only build: create the pipeline WITHOUT a checkpoint so __init__ builds/primes
    # nothing (skipping the ~10-min video warmup and the 22B transformer / VAE / upsampler
    # prime that dominate a cold run — decode_audio only touches the audio decoder + vocoder).
    # Then point it at the checkpoint and construct + prime just the audio decode chain;
    # decode_audio compiles + captures its own trace lazily on its first call.
    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=None,
        gemma_path=default_ltx_gemma(),  # built as a lazy shell only; never loaded here
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=False,
        traced=traced,
        num_frames=num_frames,
        height=height,
        width=width,
    )
    pipeline.checkpoint_name = LTXDistilledPipeline._resolve_checkpoint_file(ckpt)
    pipeline._audio_adapter = LTXAudioDecoderAdapter(
        pipeline.checkpoint_name,
        mesh_device=pipeline.mesh_device,
        vae_ccl_manager=pipeline.vae_ccl_manager,
        dit_parallel_config=pipeline.parallel_config,
        traced=pipeline._traced,
    )
    pipeline._prepare_audio_decoder()

    vps = VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=24)
    als = AudioLatentShape.from_video_pixel_shape(vps)
    # Seeded synthetic latent scaled to a real girl-clip audio latent's statistics (mean≈0.2,
    # std≈0.92). The PCC gate below is TT-vs-torch with both legs fed the same mel, so the input
    # need not be "real" content — only AUDIO_OUT .wav listening would differ. AUDIO_LATENT
    # (.npy or .pt) still overrides for anyone who wants to decode a real dumped latent.
    _lat = os.environ.get("AUDIO_LATENT")
    if _lat and os.path.exists(_lat):
        latent = (torch.from_numpy(np.load(_lat)) if _lat.endswith(".npy") else torch.load(_lat)).float()
    else:
        torch.manual_seed(0)
        latent = torch.randn(1, als.frames, pipeline.in_channels, dtype=torch.float32) * 0.92 + 0.2

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

    _save(wav, f"girl_audio_conv1d{'_traced' if traced else ''}.wav")

    # Torch-oracle quality gate vs the diffusers vocoder+BWE (real checkpoint, CPU) — the ground
    # truth. The conv1d path runs under BOTH eager and traced (gating the traced vocoder/BWE
    # output, which is otherwise unverified — exactly the gap that let a broken trace ship).
    # Per-1s-interval error-to-signal (rmse/σ, dB) + overall PCC; the absolute level (~−18 dB
    # conv1d) is the fp32/bf16 floor, gated only against a gross spike.
    from models.tt_dit.tests.models.ltx.test_audio_ltx import _build_torch_stage_c_real

    z = pipeline.tt_mel_decoder.z_channels
    audio_spatial = latent.reshape(1, latent.shape[1], z, latent.shape[2] // z).permute(0, 2, 1, 3).float()
    mel = pipeline.tt_mel_decoder(audio_spatial)  # TT mel, fed to both TT and torch
    w_conv = pipeline.tt_vocoder_with_bwe(mel).squeeze(0).float()
    with torch.no_grad():
        w_torch = _build_torch_stage_c_real(ckpt)(mel.float()).squeeze(0).float()
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
    # Gross localized spike only: an interval >4x the median AND past a clearly-audible floor
    # (rmse/σ 0.5 = −6 dB). Also gate overall PCC vs torch — a broken (e.g. mis-traced) vocoder
    # drops PCC far below the ~0.99 fp32/bf16 floor.
    assert cw < 0.5 and cw < 4 * cm + 1e-6, f"conv1d-vs-torch localized spike: worst {cw:.3e} vs median {cm:.3e}"
    assert c_pcc > 0.95, f"conv1d-vs-torch PCC {c_pcc:.4f} below floor (traced={traced}) — vocoder output is wrong"

    pipeline.release_traces()
    dur = wav.shape[-1] / sr
    print(
        f"\nAUDIO_GIRL traced={traced} "
        f"latent_frames={als.frames} out={tuple(wav.shape)} {dur:.2f}s@{sr}Hz cold={cold_ms:.1f}ms warm={warm_ms:.1f}ms",
        flush=True,
    )
    assert torch.isfinite(wav).all(), "decoded waveform has non-finite samples"
    assert abs(dur - num_frames / 24.0) < 0.2, f"duration {dur:.2f}s != ~{num_frames/24.0:.2f}s"
