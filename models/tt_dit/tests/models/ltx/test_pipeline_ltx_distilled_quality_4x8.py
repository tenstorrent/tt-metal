# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Quality gates for the served LTX-2.3 distilled configuration: 6 s clips (145 frames @ 24 fps,
1088x1920) on a 4x8 Blackhole Galaxy, Ring topology, traced.

Two gaps these close (found while chasing crackly served audio in Sep 2026):

1. ``test_pipeline_distilled`` gates video (VBench, CLIP) but nothing gates the *audio* the served
   pipeline emits. ``test_ltx_6s_audio_matches_torch_reference`` runs the real traced pipeline, captures
   the audio latent on its way into ``decode_audio``, decodes that same latent with the CPU torch
   reference (diffusers mel-VAE + vocoder + BWE, real checkpoint weights) and gates the device audio
   against it. Because the reference starts from the latent, a wrong mel decoder, vocoder, BWE, weight
   cache or trace replay all fail it; a wrong latent is caught by the whiteness/zeros sanity gates.
2. Prepared conv3d weights (``prepare_conv3d_weights``) depend on the tt-metal build, but the tt_dit
   weight cache key does not. ``test_ltx_audio_weight_cache_matches_regeneration`` regenerates the audio
   caches with the running binary and requires the live ``TT_DIT_CACHE_DIR`` copy to match byte for byte.

Run (repo root, venv active):
    LTX_CHECKPOINT=<local .safetensors> TT_DIT_CACHE_DIR=<the cache the server uses> \
    pytest models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled_quality_4x8.py -s --timeout 3600
"""

import hashlib
import os
import shutil
import tempfile

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.audio_vae.audio_decoder_ltx import LTXAudioDecoderAdapter
from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import WHITENESS_WARN_THRESHOLD, LTXDistilledPipeline
from models.tt_dit.utils.ltx import DEFAULT_LTX_PROMPT, default_ltx_checkpoint, default_ltx_gemma, traced_default
from models.tt_dit.utils.test import skip_if_unsupported_num_links

from .ltx_mesh_params import _4x8sp1tp0nl2_ring_is_fsdp0, _override_base_device_params, _ring_trace, _with_dynamic_load
from .test_audio_ltx import _decode_audio_reference, _psnr
from .test_pipeline_ltx_distilled import _ltx_checkpoint_cached

CHECKPOINT_FILE = "ltx-2.3-22b-distilled-1.1.safetensors"

# The served shape. Fixed here on purpose: these gates are calibrated for it, not parametrized.
NUM_FRAMES, FPS, HEIGHT, WIDTH = 145, 24.0, 1088, 1920
SEED = 10

# Audio gates. 28 dB is the floor test_audio_decode_e2e_psnr uses for the decode chain; on real
# generated content the bf16 chain measured 28.7 dB / PCC 0.984 against the fp32 torch reference, so the
# PCC floor is the 0.95 the vocoder-oracle test already uses (a stale mel-decoder cache measured 0.006).
AUDIO_PSNR_MIN_DB = 28.0
AUDIO_PCC_MIN = 0.95
AUDIO_RMS_MIN = 5e-3  # DEFAULT_LTX_PROMPT is a strummed guitar: silence means the decode died
AUDIO_CLIP_FRACTION_MAX = 0.01
# Replay gates: gen #2 is a pure trace replay of the same prompt/seed as gen #1 and has measured
# bit-identical audio latents; anything drifting is a replay corruption.
REPLAY_AUDIO_LATENT_PCC_MIN = 0.999
REPLAY_VIDEO_PSNR_MIN_DB = 35.0

# 4x8 BH Galaxy, Ring, 2 links, trace region + L1_SMALL -- the production mesh configuration.
QUALITY_4X8_RING_PARAMS = [
    _with_dynamic_load(_override_base_device_params(_4x8sp1tp0nl2_ring_is_fsdp0, _ring_trace), False)
]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.corrcoef(torch.stack([a.float().flatten(), b.float().flatten()]))[0, 1].item()


def _md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _make_pipeline(mesh_device, *, ckpt, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp, traced):
    return LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=ckpt,
        gemma_path=default_ltx_gemma(),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=True,
        traced=traced,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        fps=FPS,
        image_conditioning=False,
    )


@pytest.mark.skipif(not _ltx_checkpoint_cached(CHECKPOINT_FILE), reason="needs the LTX distilled checkpoint")
@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, dynamic_load",
    QUALITY_4X8_RING_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_ltx_6s_audio_matches_torch_reference(
    mesh_device, device_params, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp
):
    """Served 6 s AV generation: device audio must match the CPU torch decode of the same latent."""
    skip_if_unsupported_num_links(mesh_device, num_links)
    traced = traced_default(device_params, os.environ.get("LTX_TRACED"))
    ckpt = LTXDistilledPipeline._resolve_checkpoint_file(default_ltx_checkpoint(CHECKPOINT_FILE))
    parent = mesh_device
    mesh_device = parent.create_submesh(ttnn.MeshShape(*tuple(parent.shape)))

    pipeline = _make_pipeline(
        mesh_device,
        ckpt=ckpt,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        traced=traced,
    )

    # Capture the audio latent on its way into decode_audio; the pipeline API only returns the waveform.
    captured: dict[str, object] = {}
    orig_decode_audio = pipeline.decode_audio

    def capturing_decode_audio(audio_latent, num_frames, fps=24.0):
        captured["latent"] = audio_latent.detach().clone()
        return orig_decode_audio(audio_latent, num_frames, fps=fps)

    pipeline.decode_audio = capturing_decode_audio

    prompt = os.environ.get("PROMPT", DEFAULT_LTX_PROMPT)
    n_gens = 2 if traced else 1  # traced: gen #2 is the pure replay the server serves
    results = []
    for gen in range(n_gens):
        frames, audio = pipeline.generate(
            prompt, output_path=None, num_frames=NUM_FRAMES, height=HEIGHT, width=WIDTH, seed=SEED, fps=FPS
        )
        latent = captured.pop("latent")
        results.append((frames, audio, latent))

        # --- latent sanity (the pipeline's own corruption fingerprints, asserted instead of logged) ---
        stats = LTXDistilledPipeline._latent_stats(latent)
        logger.info(f"gen {gen} audio latent: {stats}")
        assert stats["nonfinite"] == 0, f"gen {gen}: {stats['nonfinite']} non-finite latent elements"
        assert stats["zeros"] < 0.01, f"gen {gen}: {stats['zeros']:.2%} of the audio latent is exactly zero"
        assert (
            stats["whiteness"] < WHITENESS_WARN_THRESHOLD
        ), f"gen {gen}: audio latent is ~white noise ({stats['whiteness']:.2f})"

        # --- device audio vs CPU torch reference from the SAME latent ---
        ref = _decode_audio_reference(ckpt, latent, NUM_FRAMES, fps=FPS)
        assert audio.sampling_rate == ref.sampling_rate
        n = min(audio.waveform.shape[-1], ref.waveform.shape[-1])
        dev_w, ref_w = audio.waveform[..., :n].float(), ref.waveform[..., :n].float()
        psnr, pcc = _psnr(ref_w, dev_w), _pcc(ref_w, dev_w)
        rms = dev_w.pow(2).mean().sqrt().item()
        clipped = (dev_w.abs() >= 0.999).float().mean().item()
        logger.info(
            f"gen {gen} audio vs torch reference: PSNR={psnr:.2f} dB PCC={pcc:.5f} rms={rms:.4f} clipped={clipped:.3%}"
        )
        assert torch.isfinite(dev_w).all(), f"gen {gen}: non-finite audio samples"
        assert (
            psnr >= AUDIO_PSNR_MIN_DB
        ), f"gen {gen}: audio PSNR {psnr:.2f} dB < {AUDIO_PSNR_MIN_DB} dB vs torch reference"
        assert pcc >= AUDIO_PCC_MIN, f"gen {gen}: audio PCC {pcc:.4f} < {AUDIO_PCC_MIN} vs torch reference"
        assert rms > AUDIO_RMS_MIN, f"gen {gen}: audio is near-silent (rms {rms:.4f})"
        assert clipped < AUDIO_CLIP_FRACTION_MAX, f"gen {gen}: {clipped:.2%} of samples clipped"

        # --- video sanity: right shape, not flat, not frozen ---
        assert frames.shape == (1, 3, NUM_FRAMES, HEIGHT, WIDTH), f"gen {gen}: frames {frames.shape}"
        f = frames[0].astype(np.float32)
        assert f.std() > 5.0, f"gen {gen}: video is flat (std {f.std():.2f})"
        assert np.abs(f[:, 0] - f[:, -1]).mean() > 1.0, f"gen {gen}: first and last frames are identical"

    if n_gens == 2:
        (f1, _, l1), (f2, _, l2) = results
        lat_pcc = _pcc(l1, l2)
        vid_psnr = _psnr(torch.from_numpy(f1.astype(np.float32)), torch.from_numpy(f2.astype(np.float32)))
        logger.info(f"replay vs capture: audio latent PCC={lat_pcc:.6f} video PSNR={vid_psnr:.1f} dB")
        assert lat_pcc >= REPLAY_AUDIO_LATENT_PCC_MIN, f"trace replay changed the audio latent (PCC {lat_pcc:.5f})"
        assert vid_psnr >= REPLAY_VIDEO_PSNR_MIN_DB, f"trace replay changed the video (PSNR {vid_psnr:.1f} dB)"

    if traced:
        pipeline.release_traces()


@pytest.mark.skipif(not _ltx_checkpoint_cached(CHECKPOINT_FILE), reason="needs the LTX distilled checkpoint")
@pytest.mark.skipif(
    not os.environ.get("TT_DIT_CACHE_DIR"), reason="no live weight cache to check (TT_DIT_CACHE_DIR unset)"
)
@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp, dynamic_load",
    QUALITY_4X8_RING_PARAMS,
    indirect=["mesh_device", "device_params"],
)
def test_ltx_audio_weight_cache_matches_regeneration(
    mesh_device, device_params, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp
):
    """The live audio prepared-weight caches must equal what the running binary prepares today.

    prepare_conv3d_weights lays weights out per build; the cache key carries no build id, so a cache
    written by an older build loads silently and decodes to structured garbage (right envelope, crackle).
    """
    skip_if_unsupported_num_links(mesh_device, num_links)
    live_root = os.environ["TT_DIT_CACHE_DIR"]
    ckpt = LTXDistilledPipeline._resolve_checkpoint_file(default_ltx_checkpoint(CHECKPOINT_FILE))
    model_name = os.path.basename(ckpt).removesuffix(".safetensors")
    parent = mesh_device
    mesh_device = parent.create_submesh(ttnn.MeshShape(*tuple(parent.shape)))

    scratch = tempfile.mkdtemp(prefix="tt_dit_regen_")
    try:
        os.environ["TT_DIT_CACHE_DIR"] = scratch
        # Audio-only shell (no checkpoint at construction => no DiT/VAE prime), as test_audio_decode_girl does.
        pipe = LTXDistilledPipeline.create_pipeline(
            mesh_device=mesh_device,
            checkpoint_name=None,
            gemma_path=default_ltx_gemma(),
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            num_links=num_links,
            dynamic_load=dynamic_load,
            topology=topology,
            is_fsdp=is_fsdp,
            run_warmup=False,
            traced=False,
            num_frames=NUM_FRAMES,
            height=HEIGHT,
            width=WIDTH,
            fps=FPS,
        )
        pipe.checkpoint_name = ckpt
        adapter = LTXAudioDecoderAdapter(
            ckpt,
            mesh_device=mesh_device,
            vae_ccl_manager=pipe.vae_ccl_manager,
            dit_parallel_config=pipe.parallel_config,
            traced=False,
        )
        adapter.reload_weights()  # cache miss in the scratch dir: prepares with THIS binary and writes tensorbins
    finally:
        os.environ["TT_DIT_CACHE_DIR"] = live_root

    regen_model_dir = os.path.join(scratch, model_name)
    subfolders = sorted(d for d in os.listdir(regen_model_dir) if d.startswith("audio_"))
    assert subfolders, f"regeneration wrote no audio caches under {regen_model_dir}"
    problems = []
    for sub in subfolders:
        new_dir, live_dir = os.path.join(regen_model_dir, sub), os.path.join(live_root, model_name, sub)
        if not os.path.isdir(live_dir):
            logger.info(f"{sub}: not present in the live cache ({live_dir}); nothing to compare")
            continue
        n_tot = n_bad = 0
        for dp, _, fns in os.walk(new_dir):
            for fn in fns:
                rel = os.path.relpath(os.path.join(dp, fn), new_dir)
                n_tot += 1
                live = os.path.join(live_dir, rel)
                if not os.path.exists(live) or _md5(live) != _md5(os.path.join(dp, fn)):
                    n_bad += 1
        logger.info(f"{sub}: {n_bad}/{n_tot} files differ between the live cache and a fresh regeneration")
        if n_bad:
            problems.append(f"{sub}: {n_bad}/{n_tot} files differ")
    shutil.rmtree(scratch, ignore_errors=True)
    assert not problems, (
        "live audio weight cache is stale for this binary: "
        + "; ".join(problems)
        + f". Delete {os.path.join(live_root, model_name)}/audio_* and let the pipeline regenerate it."
    )
