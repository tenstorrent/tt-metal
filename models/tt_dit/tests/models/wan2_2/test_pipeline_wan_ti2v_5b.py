# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Manual bring-up hook for Wan2.2 TI2V-5B on single BH Galaxy.

    pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b.py -k bh_4x8
"""

import os

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.events import log_event_section
from models.tt_dit.pipelines.wan.pipeline_wan_ti2v_5b import WanTI2V5BPipeline
from models.tt_dit.utils.test import ring_params_req_exact_devices, skip_if_unsupported_num_links


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params, topology",
    [
        [(4, 8), (4, 8), ring_params_req_exact_devices, ttnn.Topology.Ring],
    ],
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_ti2v_5b(mesh_device, mesh_shape, topology):
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B manual bring-up is targeting BH Galaxy first")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    skip_if_unsupported_num_links(mesh_device, 2)

    pipeline = WanTI2V5BPipeline.create_pipeline(
        mesh_device=mesh_device,
        height=704,  # 704/16=44 even; 720/16=45 odd breaks patch_size=2 patchify
        width=1280,
        num_frames=21 if os.environ.get("WAN5B_SMOKE") else 121,
        run_warmup=True,
    )
    assert pipeline.transformer_2 is None
    assert pipeline.transformer.dim == 3072
    assert pipeline.transformer.ffn_dim == 14336


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params, topology",
    [
        [(4, 8), (4, 8), ring_params_req_exact_devices, ttnn.Topology.Ring],
    ],
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_ti2v_5b_generate(mesh_device, mesh_shape, topology):
    """Real T2V generation on single BH Galaxy (4x8): denoise + Wan2.2 residual VAE
    decode, then export an .mp4. Mirrors the Wan2.2-14B test_pipeline_inference flow.

    All knobs are env-overridable::

        WAN5B_STEPS=40 WAN5B_FRAMES=121 WAN5B_HEIGHT=704 WAN5B_WIDTH=1280 \
        WAN5B_SEED=42 WAN5B_PROMPT="..." WAN5B_OUT=/home/ttuser/wan5b.mp4 \
          pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b.py \
            -k "generate and bh_4x8" -sv
    """
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B manual bring-up is targeting BH Galaxy first")

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    skip_if_unsupported_num_links(mesh_device, 2)

    height = int(os.environ.get("WAN5B_HEIGHT", 704))  # 704/16=44 even (720 breaks patch_size=2)
    width = int(os.environ.get("WAN5B_WIDTH", 1280))
    num_frames = int(os.environ.get("WAN5B_FRAMES", 121))
    num_inference_steps = int(os.environ.get("WAN5B_STEPS", 40))
    seed = int(os.environ.get("WAN5B_SEED", 42))
    prompt = os.environ.get(
        "WAN5B_PROMPT",
        "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    )
    out_path = os.environ.get("WAN5B_OUT", f"/home/ttuser/wan5b_t2v_{width}x{height}_{num_frames}f.mp4")
    traced = os.environ.get("WAN5B_TRACED", "0") == "1"
    repeat = int(os.environ.get("WAN5B_REPEAT", "2" if traced else "1"))
    fps = int(os.environ.get("WAN5B_FPS", 24))

    config_overrides = {}
    _vae_tchunk = os.environ.get("WAN5B_VAE_TCHUNK")
    if _vae_tchunk:
        config_overrides["vae_t_chunk_size"] = int(_vae_tchunk)
    if os.environ.get("WAN5B_EXPAND") is not None:
        config_overrides["expand_timesteps"] = os.environ.get("WAN5B_EXPAND", "0") == "1"
    config_overrides = config_overrides or None
    pipeline = WanTI2V5BPipeline.create_pipeline(
        mesh_device=mesh_device,
        height=height,
        width=width,
        num_frames=num_frames,
        run_warmup=True,
        config_overrides=config_overrides,
    )

    logger.info(
        f"Wan2.2 TI2V-5B generate: {height}x{width}, {num_frames} frames, " f"{num_inference_steps} steps, seed={seed}"
    )
    logger.info(f"Prompt: {prompt!r}")

    import collections
    import functools
    import time

    # Lightweight host-vs-device profiling: wrap the host-side (torch/CPU) glue and the
    # device-side text-encode / VAE-decode entry points, accumulate wall time per call.
    prof = collections.OrderedDict()

    def _wrap(obj, name, key):
        orig = getattr(obj, name)

        @functools.wraps(orig)
        def w(*a, **k):
            _t = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                prof[key] = prof.get(key, 0.0) + (time.perf_counter() - _t)

        setattr(obj, name, w)

    _tf = pipeline.transformer
    _wrap(_tf, "get_rope_features", "host_rope")
    _wrap(_tf, "preprocess_spatial_input_host", "host_patchify")
    _wrap(_tf, "postprocess_spatial_output_host", "host_unpatchify")
    _wrap(pipeline._text_encoder, "encode_cfg", "text_encode")
    _wrap(pipeline._vae, "decode", "vae_decode")

    frames = None
    for it in range(repeat):
        prof.clear()
        t0 = time.perf_counter()
        with torch.no_grad():
            frames = pipeline(
                prompts=[prompt],
                num_inference_steps=num_inference_steps,
                seed=seed,
                guidance_scale=4.0,
                guidance_scale_2=3.0,
                output_type="uint8",
                traced=traced,
                on_event=log_event_section,
            )
        dt = time.perf_counter() - t0
        if traced and it == 0:
            tag = "cold_capture"
        elif traced:
            tag = "warm_traced"
        else:
            tag = "eager"
        logger.info(
            f"E2E_PIPELINE_TIME [{tag}] iter={it}: {dt:.2f}s  "
            f"({num_frames}f / {num_inference_steps} steps, {height}x{width}) -> "
            f"{dt / max(num_inference_steps, 1) * 1000:.0f} ms/step incl. encode+decode"
        )
        host_total = prof.get("host_rope", 0) + prof.get("host_patchify", 0) + prof.get("host_unpatchify", 0)
        enc = prof.get("text_encode", 0)
        vae = prof.get("vae_decode", 0)
        logger.info(
            f"HOST_VS_DEVICE [{tag}] iter={it}: e2e={dt:.2f}s | "
            f"host_torch={host_total * 1000:.0f}ms ({host_total / dt * 100:.1f}%) "
            f"[rope={prof.get('host_rope', 0) * 1000:.0f}ms patchify={prof.get('host_patchify', 0) * 1000:.0f}ms "
            f"unpatchify={prof.get('host_unpatchify', 0) * 1000:.0f}ms] | "
            f"text_encode={enc:.2f}s | vae_decode={vae:.2f}s | "
            f"denoise+rest(device)={dt - enc - vae - host_total:.2f}s"
        )

    logger.info(f"Inference done. Output type: {type(frames)}")
    if isinstance(frames, np.ndarray):
        logger.info(f"  shape={frames.shape} range=[{frames.min():.3f}, {frames.max():.3f}]")
    elif isinstance(frames, torch.Tensor):
        logger.info(f"  shape={tuple(frames.shape)} range=[{frames.min().item():.3f}, {frames.max().item():.3f}]")

    frames = frames[0]  # drop batch dim -> (T, H, W, 3)
    if int(ttnn.distributed_context_get_rank()) == 0:
        frames_u8 = np.asarray(frames).astype(np.uint8)
        base = os.path.splitext(out_path)[0]
        # Always dump PNG previews (no ffmpeg needed).
        try:
            from PIL import Image

            t = frames_u8.shape[0]
            for tag, idx in (("first", 0), ("mid", t // 2), ("last", t - 1)):
                Image.fromarray(frames_u8[idx]).save(f"{base}_{tag}.png")
            logger.info(f"Saved preview PNGs: {base}_{{first,mid,last}}.png")
        except Exception as e:  # noqa: BLE001
            logger.info(f"PNG preview failed: {e!r}")
        # Then the mp4 (best-effort; requires imageio_ffmpeg on PYTHONPATH).
        try:
            from models.tt_dit.utils.video import export_to_video

            export_to_video(frames_u8, out_path, fps=fps)
            logger.info(f"Saved video to: {out_path}")
        except Exception as e:  # noqa: BLE001
            logger.info(f"Could not export mp4 ({e!r}); PNG previews still saved")

        # Objective quality gate: CLIP text<->image alignment on evenly-spaced frames
        # (mirrors the Wan2.2-14B check_output_with_clip). Opt out with WAN5B_CLIP=0.
        if os.environ.get("WAN5B_CLIP", "1") == "1":
            clip_threshold = float(os.environ.get("WAN5B_CLIP_THRESHOLD", "36.0"))
            try:
                from PIL import Image

                from models.tt_dit.tests.dataset_eval.clip_encoder import CLIPEncoder

                total = frames_u8.shape[0]
                idxs = np.linspace(0, total - 1, min(8, total), dtype=int)
                clip_encoder = CLIPEncoder()
                scores = []
                for i in idxs:
                    pil_img = Image.fromarray(frames_u8[i])
                    scores.append(clip_encoder.get_clip_score(prompt, pil_img).item() * 100.0)
                clip_mean = sum(scores) / len(scores)
                logger.info(
                    f"CLIP scores: min={min(scores):.2f}, max={max(scores):.2f}, "
                    f"mean={clip_mean:.2f} (threshold {clip_threshold:.2f})"
                )
                assert clip_mean >= clip_threshold, (
                    f"Mean CLIP score {clip_mean:.2f} < threshold {clip_threshold:.2f}; "
                    f"per-frame={[f'{s:.2f}' for s in scores]}"
                )
            except ImportError as e:  # open_clip missing: warn loudly, do not waste the run
                logger.warning(f"CLIP gate skipped (missing dep: {e!r}); `pip install open_clip_torch` to enable")

    assert frames.shape[0] > 0, "no frames generated"


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params, topology",
    [
        [(4, 8), (4, 8), ring_params_req_exact_devices, ttnn.Topology.Ring],
    ],
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_vae_chunk_pcc_ti2v_5b(mesh_device, mesh_shape, topology):
    """Validate that chunked temporal VAE decode is numerically equal to full-T decode.

    The production TI2V-5B config sets ``vae_t_chunk_size=7`` so long clips fit DRAM
    (full-T OOMs at 121f). Chunking carries causal-conv state across chunks via
    ``feat_cache``; this test proves it reproduces the single-pass (full-T) result.

    ``WanDecoder.forward`` selects full-T vs chunked purely from the call-time
    ``t_chunk_size`` (``_feat_cache`` is always allocated; the construction-time
    ``cached`` flag never branches compute), so we decode one fixed latent twice on
    the same adapter, flipping ``_t_chunk_size`` between ``None`` (full-T) and the
    chunk size. A small clip is used so the full-T path fits DRAM.

        WAN5B_PCC_FRAMES=57 WAN5B_PCC_TCHUNK=7 WAN5B_PCC_HEIGHT=256 WAN5B_PCC_WIDTH=512 \
          pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b.py \
            -k "chunk_pcc and bh_4x8" -sv
    """
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B manual bring-up is targeting BH Galaxy first")

    from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))
    skip_if_unsupported_num_links(mesh_device, 2)

    height = int(os.environ.get("WAN5B_PCC_HEIGHT", 256))
    width = int(os.environ.get("WAN5B_PCC_WIDTH", 512))
    num_frames = int(os.environ.get("WAN5B_PCC_FRAMES", 57))
    chunk = int(os.environ.get("WAN5B_PCC_TCHUNK", 7))
    pcc_threshold = float(os.environ.get("WAN5B_PCC_THRESHOLD", 0.999))

    pipeline = WanTI2V5BPipeline.create_pipeline(
        mesh_device=mesh_device,
        height=height,
        width=width,
        num_frames=num_frames,
        run_warmup=True,
    )

    z_dim = pipeline._vae.config.z_dim
    torch.manual_seed(0)
    latents, _ = pipeline.prepare_latents(
        batch_size=1,
        num_channels_latents=z_dim,
        height=height,
        width=width,
        num_frames=num_frames,
        device="cpu",
    )
    num_latent_frames = latents.shape[2]
    assert chunk < num_latent_frames, (
        f"chunk={chunk} >= latent T={num_latent_frames}: chunked path would degenerate to full-T. "
        f"Increase WAN5B_PCC_FRAMES or lower WAN5B_PCC_TCHUNK so chunking is actually exercised."
    )
    logger.info(
        f"VAE chunk-PCC: {height}x{width}, {num_frames}f (latent T={num_latent_frames}); "
        f"full-T vs t_chunk_size={chunk}"
    )

    def _decode(tchunk):
        orig = pipeline._vae._t_chunk_size
        pipeline._vae._t_chunk_size = tchunk
        try:
            out = pipeline._vae.decode(latents, output_type="np")
        finally:
            pipeline._vae._t_chunk_size = orig
        # decode returns a torch bf16 tensor; numpy has no bf16, so cast in-torch first
        return (out if isinstance(out, torch.Tensor) else torch.as_tensor(np.asarray(out))).float().cpu()

    full = _decode(None)
    chunked = _decode(chunk)
    assert full.shape == chunked.shape, f"shape mismatch full={tuple(full.shape)} chunked={tuple(chunked.shape)}"

    if int(ttnn.distributed_context_get_rank()) == 0:
        passed, pcc_msg = comp_pcc(full, chunked, pcc_threshold)
        max_abs = (full - chunked).abs().max().item()
        logger.info(
            f"VAE_CHUNK_PCC full-T vs chunked(t={chunk}): {pcc_msg} | "
            f"max_abs_diff={max_abs:.4e} | shape={tuple(full.shape)}"
        )
        assert passed, f"Chunked VAE decode diverges from full-T (below PCC {pcc_threshold}): {pcc_msg}"
