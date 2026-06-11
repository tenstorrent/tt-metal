# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-seed image-encode comparison for the Wan2.2 4-step distill (quad 4x32).

Builds the distill pipeline ONCE, then generates N seeds for each of four
image-encode configs, reusing the same transformers/decoder. For each config we
record the per-stage encode time (``prepare_latents``) and the full pipeline
time, save the mp4, and compute a per-frame PCC against the ``full_default``
baseline (same seed) as an automated artifact proxy — a sharp PCC drop on a late
frame is the duplicate-subject signature.

Configs (all share the same transformers; only the VAE image-encode differs):

  full_default       encode all 81 pixel frames, slow H/W=0 default blockings — control / shipped-safe.
  trunc_default      encode only 33 + replicate, slow default blockings — is truncation alone safe?
  trunc_swept        encode 33 + replicate, swept blockings (T_out_block as swept) — fast but suspected artifact.
  trunc_swept_tout1  encode 33 + replicate, swept blockings with T_out_block forced to 1 — fast + hypothesised clean.

Env:
  PROMPT_IMAGE     conditioning image (default ./prompt_image.png)
  COMPARE_SEEDS    comma-separated seeds (default "42,123,7")
  COMPARE_PROMPT   text prompt
"""
import os
import time

import numpy as np
import PIL
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.experimental.pipelines.pipeline_wan_distill import WanDistillPipelineI2V
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import ImagePrompt
from models.tt_dit.utils.test import ring_params_8k


def _pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten().to(torch.float64)
    y = y.flatten().to(torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else 1.0


def _extract_frames(result) -> np.ndarray:
    arr = result.frames if hasattr(result, "frames") else (result[0] if isinstance(result, tuple) else result)
    return np.asarray(arr)[0]  # [F, H, W, 3]


class _TimedEncoder:
    """Callable proxy around a WanEncoder that records device-synchronized
    forward time per call, isolating VAE-encode compute from the host overhead
    in prepare_latents. Delegates all other attribute access to the encoder."""

    def __init__(self, enc, mesh_device, store: list):
        self._enc = enc
        self._mesh = mesh_device
        self._store = store

    def __call__(self, *args, **kwargs):
        ttnn.synchronize_device(self._mesh)
        t0 = time.perf_counter()
        out = self._enc(*args, **kwargs)
        ttnn.synchronize_device(self._mesh)
        self._store.append(time.perf_counter() - t0)
        return out

    def __getattr__(self, name):
        return getattr(self._enc, name)


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(4, 32), (4, 32), 2, False, ring_params_8k, ttnn.Topology.Ring, False],
    ],
    ids=["bh_4x32sp1tp0_ring"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "width, height",
    [(1280, 720)],
    ids=["resolution_720p"],
)
def test_encode_compare_seeds(mesh_device, mesh_shape, num_links, dynamic_load, topology, width, height, is_fsdp):
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    pil_image = PIL.Image.open(os.environ.get("PROMPT_IMAGE", "./prompt_image.png"))
    image_prompt = [ImagePrompt(image=pil_image, frame_pos=0)]
    num_frames = 81
    seeds = [int(s) for s in os.environ.get("COMPARE_SEEDS", "42,123,7").split(",")]
    prompt = os.environ.get("COMPARE_PROMPT", "The cat in the hat runs up the hill to the house.")

    pipeline = WanDistillPipelineI2V.create_pipeline(
        mesh_device=mesh_device,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        height=height,
        width=width,
        num_frames=num_frames,
    )

    # Baseline (quality-safe) encoder the base built with H/W=0 -> default blockings.
    enc_slow = pipeline.tt_vae_encoder
    base_chunk = pipeline._encoder_t_chunk_size  # 4
    orig_encode_frames = pipeline._encode_frames_for  # bound method, returns min(33, nf)
    fast_chunk = pipeline._ENCODER_T_CHUNK_BY_MESH.get(tuple(mesh_device.shape), pipeline.DISTILL_ENCODE_FRAMES)

    # Build the two swept fast encoders (cheap vs transformers; weights cached).
    logger.info("[compare] building swept fast encoder (T_out as-swept)...")
    enc_swept = pipeline._build_fast_vae_encoder(force_t_out_block_1=False)
    logger.info("[compare] building swept fast encoder (T_out=1)...")
    enc_swept_t1 = pipeline._build_fast_vae_encoder(force_t_out_block_1=True)

    def full_frames(nf, mcp):
        return nf

    configs = [
        ("full_default", enc_slow, base_chunk, full_frames),
        ("trunc_default", enc_slow, base_chunk, orig_encode_frames),
        ("trunc_swept", enc_swept, fast_chunk, orig_encode_frames),
        ("trunc_swept_tout1", enc_swept_t1, fast_chunk, orig_encode_frames),
        # Same fast encoder as trunc_swept_tout1, but assembles the zero frames
        # on-device and only transfers the conditioned frame (host-overhead cut).
        ("trunc_swept_tout1_ondev", enc_swept_t1, fast_chunk, orig_encode_frames),
    ]

    # Time prepare_latents (= image-encode stage) via a wrapper.
    timings = {}
    orig_prepare = pipeline.prepare_latents

    def timed_prepare(*a, **k):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        r = orig_prepare(*a, **k)
        ttnn.synchronize_device(mesh_device)
        timings["enc"] = time.perf_counter() - t0
        return r

    pipeline.prepare_latents = timed_prepare

    refs: dict[int, np.ndarray] = {}  # full_default frames per seed (artifact reference)
    enc_times: dict[str, list] = {}
    total_times: dict[str, list] = {}
    fwd_times: dict[str, list] = {}
    quality: list[tuple] = []  # (config, seed, min_pcc, min_frame, mean_pcc)

    for name, enc, chunk, frames_fn in configs:
        fwd_store: list = []
        pipeline.tt_vae_encoder = _TimedEncoder(enc, mesh_device, fwd_store)
        pipeline._encoder_t_chunk_size = chunk
        pipeline._encode_frames_for = frames_fn
        # On-device conditioning assembly is gated by an env flag read per-call
        # inside _encode_image_condition; enable it only for the *_ondev config.
        os.environ["WAN_DISTILL_ONDEVICE_COND"] = "1" if name.endswith("_ondev") else "0"
        enc_times[name] = []
        total_times[name] = []
        fwd_times[name] = fwd_store

        for seed in seeds:
            timings.clear()
            t_total = time.perf_counter()
            with torch.no_grad():
                result = pipeline(
                    prompts=[prompt],
                    image_prompt=image_prompt,
                    negative_prompts=[""],
                    num_inference_steps=4,
                    seed=seed,
                    guidance_scale=1.0,
                    guidance_scale_2=1.0,
                    output_type="uint8",
                )
            total = time.perf_counter() - t_total
            frames = _extract_frames(result)

            enc_t = timings.get("enc", float("nan"))
            enc_times[name].append(enc_t)
            total_times[name].append(total)

            out_fn = f"compare_{name}_seed{seed}.mp4"
            try:
                from models.tt_dit.utils.video import export_to_video

                export_to_video(frames, out_fn, fps=16)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[compare] video export failed for {out_fn}: {e}")

            fwd_t = fwd_store[-1] if fwd_store else float("nan")
            if name == "full_default":
                refs[seed] = frames
                logger.info(
                    f"[compare] {name} seed={seed} encfwd={fwd_t:.2f}s prep={enc_t:.2f}s total={total:.2f}s -> {out_fn}"
                )
            else:
                ref = refs[seed].astype(np.float64)
                cur = frames.astype(np.float64)
                pccs = [_pcc(torch.from_numpy(ref[f]), torch.from_numpy(cur[f])) for f in range(ref.shape[0])]
                min_pcc = float(np.min(pccs))
                min_f = int(np.argmin(pccs))
                mean_pcc = float(np.mean(pccs))
                quality.append((name, seed, min_pcc, min_f, mean_pcc))
                logger.info(
                    f"[compare] {name} seed={seed} encfwd={fwd_t:.2f}s prep={enc_t:.2f}s total={total:.2f}s "
                    f"minPCC={min_pcc:.4f}@f{min_f} meanPCC={mean_pcc:.4f} -> {out_fn}"
                )

    pipeline.prepare_latents = orig_prepare
    pipeline._encode_frames_for = orig_encode_frames

    # ---- Summary ----
    logger.info("=" * 88)
    logger.info("ENCODE-COMPARE SUMMARY (quad 4x32, 720p, 81 frames, untraced)")
    logger.info(f"seeds={seeds}  prompt={prompt!r}")
    logger.info("-" * 88)
    logger.info(
        f"{'config':22s} {'encfwd*':>10s} {'prep mean':>11s} {'total mean':>12s}   (*encfwd = compiled, skips seed-1)"
    )
    for name, _, _, _ in configs:
        fwd = fwd_times[name]
        fwd_compiled = np.nanmean(fwd[1:]) if len(fwd) > 1 else (fwd[0] if fwd else float("nan"))
        logger.info(
            f"{name:22s} {fwd_compiled:>9.2f}s {np.nanmean(enc_times[name]):>10.2f}s {np.nanmean(total_times[name]):>11.2f}s"
        )
    logger.info("-" * 88)
    logger.info("Quality vs full_default (same seed) — low minPCC on a late frame = artifact:")
    for name, seed, min_pcc, min_f, mean_pcc in quality:
        flag = "  <== SUSPECT" if min_pcc < 0.92 else ""
        logger.info(f"  {name:22s} seed={seed}: minPCC={min_pcc:.4f} @frame{min_f:02d}  meanPCC={mean_pcc:.4f}{flag}")
    logger.info("=" * 88)
