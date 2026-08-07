# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Stage 2 -- VAE latents + UMT5 text embeddings -> cache, encoded on the mesh."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch
import ttnn
from PIL import Image

from pipeline_config import Config
from preprocess import load_samples, strip_style_words
from utils.tt_encoders import WanTextEncoderTT, WanVAEEncoderTT, close_mesh, make_ccl_manager, open_mesh


def _ttnn_dtype(name: str) -> ttnn.DataType:
    dtype = getattr(ttnn, name, None)
    if not isinstance(dtype, ttnn.DataType):
        raise ValueError(f"not a ttnn dtype: {name!r}")
    return dtype


def _center_crop_resize(img: Image.Image, h: int, w: int) -> Image.Image:
    iw, ih = img.size
    target_ratio, src_ratio = w / h, iw / ih
    if src_ratio > target_ratio:
        new_w = int(round(ih * target_ratio))
        x0 = (iw - new_w) // 2
        img = img.crop((x0, 0, x0 + new_w, ih))
    else:
        new_h = int(round(iw / target_ratio))
        y0 = (ih - new_h) // 2
        img = img.crop((0, y0, iw, y0 + new_h))
    return img.resize((w, h), Image.LANCZOS)


def _pil_to_video_tensor(img: Image.Image, h: int, w: int, num_frames: int = 1) -> torch.Tensor:
    """PIL RGB -> (1, 3, F, H, W) in [-1, 1]; F>1 repeats the still as a static clip."""
    img = _center_crop_resize(img, h, w)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    t = t.unsqueeze(0).unsqueeze(2)
    if num_frames > 1:
        t = t.repeat(1, 1, num_frames, 1, 1)
    return t


def _validate_res(cfg: Config, vae_config) -> None:
    spatial = 2 ** len(vae_config.temperal_downsample)
    multiple = spatial * 2
    for label, v in [("TRAIN_H", cfg.TRAIN_H), ("TRAIN_W", cfg.TRAIN_W)]:
        if v % multiple != 0:
            raise ValueError(
                f"{label}={v} must be a multiple of {multiple} (VAE spatial stride {spatial} * patch_size 2)."
            )
    if (cfg.TRAIN_FRAMES - 1) % 4 != 0:
        raise ValueError(f"TRAIN_FRAMES={cfg.TRAIN_FRAMES} must satisfy 4k+1 (VAE temporal stride 4).")


def precompute(cfg: Config) -> None:
    cache = Path(cfg.CACHE_DIR)
    samples_dir = cache / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(cfg.DATA_DIR)
    if cfg.SUBSET_SIZE and cfg.SUBSET_SIZE > 0:
        samples = samples[: cfg.SUBSET_SIZE]
    print(f"[pre] {len(samples)} (image, caption) pairs from {cfg.DATA_DIR}")

    print(f"[pre] opening mesh device {tuple(cfg.MESH_SHAPE)} ...")
    mesh_device = open_mesh(cfg.MESH_SHAPE)
    try:
        ccl_manager = make_ccl_manager(mesh_device)

        print(f"[pre] building Wan VAE encoder on device (dtype={cfg.VAE_DTYPE}) ...")
        vae = WanVAEEncoderTT(
            checkpoint_name=cfg.MODEL_ID,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            height=cfg.TRAIN_H,
            width=cfg.TRAIN_W,
            num_frames=cfg.TRAIN_FRAMES,
            dtype=_ttnn_dtype(cfg.VAE_DTYPE),
        )
        _validate_res(cfg, vae.config)

        metadata: list[dict] = []
        print(f"[pre] VAE-encoding {len(samples)} images at {cfg.TRAIN_H}x{cfg.TRAIN_W} ...")
        for i, (img, caption) in enumerate(samples):
            video = _pil_to_video_tensor(img, cfg.TRAIN_H, cfg.TRAIN_W, cfg.TRAIN_FRAMES)
            latent = vae.encode(video)
            if cfg.STRIP_STYLE_WORDS:
                caption = strip_style_words(caption)
            triggered = cfg.TRIGGER + caption
            torch.save({"latent": latent, "caption": triggered}, samples_dir / f"sample_{i:04d}.pt")
            metadata.append({"idx": i, "caption": triggered})
            if (i + 1) % 8 == 0 or i == len(samples) - 1:
                print(f"  [pre] {i + 1}/{len(samples)} latent.shape={tuple(latent.shape)}")

        (cache / "metadata.json").write_text(json.dumps(metadata, indent=2))
        del vae
        gc.collect()

        unique_captions = sorted({m["caption"] for m in metadata})
        if "" not in unique_captions:
            unique_captions.append("")  # CFG drop caption

        print("[pre] building UMT5 text encoder on device ...")
        text_encoder = WanTextEncoderTT(
            checkpoint_name=cfg.MODEL_ID,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            max_sequence_length=cfg.MAX_SEQ,
        )
        print(f"[pre] T5-encoding {len(unique_captions)} unique captions ...")
        embeds = text_encoder.encode(unique_captions)
        del text_encoder
        gc.collect()
    finally:
        close_mesh(mesh_device)

    torch.save(embeds, cache / "embeds.pt")
    print(f"[pre] done. cache at {cache.resolve()} — {len(metadata)} samples, {len(embeds)} embeds.")
