# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Encode the cached canvas images with the MiniMax-H3 video VAE encoder on a Tenstorrent mesh.

    PYTHONPATH=$TT_METAL_HOME python encode_vae.py --cache cache/pixelart-150 --weights <hf snapshot>/vae --mesh 1,8

Uses the in-tree tt_dit encoder (models/tt_dit/models/vae/minimax_h3): one 256px tile per device, spatially
stitched, the reference `_encode_clip` at T=1. Torch is only the host container the module takes.

Writes per image `N`: `moments/N.npy` float32 `(48, h, w)` (mean | logvar) and `samples/N.npy` float16
`(rows, 96)`: a posterior sample, normalised by latents_mean/std and patchified, the GPU cache's `video_rows`.
`--reference-dir` holds `N.npy` GPU moments to compare against.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

log = logging.getLogger("minimax_h3.encode_vae")

MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
DEFAULT_WEIGHTS = (
    Path.home() / ".cache/huggingface/hub/models--MiniMaxAI--MiniMax-H3/snapshots" / MODEL_REVISION / "vae"
)
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
PATCH = (1, 2, 2)


def _import_tt_dit():
    root = os.environ.get("TT_METAL_HOME")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    from models.tt_dit.models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3Vae, MiniMaxH3VaeConfig

    return MiniMaxH3Vae, MiniMaxH3VaeConfig


def load_encoder_state(weights_dir: Path) -> dict[str, torch.Tensor]:
    """Only the encoder and quant_conv tensors; the ViT decoder is 9 of the 10 GB."""
    from safetensors import safe_open

    index = json.loads((weights_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    wanted = {k: v for k, v in index.items() if k.startswith("encoder.") or k.startswith("quant_conv.")}
    state = {}
    for shard in sorted(set(wanted.values())):
        with safe_open(str(weights_dir / shard), framework="pt") as f:
            for key, src in wanted.items():
                if src == shard:
                    state[key] = f.get_tensor(key)
    return state


def patchify(latents: np.ndarray) -> np.ndarray:
    """(C, H, W) -> (rows, 4C), row-major over (H/2, W/2); the diffusers patchify at T=1."""
    c, h, w = latents.shape
    x = latents.reshape(c, h // 2, 2, w // 2, 2).transpose(1, 3, 0, 2, 4)
    return np.ascontiguousarray(x.reshape(-1, c * 4))


def sample_rows(moments: np.ndarray, latents_mean, latents_std, rng: np.random.Generator) -> np.ndarray:
    """Posterior sample -> fp16 round trip -> normalise -> patchify, as the GPU `encode_image`."""
    mean, logvar = moments[:24], np.clip(moments[24:], -30.0, 20.0)
    z = mean + np.exp(0.5 * logvar) * rng.standard_normal(mean.shape, dtype=np.float32)
    z = z.astype(np.float16).astype(np.float32)
    z = (z - np.asarray(latents_mean, np.float32)[:, None, None]) / np.asarray(latents_std, np.float32)[:, None, None]
    return patchify(z).astype(np.float16)


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel() - a.astype(np.float64).mean()
    b = b.astype(np.float64).ravel() - b.astype(np.float64).mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="HF vae/ folder")
    p.add_argument("--mesh", default="1,8", help="rows,cols; one tile per device, no collectives")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=0, help="posterior noise seed; per-sample streams derive from it")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--reference-dir", default=None, help="N.npy float32 GPU moments to compare against")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    weights = Path(args.weights)
    if not (weights / "config.json").exists():
        raise SystemExit(
            f"no vae weights at {weights}\n"
            f"  hf download MiniMaxAI/MiniMax-H3 --revision {MODEL_REVISION} --include 'vae/*'"
        )
    cache = Path(args.cache)
    manifest = json.loads((cache / "manifest.json").read_text())
    keys = sorted(manifest, key=lambda k: int(k) if k.isdigit() else k)
    if args.limit:
        keys = keys[: args.limit]
    (cache / "moments").mkdir(exist_ok=True)
    (cache / "samples").mkdir(exist_ok=True)

    MiniMaxH3Vae, MiniMaxH3VaeConfig = _import_tt_dit()
    rows, cols = (int(v) for v in args.mesh.split(","))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    try:
        config = MiniMaxH3VaeConfig.from_pretrained(weights)
        latents_mean, latents_std = tuple(config.latents_mean), tuple(config.latents_std)
        t0 = time.perf_counter()
        vae = MiniMaxH3Vae(
            config,
            task="t2va",
            mesh_device=mesh,
            dtype=ttnn.bfloat16 if args.dtype == "bf16" else ttnn.float32,
            pixel_norm=(PIXEL_MEAN, PIXEL_STD),
        )
        vae.load_state(load_encoder_state(weights))
        log.info(
            "vae encoder ready in %.0f s (%d devices, %s)", time.perf_counter() - t0, mesh.get_num_devices(), args.dtype
        )

        scores = []
        t0 = time.perf_counter()
        done = 0
        for key in keys:
            out_moments, out_rows = cache / "moments" / f"{key}.npy", cache / "samples" / f"{key}.npy"
            if out_rows.exists() and not args.overwrite and not args.reference_dir:
                continue
            pixels = np.load(cache / "pixels" / f"{key}.npy")  # (H, W, 3) uint8
            x = torch.from_numpy(pixels).permute(2, 0, 1)[None, :, None].contiguous()  # (1, 3, 1, H, W) uint8
            moments = vae.encode_clip(x)[0, :, 0].float().cpu().numpy()  # (48, h, w)
            entry = manifest[key]
            if moments.shape != (48, entry["latent_height"], entry["latent_width"]):
                raise RuntimeError(
                    f"{key}: moments {moments.shape} vs manifest {entry['latent_height']}x{entry['latent_width']}"
                )
            np.save(out_moments, moments.astype(np.float32))
            rng = np.random.default_rng([args.seed, int(key) if key.isdigit() else abs(hash(key))])
            np.save(out_rows, sample_rows(moments, latents_mean, latents_std, rng))
            done += 1
            if args.reference_dir:
                ref = np.load(Path(args.reference_dir) / f"{key}.npy").astype(np.float32)
                p_mean, p_logvar = pcc(moments[:24], ref[:24]), pcc(moments[24:], ref[24:])
                lm = np.asarray(latents_mean, np.float32)[:, None, None]
                ls = np.asarray(latents_std, np.float32)[:, None, None]
                rel = np.linalg.norm((moments[:24] - ref[:24]) / ls) / np.linalg.norm((ref[:24] - lm) / ls)
                scores.append(p_mean)
                log.info(
                    "%s: %dx%d pcc mean %.5f logvar %.5f | normalised-mean rel err %.4f | max|d mean| %.3f",
                    key,
                    moments.shape[1],
                    moments.shape[2],
                    p_mean,
                    p_logvar,
                    rel,
                    np.abs(moments[:24] - ref[:24]).max(),
                )
            if done % 10 == 0 or done == len(keys):
                log.info("[%d/%d] %.2f s/sample", done, len(keys), (time.perf_counter() - t0) / max(done, 1))
        if scores:
            log.info(
                "reference pcc(mean): worst %.5f mean %.5f over %d samples",
                min(scores),
                float(np.mean(scores)),
                len(scores),
            )
    finally:
        ttnn.close_mesh_device(mesh)
    log.info("done: moments/ and samples/ in %s", cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
