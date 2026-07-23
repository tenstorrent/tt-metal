# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# VAE-only decode: random OR saved real latent -> PyTorch fp32 VAE -> PNG.
# Skips DiT/denoise and Tenstorrent devices; use to check VAE decode in isolation.
#
# Real latent (1024x1024):
#   HY_LATENT=models/experimental/hunyuan_image_3_0/real_latent_1024.pt \
#     HY_OUT=models/experimental/hunyuan_image_3_0/vae_real_1024.png \
#     python_env/bin/python models/experimental/hunyuan_image_3_0/demo/vae_decode_random.py
#
# Random latent (1024x1024):
#   python_env/bin/python models/experimental/hunyuan_image_3_0/demo/vae_decode_random.py
#
# Fast smoke (64x64 image, random only):
#   HY_GRID=4 python_env/bin/python models/experimental/hunyuan_image_3_0/demo/vae_decode_random.py
#
# Env:
#   HUNYUAN_MODEL_DIR  checkpoint dir (default: ensure_base_weights() / HF hub)
#   HY_LATENT          path to .pt from export_latent.py or demo (HY_SAVE_LATENT)
#   HY_GRID            latent side G -> image G*16 (default 64 => 1024; ignored if HY_LATENT set)
#   HY_SEED            RNG seed for random mode (default 42)
#   HY_OUT             output PNG path

import os
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = str(Path(__file__).resolve().parents[4])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.experimental.hunyuan_image_3_0.ref.weights import ensure_base_weights
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import Z_CHANNELS, load_decoder, vae_decode_output_to_rgb
from models.experimental.hunyuan_image_3_0.ref.model_config import VAE_SCALING_FACTOR

WEIGHTS = ensure_base_weights()
os.environ.setdefault("HUNYUAN_MODEL_DIR", str(WEIGHTS))

LATENT_PATH = os.environ.get("HY_LATENT")
GRID = int(os.environ.get("HY_GRID", "64"))
SEED = int(os.environ.get("HY_SEED", "42"))
OUT_PNG = os.environ.get("HY_OUT", str(Path(__file__).resolve().parent.parent / "vae_random_torch.png"))
DEFAULT_SCALING = VAE_SCALING_FACTOR


def _load_latent():
    scaling = DEFAULT_SCALING
    if not LATENT_PATH:
        torch.manual_seed(SEED)
        latent = torch.randn(1, Z_CHANNELS, GRID, GRID, dtype=torch.float32)
        return latent, GRID, "random", scaling

    obj = torch.load(LATENT_PATH, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "latent" in obj:
        latent = obj["latent"].float()
        scaling = float(obj.get("scaling_factor", DEFAULT_SCALING))
        meta = {k: obj.get(k) for k in ("prompt", "seed", "image_size", "grid")}
        print(f"[vae_decode] loaded metadata: {meta}", flush=True)
    else:
        latent = obj.float() if isinstance(obj, torch.Tensor) else obj["latent"].float()
    if latent.ndim != 4 or latent.shape[1] != Z_CHANNELS:
        raise SystemExit(f"expected latent [B,{Z_CHANNELS},H,W], got {tuple(latent.shape)}")
    grid = int(latent.shape[-1])
    assert latent.shape[-2] == grid, f"expected square latent grid, got {tuple(latent.shape)}"
    return latent, grid, LATENT_PATH, scaling


@torch.no_grad()
def _decode_torch(latent_bchw, scaling):
    z_bcthw = (latent_bchw / scaling).unsqueeze(2)
    out = load_decoder()(z_bcthw)
    return vae_decode_output_to_rgb(out)  # [B, 3, H, W]


def main():
    latent, grid, source, scaling = _load_latent()
    print(f"[vae_decode] backend=torch  weights={WEIGHTS}", flush=True)
    print(
        f"[vae_decode] source={source!r}  latent {tuple(latent.shape)}  "
        f"grid={grid}  image={grid * 16}x{grid * 16}  scaling={scaling}",
        flush=True,
    )

    rgb = _decode_torch(latent, scaling)
    print(
        f"[vae_decode] rgb {tuple(rgb.shape)}  min={rgb.min():.4f}  max={rgb.max():.4f}  mean={rgb.mean():.4f}",
        flush=True,
    )

    out_path = Path(OUT_PNG)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = (rgb[0].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    Image.fromarray(arr).save(out_path)
    print(f"[vae_decode] saved -> {out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
