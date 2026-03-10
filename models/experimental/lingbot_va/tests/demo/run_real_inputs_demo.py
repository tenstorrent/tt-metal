#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Lingbot-VA PyTorch demo with all real inputs.

Uses real checkpoints (transformer, tokenizer, text_encoder, optional VAE),
real text prompt, and real video (or a single image repeated as video) encoded
via VAE to produce real latents, then runs the transformer in video mode.

Usage:
    # With real video (directory of frames or .mp4/.avi):
    python run_real_inputs_demo.py --video /path/to/frames_or_video.mp4 --prompt "Pick up the bottle."

    # With a single real image (repeated as 8 frames):
    python run_real_inputs_demo.py --image /path/to/image.png --prompt "Pick up the bottle."

    # Checkpoints must exist; run download_pretrained_weights.py first.
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent paths for imports
DEMO_DIR = Path(__file__).parent
sys.path.insert(0, str(DEMO_DIR.parent.parent.parent.parent))

from models.experimental.lingbot_va.reference.model import (
    WanTransformer3DModel,
    load_vae,
    WanVAEStreamingWrapper,
)

# Transformer expects latent shape (B, 48, F, H, W) with patch_size [1,2,2]
LATENT_C = 48
LATENT_F = 8
LATENT_H = 24
LATENT_W = 24
PATCH_SIZE = [1, 2, 2]


def get_mesh_id(f, h, w, t, f_w=1, f_shift=0, action=False):
    f_idx = torch.arange(f_shift, f + f_shift) * f_w
    h_idx = torch.arange(h)
    w_idx = torch.arange(w)
    ff, hh, ww = torch.meshgrid(f_idx, h_idx, w_idx, indexing="ij")
    if action:
        ff_offset = (torch.ones([h]).cumsum(0) / (h + 1)).view(1, -1, 1)
        ff = ff + ff_offset
        hh = torch.ones_like(hh) * -1
        ww = torch.ones_like(ww) * -1
    grid_id = torch.cat([ff.unsqueeze(0), hh.unsqueeze(0), ww.unsqueeze(0)], dim=0).flatten(1)
    grid_id = torch.cat([grid_id, torch.full_like(grid_id[:1], t)], dim=0)
    return grid_id


def load_frames_from_dir(dir_path: Path, num_frames: int, target_h: int, target_w: int, device, dtype):
    """Load image frames from a directory (e.g. frame_000.png, frame_001.png)."""
    from PIL import Image
    import numpy as np

    dir_path = Path(dir_path)
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    paths = []
    for ext in exts:
        paths.extend(sorted(dir_path.glob(ext)))
    if not paths:
        raise FileNotFoundError(f"No images found in {dir_path}")
    # Take evenly spaced frames up to num_frames
    indices = torch.linspace(0, len(paths) - 1, num_frames).long().tolist()
    frames = []
    for i in indices:
        img = Image.open(paths[i]).convert("RGB")
        arr = torch.from_numpy(np.array(img)).float() / 255.0  # [H,W,3]
        arr = arr.permute(2, 0, 1)  # [3,H,W]
        arr = torch.nn.functional.interpolate(
            arr.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False
        ).squeeze(0)
        frames.append(arr)
    # [F, 3, H, W] -> [1, 3, F, H, W] (B, C, F, H, W for VAE)
    stacked = torch.stack(frames, dim=0).to(device=device, dtype=dtype)
    video = stacked.permute(1, 0, 2, 3).unsqueeze(0)  # (3, F, H, W) -> (1, 3, F, H, W)
    # VAE often expects [0,1] or [-1,1]; Wan typically uses [-1, 1]
    video = video * 2.0 - 1.0
    return video


def load_frames_from_image(image_path: Path, num_frames: int, target_h: int, target_w: int, device, dtype):
    """Load a single image and repeat as num_frames (real image, repeated for temporal dimension)."""
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert("RGB")
    arr = torch.from_numpy(np.array(img)).float() / 255.0
    arr = arr.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    arr = torch.nn.functional.interpolate(arr, size=(target_h, target_w), mode="bilinear", align_corners=False)
    # Repeat for F frames: [1,3,H,W] -> [1,3,F,H,W]
    video = arr.unsqueeze(2).repeat(1, 1, num_frames, 1, 1).to(device=device, dtype=dtype)
    video = video * 2.0 - 1.0
    return video


def load_frames_from_video_file(video_path: Path, num_frames: int, target_h: int, target_w: int, device, dtype):
    """Load frames from a video file (.mp4, .avi, etc.) using torchvision or decord if available."""
    try:
        import torchvision.io as tv_io
    except ImportError:
        raise ImportError("torchvision is required for video file input: pip install torchvision")
    # torchvision read_video returns T,H,W,C in 0-255
    v, _, _ = tv_io.read_video(str(video_path), pts_unit="sec", output_format="TCHW")
    if v is None or v.numel() == 0:
        raise ValueError(f"Could not read video: {video_path}")
    v = v.float() / 255.0  # [T,3,H,W]
    T = v.size(0)
    indices = torch.linspace(0, T - 1, num_frames).long().tolist()
    frames = v[indices]
    frames = torch.nn.functional.interpolate(
        frames, size=(target_h, target_w), mode="bilinear", align_corners=False
    ).to(device=device, dtype=dtype)
    # (F, 3, H, W) -> (1, 3, F, H, W) for VAE
    video = frames.permute(1, 0, 2, 3).unsqueeze(0)
    video = video * 2.0 - 1.0
    return video


def encode_video_with_vae(vae_wrapper, video, latent_f, latent_h, latent_w):
    """
    Encode video (1, 3, F, H, W) to latents. Returns (1, C, F', H', W').
    If VAE output shape does not match (1, 48, latent_f, latent_h, latent_w), we interpolate.
    """
    vae_wrapper.clear_cache()
    # Encode in one chunk (full video)
    latents = vae_wrapper.encode_chunk(video)
    # latents: typically (1, C, f, h, w)
    B, C, f, h, w = latents.shape
    if (f, h, w) != (latent_f, latent_h, latent_w):
        latents = torch.nn.functional.interpolate(
            latents.reshape(B, C * f, h, w),
            size=(latent_h, latent_w),
            mode="bilinear",
            align_corners=False,
        )
        latents = latents.reshape(B, C, f, latent_h, latent_w)
        if f != latent_f:
            latents = torch.nn.functional.interpolate(
                latents.permute(0, 1, 3, 4, 2),  # B,C,H,W,F
                size=(latent_h, latent_w, latent_f),
                mode="trilinear",
                align_corners=False,
            ).permute(0, 1, 4, 2, 3)
    return latents.to(video.dtype)


def main():
    parser = argparse.ArgumentParser(description="Lingbot-VA PyTorch demo with all real inputs")
    parser.add_argument("--video", type=str, default=None, help="Path to video file or directory of frames")
    parser.add_argument("--image", type=str, default=None, help="Path to single image (repeated as 8 frames)")
    parser.add_argument("--prompt", type=str, default="Pick up the bottle.", help="Text prompt")
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None, help="Checkpoint root (default: lingbot_va/reference/checkpoints)"
    )
    parser.add_argument("--no-vae", action="store_true", help="Skip VAE; use random latents (then only text is real)")
    args = parser.parse_args()

    # lingbot_va/reference/checkpoints (same as download_pretrained_weights.py)
    default_ckpt = DEMO_DIR.parent.parent / "reference" / "checkpoints"
    checkpoint_dir = Path(args.checkpoint_dir or str(default_ckpt)).resolve()
    ckpt_transformer = checkpoint_dir / "transformer"
    ckpt_tokenizer = checkpoint_dir / "tokenizer"
    ckpt_text_encoder = checkpoint_dir / "text_encoder"
    ckpt_vae = checkpoint_dir / "vae"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print("=" * 60)
    print("  Lingbot-VA PyTorch demo — real inputs")
    print("=" * 60)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f'Prompt: "{args.prompt}"')
    print()

    if not ckpt_transformer.exists():
        print(f"Transformer checkpoint not found: {ckpt_transformer}")
        print("Run: python models/experimental/lingbot_va/tests/download_pretrained_weights.py")
        return 1
    if not ckpt_tokenizer.exists() or not ckpt_text_encoder.exists():
        print("Tokenizer or text_encoder checkpoint not found. Run download_pretrained_weights.py")
        return 1

    # --- Load real models ---
    print("Loading transformer...")
    transformer = WanTransformer3DModel.from_pretrained(str(ckpt_transformer), torch_dtype=dtype, attn_mode="torch").to(
        device
    )
    transformer.eval()

    print("Loading tokenizer and text encoder...")
    try:
        from models.experimental.lingbot_va.reference.model import load_tokenizer, load_text_encoder

        tokenizer = load_tokenizer(str(ckpt_tokenizer))
        text_encoder = load_text_encoder(str(ckpt_text_encoder), dtype, device)
    except Exception:
        from transformers import T5Tokenizer, T5EncoderModel

        tokenizer = T5Tokenizer.from_pretrained(str(ckpt_tokenizer))
        text_encoder = T5EncoderModel.from_pretrained(str(ckpt_text_encoder), torch_dtype=dtype).to(device)
    text_encoder.eval()

    # --- Real text embeddings ---
    text_inputs = tokenizer(
        args.prompt, return_tensors="pt", padding=True, truncation=True, max_length=64
    ).input_ids.to(device)
    with torch.no_grad():
        text_emb = text_encoder(text_inputs).last_hidden_state
    print(f"Text embedding shape: {text_emb.shape}")

    # --- Real video -> real latents ---
    use_real_latents = False
    if args.no_vae:
        noisy_latents = torch.randn(1, LATENT_C, LATENT_F, LATENT_H, LATENT_W, dtype=dtype, device=device)
        print("Using random latents (--no-vae). For real latents, provide --video or --image and VAE checkpoint.")
    elif args.video or args.image:
        video_path = Path(args.video) if args.video else Path(args.image)
        if not video_path.exists():
            print(f"Path not found: {video_path}")
            return 1
        # Video spatial size for VAE input (common 256 or match VAE config if known)
        vae_h, vae_w = 256, 256
        num_frames = LATENT_F
        if video_path.is_dir():
            print(f"Loading frames from directory: {video_path}")
            video = load_frames_from_dir(video_path, num_frames, vae_h, vae_w, device, torch.float32)
        elif video_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
            print(f"Loading video file: {video_path}")
            video = load_frames_from_video_file(video_path, num_frames, vae_h, vae_w, device, torch.float32)
        else:
            print(f"Loading single image: {video_path}")
            video = load_frames_from_image(video_path, num_frames, vae_h, vae_w, device, torch.float32)
        print(f"Video tensor shape: {video.shape}")

        if ckpt_vae.exists():
            print("Loading VAE and encoding video to latents...")
            vae = load_vae(str(ckpt_vae), dtype, device)
            vae_wrapper = WanVAEStreamingWrapper(vae)
            # VAE weights are in dtype (bfloat16); input must match
            video = video.to(dtype)
            with torch.no_grad():
                noisy_latents = encode_video_with_vae(vae_wrapper, video, LATENT_F, LATENT_H, LATENT_W)
            use_real_latents = True
            print(f"Real latents shape: {noisy_latents.shape}")
        else:
            print(f"VAE checkpoint not found at {ckpt_vae}; using random latents.")
            noisy_latents = torch.randn(1, LATENT_C, LATENT_F, LATENT_H, LATENT_W, dtype=dtype, device=device)
    else:
        noisy_latents = torch.randn(1, LATENT_C, LATENT_F, LATENT_H, LATENT_W, dtype=dtype, device=device)
        print("No --video or --image provided. Using random latents.")
        print("For all real inputs: --video /path/to/frames_or_video.mp4 or --image /path/to/image.png")

    patch_f, patch_h, patch_w = PATCH_SIZE
    F_patched = LATENT_F // patch_f
    latent_grid_id = get_mesh_id(
        LATENT_F // patch_f, LATENT_H // patch_h, LATENT_W // patch_w, t=0, f_w=1, f_shift=0, action=False
    ).to(device)
    latent_grid_id = latent_grid_id[:3].unsqueeze(0).repeat(1, 1, 1)

    # Fixed timestep (real diffusion step, not random)
    timesteps = torch.full((1, F_patched), 500.0, device=device, dtype=torch.float32)

    input_dict = {
        "noisy_latents": noisy_latents.to(dtype),
        "text_emb": text_emb,
        "timesteps": timesteps,
        "grid_id": latent_grid_id,
    }

    print("\nRunning transformer forward (video mode)...")
    with torch.no_grad():
        out = transformer(input_dict, update_cache=0, cache_name="pos", action_mode=False)

    print(f"Output shape: {out.shape}")
    print("\nDone.")
    if use_real_latents:
        print("(Video and text were real; transformer ran on real latent + real text conditioning.)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
