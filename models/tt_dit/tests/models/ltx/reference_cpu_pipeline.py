"""CPU-only LTX-2 reference pipeline for debugging TT-DiT output.

Requires LTX-2 cloned beside tt-metal and checkpoint/Gemma weights on disk.
Run from TT_METAL_HOME root.
"""

import argparse
import gc
import os
import sys
import time

import imageio
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
_LTX_ROOT = os.path.join(_REPO_ROOT, "LTX-2")
sys.path.insert(0, os.path.join(_LTX_ROOT, "packages/ltx-core/src"))
sys.path.insert(0, os.path.join(_LTX_ROOT, "packages/ltx-pipelines/src"))
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda *a, **kw: None
    torch.cuda.empty_cache = lambda *a, **kw: None

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationConfig
from ltx_core.model.transformer.model import Modality
from ltx_core.types import VideoLatentShape
from ltx_pipelines.utils.helpers import encode_prompts
from ltx_pipelines.utils.model_ledger import ModelLedger


def parse_args():
    parser = argparse.ArgumentParser(description="Run LTX-2 CPU reference pipeline")
    parser.add_argument(
        "--prompt", default="A golden retriever running through a field of sunflowers, cinematic lighting"
    )
    parser.add_argument(
        "--checkpoint", default=os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors")
    )
    parser.add_argument("--gemma-path", default=os.environ.get("GEMMA_PATH", ""))
    parser.add_argument("--output", default="ltx_reference_cpu.mp4")
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--cfg", type=float, default=3.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.gemma_path:
        raise SystemExit("Set --gemma-path or GEMMA_PATH to Gemma-3 weights")

    fps = 24.0
    latent_frames = (args.num_frames - 1) // 8 + 1
    lh = args.height // 32
    lw = args.width // 32
    n_tokens = latent_frames * lh * lw

    print("Creating ModelLedger...")
    t0 = time.time()
    ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=args.checkpoint,
        gemma_root_path=args.gemma_path,
    )
    print(f"ModelLedger created in {time.time() - t0:.1f}s")

    print("Encoding prompts...")
    t0 = time.time()
    results = encode_prompts([args.prompt, ""], ledger)
    v_ctx = results[0].video_encoding
    v_ctx_neg = results[1].video_encoding
    print(f"Video context: {v_ctx.shape} (encoded in {time.time() - t0:.1f}s)")

    print("Loading transformer...")
    t0 = time.time()
    transformer = ledger.transformer()
    print(f"Transformer loaded in {time.time() - t0:.1f}s")

    scheduler = LTX2Scheduler()
    sigmas = scheduler.execute(steps=args.steps, latent=torch.randn(1, 1, n_tokens))
    print(f"Sigmas: {sigmas}")

    torch.manual_seed(args.seed)
    latent = torch.randn(1, n_tokens, 128, dtype=torch.bfloat16) * sigmas[0]

    patchifier = VideoLatentPatchifier(patch_size=1)
    target_shape = VideoLatentShape(batch=1, channels=128, frames=latent_frames, height=lh, width=lw)
    latent_coords = patchifier.get_patch_grid_bounds(output_shape=target_shape, device="cpu")
    pixel_coords = get_pixel_coords(latent_coords, (8, 32, 32), causal_fix=True)
    video_pos = pixel_coords.float()
    video_pos[:, 0, ...] = video_pos[:, 0, ...] / fps
    video_pos = video_pos.bfloat16()

    stepper = EulerDiffusionStep()
    for step_idx in range(args.steps):
        sigma = sigmas[step_idx]
        t_step_start = time.time()
        timesteps = (torch.ones(1, n_tokens) * sigma).unsqueeze(-1)
        vm = Modality(
            latent=latent,
            sigma=torch.tensor([sigma.item()]),
            timesteps=timesteps,
            positions=video_pos,
            context=v_ctx,
            enabled=True,
            context_mask=None,
            attention_mask=None,
        )
        perturb = BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=None)])
        with torch.no_grad():
            denoised, _ = transformer(video=vm, audio=None, perturbations=perturb)
        vm_neg = Modality(
            latent=latent,
            sigma=torch.tensor([sigma.item()]),
            timesteps=timesteps,
            positions=video_pos,
            context=v_ctx_neg,
            enabled=True,
            context_mask=None,
            attention_mask=None,
        )
        with torch.no_grad():
            uncond, _ = transformer(video=vm_neg, audio=None, perturbations=perturb)
        pred = denoised + (args.cfg - 1) * (denoised - uncond)
        factor = 0.7 * (denoised.std() / pred.std()) + 0.3
        denoised = pred * factor
        latent = stepper.step(latent.float(), denoised.float(), sigmas, step_idx).bfloat16()
        print(
            f"Step {step_idx + 1}/{args.steps}: sigma {sigma:.4f}, "
            f"range [{latent.float().min():.3f}, {latent.float().max():.3f}], "
            f"took {time.time() - t_step_start:.1f}s"
        )

    del transformer
    gc.collect()

    print("Decoding VAE...")
    vae = ledger.video_decoder()
    latent_spatial = latent.float().reshape(1, latent_frames, lh, lw, 128).permute(0, 4, 1, 2, 3)
    with torch.no_grad():
        pixels = vae(latent_spatial.bfloat16())

    pixels = pixels.float().clamp(-1, 1)
    pixels = (pixels + 1) / 2
    video_np = (pixels[0].permute(1, 2, 3, 0).numpy() * 255).astype("uint8")
    imageio.mimwrite(args.output, video_np, fps=fps, codec="libx264")
    print(f"Saved {args.output}")
    print(f"Frames: {video_np.shape[0]}, resolution: {video_np.shape[1]}x{video_np.shape[2]}")


if __name__ == "__main__":
    main()
