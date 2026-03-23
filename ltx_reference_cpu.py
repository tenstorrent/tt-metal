import sys, os, torch, time

sys.path.insert(0, "LTX-2/packages/ltx-core/src")
sys.path.insert(0, "LTX-2/packages/ltx-pipelines/src")
os.environ["HF_TOKEN"] = "REDACTED"

# Monkey-patch torch.cuda for CPU-only execution
import types

if not torch.cuda.is_available():
    torch.cuda.synchronize = lambda *a, **kw: None
    torch.cuda.empty_cache = lambda *a, **kw: None

from ltx_pipelines.utils.model_ledger import ModelLedger
from ltx_pipelines.utils.helpers import encode_prompts
from ltx_core.model.transformer.model import Modality
from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationConfig
from ltx_core.utils import to_denoised
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.diffusion_steps import EulerDiffusionStep

ckpt = os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.3-22b-dev.safetensors")
gemma_path = "/home/kevinmi/.cache/huggingface/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80"

prompt = "A golden retriever running through a field of sunflowers, cinematic lighting"

# 1. Encode prompt
print("Creating ModelLedger...")
t0 = time.time()
ledger = ModelLedger(dtype=torch.bfloat16, device=torch.device("cpu"), checkpoint_path=ckpt, gemma_root_path=gemma_path)
print(f"ModelLedger created in {time.time()-t0:.1f}s")

print("Encoding prompts...")
t0 = time.time()
results = encode_prompts([prompt, ""], ledger)
v_ctx = results[0].video_encoding
v_ctx_neg = results[1].video_encoding
print(f"Video context: {v_ctx.shape} (encoded in {time.time()-t0:.1f}s)")

# 2. Load transformer
print("Loading transformer...")
t0 = time.time()
transformer = ledger.transformer()
print(f"Transformer loaded in {time.time()-t0:.1f}s")

# 3. Small test: 33 frames @ 256x256, 5 steps for speed
num_frames = 33
H = 256
W = 256
steps = 5
seed = 10
fps = 24.0
sigma_cfg = 3.0
latent_frames = (num_frames - 1) // 8 + 1
lh = H // 32
lw = W // 32
N = latent_frames * lh * lw
print(f"Latent: {latent_frames}x{lh}x{lw} = {N} tokens")

# 4. Sigma schedule
scheduler = LTX2Scheduler()
dummy_latent = torch.randn(1, 1, N)
sigmas = scheduler.execute(steps=steps, latent=dummy_latent)
print(f"Sigmas: {sigmas}")

# 5. Initial noise
torch.manual_seed(seed)
latent = torch.randn(1, N, 128, dtype=torch.bfloat16) * sigmas[0]

# 6. Positions — use official pipeline position computation
from ltx_core.components.patchifiers import get_pixel_coords, VideoLatentPatchifier
from ltx_core.types import VideoLatentShape

patchifier = VideoLatentPatchifier(patch_size=1)
target_shape = VideoLatentShape(batch=1, channels=128, frames=latent_frames, height=lh, width=lw)
latent_coords = patchifier.get_patch_grid_bounds(output_shape=target_shape, device="cpu")
scale_factors = (8, 32, 32)
pixel_coords = get_pixel_coords(latent_coords, scale_factors, causal_fix=True)
video_pos = pixel_coords.float()
video_pos[:, 0, ...] = video_pos[:, 0, ...] / fps
video_pos = video_pos.bfloat16()
print(f"Positions shape: {video_pos.shape}")

# 7. Denoising
stepper = EulerDiffusionStep()
for step_idx in range(steps):
    sigma = sigmas[step_idx]
    t_step_start = time.time()

    timesteps = (torch.ones(1, N) * sigma).unsqueeze(-1)  # (1, N, 1) for broadcasting with (1, N, D)
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

    # CFG
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

    # Reference CFG with rescale
    raw_denoised = denoised  # Save for logging
    pred = denoised + (sigma_cfg - 1) * (denoised - uncond)
    factor = 0.7 * (denoised.std() / pred.std()) + 0.3
    denoised = pred * factor

    print(
        f"  DBG: raw_den=[{raw_denoised.float().min():.3f},{raw_denoised.float().max():.3f}] unc=[{uncond.float().min():.3f},{uncond.float().max():.3f}] factor={factor:.4f}"
    )
    latent = stepper.step(latent.float(), denoised.float(), sigmas, step_idx).bfloat16()
    elapsed = time.time() - t_step_start
    print(
        f"Step {step_idx+1}/{steps}: sigma {sigma:.4f}, range [{latent.float().min():.3f}, {latent.float().max():.3f}], took {elapsed:.1f}s"
    )

print(f"Final latent: {latent.shape}, range [{latent.float().min():.3f}, {latent.float().max():.3f}]")

# 8. Free transformer to save memory for VAE
del transformer
import gc

gc.collect()

# 8. VAE decode
print("Loading VAE decoder...")
t0 = time.time()
vae = ledger.video_decoder()
print(f"VAE loaded in {time.time()-t0:.1f}s")

latent_spatial = latent.float().reshape(1, latent_frames, lh, lw, 128).permute(0, 4, 1, 2, 3)
print(f"Latent spatial shape: {latent_spatial.shape}")
with torch.no_grad():
    pixels = vae(latent_spatial.bfloat16())
print(f"Video: {pixels.shape}")

# 9. Export
pixels = pixels.float().clamp(-1, 1)
pixels = (pixels + 1) / 2
video_np = (pixels[0].permute(1, 2, 3, 0).numpy() * 255).astype("uint8")
import imageio

imageio.mimwrite("models/tt_dit/demos/ltx/ltx_reference_cpu.mp4", video_np, fps=24, codec="libx264")
print("Saved reference video")

# Check stats
import numpy as np

print(f"F0: mean={video_np[0].mean():.1f} std={video_np[0].std():.1f}")
print(f"Fmid: mean={video_np[16].mean():.1f} std={video_np[16].std():.1f}")
print(f"Total frames: {video_np.shape[0]}, resolution: {video_np.shape[1]}x{video_np.shape[2]}")
