#!/usr/bin/env python3
"""
Diagnose TTNN vs PyTorch accuracy at each pipeline stage:
  1. VAE encoder (encode_chunk)
  2. Transformer (single forward pass)

Runs on-device. Uses the same model loading as inference_ttnn.py.
PyTorch reference runs on CPU (slow for VAE, but only needs 1 call).
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_TT_METAL_ROOT = os.environ.get("TT_METAL_HOME") or str(_REPO_ROOT.parent.parent.parent)
if os.path.isdir(_TT_METAL_ROOT) and _TT_METAL_ROOT not in sys.path:
    sys.path.insert(0, _TT_METAL_ROOT)

import ttnn
from models.experimental.lingbot_va.reference.utils import (
    load_vae,
    WanVAEStreamingWrapper as RefStreamingWrapper,
    VA_CONFIGS,
)
from models.experimental.lingbot_va.tt.utils import (
    WanVAEStreamingWrapper as TTStreamingWrapper,
)
from models.tt_dit.parallel.config import VaeHWParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    return float(np.corrcoef(a_f.detach().numpy(), b_f.detach().numpy())[0, 1])


def report(name, tt_out, ref_out):
    tt_f = tt_out.float().detach()
    ref_f = ref_out.float().detach()
    p = pcc(tt_f, ref_f)
    diff = (tt_f - ref_f).abs()
    print(f"  {name}:")
    print(f"    PCC={p:.6f}  MaxAbsDiff={diff.max():.4f}  MeanAbsDiff={diff.mean():.6f}")
    print(f"    TT  : min={tt_f.min():.4f} max={tt_f.max():.4f} mean={tt_f.mean():.4f} std={tt_f.std():.4f}")
    print(f"    Ref : min={ref_f.min():.4f} max={ref_f.max():.4f} mean={ref_f.mean():.4f} std={ref_f.std():.4f}")
    return p


def diagnose_vae_encoder():
    """Compare TTNN vs PyTorch VAE encode_chunk on real images."""
    print("\n" + "=" * 60)
    print("STAGE 1: VAE ENCODER (encode_chunk)")
    print("=" * 60)

    ckpt = str(_REPO_ROOT / "reference" / "checkpoints")
    img_dir = str(_REPO_ROOT / "tests" / "demo" / "sample_images" / "robotwin")
    config = VA_CONFIGS["robotwin"]
    dtype = torch.bfloat16

    print("Loading PyTorch VAE...")
    t0 = time.time()
    vae = load_vae(os.path.join(ckpt, "vae"), dtype, "cpu")
    print(f"  VAE loaded in {time.time()-t0:.1f}s")

    # Load images
    obs_cam_keys = config.obs_cam_keys
    height, width = config.height, config.width
    videos = []
    for k_i, k in enumerate(obs_cam_keys):
        height_i, width_i = (height, width) if k_i == 0 else (height // 2, width // 2)
        img = np.array(Image.open(os.path.join(img_dir, f"{k}.png")).convert("RGB"))
        history_video_k = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(1)
        history_video_k = F.interpolate(
            history_video_k, size=(height_i, width_i), mode="bilinear", align_corners=False
        ).unsqueeze(0)
        videos.append(history_video_k)
    videos_high = videos[0] / 255.0 * 2.0 - 1.0
    videos_lr = torch.cat(videos[1:], dim=0) / 255.0 * 2.0 - 1.0
    print(f"  videos_high: {videos_high.shape}, videos_lr: {videos_lr.shape}")

    # PyTorch reference encode
    print("\nRunning PyTorch encode_chunk (CPU, may take a few minutes)...")
    ref_vae = RefStreamingWrapper(vae)
    ref_vae_half = RefStreamingWrapper(vae)
    t0 = time.time()
    with torch.no_grad():
        ref_enc_high = ref_vae.encode_chunk(videos_high.to(dtype))
        ref_enc_lr = ref_vae_half.encode_chunk(videos_lr.to(dtype))
    print(f"  PyTorch encode done in {time.time()-t0:.1f}s")

    # TTNN encode
    print("Running TTNN encode_chunk (on device)...")
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    ccl_mgr = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    par_cfg = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=1, mesh_axis=0),
        width_parallel=ParallelFactor(factor=1, mesh_axis=1),
    )
    tt_vae = TTStreamingWrapper(vae, mesh_device, ccl_mgr, par_cfg)
    tt_vae_half = TTStreamingWrapper(vae, mesh_device, ccl_mgr, par_cfg)
    t0 = time.time()
    tt_enc_high = tt_vae.encode_chunk(videos_high.to(dtype))
    tt_enc_lr = tt_vae_half.encode_chunk(videos_lr.to(dtype))
    print(f"  TTNN encode done in {time.time()-t0:.1f}s")

    # Compare
    print("\nResults:")
    p_high = report("encode_chunk (high-res cam)", tt_enc_high, ref_enc_high)
    p_lr = report("encode_chunk (low-res cams)", tt_enc_lr, ref_enc_lr)

    # Combined output (same post-processing as _encode_obs)
    ref_enc_out = torch.cat([torch.cat(ref_enc_lr.split(1, dim=0), dim=-1), ref_enc_high], dim=-2)
    tt_enc_out = torch.cat([torch.cat(tt_enc_lr.split(1, dim=0), dim=-1), tt_enc_high], dim=-2)
    ref_mu, _ = torch.chunk(ref_enc_out, 2, dim=1)
    tt_mu, _ = torch.chunk(tt_enc_out, 2, dim=1)
    p_mu = report("combined mu (init_latent input)", tt_mu, ref_mu)

    ttnn.close_mesh_device(mesh_device)
    return {"pcc_high": p_high, "pcc_lr": p_lr, "pcc_mu": p_mu}


if __name__ == "__main__":
    results = diagnose_vae_encoder()
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in results.items():
        status = "PASS" if v > 0.99 else ("WARN" if v > 0.9 else "FAIL")
        print(f"  {k}: {v:.6f}  [{status}]")
