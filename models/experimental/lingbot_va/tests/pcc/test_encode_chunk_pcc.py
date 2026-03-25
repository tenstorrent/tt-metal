"""
Diagnostic: compare TTNN WanVAEStreamingWrapper.encode_chunk vs PyTorch reference
on real demo images. Runs on-device.
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import ttnn
from diffusers import AutoencoderKLWan
from models.experimental.lingbot_va.reference.utils import (
    WanVAEStreamingWrapper as RefStreamingWrapper,
    VA_CONFIGS,
)
from models.experimental.lingbot_va.tt.utils import (
    WanVAEStreamingWrapper as TTStreamingWrapper,
)
from models.tt_dit.parallel.config import VaeHWParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager


def load_images():
    config = VA_CONFIGS["robotwin"]
    img_dir = "models/experimental/lingbot_va/tests/demo/sample_images/robotwin"
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
    videos_left_and_right = torch.cat(videos[1:], dim=0) / 255.0 * 2.0 - 1.0
    return videos_high, videos_left_and_right


def run_test():
    ckpt = "models/experimental/lingbot_va/reference/checkpoints"
    dtype = torch.bfloat16

    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(os.path.join(ckpt, "vae"), torch_dtype=dtype)

    print("Loading images...")
    videos_high, videos_lr = load_images()
    print(f"  videos_high: {videos_high.shape}, videos_lr: {videos_lr.shape}")

    # ---- PyTorch reference ----
    print("\n=== PyTorch Reference ===")
    ref_streaming = RefStreamingWrapper(vae)
    ref_streaming_half = RefStreamingWrapper(vae)

    t0 = time.time()
    ref_enc_high = ref_streaming.encode_chunk(videos_high.to(dtype))
    ref_enc_lr = ref_streaming_half.encode_chunk(videos_lr.to(dtype))
    print(f"PyTorch encode time: {time.time() - t0:.1f}s")
    print(f"  ref_enc_high: {ref_enc_high.shape}, dtype={ref_enc_high.dtype}")
    print(
        f"    min={ref_enc_high.float().min():.4f}, max={ref_enc_high.float().max():.4f}, mean={ref_enc_high.float().mean():.4f}"
    )
    print(f"  ref_enc_lr: {ref_enc_lr.shape}, dtype={ref_enc_lr.dtype}")
    print(
        f"    min={ref_enc_lr.float().min():.4f}, max={ref_enc_lr.float().max():.4f}, mean={ref_enc_lr.float().mean():.4f}"
    )

    ref_enc_out = torch.cat(
        [torch.cat(ref_enc_lr.split(1, dim=0), dim=-1), ref_enc_high],
        dim=-2,
    )
    ref_mu, _ = torch.chunk(ref_enc_out, 2, dim=1)
    print(f"  ref_mu: {ref_mu.shape}")
    print(
        f"    min={ref_mu.float().min():.4f}, max={ref_mu.float().max():.4f}, mean={ref_mu.float().mean():.4f}, std={ref_mu.float().std():.4f}"
    )

    # ---- TTNN ----
    print("\n=== TTNN ===")
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=1, mesh_axis=0),
        width_parallel=ParallelFactor(factor=1, mesh_axis=1),
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_streaming = TTStreamingWrapper(vae, mesh_device, ccl_manager, parallel_config)
    tt_streaming_half = TTStreamingWrapper(vae, mesh_device, ccl_manager, parallel_config)

    t0 = time.time()
    tt_enc_high = tt_streaming.encode_chunk(videos_high.to(dtype))
    tt_enc_lr = tt_streaming_half.encode_chunk(videos_lr.to(dtype))
    print(f"TTNN encode time: {time.time() - t0:.1f}s")
    print(f"  tt_enc_high: {tt_enc_high.shape}, dtype={tt_enc_high.dtype}")
    print(
        f"    min={tt_enc_high.float().min():.4f}, max={tt_enc_high.float().max():.4f}, mean={tt_enc_high.float().mean():.4f}"
    )
    print(f"  tt_enc_lr: {tt_enc_lr.shape}, dtype={tt_enc_lr.dtype}")
    print(
        f"    min={tt_enc_lr.float().min():.4f}, max={tt_enc_lr.float().max():.4f}, mean={tt_enc_lr.float().mean():.4f}"
    )

    # Compare high-res encode
    a = tt_enc_high.float().detach().flatten().numpy()
    b = ref_enc_high.float().detach().flatten().numpy()
    pcc_high = float(np.corrcoef(a, b)[0, 1])
    print(f"\n  PCC (high-res encode_chunk): {pcc_high:.6f}")
    print(f"  Max abs diff: {np.abs(a - b).max():.4f}, Mean abs diff: {np.abs(a - b).mean():.4f}")

    # Compare low-res encode
    a = tt_enc_lr.float().detach().flatten().numpy()
    b = ref_enc_lr.float().detach().flatten().numpy()
    pcc_lr = float(np.corrcoef(a, b)[0, 1])
    print(f"  PCC (low-res encode_chunk): {pcc_lr:.6f}")
    print(f"  Max abs diff: {np.abs(a - b).max():.4f}, Mean abs diff: {np.abs(a - b).mean():.4f}")

    # Full combined output
    tt_enc_out = torch.cat(
        [torch.cat(tt_enc_lr.split(1, dim=0), dim=-1), tt_enc_high],
        dim=-2,
    )
    tt_mu, _ = torch.chunk(tt_enc_out, 2, dim=1)

    a = tt_mu.float().detach().flatten().numpy()
    b = ref_mu.float().detach().flatten().numpy()
    pcc_mu = float(np.corrcoef(a, b)[0, 1])
    print(f"\n  PCC (combined mu): {pcc_mu:.6f}")
    print(f"  Max abs diff: {np.abs(a - b).max():.4f}, Mean abs diff: {np.abs(a - b).mean():.4f}")

    # Now test encoder-only (no quant_conv) to isolate the issue
    print("\n=== Encoder-only comparison (no quant_conv) ===")
    from models.experimental.lingbot_va.tt.utils import patchify as tt_patchify
    from models.tt_dit.utils.conv3d import conv_pad_in_channels, conv_pad_height, conv_unpad_height

    test_input = videos_high.to(dtype).clone()
    test_input_patchified = tt_patchify(test_input, vae.config.patch_size)
    print(f"  Input after patchify: {test_input_patchified.shape}")

    # PyTorch encoder only
    with torch.no_grad():
        ref_enc_only = vae.encoder(test_input_patchified, feat_cache=[None] * 26, feat_idx=[0])
    print(f"  PyTorch encoder-only output: {ref_enc_only.shape}")
    print(
        f"    min={ref_enc_only.float().min():.4f}, max={ref_enc_only.float().max():.4f}, mean={ref_enc_only.float().mean():.4f}"
    )

    # TTNN encoder only
    tt_streaming.clear_cache()
    video_BTHWC = test_input_patchified.permute(0, 2, 3, 4, 1)
    video_BTHWC = conv_pad_in_channels(video_BTHWC)
    video_BTHWC, logical_h = conv_pad_height(video_BTHWC, parallel_config.height_parallel.factor)
    tt_input = ttnn.from_torch(video_BTHWC, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=mesh_device)
    feat_idx = [0]
    tt_enc_only, out_logical_h = tt_streaming.encoder(
        tt_input, logical_h, feat_cache=tt_streaming.feat_cache, feat_idx=feat_idx
    )
    ttnn.synchronize_device(mesh_device)
    tt_enc_only_torch = ttnn.to_torch(tt_enc_only)
    tt_enc_only_torch = conv_unpad_height(tt_enc_only_torch, out_logical_h)
    tt_enc_only_torch = tt_enc_only_torch.permute(0, 4, 1, 2, 3)
    print(f"  TTNN encoder-only output: {tt_enc_only_torch.shape}")
    print(
        f"    min={tt_enc_only_torch.float().min():.4f}, max={tt_enc_only_torch.float().max():.4f}, mean={tt_enc_only_torch.float().mean():.4f}"
    )

    min_c = min(tt_enc_only_torch.shape[1], ref_enc_only.shape[1])
    min_h = min(tt_enc_only_torch.shape[3], ref_enc_only.shape[3])
    min_w = min(tt_enc_only_torch.shape[4], ref_enc_only.shape[4])
    tt_trim = tt_enc_only_torch[:, :min_c, :, :min_h, :min_w]
    ref_trim = ref_enc_only[:, :min_c, :, :min_h, :min_w]

    a = tt_trim.float().detach().flatten().numpy()
    b = ref_trim.float().detach().flatten().numpy()
    pcc_enc = float(np.corrcoef(a, b)[0, 1])
    print(f"\n  PCC (encoder-only, real images): {pcc_enc:.6f}")
    print(f"  Max abs diff: {np.abs(a - b).max():.4f}, Mean abs diff: {np.abs(a - b).mean():.4f}")

    ttnn.close_mesh_device(mesh_device)

    print("\n=== SUMMARY ===")
    print(f"Encoder-only PCC: {pcc_enc:.6f}")
    print(f"Full encode_chunk PCC (high): {pcc_high:.6f}")
    print(f"Full encode_chunk PCC (low): {pcc_lr:.6f}")
    print(f"Combined mu PCC: {pcc_mu:.6f}")


if __name__ == "__main__":
    run_test()
