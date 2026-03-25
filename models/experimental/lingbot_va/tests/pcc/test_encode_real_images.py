"""
Test TTNN VAE encoder with real images (not random data).
Uses the same setup as test_vae_encoder.py but with actual demo images.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import ttnn
from diffusers import AutoencoderKLWan
from models.experimental.lingbot_va.tt.utils import patchify
from models.experimental.lingbot_va.tt.vae_encoder import WanVAEEncoder
from models.tt_dit.parallel.config import VaeHWParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import conv_pad_in_channels, conv_pad_height, conv_unpad_height


def main():
    ckpt = "models/experimental/lingbot_va/reference/checkpoints"
    img_dir = "models/experimental/lingbot_va/tests/demo/sample_images/robotwin"
    dtype_torch = torch.bfloat16

    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(os.path.join(ckpt, "vae"), torch_dtype=dtype_torch)

    # Load just the "head" camera for simplicity (256x320)
    img = np.array(Image.open(os.path.join(img_dir, "observation.images.cam_high.png")).convert("RGB"))
    video = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # [1, 3, 1, H, W]
    video = F.interpolate(video.squeeze(2), size=(256, 320), mode="bilinear", align_corners=False).unsqueeze(2)
    video = video / 255.0 * 2.0 - 1.0  # normalize to [-1, 1]
    video = video.to(dtype_torch)
    print(f"Input video: {video.shape}, dtype={video.dtype}")
    print(f"  min={video.float().min():.4f}, max={video.float().max():.4f}")

    # Patchify
    if hasattr(vae.config, "patch_size") and vae.config.patch_size:
        video_patched = patchify(video, vae.config.patch_size)
    else:
        video_patched = video
    print(f"After patchify: {video_patched.shape}")

    # PyTorch encoder (no quant_conv, matching test_vae_encoder.py)
    print("\n=== PyTorch encoder-only ===")
    with torch.no_grad():
        torch_out = vae.encoder(video_patched.clone())
    print(f"  Output: {torch_out.shape}, dtype={torch_out.dtype}")
    print(
        f"  min={torch_out.float().min():.4f}, max={torch_out.float().max():.4f}, mean={torch_out.float().mean():.4f}"
    )

    # TTNN encoder (same as test_vae_encoder.py)
    print("\n=== TTNN encoder-only ===")
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=1, mesh_axis=0),
        width_parallel=ParallelFactor(factor=1, mesh_axis=1),
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_encoder = WanVAEEncoder(
        in_channels=video_patched.shape[1],
        dim=vae.config.base_dim,
        z_dim=vae.config.z_dim * 2,
        dim_mult=list(vae.config.dim_mult),
        num_res_blocks=vae.config.num_res_blocks,
        attn_scales=list(vae.config.attn_scales),
        temperal_downsample=list(vae.config.temperal_downsample),
        is_residual=getattr(vae.config, "is_residual", False),
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    state = {k: v.cpu() for k, v in vae.encoder.state_dict().items()}
    tt_encoder.load_torch_state_dict(state)

    video_BTHWC = video_patched.permute(0, 2, 3, 4, 1)
    video_BTHWC = conv_pad_in_channels(video_BTHWC)
    video_BTHWC, logical_h = conv_pad_height(video_BTHWC, parallel_config.height_parallel.factor)

    tt_input = ttnn.from_torch(video_BTHWC, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=mesh_device)
    tt_out_BTHWC, out_logical_h = tt_encoder(tt_input, logical_h)
    ttnn.synchronize_device(mesh_device)

    ttnn_out = ttnn.to_torch(tt_out_BTHWC)
    ttnn_out = conv_unpad_height(ttnn_out, out_logical_h)
    ttnn_out = ttnn_out.permute(0, 4, 1, 2, 3)
    print(f"  Output: {ttnn_out.shape}")
    print(f"  min={ttnn_out.float().min():.4f}, max={ttnn_out.float().max():.4f}, mean={ttnn_out.float().mean():.4f}")

    # Compare
    min_c = min(torch_out.shape[1], ttnn_out.shape[1])
    min_h = min(torch_out.shape[3], ttnn_out.shape[3])
    min_w = min(torch_out.shape[4], ttnn_out.shape[4])
    torch_trim = torch_out[:, :min_c, :, :min_h, :min_w].float().detach()
    ttnn_trim = ttnn_out[:, :min_c, :, :min_h, :min_w].float().detach()

    pcc = float(np.corrcoef(ttnn_trim.flatten().numpy(), torch_trim.flatten().numpy())[0, 1])
    diff = (ttnn_trim - torch_trim).abs()
    print(f"\n=== RESULTS ===")
    print(f"PCC: {pcc:.6f}")
    print(f"Max abs diff: {diff.max():.4f}")
    print(f"Mean abs diff: {diff.mean():.4f}")

    # Also test with random data for comparison
    print("\n\n=== Random data comparison ===")
    torch.manual_seed(42)
    random_video = (torch.randn(1, 3, 1, 256, 320, dtype=torch.float32) * 2.0 - 1.0).to(dtype_torch)
    if hasattr(vae.config, "patch_size") and vae.config.patch_size:
        random_patched = patchify(random_video, vae.config.patch_size)
    else:
        random_patched = random_video

    with torch.no_grad():
        torch_rand_out = vae.encoder(random_patched.clone())

    rand_BTHWC = random_patched.permute(0, 2, 3, 4, 1)
    rand_BTHWC = conv_pad_in_channels(rand_BTHWC)
    rand_BTHWC, rand_lh = conv_pad_height(rand_BTHWC, parallel_config.height_parallel.factor)
    tt_rand_in = ttnn.from_torch(rand_BTHWC, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=mesh_device)
    tt_rand_out, rand_out_lh = tt_encoder(tt_rand_in, rand_lh)
    ttnn.synchronize_device(mesh_device)
    ttnn_rand = ttnn.to_torch(tt_rand_out)
    ttnn_rand = conv_unpad_height(ttnn_rand, rand_out_lh)
    ttnn_rand = ttnn_rand.permute(0, 4, 1, 2, 3)

    min_c = min(torch_rand_out.shape[1], ttnn_rand.shape[1])
    min_h = min(torch_rand_out.shape[3], ttnn_rand.shape[3])
    min_w = min(torch_rand_out.shape[4], ttnn_rand.shape[4])
    torch_rand_trim = torch_rand_out[:, :min_c, :, :min_h, :min_w].float().detach()
    ttnn_rand_trim = ttnn_rand[:, :min_c, :, :min_h, :min_w].float().detach()

    pcc_rand = float(np.corrcoef(ttnn_rand_trim.flatten().numpy(), torch_rand_trim.flatten().numpy())[0, 1])
    diff_rand = (ttnn_rand_trim - torch_rand_trim).abs()
    print(f"PCC (random): {pcc_rand:.6f}")
    print(f"Max abs diff (random): {diff_rand.max():.4f}")
    print(f"Mean abs diff (random): {diff_rand.mean():.4f}")

    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
