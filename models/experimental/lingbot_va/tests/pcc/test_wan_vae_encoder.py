# test_encode_one_video_pcc.py (or add to test_wan_vae_encoder.py)
# Compares torch AutoencoderKLWan encoder vs TTNN VaeWanEncoder with PCC, same pattern as decoder PCC test.

import sys
import time
import torch
import ttnn

from diffusers import AutoencoderKLWan
from models.experimental.lingbot_va.tt.vae_encoder import VaeWanEncoder
from models.common.metrics import compute_pcc
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.parallel.config import VaeHWParallelConfig, ParallelFactor
from models.tt_dit.utils.conv3d import conv_pad_in_channels, conv_pad_height, conv_unpad_height

CHECKPOINT_PATH = "models/experimental/lingbot_va/reference/checkpoints/vae"
PCC_THRESHOLD = 0.99
BATCH_SIZE = 1
# Small video for fast CPU encode; H,W divisible by 32 and by patch_size if used
VIDEO_T = 1
VIDEO_H = 256
VIDEO_W = 320


def patchify(x, patch_size):
    """Match diffusers AutoencoderKLWan patchify; no-op if patch_size is None or 1."""
    if patch_size is None or patch_size == 1:
        return x
    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
    B, C, F, H, W = x.shape
    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError(f"H ({H}) and W ({W}) must be divisible by patch_size ({patch_size})")
    x = x.view(B, C, F, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(B, C * patch_size * patch_size, F, H // patch_size, W // patch_size)
    return x


# ─────────────────────────────────────────────
# Torch reference encode (encoder only, same input as _encode)
# ─────────────────────────────────────────────


def encode_torch(vae, video):
    """Run the diffusers AutoencoderKLWan encoder (torch, CPU). Returns encoder output before quant_conv."""
    video = video.to(vae.dtype)
    ps = getattr(vae.config, "patch_size", None)
    if ps and ps > 1:
        video = patchify(video, ps)
    with torch.no_grad():
        out = vae.encoder(video)
    return out  # [B, z_dim*2, T, H, W]


# ─────────────────────────────────────────────
# TTNN encode
# ─────────────────────────────────────────────


def encode_ttnn(vae, video, mesh_device):
    """Run the TTNN VaeWanEncoder on device, return torch tensor (encoder output before quant_conv)."""
    video = video.to(vae.dtype)
    ps = getattr(vae.config, "patch_size", None)
    if ps and ps > 1:
        video = patchify(video, ps)

    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=1, mesh_axis=0),
        width_parallel=ParallelFactor(factor=1, mesh_axis=1),
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_vae_encoder = VaeWanEncoder(
        in_channels=video.shape[1],
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
    tt_vae_encoder.load_torch_state_dict(state)

    # BCTHW -> BTHWC, pad for ttnn
    video_BTHWC = video.permute(0, 2, 3, 4, 1)
    video_BTHWC = conv_pad_in_channels(video_BTHWC)
    video_BTHWC, logical_h = conv_pad_height(video_BTHWC, parallel_config.height_parallel.factor)

    tt_input = ttnn.from_torch(
        video_BTHWC,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh_device,
    )

    tt_out_BTHWC, out_logical_h = tt_vae_encoder(tt_input, logical_h)
    ttnn.synchronize_device(mesh_device)

    out_torch = ttnn.to_torch(tt_out_BTHWC)
    out_torch = conv_unpad_height(out_torch, out_logical_h)
    out_torch = out_torch.permute(0, 4, 1, 2, 3)
    return out_torch


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────


def test_encode_one_video_pcc():
    """Compare torch vs TTNN VAE encoder outputs with PCC."""

    print("=" * 60)
    print("test_encode_one_video_pcc")
    print("=" * 60)

    print(f"Loading AutoencoderKLWan from {CHECKPOINT_PATH}...")
    vae = AutoencoderKLWan.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=torch.bfloat16,
    ).to(device="cpu")
    vae.eval()

    z_dim = vae.config.z_dim
    print(
        f"VAE config: z_dim={z_dim}, base_dim={vae.config.base_dim}, "
        f"patch_size={getattr(vae.config, 'patch_size', None)}, "
        f"is_residual={getattr(vae.config, 'is_residual', False)}"
    )

    torch.manual_seed(42)
    video = torch.randn(BATCH_SIZE, 3, VIDEO_T, VIDEO_H, VIDEO_W, dtype=torch.float32) * 2.0 - 1.0
    print(f"Input video shape: {video.shape}")

    print("\nRunning torch encode...")
    t0 = time.time()
    torch_out = encode_torch(vae, video.clone())
    t_torch = time.time() - t0
    print(f"Torch encoder output shape: {torch_out.shape}, time: {t_torch:.1f}s")

    print("\nSetting up TT device...")
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    mesh_device.enable_program_cache()

    try:
        print("Running TTNN encode...")
        t0 = time.time()
        ttnn_out = encode_ttnn(vae, video.clone(), mesh_device)
        t_ttnn = time.time() - t0
        print(f"TTNN encoder output shape: {ttnn_out.shape}, time: {t_ttnn:.1f}s")

        torch_out_f = torch_out.float()
        ttnn_out_f = ttnn_out.float()

        min_c = min(torch_out_f.shape[1], ttnn_out_f.shape[1])
        min_t = min(torch_out_f.shape[2], ttnn_out_f.shape[2])
        min_h = min(torch_out_f.shape[3], ttnn_out_f.shape[3])
        min_w = min(torch_out_f.shape[4], ttnn_out_f.shape[4])
        torch_trimmed = torch_out_f[:, :min_c, :min_t, :min_h, :min_w]
        ttnn_trimmed = ttnn_out_f[:, :min_c, :min_t, :min_h, :min_w]

        if torch_trimmed.shape != ttnn_trimmed.shape:
            print(f"Shape mismatch after trim: torch={torch_trimmed.shape}, ttnn={ttnn_trimmed.shape}")

        pcc = compute_pcc(ttnn_trimmed, torch_trimmed)
        max_err = (torch_trimmed - ttnn_trimmed).abs().max().item()
        mean_err = (torch_trimmed - ttnn_trimmed).abs().mean().item()

        print("\n" + "=" * 60)
        print("COMPARISON RESULTS")
        print("=" * 60)
        print(f"Torch  output shape    : {torch_out.shape}")
        print(f"TTNN   output shape    : {ttnn_out.shape}")
        print(f"Compared region       : {torch_trimmed.shape}")
        print(f"PCC                   : {pcc:.6f}  (threshold={PCC_THRESHOLD})")
        print(f"Max  absolute error   : {max_err:.6f}")
        print(f"Mean absolute error   : {mean_err:.6f}")
        print(f"Torch encode time     : {t_torch:.1f}s")
        print(f"TTNN  encode time     : {t_ttnn:.1f}s")
        print("=" * 60)

        passed = pcc >= PCC_THRESHOLD
        if passed:
            print("TEST PASSED")
        else:
            print(f"TEST FAILED - PCC {pcc:.6f} < {PCC_THRESHOLD}")

        assert passed, f"PCC {pcc:.6f} is below threshold {PCC_THRESHOLD}"

        return pcc, max_err, mean_err

    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    try:
        pcc, max_err, mean_err = test_encode_one_video_pcc()
        sys.exit(0)
    except Exception as e:
        print(f"\nTest failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
