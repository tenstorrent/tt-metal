# test_encode_one_video_pcc.py
# Compares torch AutoencoderKLWan encoder vs TTNN WanVAEEncoder with PCC

import time
import pytest
import torch
import ttnn

from diffusers import AutoencoderKLWan
from models.experimental.lingbot_va.tt.vae_encoder import WanVAEEncoder
from models.common.metrics import compute_pcc
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.parallel.config import VaeHWParallelConfig, ParallelFactor
from models.tt_dit.utils.conv3d import conv_pad_in_channels, conv_pad_height, conv_unpad_height


CHECKPOINT_PATH = "models/experimental/lingbot_va/reference/checkpoints/vae"
PCC_THRESHOLD = 0.99
BATCH_SIZE = 1
VIDEO_T = 1
VIDEO_H = 256
VIDEO_W = 320


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def mesh_device():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    device.enable_program_cache()
    yield device
    ttnn.close_mesh_device(device)


@pytest.fixture(scope="module")
def vae():
    model = AutoencoderKLWan.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=torch.bfloat16,
    ).to(device="cpu")
    model.eval()
    return model


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x

    B, C, F, H, W = x.shape
    x = x.view(B, C, F, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(B, C * patch_size * patch_size, F, H // patch_size, W // patch_size)
    return x


def encode_torch(vae, video):
    video = video.to(vae.dtype)

    ps = getattr(vae.config, "patch_size", None)
    if ps and ps > 1:
        video = patchify(video, ps)

    with torch.no_grad():
        out = vae.encoder(video)

    return out


def encode_ttnn(vae, video, mesh_device):
    video = video.to(vae.dtype)

    ps = getattr(vae.config, "patch_size", None)
    if ps and ps > 1:
        video = patchify(video, ps)

    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=1, mesh_axis=0),
        width_parallel=ParallelFactor(factor=1, mesh_axis=1),
    )

    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    tt_encoder = WanVAEEncoder(
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
    tt_encoder.load_torch_state_dict(state)

    video_BTHWC = video.permute(0, 2, 3, 4, 1)
    video_BTHWC = conv_pad_in_channels(video_BTHWC)
    video_BTHWC, logical_h = conv_pad_height(video_BTHWC, parallel_config.height_parallel.factor)

    tt_input = ttnn.from_torch(
        video_BTHWC,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=mesh_device,
    )

    tt_out_BTHWC, out_logical_h = tt_encoder(tt_input, logical_h)

    ttnn.synchronize_device(mesh_device)

    out = ttnn.to_torch(tt_out_BTHWC)
    out = conv_unpad_height(out, out_logical_h)
    out = out.permute(0, 4, 1, 2, 3)

    return out


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────


@pytest.mark.timeout(0)
def test_encode_one_video_pcc(mesh_device, vae):
    torch.manual_seed(42)

    video = (
        torch.randn(
            BATCH_SIZE,
            3,
            VIDEO_T,
            VIDEO_H,
            VIDEO_W,
            dtype=torch.float32,
        )
        * 2.0
        - 1.0
    )

    # Torch reference
    t0 = time.time()
    torch_out = encode_torch(vae, video.clone())
    torch_time = time.time() - t0

    # TTNN
    t0 = time.time()
    ttnn_out = encode_ttnn(vae, video.clone(), mesh_device)
    ttnn_time = time.time() - t0

    torch_out = torch_out.float()
    ttnn_out = ttnn_out.float()

    min_c = min(torch_out.shape[1], ttnn_out.shape[1])
    min_t = min(torch_out.shape[2], ttnn_out.shape[2])
    min_h = min(torch_out.shape[3], ttnn_out.shape[3])
    min_w = min(torch_out.shape[4], ttnn_out.shape[4])

    torch_trim = torch_out[:, :min_c, :min_t, :min_h, :min_w]
    ttnn_trim = ttnn_out[:, :min_c, :min_t, :min_h, :min_w]

    assert torch_trim.shape == ttnn_trim.shape

    pcc = compute_pcc(ttnn_trim, torch_trim)
    max_err = (torch_trim - ttnn_trim).abs().max().item()
    mean_err = (torch_trim - ttnn_trim).abs().mean().item()

    print("\n================================================")
    print("ENCODER PCC COMPARISON")
    print("================================================")
    print(f"PCC               : {pcc:.6f}")
    print(f"Max absolute err  : {max_err:.6f}")
    print(f"Mean absolute err : {mean_err:.6f}")
    print(f"Torch time        : {torch_time:.2f}s")
    print(f"TTNN time         : {ttnn_time:.2f}s")
    print("================================================")

    assert pcc >= PCC_THRESHOLD, (
        f"PCC {pcc:.6f} < threshold {PCC_THRESHOLD} " f"(max_err={max_err:.6f}, mean_err={mean_err:.6f})"
    )
