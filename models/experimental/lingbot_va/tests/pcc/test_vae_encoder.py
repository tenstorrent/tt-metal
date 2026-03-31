# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC: diffusers AutoencoderKLWan encoder vs TTNN WanVAEEncoder (reference VAE in fp32 on CPU for speed).

Multi-device setup aligned with ``test_encoder_wan.py`` / ``models/tt_dit/tests/encoders/umt5/test_umt5.py``:

- ``device_params`` with ``fabric_config=FABRIC_1D`` (indirect) for fabric-backed dispatch / CCL.
- ``num_links`` (indirect) passed into ``CCLManager`` (default ``1``).
- ``mesh_device`` from ``mesh_shape_request_param()`` (``MESH_DEVICE`` / device count), same as other Lingbot PCC tests.

``vae_hw_parallel_config_for_mesh`` supplies H/W mesh axes for ``WanVAEEncoder`` (rows×cols on multi-chip meshes).

**Wall time:** ``pytestmark = pytest.mark.timeout(600)``.
"""

import pytest
import torch
import ttnn
from diffusers import AutoencoderKLWan

from models.common.metrics import compute_pcc
from models.experimental.lingbot_va.tests.mesh_utils import (
    mesh_shape_request_param,
    vae_bthwc_to_torch,
    vae_hw_parallel_config_for_mesh,
)
from models.experimental.lingbot_va.tt.vae_encoder import WanVAEEncoder
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_in_channels, conv_unpad_height

pytestmark = pytest.mark.timeout(600)

CHECKPOINT_PATH = "models/experimental/lingbot_va/reference/checkpoints/vae"
PCC_THRESHOLD = 0.99
BATCH_SIZE = 1
VIDEO_T = 1
VIDEO_H = 256
VIDEO_W = 320


@pytest.fixture
def num_links(request):
    """CCL link count; indirect param (aligned with ``test_encoder_wan`` / tt_dit UMT5 tests)."""
    return request.param


@pytest.fixture
def vae_ccl_manager(mesh_device, num_links):
    """Fabric-backed CCL for VAE conv / halo paths on multi-device meshes."""
    return CCLManager(
        mesh_device=mesh_device,
        num_links=num_links,
        topology=ttnn.Topology.Linear,
    )


@pytest.fixture(scope="module")
def vae():
    # bf16 conv on CPU is very slow; fp32 uses fast MKL paths (same idea as test_vae_decoder.py).
    model = AutoencoderKLWan.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=torch.float32,
    ).to(device="cpu")
    model.eval()
    return model


def _patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    B, C, F, H, W = x.shape
    x = x.view(B, C, F, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    return x.view(B, C * patch_size * patch_size, F, H // patch_size, W // patch_size)


def encode_torch(vae, video):
    video = video.to(vae.dtype)
    ps = getattr(vae.config, "patch_size", None)
    if ps and ps > 1:
        video = _patchify(video, ps)
    with torch.no_grad():
        return vae.encoder(video)


def encode_ttnn(vae, video, mesh_device, ccl_manager):
    video = video.to(vae.dtype)
    ps = getattr(vae.config, "patch_size", None)
    if ps and ps > 1:
        video = _patchify(video, ps)

    parallel_config = vae_hw_parallel_config_for_mesh(mesh_device)

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

    out = vae_bthwc_to_torch(tt_out_BTHWC, mesh_device, parallel_config, ccl_manager)
    out = conv_unpad_height(out, out_logical_h)
    return out.permute(0, 4, 1, 2, 3)


@pytest.mark.parametrize(
    "mesh_device",
    [mesh_shape_request_param()],
    indirect=True,
)
@pytest.mark.parametrize("num_links", [1], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_encode_one_video_pcc(mesh_device, num_links, vae_ccl_manager, vae):
    mesh_device.enable_program_cache()
    torch.manual_seed(42)
    video = torch.randn(BATCH_SIZE, 3, VIDEO_T, VIDEO_H, VIDEO_W, dtype=torch.float32) * 2.0 - 1.0

    torch_out = encode_torch(vae, video.clone())
    ttnn_out = encode_ttnn(vae, video.clone(), mesh_device, vae_ccl_manager)

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

    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < {PCC_THRESHOLD} (max_err={max_err:.6f}, mean_err={mean_err:.6f})"
