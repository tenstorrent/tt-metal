# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC test for the Cosmos3-I2V VAE encoder + decoder adapters.

Both adapters wrap the existing TT-NN `WanEncoder` / `WanVAEDecoderAdapter`
plus host-side spatial patchify/unpatchify for `patch_size=(1, 2, 2)` (the
Cosmos3 VAE config). This test verifies that the round trip matches
`AutoencoderKLWan`'s host PyTorch reference on subsets of a WH LoudBox (1, 8)
mesh — same convention as the rest of cosmos3_i2v's PCC suite.

The host VAE is *very* expensive to load (~5 GB). To keep CI cycle time
tolerable, the tests use a tiny `AutoencoderKLWan` constructed with the
Cosmos3 dim_mult / z_dim / patch_size but random weights and a small spatial
resolution. Bigger configs and real weights are exercised by the Wan2.2
test suite already.
"""

from __future__ import annotations

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan as TorchAutoencoderKLWan
from loguru import logger

import ttnn
from models.tt_dit.experimental.cosmos3_i2v.tokenizer.vae_cosmos3 import (
    Cosmos3VAEDecoderAdapter,
    Cosmos3VAEEncoderAdapter,
    _host_patchify_spatial,
    _host_unpatchify_spatial,
)
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.test import line_params


def _build_tiny_cosmos3_vae(base_dim: int = 32) -> TorchAutoencoderKLWan:
    """A small AutoencoderKLWan with Cosmos3-shaped dims (z_dim=48, patch=(1,2,2))
    but random weights. `base_dim` is the only knob we vary; everything else
    matches real Cosmos3 layout."""
    return TorchAutoencoderKLWan(
        base_dim=base_dim,
        # Real Cosmos3 ships cfg.in_channels=12 (post-patchify: 3 RGB * 2*2).
        in_channels=12,
        z_dim=48,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
        out_channels=3,
        is_residual=True,
        # diffusers 0.35.1 `patchify` is scalar-only; the tuple form lives on
        # diffusers main. Scalar 2 produces the same downstream tensors here.
        patch_size=2,
        # Cosmos3 latents_mean/std lengths must match z_dim=48; populate with
        # neutral values (mean=0, std=1) so normalization isn't load-bearing in
        # the encoder PCC.
        latents_mean=[0.0] * 48,
        latents_std=[1.0] * 48,
    )


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((2, 4), line_params, id="wh_loudbox_2x4"),
        pytest.param((4, 8), line_params, id="bh_galaxy_4x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "submesh_shape",
    [(1, 1), (1, 2), (1, 4), (2, 4), (4, 8)],
    ids=["sub_1x1", "sub_1x2", "sub_1x4", "sub_2x4", "sub_4x8"],
)
@pytest.mark.parametrize("base_dim", [32, 160], ids=["dim32", "dim160"])
@pytest.mark.timeout(900)
def test_cosmos3_vae_encoder_pcc(mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int], base_dim: int) -> None:
    """Cosmos3 VAE encoder PCC: TT raw_mu vs torch raw_mu, full host patchify path.

    Env vars:
      COSMOS3_VAE_REAL_WEIGHTS=1  Load nvidia/Cosmos3-Super-Image2Video VAE weights
                                  instead of random. Slow (~30s + disk for HF cache)
                                  but exposes weight-magnitude-sensitive bugs.
    """
    import os

    parent_shape = tuple(mesh_device.shape)
    if submesh_shape[0] > parent_shape[0] or submesh_shape[1] > parent_shape[1]:
        pytest.skip(f"submesh {submesh_shape} doesn't fit in parent {parent_shape}")
    # Use parent mesh directly when submesh == parent. The smoke uses the parent
    # mesh, not a submesh, and that's what we need to reproduce.
    if submesh_shape == parent_shape:
        submesh = mesh_device
    else:
        submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    sub_shape = tuple(submesh.shape)
    tp_axis = max(range(len(sub_shape)), key=lambda i: sub_shape[i])
    sp_axis = 1 - tp_axis if len(sub_shape) == 2 else 0
    h_axis, w_axis = tp_axis, sp_axis
    # Match the smoke's num_links picking: 2 on BH, 4 on WH 4x8, else 1.
    if ttnn.device.is_blackhole():
        num_links = 2
    elif sub_shape == (4, 8):
        num_links = 4
    else:
        num_links = 1
    torch.manual_seed(0)
    if os.environ.get("COSMOS3_VAE_REAL_WEIGHTS") == "1":
        from models.tt_dit.experimental.cosmos3_i2v.model_config import HF_REPO

        torch_vae = TorchAutoencoderKLWan.from_pretrained(HF_REPO, subfolder="vae", torch_dtype=torch.bfloat16).eval()
        logger.info(f"loaded real Cosmos3 VAE; base_dim={torch_vae.config.base_dim} (ignored param={base_dim})")
    else:
        torch_vae = _build_tiny_cosmos3_vae(base_dim=base_dim).to(torch.bfloat16).eval()

    ccl_manager = CCLManager(submesh, topology=ttnn.Topology.Linear, num_links=num_links)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(submesh.shape)[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=tuple(submesh.shape)[w_axis], mesh_axis=w_axis),
    )
    mesh_device = submesh  # downstream code uses `mesh_device`

    # Construct the adapter with the in-memory torch VAE injected so we don't
    # hit HF Hub. Use a sentinel checkpoint_name (only used as a cache subdir).
    adapter = Cosmos3VAEEncoderAdapter(
        checkpoint_name="cosmos3-vae-pcc-test",
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        encoder_t_chunk_size=None,
        vae_dtype=ttnn.bfloat16,
        torch_vae=torch_vae,
    )

    # Match vae_tt_smoke exactly: same image, same preprocess, same tiling.
    # Default is ~/ref.jpg; override with COSMOS3_VAE_TEST_IMAGE=<path>.
    B, C, T = 1, 3, 5
    H = W = 256
    image_path = os.environ.get("COSMOS3_VAE_TEST_IMAGE") or os.path.expanduser("~/ref.jpg")
    from diffusers.video_processor import VideoProcessor
    from PIL import Image

    sf_s = int(torch_vae.config.scale_factor_spatial)
    vp = VideoProcessor(vae_scale_factor=sf_s, resample="bilinear")
    img = Image.open(image_path).convert("RGB").resize((W, H))
    frame_2d = vp.preprocess(img, height=H, width=W).to(dtype=torch.bfloat16)
    x = frame_2d.unsqueeze(2).expand(-1, -1, T, -1, -1).contiguous()

    logger.info(f"Running torch reference encode on input {tuple(x.shape)}")
    with torch.no_grad():
        torch_mu = torch_vae.encode(x).latent_dist.mode()
    logger.info(f"torch raw_mu shape: {tuple(torch_mu.shape)}")

    logger.info("Running TT encoder")
    tt_mu = adapter.encode(x)
    logger.info(f"tt raw_mu shape: {tuple(tt_mu.shape)}")

    # Trim any padding the TT path may have introduced before comparing.
    tt_mu = tt_mu[
        : torch_mu.shape[0], : torch_mu.shape[1], : torch_mu.shape[2], : torch_mu.shape[3], : torch_mu.shape[4]
    ]
    assert_quality(torch_mu.to(torch.float32), tt_mu.to(torch.float32), pcc=0.99, relative_rmse=0.2)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param((2, 4), line_params, id="wh_loudbox_2x4"),
        pytest.param((4, 8), line_params, id="bh_galaxy_4x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "submesh_shape",
    [(1, 1), (1, 2), (1, 4), (2, 4), (4, 8)],
    ids=["sub_1x1", "sub_1x2", "sub_1x4", "sub_2x4", "sub_4x8"],
)
@pytest.mark.timeout(900)
def test_cosmos3_vae_decoder_pcc(mesh_device: ttnn.MeshDevice, submesh_shape: tuple[int, int]) -> None:
    """Cosmos3 VAE decoder PCC: TT decode vs torch decode.

    Mirrors the encoder test harness — same submesh/num_links/sharding rules
    so any pass/fail pattern is comparable. COSMOS3_VAE_REAL_WEIGHTS=1 loads
    real Cosmos3 weights.
    """
    import os

    parent_shape = tuple(mesh_device.shape)
    if submesh_shape[0] > parent_shape[0] or submesh_shape[1] > parent_shape[1]:
        pytest.skip(f"submesh {submesh_shape} doesn't fit in parent {parent_shape}")
    if submesh_shape == parent_shape:
        submesh = mesh_device
    else:
        submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    sub_shape = tuple(submesh.shape)
    tp_axis = max(range(len(sub_shape)), key=lambda i: sub_shape[i])
    sp_axis = 1 - tp_axis if len(sub_shape) == 2 else 0
    h_axis, w_axis = tp_axis, sp_axis
    if ttnn.device.is_blackhole():
        num_links = 2
    elif sub_shape == (4, 8):
        num_links = 4
    else:
        num_links = 1
    torch.manual_seed(0)
    if os.environ.get("COSMOS3_VAE_REAL_WEIGHTS") == "1":
        from models.tt_dit.experimental.cosmos3_i2v.model_config import HF_REPO

        torch_vae = TorchAutoencoderKLWan.from_pretrained(HF_REPO, subfolder="vae", torch_dtype=torch.bfloat16).eval()
    else:
        torch_vae = _build_tiny_cosmos3_vae(base_dim=32).to(torch.bfloat16).eval()

    ccl_manager = CCLManager(submesh, topology=ttnn.Topology.Linear, num_links=num_links)
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=sub_shape[h_axis], mesh_axis=h_axis),
        width_parallel=ParallelFactor(factor=sub_shape[w_axis], mesh_axis=w_axis),
    )
    mesh_device = submesh

    # Smoke-equivalent latent shape: post-encode of (1,3,5,256,256) gives
    # (1, z_dim, 2, 16, 16). COSMOS3_VAE_TEST_T_LAT=1 forces a single-frame
    # latent which avoids the host's chunked-T-encode behavior so per-level
    # stats are directly comparable.
    z_dim = int(torch_vae.config.z_dim)
    T_lat = int(os.environ.get("COSMOS3_VAE_TEST_T_LAT", "2"))
    B, H_lat, W_lat = 1, 16, 16
    z = torch.randn(B, z_dim, T_lat, H_lat, W_lat, dtype=torch.bfloat16)

    pixel_height = H_lat * int(torch_vae.config.scale_factor_spatial)
    pixel_width = W_lat * int(torch_vae.config.scale_factor_spatial)
    pixel_frames = (T_lat - 1) * int(torch_vae.config.scale_factor_temporal) + 1

    # WanVAEDecoderAdapter unconditionally calls AutoencoderKLWan.from_pretrained
    # in its __init__ — must be a resolvable HF repo even when torch_vae is
    # injected. Use HF_REPO when on the real-weights path; the encoder test
    # gets away with a sentinel because Cosmos3VAEEncoderAdapter's `or` short-
    # circuits when torch_vae is passed.
    if os.environ.get("COSMOS3_VAE_REAL_WEIGHTS") == "1":
        from models.tt_dit.experimental.cosmos3_i2v.model_config import HF_REPO

        ckpt_name = HF_REPO
    else:
        pytest.skip("decoder PCC requires real weights for now (WanVAEDecoderAdapter needs a resolvable repo)")

    adapter = Cosmos3VAEDecoderAdapter(
        checkpoint_name=ckpt_name,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        height=pixel_height,
        width=pixel_width,
        num_frames=pixel_frames,
        vae_t_chunk_size=None,
        vae_dtype=ttnn.bfloat16,
        torch_vae=torch_vae,
    )

    # Cosmos3VAEDecoderAdapter neutralizes its inner denorm, matching the
    # production pipeline which pre-denormalizes before calling vae.decode.
    # Both sides see z directly.
    z_raw_host = z

    logger.info(f"Running torch reference decode on latent {tuple(z.shape)}")
    if os.environ.get("TT_DIT_VAE_DEBUG") == "1":
        hooks = []
        host_log: list[tuple[str, tuple, float, float, float, float]] = []
        td = torch_vae.decoder

        def _make_hook(label):
            def _h(_m, _i, out):
                o = out.detach().to(torch.float32).cpu()
                host_log.append(
                    (label, tuple(o.shape), o.min().item(), o.max().item(), o.mean().item(), o.std().item())
                )

            return _h

        hooks.append(td.conv_in.register_forward_hook(_make_hook("host:post_conv_in")))
        hooks.append(td.mid_block.register_forward_hook(_make_hook("host:post_mid_block")))
        for i, ub in enumerate(td.up_blocks):
            hooks.append(ub.register_forward_hook(_make_hook(f"host:up_block[{i}]:{type(ub).__name__}")))
        try:
            with torch.no_grad():
                torch_video = torch_vae.decode(z_raw_host).sample
        finally:
            for h in hooks:
                h.remove()
        for label, shape, mn, mx, mean, std in host_log:
            print(
                f"[wan-decoder-dbg] {label}: shape={shape} min={mn:.4f} max={mx:.4f} mean={mean:.4f} std={std:.4f}",
                flush=True,
            )
    else:
        with torch.no_grad():
            torch_video = torch_vae.decode(z).sample
    logger.info(f"torch decoded shape: {tuple(torch_video.shape)}")

    logger.info("Running TT decoder")
    tt_video = adapter.decode(z, output_type="pt")
    logger.info(f"tt decoded shape: {tuple(tt_video.shape)}")

    tt_video = tt_video[
        : torch_video.shape[0],
        : torch_video.shape[1],
        : torch_video.shape[2],
        : torch_video.shape[3],
        : torch_video.shape[4],
    ]
    assert_quality(torch_video.to(torch.float32), tt_video.to(torch.float32), pcc=0.99, relative_rmse=0.2)


def test_host_patchify_roundtrip_cpu() -> None:
    """Pure-host sanity check for the patchify/unpatchify pair (no device)."""
    torch.manual_seed(0)
    x = torch.randn(1, 3, 5, 16, 16)
    p = 2
    y = _host_patchify_spatial(x, p)
    assert y.shape == (1, 12, 5, 8, 8)
    z = _host_unpatchify_spatial(y, p)
    assert z.shape == x.shape
    torch.testing.assert_close(z, x)
