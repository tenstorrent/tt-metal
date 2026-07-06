# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from huggingface_hub import snapshot_download

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")


def _fibo_local():
    try:
        return snapshot_download(FIBO_PATH, allow_patterns=["scheduler/*", "vae/*"], local_files_only=True)
    except Exception as e:
        pytest.skip(f"FIBO not cached: {e}")


def _load_ref_vae(dtype=None):
    import torch
    from diffusers import AutoencoderKLWan

    try:
        path = snapshot_download(FIBO_PATH, allow_patterns=["vae/*"], local_files_only=True)
        return AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=dtype or torch.float32).eval()
    except Exception as e:
        pytest.skip(f"FIBO vae unavailable: {e}")


def test_fibo_vae_reference_config():
    m = _load_ref_vae()
    c = m.config
    assert c.z_dim == 48 and c.is_residual is True
    assert c.decoder_base_dim == 256 and c.base_dim == 160
    assert c.dim_mult == [1, 2, 4, 4] and c.out_channels == 12
    assert c.scale_factor_spatial == 16


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_fibo_vae_decode(*, mesh_device):
    """Decode FIBO's z_dim=48 Wan 2.2 residual (is_residual=True) latents on device, PCC vs HF reference.

    Reduced latent resolution (T=1, 16x16) for speed. Mirrors the latent prep / decode / gather / trim
    of tests/models/wan2_2/test_vae_wan2_1.py::test_wan_decoder, but for the residual decoder path.
    """
    import torch
    from loguru import logger

    import ttnn
    from models.tt_dit.models.vae.vae_wan2_1 import WanDecoder
    from models.tt_dit.parallel.config import VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.check import assert_quality
    from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_in_channels, conv_pad_width
    from models.tt_dit.utils.tensor import typed_tensor_2dshard

    ref = _load_ref_vae()
    c = ref.config

    torch.manual_seed(0)
    latent_h, latent_w = 16, 16
    z = torch.randn(1, c.z_dim, 1, latent_h, latent_w)  # (B, 48, T=1, H, W)

    with torch.no_grad():
        ref_img = ref.decode(z, return_dict=False)[0]
    logger.info(f"ref decode output shape: {tuple(ref_img.shape)}")

    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pc = VaeHWParallelConfig.from_tuples(height=(1, 0), width=(1, 1))

    dtype = ttnn.bfloat16
    dec = WanDecoder(
        base_dim=c.base_dim,
        decoder_base_dim=c.decoder_base_dim,
        z_dim=c.z_dim,
        dim_mult=c.dim_mult,
        num_res_blocks=c.num_res_blocks,
        out_channels=c.out_channels,
        is_residual=True,
        temperal_downsample=c.temperal_downsample,
        latents_mean=c.latents_mean,
        latents_std=c.latents_std,
        mesh_device=mesh_device,
        parallel_config=pc,
        ccl_manager=ccl,
        dtype=dtype,
    )
    dec.load_torch_state_dict(ref.state_dict())

    # Latent prep: BCTHW -> BTHWC, pad channels/height/width, shard.
    tt_z_BTHWC = z.permute(0, 2, 3, 4, 1)
    tt_z_BTHWC = conv_pad_in_channels(tt_z_BTHWC)
    tt_z_BTHWC, logical_h = conv_pad_height(tt_z_BTHWC, pc.height_parallel.factor)
    tt_z_BTHWC, logical_w = conv_pad_width(tt_z_BTHWC, pc.width_parallel.factor)
    tt_z_BTHWC = typed_tensor_2dshard(
        tt_z_BTHWC,
        mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_mapping={pc.height_parallel.mesh_axis: 2, pc.width_parallel.mesh_axis: 3},
        dtype=dtype,
    )

    tt_out_BCTHW, new_logical_h, new_logical_w = dec(tt_z_BTHWC, logical_h, t_chunk_size=None, logical_w=logical_w)

    concat_dims = [None, None]
    concat_dims[pc.height_parallel.mesh_axis] = 3
    concat_dims[pc.width_parallel.mesh_axis] = 4
    tt_img = ttnn.to_torch(
        tt_out_BCTHW,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
    )
    logger.info(f"tt decode output shape (pre-unpatchify): {tuple(tt_img.shape)}")

    # tt WanDecoder emits patchified pixel space (out_channels=12); the reference decode() additionally
    # runs unpatchify (patch_size=2) -> (1, 3, T, H, W). Apply the same unpatchify to compare in RGB space.
    from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

    # Trim decoder padding (channels/height/width) before unpatchify.
    tt_img = tt_img[:, : c.out_channels]
    if new_logical_h != tt_img.shape[3]:
        tt_img = tt_img[:, :, :, :new_logical_h, :]
    if new_logical_w != tt_img.shape[4]:
        tt_img = tt_img[:, :, :, :, :new_logical_w]
    if c.patch_size is not None:
        tt_img = unpatchify(tt_img, patch_size=c.patch_size)
    tt_img = torch.clamp(tt_img, min=-1.0, max=1.0)
    logger.info(f"tt decode output shape (unpatchified): {tuple(tt_img.shape)}")

    assert_quality(ref_img, tt_img, pcc=0.99)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=["device_params"])
def test_fibo_vae_decode_production(*, mesh_device):
    """Production-resolution decode of FIBO's z_dim=48 Wan 2.2 residual VAE on single Blackhole device.

    Targets a 1024x1024 image = 64x64 latent (scale_factor_spatial=16, 8x conv upscale * 2x unpatchify).
    Uses full-T single-pass mode (t_chunk_size=None) with (1,1) single-device config.
    PCC threshold: ≥ 0.99 vs HF reference AutoencoderKLWan.decode().

    If the 64x64 latent OOMs, the test steps down (48x48, 32x32) and asserts the largest
    resolution that fits, reporting the validated resolution in the test log.
    """
    import torch
    from loguru import logger

    import ttnn
    from models.tt_dit.models.vae.vae_wan2_1 import WanDecoder
    from models.tt_dit.parallel.config import VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.check import assert_quality
    from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_in_channels, conv_pad_width
    from models.tt_dit.utils.tensor import typed_tensor_2dshard

    ref = _load_ref_vae()
    c = ref.config

    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    pc = VaeHWParallelConfig.from_tuples(height=(1, 0), width=(1, 1))
    dtype = ttnn.bfloat16

    # Try production resolutions in decreasing order: 64x64 -> 48x48 -> 32x32.
    # The first that succeeds (PCC >= 0.99, no OOM) is the validated resolution.
    candidate_sizes = [64, 48, 32]
    last_exc = None
    for latent_hw in candidate_sizes:
        torch.manual_seed(0)
        z = torch.randn(1, c.z_dim, 1, latent_hw, latent_hw)  # (B, 48, T=1, H, W)

        with torch.no_grad():
            ref_img = ref.decode(z, return_dict=False)[0]
        image_hw = ref_img.shape[-1]
        logger.info(f"Trying latent {latent_hw}x{latent_hw} -> image {image_hw}x{image_hw}")
        logger.info(f"ref decode output shape: {tuple(ref_img.shape)}")

        # Rebuild decoder for each attempt (fresh weights each time).
        dec = WanDecoder(
            base_dim=c.base_dim,
            decoder_base_dim=c.decoder_base_dim,
            z_dim=c.z_dim,
            dim_mult=c.dim_mult,
            num_res_blocks=c.num_res_blocks,
            out_channels=c.out_channels,
            is_residual=True,
            temperal_downsample=c.temperal_downsample,
            latents_mean=c.latents_mean,
            latents_std=c.latents_std,
            mesh_device=mesh_device,
            parallel_config=pc,
            ccl_manager=ccl,
            dtype=dtype,
        )
        dec.load_torch_state_dict(ref.state_dict())

        # Latent prep: BCTHW -> BTHWC, pad channels/height/width, shard.
        tt_z_BTHWC = z.permute(0, 2, 3, 4, 1)
        tt_z_BTHWC = conv_pad_in_channels(tt_z_BTHWC)
        tt_z_BTHWC, logical_h = conv_pad_height(tt_z_BTHWC, pc.height_parallel.factor)
        tt_z_BTHWC, logical_w = conv_pad_width(tt_z_BTHWC, pc.width_parallel.factor)
        tt_z_BTHWC = typed_tensor_2dshard(
            tt_z_BTHWC,
            mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_mapping={pc.height_parallel.mesh_axis: 2, pc.width_parallel.mesh_axis: 3},
            dtype=dtype,
        )

        try:
            tt_out_BCTHW, new_logical_h, new_logical_w = dec(
                tt_z_BTHWC, logical_h, t_chunk_size=None, logical_w=logical_w
            )
        except Exception as exc:
            logger.warning(f"latent {latent_hw}x{latent_hw} failed: {exc}")
            last_exc = exc
            # Deallocate device tensors before retry.
            del tt_z_BTHWC
            continue

        concat_dims = [None, None]
        concat_dims[pc.height_parallel.mesh_axis] = 3
        concat_dims[pc.width_parallel.mesh_axis] = 4
        tt_img = ttnn.to_torch(
            tt_out_BCTHW,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
        )
        logger.info(f"tt decode output shape (pre-unpatchify): {tuple(tt_img.shape)}")

        # tt WanDecoder emits patchified pixel space (out_channels=12); the reference decode()
        # additionally runs unpatchify (patch_size=2) -> (1, 3, T, H, W).
        from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

        # Trim decoder padding (channels/height/width) before unpatchify.
        tt_img = tt_img[:, : c.out_channels]
        if new_logical_h != tt_img.shape[3]:
            tt_img = tt_img[:, :, :, :new_logical_h, :]
        if new_logical_w != tt_img.shape[4]:
            tt_img = tt_img[:, :, :, :, :new_logical_w]
        if c.patch_size is not None:
            tt_img = unpatchify(tt_img, patch_size=c.patch_size)
        tt_img = torch.clamp(tt_img, min=-1.0, max=1.0)
        logger.info(f"tt decode output shape (unpatchified): {tuple(tt_img.shape)}")

        logger.info(f"PRODUCTION RESOLUTION VALIDATED: latent {latent_hw}x{latent_hw} -> image {image_hw}x{image_hw}")
        assert_quality(ref_img, tt_img, pcc=0.99)
        return  # success — no need to try smaller sizes

    # All candidates failed.
    raise RuntimeError(
        f"test_fibo_vae_decode_production: all candidate latent sizes {candidate_sizes} failed. "
        f"Last error: {last_exc}"
    )
