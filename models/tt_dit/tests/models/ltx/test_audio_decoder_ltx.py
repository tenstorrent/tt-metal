# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 audio mel-VAE decoder Stage A test.

Builds the torch reference ``AudioDecoder`` with the production config
(ch=128, ch_mult=[1,2,4], num_res_blocks=2, no attention, pixel norm, causal
height) and the matching tt-dit ``LTXAudioDecoder``. Random weights, identity
per-channel statistics, latent shape (1, 8, T=64, 64).

PCC ≥ 0.998 vs the torch reference on a 1×1 mesh device.
"""

import sys

import pytest
import torch
from loguru import logger

import ttnn

# Make the ltx-core package importable (matches the pattern used in
# test_vae_ltx.py).
sys.path.insert(0, "LTX-2/packages/ltx-core/src")

from models.tt_dit.models.audio_vae.audio_decoder_ltx import LTXAudioDecoder
from models.tt_dit.utils.check import assert_quality

# Single-chip — Conv2dViaConv3d is single-device. No fabric.
_AUDIO_DECODER_MESH_DEVICE_PARAMS = [
    ((1, 1), {}),
]


# Production config from the LTX-2.3 22B distilled checkpoint's
# audio_vae.model.params.ddconfig (see CLAUDE.md task prompt).
_PROD_CONFIG = dict(
    ch=128,
    out_ch=2,
    ch_mult=(1, 2, 4),
    num_res_blocks=2,
    attn_resolutions=(),
    mid_block_add_attention=False,
    z_channels=8,
    resolution=64,  # mel_bins / 2**(num_resolutions-1) * 2**(num_resolutions-1) -> any value; used by encoder
    mel_bins=64,
    sample_rate=16000,
    mel_hop_length=160,
    is_causal=True,
)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    _AUDIO_DECODER_MESH_DEVICE_PARAMS,
    ids=["1x1"],
    indirect=["mesh_device", "device_params"],
)
def test_ltx_audio_decoder(mesh_device: ttnn.MeshDevice):
    """Stage A: tt-dit ``LTXAudioDecoder`` vs torch reference, random weights."""
    from ltx_core.model.audio_vae.audio_vae import AudioDecoder as TorchAudioDecoder
    from ltx_core.model.audio_vae.causality_axis import CausalityAxis
    from ltx_core.model.common.normalization import NormType

    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # Build torch reference.
    # ------------------------------------------------------------------
    torch_decoder = TorchAudioDecoder(
        ch=_PROD_CONFIG["ch"],
        out_ch=_PROD_CONFIG["out_ch"],
        ch_mult=_PROD_CONFIG["ch_mult"],
        num_res_blocks=_PROD_CONFIG["num_res_blocks"],
        attn_resolutions=set(_PROD_CONFIG["attn_resolutions"]),
        resolution=_PROD_CONFIG["resolution"],
        z_channels=_PROD_CONFIG["z_channels"],
        norm_type=NormType.PIXEL,
        causality_axis=CausalityAxis.HEIGHT,
        dropout=0.0,
        mid_block_add_attention=_PROD_CONFIG["mid_block_add_attention"],
        sample_rate=_PROD_CONFIG["sample_rate"],
        mel_hop_length=_PROD_CONFIG["mel_hop_length"],
        is_causal=_PROD_CONFIG["is_causal"],
        mel_bins=_PROD_CONFIG["mel_bins"],
    )
    torch_decoder.eval()

    # Identity per-channel statistics — so denormalize is x*1 + 0 = x.
    #
    # The reference ``PerChannelStatistics`` is constructed with
    # ``latent_channels=ch=128``, but the decoder calls ``un_normalize`` on a
    # ``(B, T, z*F) = (1, T, 8*64=512)`` patched tensor — the stats need to
    # broadcast against the last dim (z*F). We override both buffers directly
    # at the correct shape so the broadcast works and yields an identity.
    z_times_f = _PROD_CONFIG["z_channels"] * _PROD_CONFIG["mel_bins"]
    torch_decoder.per_channel_statistics.__dict__["_buffers"]["std-of-means"] = torch.ones(z_times_f)
    torch_decoder.per_channel_statistics.__dict__["_buffers"]["mean-of-means"] = torch.zeros(z_times_f)

    # ------------------------------------------------------------------
    # Build tt-dit decoder, load same state dict.
    # ------------------------------------------------------------------
    tt_decoder = LTXAudioDecoder(
        ch=_PROD_CONFIG["ch"],
        out_ch=_PROD_CONFIG["out_ch"],
        ch_mult=_PROD_CONFIG["ch_mult"],
        num_res_blocks=_PROD_CONFIG["num_res_blocks"],
        attn_resolutions=_PROD_CONFIG["attn_resolutions"],
        resolution=_PROD_CONFIG["resolution"],
        z_channels=_PROD_CONFIG["z_channels"],
        mid_block_add_attention=_PROD_CONFIG["mid_block_add_attention"],
        sample_rate=_PROD_CONFIG["sample_rate"],
        mel_hop_length=_PROD_CONFIG["mel_hop_length"],
        is_causal=_PROD_CONFIG["is_causal"],
        mel_bins=_PROD_CONFIG["mel_bins"],
        mesh_device=mesh_device,
    )
    tt_decoder.load_torch_state_dict(torch_decoder.state_dict())

    # ------------------------------------------------------------------
    # Run.
    # ------------------------------------------------------------------
    B = 1
    T = 64
    F = _PROD_CONFIG["mel_bins"]
    z = _PROD_CONFIG["z_channels"]
    latent = torch.randn(B, z, T, F, dtype=torch.float32)

    with torch.no_grad():
        torch_out = torch_decoder(latent)

    tt_out = tt_decoder(latent)

    logger.info(f"Audio decoder: {tuple(latent.shape)} -> torch {tuple(torch_out.shape)}, tt {tuple(tt_out.shape)}")
    assert torch_out.shape == tt_out.shape, f"shape mismatch: torch {torch_out.shape}, tt {tt_out.shape}"
    assert_quality(torch_out, tt_out, pcc=0.998, relative_rmse=0.05)
    logger.info("PASSED: LTXAudioDecoder matches torch reference (PCC ≥ 0.998)")
