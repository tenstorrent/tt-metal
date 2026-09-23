# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC tests: Hunyuan VAE encoder and decoder, PyTorch ref vs TTNN.

Encoder and decoder share the same rig (2x2 mesh + ``vae_helpers.run_bcthw_module``)
but keep separate PCC gates — see ENCODER_PCC_THRESHOLD / DECODER_PCC_THRESHOLD.

The ref modules ``ref.vae.encoder`` and ``ref.vae.decoder`` both export ``get_input``,
``get_mid_input``, ``load_conv_in`` and ``load_mid``, so they are imported module-
qualified as ``ref_enc`` / ``ref_dec`` rather than by name.

Run:
  python_env/bin/python -m pytest \
    models/experimental/hunyuan_image_3_0/tests/pcc/test_vae.py -v
  # one side only:
  ... -m pytest .../test_vae.py -k encoder -v
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.hunyuan_image_3_0.ref.vae import decoder as ref_dec, encoder as ref_enc
from models.experimental.hunyuan_image_3_0.tests.pcc.vae_helpers import (
    pad_encoder_channels_bcthw,
    run_bcthw_module,
)
from models.experimental.hunyuan_image_3_0.ttnn.vae.decoder import (
    ConvInTTNN,
    ConvOutTTNN,
    DecoderUpTTNN,
    MidBlockTTNN,
    NormOutTTNN,
    UpBlockTTNN,
    VAEDecoderTTNN,
    VAEDecoderUpTailTTNN,
)
from models.experimental.hunyuan_image_3_0.ttnn.vae.encoder import (
    DownBlockTTNN,
    EncoderConvInTTNN,
    EncoderDownTTNN,
    EncoderHeadTTNN,
    EncoderMidBlockTTNN,
    VAEEncoderTTNN,
)
from models.experimental.hunyuan_image_3_0.ttnn.vae.weights import (
    load_conv_out as load_tt_conv_out,
    load_down_block as load_tt_down_block,
    load_norm_out as load_tt_norm_out,
    load_up_block as load_tt_up_block,
)

ENCODER_PCC_THRESHOLD = 0.998
# Stacked decoder_up can land ~0.9979 under BF16; keep above pipeline gate (0.99).
DECODER_PCC_THRESHOLD = 0.997


def _assert_pcc(pt_out: torch.Tensor, tt_out: torch.Tensor, threshold: float, label: str) -> float:
    passing, pcc = comp_pcc(pt_out, tt_out, threshold)
    name = f" ({label})" if label else ""
    logger.info(f"PCC{name}: {pcc:.6f}")
    assert passing, f"PCC {pcc:.6f} < {threshold}"
    return pcc


def assert_encoder_pcc(pt_out: torch.Tensor, tt_out: torch.Tensor, *, label: str = "") -> float:
    return _assert_pcc(pt_out, tt_out, ENCODER_PCC_THRESHOLD, label)


def assert_decoder_pcc(pt_out: torch.Tensor, tt_out: torch.Tensor, *, label: str = "") -> float:
    return _assert_pcc(pt_out, tt_out, DECODER_PCC_THRESHOLD, label)


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_encoder_conv_in_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = pad_encoder_channels_bcthw(ref_enc.get_input())
    with torch.no_grad():
        pt_out = ref_enc.load_conv_in()(ref_enc.get_input())
    tt_out = run_bcthw_module(mesh_device, EncoderConvInTTNN(mesh_device), x)
    assert (
        pt_out.shape
        == tt_out.shape
        == (1, ref_enc.BLOCK_OUT_CHANNELS[0], ref_enc.PIXEL_T, ref_enc.PIXEL_H, ref_enc.PIXEL_W)
    )
    assert_encoder_pcc(pt_out, tt_out, label="conv_in")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_down_block_vs_pytorch(mesh_device, level):
    mesh_device.enable_program_cache()
    spec = ref_enc.encoder_down_level_specs()[level]
    x = ref_enc.get_down_level_input(level)
    with torch.no_grad():
        pt_out = ref_enc.load_down_block(level)(x)
    tt_down = DownBlockTTNN(spec, mesh_device)
    load_tt_down_block(tt_down, ref_enc.load_down_block(level))
    tt_out = run_bcthw_module(mesh_device, tt_down, x)
    if spec.has_downsample:
        r1 = 2 if spec.add_temporal_downsample else 1
        expected_shape = (
            1,
            spec.downsample_out_channels,
            spec.t // r1,
            spec.h // 2,
            spec.w // 2,
        )
    else:
        expected_shape = (1, spec.block_channels, spec.t, spec.h, spec.w)
    assert pt_out.shape == tt_out.shape == expected_shape
    assert_encoder_pcc(pt_out, tt_out, label=f"down_block_{level}")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_encoder_down_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = ref_enc.get_encoder_down_input()
    with torch.no_grad():
        pt_out = ref_enc.load_encoder_down()(x)
    tt_out = run_bcthw_module(mesh_device, EncoderDownTTNN(mesh_device), x)
    head_t, head_h, head_w, head_c = ref_enc.encoder_head_shape()
    assert pt_out.shape == tt_out.shape == (1, head_c, head_t, head_h, head_w)
    assert_encoder_pcc(pt_out, tt_out, label="encoder_down")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_encoder_mid_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = ref_enc.get_mid_input()
    with torch.no_grad():
        pt_out = ref_enc.load_mid()(x)
    tt_out = run_bcthw_module(mesh_device, EncoderMidBlockTTNN(mesh_device), x)
    head_t, head_h, head_w, head_c = ref_enc.encoder_head_shape()
    assert pt_out.shape == tt_out.shape == (1, head_c, head_t, head_h, head_w)
    assert_encoder_pcc(pt_out, tt_out, label="mid")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_encoder_head_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = ref_enc.get_encoder_head_input()
    with torch.no_grad():
        pt_out = ref_enc.load_encoder_head()(x)
    tt_out = run_bcthw_module(mesh_device, EncoderHeadTTNN(mesh_device), x)
    assert (
        pt_out.shape
        == tt_out.shape
        == (1, ref_enc.OUT_PARAM_CHANNELS, ref_enc.LATENT_T, ref_enc.LATENT_H, ref_enc.LATENT_W)
    )
    assert_encoder_pcc(pt_out, tt_out, label="encoder_head")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_full_encoder_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = pad_encoder_channels_bcthw(ref_enc.get_input())
    with torch.no_grad():
        pt_out = ref_enc.load_encoder()(ref_enc.get_input())
    tt_out = run_bcthw_module(mesh_device, VAEEncoderTTNN(mesh_device), x)
    assert (
        pt_out.shape
        == tt_out.shape
        == (1, ref_enc.OUT_PARAM_CHANNELS, ref_enc.LATENT_T, ref_enc.LATENT_H, ref_enc.LATENT_W)
    )
    assert_encoder_pcc(pt_out, tt_out, label="full_encoder")


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_decoder_conv_in_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    z = ref_dec.get_input()
    with torch.no_grad():
        pt_out = ref_dec.load_conv_in()(z)
    tt_out = run_bcthw_module(mesh_device, ConvInTTNN(mesh_device), z)
    assert pt_out.shape == tt_out.shape == (1, 1024, 1, 64, 64)
    assert_decoder_pcc(pt_out, tt_out, label="conv_in")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_decoder_mid_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = ref_dec.get_mid_input()
    with torch.no_grad():
        pt_out = ref_dec.load_mid()(x)
    tt_out = run_bcthw_module(mesh_device, MidBlockTTNN(mesh_device), x)
    assert pt_out.shape == tt_out.shape == (1, 1024, 1, 64, 64)
    assert_decoder_pcc(pt_out, tt_out, label="mid")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_up_block_vs_pytorch(mesh_device, level):
    mesh_device.enable_program_cache()
    spec = ref_dec.decoder_up_level_specs()[level]
    x = ref_dec.get_up_level_input(level)
    with torch.no_grad():
        pt_out = ref_dec.load_up_block(level)(x)
    tt_up = UpBlockTTNN(spec, mesh_device)
    load_tt_up_block(tt_up, ref_dec.load_up_block(level))
    tt_out = run_bcthw_module(mesh_device, tt_up, x)
    if spec.has_upsample:
        r1 = 2 if spec.add_temporal_upsample else 1
        expected_shape = (1, spec.upsample_out_channels, spec.t * r1, spec.h * 2, spec.w * 2)
    else:
        expected_shape = (1, spec.block_channels, spec.t, spec.h, spec.w)
    assert pt_out.shape == tt_out.shape == expected_shape
    assert_decoder_pcc(pt_out, tt_out, label=f"up_block_{level}")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_norm_out_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    tail_t, tail_h, tail_w, tail_c = ref_dec.decoder_tail_shape()
    x = ref_dec.get_decoder_tail_input()
    with torch.no_grad():
        pt_out = ref_dec.load_norm_out()(x)
    tt_norm = NormOutTTNN(tail_c, mesh_device, t=tail_t, h=tail_h, w=tail_w)
    load_tt_norm_out(tt_norm, ref_dec.load_norm_out())
    tt_out = run_bcthw_module(mesh_device, tt_norm, x)
    assert pt_out.shape == tt_out.shape == (1, tail_c, tail_t, tail_h, tail_w)
    assert_decoder_pcc(pt_out, tt_out, label="norm_out")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_conv_out_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    tail_t, tail_h, tail_w, tail_c = ref_dec.decoder_tail_shape()
    with torch.no_grad():
        x = ref_dec.load_norm_out()(ref_dec.get_decoder_tail_input())
        pt_out = ref_dec.load_conv_out()(x)
    tt_conv = ConvOutTTNN(tail_c, mesh_device, t=tail_t, h=tail_h, w=tail_w)
    load_tt_conv_out(tt_conv, ref_dec.load_conv_out())
    tt_out = run_bcthw_module(mesh_device, tt_conv, x)
    assert pt_out.shape == tt_out.shape == (1, ref_dec.OUT_CHANNELS, tail_t, tail_h, tail_w)
    assert_decoder_pcc(pt_out, tt_out, label="conv_out")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_decoder_up_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = ref_dec.get_decoder_up_input()
    with torch.no_grad():
        pt_out = ref_dec.load_decoder_up()(x)
    tt_out = run_bcthw_module(mesh_device, DecoderUpTTNN(mesh_device), x)
    tail_t, tail_h, tail_w, tail_c = ref_dec.decoder_tail_shape()
    assert pt_out.shape == tt_out.shape == (1, tail_c, tail_t, tail_h, tail_w)
    assert_decoder_pcc(pt_out, tt_out, label="decoder_up")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_decoder_up_tail_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = ref_dec.get_decoder_up_input()
    with torch.no_grad():
        pt_up = ref_dec.load_decoder_up()(x)
        pt_out = ref_dec.load_decoder_tail()(pt_up)
    tt_out = run_bcthw_module(mesh_device, VAEDecoderUpTailTTNN(mesh_device), x)
    tail_t, tail_h, tail_w, _ = ref_dec.decoder_tail_shape()
    assert pt_out.shape == tt_out.shape == (1, ref_dec.OUT_CHANNELS, tail_t, tail_h, tail_w)
    assert_decoder_pcc(pt_out, tt_out, label="decoder_up_tail")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_full_decoder_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    z = ref_dec.get_input()
    with torch.no_grad():
        pt_out = ref_dec.load_decoder()(z)
    tt_out = run_bcthw_module(mesh_device, VAEDecoderTTNN(mesh_device), z)
    tail_t, tail_h, tail_w, _ = ref_dec.decoder_tail_shape()
    assert pt_out.shape == tt_out.shape == (1, ref_dec.OUT_CHANNELS, tail_t, tail_h, tail_w)
    assert_decoder_pcc(pt_out, tt_out, label="full_decoder")
