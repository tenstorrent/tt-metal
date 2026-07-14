# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC tests: Hunyuan VAE decoder PyTorch ref vs TTNN (single test module)."""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import (
    OUT_CHANNELS,
    decoder_tail_shape,
    decoder_up_level_specs,
    get_decoder_tail_input,
    get_decoder_up_input,
    get_input,
    get_mid_input,
    get_up_level_input,
    load_conv_in,
    load_conv_out,
    load_decoder,
    load_decoder_tail,
    load_decoder_up,
    load_mid,
    load_norm_out as load_ref_norm_out,
    load_up_block as load_ref_up_block,
)
from models.experimental.hunyuan_image_3_0.tt.vae.decoder_weights import (
    load_conv_out as load_tt_conv_out,
    load_norm_out as load_tt_norm_out,
    load_up_block as load_tt_up_block,
)
from models.experimental.hunyuan_image_3_0.tests.vae.test_utils import run_bcthw_module
from models.experimental.hunyuan_image_3_0.tt.vae.decoder import (
    ConvInTTNN,
    ConvOutTTNN,
    DecoderUpTTNN,
    MidBlockTTNN,
    NormOutTTNN,
    UpBlockTTNN,
    VAEDecoderTTNN,
    VAEDecoderUpTailTTNN,
)

# Stacked decoder_up can land ~0.9979 under BF16; keep above pipeline gate (0.99).
PCC_THRESHOLD = 0.997


def assert_pcc(pt_out: torch.Tensor, tt_out: torch.Tensor, *, label: str = "") -> float:
    passing, pcc = comp_pcc(pt_out, tt_out, PCC_THRESHOLD)
    name = f" ({label})" if label else ""
    logger.info(f"PCC{name}: {pcc:.6f}")
    assert passing, f"PCC {pcc:.6f} < {PCC_THRESHOLD}"
    return pcc


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_conv_in_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    z = get_input()
    with torch.no_grad():
        pt_out = load_conv_in()(z)
    tt_out = run_bcthw_module(mesh_device, ConvInTTNN(mesh_device), z)
    assert pt_out.shape == tt_out.shape == (1, 1024, 1, 64, 64)
    assert_pcc(pt_out, tt_out, label="conv_in")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_mid_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = get_mid_input()
    with torch.no_grad():
        pt_out = load_mid()(x)
    tt_out = run_bcthw_module(mesh_device, MidBlockTTNN(mesh_device), x)
    assert pt_out.shape == tt_out.shape == (1, 1024, 1, 64, 64)
    assert_pcc(pt_out, tt_out, label="mid")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_up_block_vs_pytorch(mesh_device, level):
    mesh_device.enable_program_cache()
    spec = decoder_up_level_specs()[level]
    x = get_up_level_input(level)
    with torch.no_grad():
        pt_out = load_ref_up_block(level)(x)
    tt_up = UpBlockTTNN(spec, mesh_device)
    load_tt_up_block(tt_up, load_ref_up_block(level))
    tt_out = run_bcthw_module(mesh_device, tt_up, x)
    if spec.has_upsample:
        r1 = 2 if spec.add_temporal_upsample else 1
        expected_shape = (1, spec.upsample_out_channels, spec.t * r1, spec.h * 2, spec.w * 2)
    else:
        expected_shape = (1, spec.block_channels, spec.t, spec.h, spec.w)
    assert pt_out.shape == tt_out.shape == expected_shape
    assert_pcc(pt_out, tt_out, label=f"up_block_{level}")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_norm_out_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    tail_t, tail_h, tail_w, tail_c = decoder_tail_shape()
    x = get_decoder_tail_input()
    with torch.no_grad():
        pt_out = load_ref_norm_out()(x)
    tt_norm = NormOutTTNN(tail_c, mesh_device, t=tail_t, h=tail_h, w=tail_w)
    load_tt_norm_out(tt_norm, load_ref_norm_out())
    tt_out = run_bcthw_module(mesh_device, tt_norm, x)
    assert pt_out.shape == tt_out.shape == (1, tail_c, tail_t, tail_h, tail_w)
    assert_pcc(pt_out, tt_out, label="norm_out")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_conv_out_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    tail_t, tail_h, tail_w, tail_c = decoder_tail_shape()
    with torch.no_grad():
        x = load_ref_norm_out()(get_decoder_tail_input())
        pt_out = load_conv_out()(x)
    tt_conv = ConvOutTTNN(tail_c, mesh_device, t=tail_t, h=tail_h, w=tail_w)
    load_tt_conv_out(tt_conv, load_conv_out())
    tt_out = run_bcthw_module(mesh_device, tt_conv, x)
    assert pt_out.shape == tt_out.shape == (1, OUT_CHANNELS, tail_t, tail_h, tail_w)
    assert_pcc(pt_out, tt_out, label="conv_out")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_decoder_up_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = get_decoder_up_input()
    with torch.no_grad():
        pt_out = load_decoder_up()(x)
    tt_out = run_bcthw_module(mesh_device, DecoderUpTTNN(mesh_device), x)
    tail_t, tail_h, tail_w, tail_c = decoder_tail_shape()
    assert pt_out.shape == tt_out.shape == (1, tail_c, tail_t, tail_h, tail_w)
    assert_pcc(pt_out, tt_out, label="decoder_up")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_decoder_up_tail_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = get_decoder_up_input()
    with torch.no_grad():
        pt_up = load_decoder_up()(x)
        pt_out = load_decoder_tail()(pt_up)
    tt_out = run_bcthw_module(mesh_device, VAEDecoderUpTailTTNN(mesh_device), x)
    tail_t, tail_h, tail_w, _ = decoder_tail_shape()
    assert pt_out.shape == tt_out.shape == (1, OUT_CHANNELS, tail_t, tail_h, tail_w)
    assert_pcc(pt_out, tt_out, label="decoder_up_tail")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_full_decoder_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    z = get_input()
    with torch.no_grad():
        pt_out = load_decoder()(z)
    tt_out = run_bcthw_module(mesh_device, VAEDecoderTTNN(mesh_device), z)
    tail_t, tail_h, tail_w, _ = decoder_tail_shape()
    assert pt_out.shape == tt_out.shape == (1, OUT_CHANNELS, tail_t, tail_h, tail_w)
    assert_pcc(pt_out, tt_out, label="full_decoder")
