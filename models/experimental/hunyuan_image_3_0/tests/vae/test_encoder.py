# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC tests: Hunyuan VAE encoder PyTorch ref vs TTNN (single test module)."""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.hunyuan_image_3_0.ref.vae.encoder import (
    BLOCK_OUT_CHANNELS,
    LATENT_H,
    LATENT_T,
    LATENT_W,
    OUT_PARAM_CHANNELS,
    PIXEL_H,
    PIXEL_T,
    PIXEL_W,
    encoder_down_level_specs,
    encoder_head_shape,
    get_down_level_input,
    get_encoder_down_input,
    get_encoder_head_input,
    get_input,
    get_mid_input,
    load_conv_in,
    load_down_block as load_ref_down_block,
    load_encoder,
    load_encoder_down,
    load_encoder_head,
    load_mid,
)
from models.experimental.hunyuan_image_3_0.tests.vae.test_utils import (
    pad_encoder_channels_bcthw,
    run_bcthw_module,
)
from models.experimental.hunyuan_image_3_0.tt.vae.encoder import (
    DownBlockTTNN,
    EncoderConvInTTNN,
    EncoderDownTTNN,
    EncoderHeadTTNN,
    EncoderMidBlockTTNN,
    VAEEncoderTTNN,
)
from models.experimental.hunyuan_image_3_0.tt.vae.encoder_weights import load_down_block as load_tt_down_block

PCC_THRESHOLD = 0.998


def assert_pcc(pt_out: torch.Tensor, tt_out: torch.Tensor, *, label: str = "") -> float:
    passing, pcc = comp_pcc(pt_out, tt_out, PCC_THRESHOLD)
    name = f" ({label})" if label else ""
    logger.info(f"PCC{name}: {pcc:.6f}")
    assert passing, f"PCC {pcc:.6f} < {PCC_THRESHOLD}"
    return pcc


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_conv_in_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = pad_encoder_channels_bcthw(get_input())
    with torch.no_grad():
        pt_out = load_conv_in()(get_input())
    tt_out = run_bcthw_module(mesh_device, EncoderConvInTTNN(mesh_device), x)
    assert pt_out.shape == tt_out.shape == (1, BLOCK_OUT_CHANNELS[0], PIXEL_T, PIXEL_H, PIXEL_W)
    assert_pcc(pt_out, tt_out, label="conv_in")


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_down_block_vs_pytorch(mesh_device, level):
    mesh_device.enable_program_cache()
    spec = encoder_down_level_specs()[level]
    x = get_down_level_input(level)
    with torch.no_grad():
        pt_out = load_ref_down_block(level)(x)
    tt_down = DownBlockTTNN(spec, mesh_device)
    load_tt_down_block(tt_down, load_ref_down_block(level))
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
    assert_pcc(pt_out, tt_out, label=f"down_block_{level}")


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_encoder_down_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = get_encoder_down_input()
    with torch.no_grad():
        pt_out = load_encoder_down()(x)
    tt_out = run_bcthw_module(mesh_device, EncoderDownTTNN(mesh_device), x)
    head_t, head_h, head_w, head_c = encoder_head_shape()
    assert pt_out.shape == tt_out.shape == (1, head_c, head_t, head_h, head_w)
    assert_pcc(pt_out, tt_out, label="encoder_down")


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_mid_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = get_mid_input()
    with torch.no_grad():
        pt_out = load_mid()(x)
    tt_out = run_bcthw_module(mesh_device, EncoderMidBlockTTNN(mesh_device), x)
    head_t, head_h, head_w, head_c = encoder_head_shape()
    assert pt_out.shape == tt_out.shape == (1, head_c, head_t, head_h, head_w)
    assert_pcc(pt_out, tt_out, label="mid")


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_encoder_head_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = get_encoder_head_input()
    with torch.no_grad():
        pt_out = load_encoder_head()(x)
    tt_out = run_bcthw_module(mesh_device, EncoderHeadTTNN(mesh_device), x)
    assert pt_out.shape == tt_out.shape == (1, OUT_PARAM_CHANNELS, LATENT_T, LATENT_H, LATENT_W)
    assert_pcc(pt_out, tt_out, label="encoder_head")


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_full_encoder_vs_pytorch(mesh_device):
    mesh_device.enable_program_cache()
    x = pad_encoder_channels_bcthw(get_input())
    with torch.no_grad():
        pt_out = load_encoder()(get_input())
    tt_out = run_bcthw_module(mesh_device, VAEEncoderTTNN(mesh_device), x)
    assert pt_out.shape == tt_out.shape == (1, OUT_PARAM_CHANNELS, LATENT_T, LATENT_H, LATENT_W)
    assert_pcc(pt_out, tt_out, label="full_encoder")
