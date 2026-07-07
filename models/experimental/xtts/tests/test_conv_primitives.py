# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for the XTTS-v2 HiFi-GAN decoder conv primitives.

Validates :class:`TtConv1d` and :class:`TtConvTranspose1d` against ``torch.nn``
on the exact layer shapes the XTTS ``waveform_decoder`` uses (conv_pre, the
dilated ResBlock1 convs, the 1x1 conditioning convs, conv_post, and the
ConvTranspose1d upsamplers). No checkpoint download — random weights are enough
to prove the op math.

Run:
    source python_env/bin/activate
    export TT_METAL_HOME=$(pwd)
    export PYTHONPATH=$(pwd)
    pytest models/experimental/xtts/tests/test_conv_primitives.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.xtts.tt.xtts_conv import TtConv1d, TtConvTranspose1d


def _to_device_blc(torch_ncl: torch.Tensor, device) -> ttnn.Tensor:
    """torch [N, C, L] -> ttnn channels-last [N, L, C] fp32 ROW_MAJOR on device."""
    return ttnn.from_torch(
        torch_ncl.permute(0, 2, 1).contiguous().float(),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.float32,
    )


def _from_device_blc(tt_blc: ttnn.Tensor) -> torch.Tensor:
    """ttnn [N, L, C] -> torch [N, C, L]."""
    return ttnn.to_torch(tt_blc).float().permute(0, 2, 1)


# (name, in_channels, out_channels, kernel_size, dilation, bias) — "same" padding.
CONV1D_SHAPES = [
    ("conv_pre", 1024, 512, 7, 1, True),
    ("resblock_k3_d1", 256, 256, 3, 1, True),
    ("resblock_k3_d3", 256, 256, 3, 3, True),
    ("resblock_k7_d3", 128, 128, 7, 3, True),
    ("resblock_k11_d5", 128, 128, 11, 5, True),
    ("cond_1x1", 512, 256, 1, 1, True),
    ("conv_post", 32, 1, 7, 1, False),
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_length", [256])
@pytest.mark.parametrize("name,in_ch,out_ch,k,dilation,bias", CONV1D_SHAPES)
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_conv1d(device, reset_seeds, name, in_ch, out_ch, k, dilation, bias, input_length, pcc):
    padding = dilation * (k - 1) // 2  # "same" length
    ref = torch.nn.Conv1d(in_ch, out_ch, k, stride=1, padding=padding, dilation=dilation, bias=bias).eval()

    torch_input = torch.randn(1, in_ch, input_length)
    with torch.no_grad():
        ref_out = ref(torch_input)

    tt_conv = TtConv1d(
        device,
        ref.weight.detach(),
        ref.bias.detach() if bias else None,
        padding=padding,
        dilation=dilation,
    )
    tt_out = _from_device_blc(tt_conv(_to_device_blc(torch_input, device)))

    assert tt_out.shape == ref_out.shape, f"{name}: shape {tuple(tt_out.shape)} != {tuple(ref_out.shape)}"
    does_pass, msg = comp_pcc(ref_out, tt_out, pcc)
    logger.info(comp_allclose(ref_out, tt_out))
    logger.info(f"conv1d[{name}] k={k} d={dilation}: {msg}")
    assert does_pass, f"conv1d[{name}] PCC below {pcc}: {msg}"


# (name, in_channels, out_channels, kernel_size, stride) — XTTS upsample layers.
CONVT1D_SHAPES = [
    ("ups0", 512, 256, 16, 8),
    ("ups1", 256, 128, 16, 8),
    ("ups2", 128, 64, 4, 2),
    ("ups3", 64, 32, 4, 2),
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("input_length", [32])
@pytest.mark.parametrize("name,in_ch,out_ch,k,stride", CONVT1D_SHAPES)
@pytest.mark.parametrize("pcc", [0.99])
def test_tt_conv_transpose1d(device, reset_seeds, name, in_ch, out_ch, k, stride, input_length, pcc):
    padding = (k - stride) // 2  # HiFi-GAN convention -> exact stride x upsample
    ref = torch.nn.ConvTranspose1d(in_ch, out_ch, k, stride=stride, padding=padding, bias=True).eval()

    torch_input = torch.randn(1, in_ch, input_length)
    with torch.no_grad():
        ref_out = ref(torch_input)

    tt_convt = TtConvTranspose1d(device, ref.weight.detach(), ref.bias.detach(), stride=stride)
    tt_out = _from_device_blc(tt_convt(_to_device_blc(torch_input, device)))

    assert tt_out.shape == ref_out.shape, f"{name}: shape {tuple(tt_out.shape)} != {tuple(ref_out.shape)}"
    does_pass, msg = comp_pcc(ref_out, tt_out, pcc)
    logger.info(comp_allclose(ref_out, tt_out))
    logger.info(f"conv_transpose1d[{name}] k={k} s={stride}: {msg}")
    assert does_pass, f"conv_transpose1d[{name}] PCC below {pcc}: {msg}"
