# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Kokoro ISTFTNet AdainResBlk1d PCC (TTNN vs PyTorch)

TTNN `AdainResBlk1d` against PyTorch for `decoder.encode`, `decoder.decode[0..2]`,
and `decoder.decode[3]` (nearest 2× shortcut + depthwise `ConvTranspose1d` pool).
Expect PCC >= 0.99 vs CPU reference (`comp_pcc`).

Setup:
    cd <tt-metal-root>
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate

Usage:
    pytest models/experimental/kokoro/tests/test_adain_encode_pcc.py --confcutdir=models/experimental/kokoro -v
"""

import sys
from pathlib import Path

import pytest
import torch

# Add tt-metal root directory to path (parents[4] from tests/ → tt-metal root)
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

ttnn = pytest.importorskip("ttnn")

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.kokoro_istftnet import load_decoder_from_huggingface
from models.experimental.kokoro.tt import (
    AdainResBlk1d,
    infer_adain_resblk1d_dims,
    preprocess_adain_resblk1d_parameters,
)


@pytest.fixture
def ttnn_device():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield device
    ttnn.close_device(device)


@pytest.fixture
def kokoro_decoder_cpu():
    return load_decoder_from_huggingface(device="cpu")


def assert_adain_resblk_pcc(device, pytorch_block, batch: int, time_len: int, seed: int) -> None:
    dim_in, dim_out, style_dim = infer_adain_resblk1d_dims(pytorch_block)
    torch.manual_seed(seed)
    x = torch.randn(batch, dim_in, time_len, dtype=torch.float32)
    s = torch.randn(batch, style_dim, dtype=torch.float32)

    with torch.no_grad():
        y_ref = pytorch_block(x, s)

    params = preprocess_adain_resblk1d_parameters(pytorch_block, device)
    tt_block = AdainResBlk1d(device, params, dim_in, dim_out, style_dim)

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    s_tt = ttnn.from_torch(
        s,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    y_tt = tt_block(x_tt, s_tt)
    y_hat = ttnn.to_torch(y_tt).reshape(y_ref.shape)
    ok, p = comp_pcc(y_ref, y_hat, pcc=0.99)
    print(f"PCC={p:.6f} dim_in={dim_in} dim_out={dim_out} pass={ok}")
    assert ok, f"PCC {p} expected >= 0.99"


def test_adain_resblk_encode_pcc(ttnn_device, kokoro_decoder_cpu):
    """decoder.encode: 514→1024; KokoroModules encode block."""
    block = kokoro_decoder_cpu.decoder.encode
    dim_in, dim_out, style_dim = infer_adain_resblk1d_dims(block)
    assert dim_in == 514 and dim_out == 1024 and style_dim == 128
    assert_adain_resblk_pcc(ttnn_device, block, batch=2, time_len=64, seed=0)


@pytest.mark.parametrize("decode_idx", (0, 1, 2, 3))
def test_adain_resblk_decode_stage_pcc(ttnn_device, kokoro_decoder_cpu, decode_idx: int):
    """decoder.decode[*]: KokoroModules decode ModuleList (1090→1024 or 1090→512 + upsample on [3])."""
    block = kokoro_decoder_cpu.decoder.decode[decode_idx]
    dim_in, dim_out, style_dim = infer_adain_resblk1d_dims(block)
    assert dim_in == 1090 and style_dim == 128
    if decode_idx == 3:
        assert block.upsample_type != "none"
        assert dim_out == 512
    else:
        assert block.upsample_type == "none"
        assert dim_out == 1024
    assert_adain_resblk_pcc(ttnn_device, block, batch=2, time_len=64, seed=1 + decode_idx)
