# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for TTNN AdainResBlk1d vs PyTorch (encoder encode block and decoder stages).

Decode stages 0–2 match KokoroModules.txt: Conv1d(1090→1024), AdaIN on 1090 / 1024 channels,
style Linear 128→2180 / 128→2048 (decode[3] uses upsample=True and is not covered here).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))


def infer_adain_resblk_dims(block):
    from models.experimental.kokoro.tt import infer_adain_resblk1d_dims

    return infer_adain_resblk1d_dims(block)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    ac = a - a.mean()
    bc = b - b.mean()
    den = (ac.square().sum().sqrt() * bc.square().sum().sqrt()).clamp(min=1e-12)
    return ((ac * bc).sum() / den).item()


def _ttnn():
    return pytest.importorskip("ttnn")


def _ttnn_and_device():
    ttnn = _ttnn()
    if not hasattr(ttnn, "open_device"):
        pytest.skip("TTNN has no open_device (install full tt-metal runtime)")
    return ttnn, ttnn.open_device(device_id=0, l1_small_size=24576)


@pytest.fixture
def ttnn_device():
    ttnn, dev = _ttnn_and_device()
    yield ttnn, dev
    ttnn.close_device(dev)


@pytest.fixture
def kokoro_decoder_cpu():
    from models.experimental.kokoro.reference.kokoro_istftnet import load_decoder_from_huggingface

    return load_decoder_from_huggingface(device="cpu")


def _assert_adain_resblk_pcc(ttnn, device, pytorch_block, batch: int, time_len: int, seed: int) -> None:
    from models.experimental.kokoro.tt import AdainResBlk1d, preprocess_adain_resblk1d_parameters

    dim_in, dim_out, style_dim = infer_adain_resblk_dims(pytorch_block)
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
    p = _pcc(y_ref, y_hat)
    assert p > 0.99, f"PCC {p} expected > 0.99"


def test_adain_resblk_encode_pcc(ttnn_device, kokoro_decoder_cpu):
    """decoder.encode: 514→1024 (dim_in+2 over ASR stack); KokoroModules encode block."""
    ttnn, device = ttnn_device
    block = kokoro_decoder_cpu.decoder.encode
    dim_in, dim_out, style_dim = infer_adain_resblk_dims(block)
    assert dim_in == 514 and dim_out == 1024 and style_dim == 128
    _assert_adain_resblk_pcc(ttnn, device, block, batch=2, time_len=64, seed=0)


@pytest.mark.parametrize("decode_idx", (0, 1, 2))
def test_adain_resblk_decode_stage_pcc(ttnn_device, kokoro_decoder_cpu, decode_idx: int):
    """decoder.decode[0..2]: 1090→1024 per KokoroModules.txt (conv1 1090→1024, norm1 inst 1090, norm2 inst 1024)."""
    ttnn, device = ttnn_device
    block = kokoro_decoder_cpu.decoder.decode[decode_idx]
    assert block.upsample_type == "none"
    dim_in, dim_out, style_dim = infer_adain_resblk_dims(block)
    assert dim_in == 1090 and dim_out == 1024 and style_dim == 128
    _assert_adain_resblk_pcc(ttnn, device, block, batch=2, time_len=64, seed=1 + decode_idx)
