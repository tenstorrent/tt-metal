# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :class:`~models.experimental.kokoro.tt.tt_adain_resblk_1d.TTAdainResBlk1d` vs PyTorch reference block."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import ttnn

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.istftnet import AdainResBlk1d
from models.experimental.kokoro.tt.tt_adain_resblk_1d import TTAdainResBlk1d, preprocess_tt_adain_resblk_1d


def _pcc_nlc(y_ref_bcl: torch.Tensor, y_tt) -> float:
    y_ref_nlc = y_ref_bcl.transpose(1, 2).contiguous()
    y_hat = ttnn.to_torch(y_tt).float()
    _, pcc = comp_pcc(y_ref_nlc, y_hat, pcc=0.0)
    return pcc


def test_tt_adain_resblk_1d_no_upsample_equal_channels(device):
    torch.manual_seed(0)
    dim_in = dim_out = 48
    style_dim = 32
    b, l = 2, 40
    ref = AdainResBlk1d(dim_in, dim_out, style_dim=style_dim, upsample="none")
    ref.eval()
    params = preprocess_tt_adain_resblk_1d(ref, device)
    tt_blk = TTAdainResBlk1d(device, params)

    x_bcl = torch.randn(b, dim_in, l)
    s = torch.randn(b, style_dim)
    with torch.no_grad():
        y_ref = ref(x_bcl, s)

    x_nlc = x_bcl.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_blk(x_tt, s_tt)
    pcc = _pcc_nlc(y_ref, y_tt)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)

    print(f"TTAdainResBlk1d (no up, C_in=C_out) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_adain_resblk_1d_no_upsample_learned_sc(device):
    torch.manual_seed(1)
    dim_in, dim_out, style_dim = 40, 72, 48
    b, l = 1, 36
    ref = AdainResBlk1d(dim_in, dim_out, style_dim=style_dim, upsample="none")
    ref.eval()
    params = preprocess_tt_adain_resblk_1d(ref, device)
    tt_blk = TTAdainResBlk1d(device, params)

    x_bcl = torch.randn(b, dim_in, l)
    s = torch.randn(b, style_dim)
    with torch.no_grad():
        y_ref = ref(x_bcl, s)

    x_nlc = x_bcl.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_blk(x_tt, s_tt)
    pcc = _pcc_nlc(y_ref, y_tt)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)

    print(f"TTAdainResBlk1d (learned_sc) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_adain_resblk_1d_with_upsample(device):
    torch.manual_seed(2)
    # Use ``dim >= 48`` so ``conv1``/``conv2`` hit the same TTNN ``conv1d`` path as the encoder
    # (``C==32`` mis-compares vs PyTorch for this op on Wormhole; Kokoro decode blocks use 512+).
    dim_in = dim_out = 48
    style_dim = 24
    b, l = 1, 24
    ref = AdainResBlk1d(dim_in, dim_out, style_dim=style_dim, upsample="nearest")
    ref.eval()
    params = preprocess_tt_adain_resblk_1d(ref, device)
    tt_blk = TTAdainResBlk1d(device, params)

    x_bcl = torch.randn(b, dim_in, l)
    s = torch.randn(b, style_dim)
    with torch.no_grad():
        y_ref = ref(x_bcl, s)

    x_nlc = x_bcl.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_blk(x_tt, s_tt)
    pcc = _pcc_nlc(y_ref, y_tt)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)

    print(f"TTAdainResBlk1d (upsample+pool) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_adain_resblk_1d_decoder_encode_config(device):
    """``Decoder.encode`` configuration: ``AdainResBlk1d(514, 1024, 128)``.

    Matches the exact ``dim_in=512+2``, ``dim_out=1024``, ``style_dim=128`` used by
    :class:`~models.experimental.kokoro.reference.istftnet.Decoder` to encode the
    concatenated ``[asr, F0, N]`` feature before the decode chain.
    """
    torch.manual_seed(3)
    # Exact Decoder.encode dimensions: dim_in = hidden_dim + 2 = 514, dim_out = 1024
    dim_in, dim_out, style_dim = 514, 1024, 128
    b, l = 1, 5
    ref = AdainResBlk1d(dim_in, dim_out, style_dim=style_dim, upsample="none")
    ref.eval()
    params = preprocess_tt_adain_resblk_1d(ref, device, weights_dtype=ttnn.float32, conv_weights_dtype=ttnn.float32)
    tt_blk = TTAdainResBlk1d(device, params)

    x_bcl = torch.randn(b, dim_in, l)
    s = torch.randn(b, style_dim)
    with torch.no_grad():
        y_ref = ref(x_bcl, s)

    x_nlc = x_bcl.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_blk(x_tt, s_tt)
    pcc = _pcc_nlc(y_ref, y_tt)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)

    print(f"TTAdainResBlk1d (Decoder.encode: 514→1024, style=128) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_adain_resblk_1d_decoder_decode_block_config(device):
    """Decode block configuration: ``AdainResBlk1d(1090, 1024, 128)``.

    Matches blocks 0-2 in :class:`~models.experimental.kokoro.reference.istftnet.Decoder`
    which receive the concatenated ``[x, asr_res, F0, N]`` feature (1024+64+1+1 = 1090 ch).
    """
    torch.manual_seed(4)
    # Decoder.decode blocks 0-2: dim_in = 1024 + 64 + 1 + 1 = 1090, dim_out = 1024
    dim_in, dim_out, style_dim = 1090, 1024, 128
    b, l = 1, 5
    ref = AdainResBlk1d(dim_in, dim_out, style_dim=style_dim, upsample="none")
    ref.eval()
    params = preprocess_tt_adain_resblk_1d(ref, device, weights_dtype=ttnn.float32, conv_weights_dtype=ttnn.float32)
    tt_blk = TTAdainResBlk1d(device, params)

    x_bcl = torch.randn(b, dim_in, l)
    s = torch.randn(b, style_dim)
    with torch.no_grad():
        y_ref = ref(x_bcl, s)

    x_nlc = x_bcl.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_blk(x_tt, s_tt)
    pcc = _pcc_nlc(y_ref, y_tt)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)

    print(f"TTAdainResBlk1d (Decoder.decode block: 1090→1024, style=128) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"
