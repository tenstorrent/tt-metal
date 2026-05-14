# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC tests for :class:`~models.experimental.kokoro.tt.tt_text_encoder.TTTextEncoder`."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.modules import TextEncoder
from models.experimental.kokoro.tt import TTTextEncoder, preprocess_tt_text_encoder


def _make_reference_encoder(*, channels: int = 512, kernel_size: int = 5, depth: int = 3, n_symbols: int = 178):
    enc = TextEncoder(channels, kernel_size, depth, n_symbols)
    enc.eval()
    return enc


def test_tt_text_encoder_full_sequence_matches_torch(device):
    """Full-length batch: no padding mask, PCC vs reference."""
    torch_enc = _make_reference_encoder()
    params = preprocess_tt_text_encoder(torch_enc, device)
    tt_enc = TTTextEncoder(device, params)

    torch.manual_seed(1)
    B, T = 2, 48
    input_ids = torch.randint(0, torch_enc.embedding.num_embeddings, (B, T), dtype=torch.long)
    input_lengths = torch.tensor([T, T], dtype=torch.long)
    text_mask = torch.zeros((B, T), dtype=torch.bool)

    with torch.no_grad():
        ref = torch_enc(input_ids, input_lengths, text_mask)

    tt_out = tt_enc(input_ids, input_lengths, text_mask)
    tt_torch = ttnn.to_torch(tt_out).float()
    ttnn.deallocate(tt_out)

    assert ref.shape == tt_torch.shape
    _, pcc = comp_pcc(ref, tt_torch, pcc=0.0)
    print(f"TTTextEncoder full-seq PCC: {pcc:.6f}")
    assert pcc >= 0.99


def test_tt_text_encoder_variable_length_packed_lstm(device):
    """Shorter valid lengths exercise packed LSTM path (per-row ``input_lengths``)."""
    torch_enc = _make_reference_encoder()
    params = preprocess_tt_text_encoder(torch_enc, device)
    tt_enc = TTTextEncoder(device, params)

    torch.manual_seed(2)
    B, T = 3, 40
    input_ids = torch.randint(0, 120, (B, T), dtype=torch.long)
    lengths = torch.tensor([38, 12, 25], dtype=torch.long)
    text_mask = torch.zeros((B, T), dtype=torch.bool)
    for b, le in enumerate(lengths.tolist()):
        text_mask[b, le:] = True

    with torch.no_grad():
        ref = torch_enc(input_ids, lengths, text_mask)

    tt_out = tt_enc(input_ids, lengths, text_mask)
    tt_torch = ttnn.to_torch(tt_out).float()
    ttnn.deallocate(tt_out)

    _, pcc = comp_pcc(ref, tt_torch, pcc=0.0)
    print(f"TTTextEncoder variable-length PCC: {pcc:.6f}")
    assert pcc >= 0.99


def test_tt_text_encoder_conv_block_is_callable(device):
    """Smoke: submodule class runs one stage without raising."""
    torch_enc = _make_reference_encoder(depth=1)
    params = preprocess_tt_text_encoder(torch_enc, device)
    from models.experimental.kokoro.tt.tt_text_encoder import TTTextEncoderConvLNBlock

    blk = TTTextEncoderConvLNBlock(
        device=device,
        params=params.blocks[0],
        ln_eps=params.ln_eps,
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        ),
    )
    torch.manual_seed(3)
    x = torch.randn(1, 16, 512)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    m = torch.zeros(1, 16, dtype=torch.bool)
    mk = ttnn.from_torch(
        (~m).to(torch.float32).unsqueeze(-1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    y = blk.forward(x_tt, mk)
    assert y.shape == x_tt.shape
    ttnn.deallocate(y)
    ttnn.deallocate(mk)
    ttnn.deallocate(x_tt)
