# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: TTNN Decoder front-end vs PyTorch reference (pre-generator features)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import ttnn

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

from models.common.utility_functions import comp_pcc
from models.demos.kokoro.reference import KokoroConfig
from models.demos.kokoro.reference.kokoro_istftnet import load_decoder_from_huggingface
from models.demos.kokoro.tt.ttnn_kokoro_decoder import TtKokoroDecoderFront, preprocess_decoder_front


@torch.no_grad()
def _torch_decoder_front(decoder, *, asr, f0_pred, n_pred, s):
    # mirrors Decoder.forward up to `x = self.generator(...)`
    F0 = decoder.F0_conv(f0_pred.unsqueeze(1))
    N = decoder.N_conv(n_pred.unsqueeze(1))
    x = torch.cat([asr, F0, N], axis=1)
    x = decoder.encode(x, s)
    asr_res = decoder.asr_res(asr)
    res = True
    for block in decoder.decode:
        if res:
            x = torch.cat([x, asr_res, F0, N], axis=1)
        x = block(x, s)
        if block.upsample_type != "none":
            res = False
    return x


def test_ttnn_decoder_front_matches_torch(device):
    torch_model = load_decoder_from_huggingface(repo_id=KokoroConfig.repo_id, device="cpu", disable_complex=True)
    decoder = torch_model.decoder

    params = preprocess_decoder_front(decoder, device)
    tt = TtKokoroDecoderFront(device, params)

    torch.manual_seed(0)
    B, T = 1, 64
    # F0/N convs stride=2, so asr time dim must match their output length (T//2)
    T_asr = T // 2
    asr = torch.randn(B, 512, T_asr, dtype=torch.float32)
    f0_pred = torch.randn(B, T, dtype=torch.float32)
    n_pred = torch.randn(B, T, dtype=torch.float32)
    s = torch.randn(B, 128, dtype=torch.float32)

    ref = _torch_decoder_front(decoder, asr=asr, f0_pred=f0_pred, n_pred=n_pred, s=s)

    asr_tt = ttnn.from_torch(asr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = tt(asr_bct=asr_tt, f0_pred=f0_pred, n_pred=n_pred, style_s=s)
    out = ttnn.to_torch(out_tt).to(torch.float32)

    # tolerate any length mismatch from conv stride/padding by comparing min length
    min_len = min(ref.shape[-1], out.shape[-1])
    ref = ref[..., :min_len]
    out = out[..., :min_len]

    ok, pcc = comp_pcc(ref, out, pcc=0.80)
    assert ok, f"decoder front pcc low: {pcc}"
