# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import output_proj_mel_ncl_reference_bf16
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.audio_tokenizer.model import (
    VoxtralTTAudioTokenizer,
    extract_audio_tokenizer_state_dict,
)
from models.experimental.voxtraltts.tt.voxtral_tt_args import _load_safetensors_state_dict


def _btd_to_tt_b1td(device, x_btd: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        x_btd.unsqueeze(1).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@torch.no_grad()
@pytest.mark.parametrize("time_len", [16, 32])
def test_audio_tokenizer_output_proj_pcc(device, reset_seeds, time_len):
    """``output_proj`` causal conv vs ``output_proj_mel_ncl_reference_bf16`` (checkpoint weights)."""
    model_name = resolve_voxtral_model_name_or_skip()
    try:
        full = _load_safetensors_state_dict(model_name)
    except Exception as exc:
        pytest.skip(f"No checkpoint available: {exc}")

    cfg = load_voxtral_config(model_name).audio_tokenizer_args
    sd = extract_audio_tokenizer_state_dict(full)
    try:
        tok = VoxtralTTAudioTokenizer(device, state_dict=sd, tokenizer_cfg=cfg)
    except Exception as exc:
        pytest.skip(f"Unable to build VoxtralTTAudioTokenizer: {exc}")
    if tok.output_proj_conv is None:
        pytest.skip("output_proj weights not in audio_tokenizer checkpoint.")

    b, d = 1, cfg.dim
    hidden = torch.randn(b, time_len, d, dtype=torch.bfloat16)
    ref_ncl = output_proj_mel_ncl_reference_bf16(hidden, sd)
    ref_btc = ref_ncl.permute(0, 2, 1).contiguous()

    h_tt = _btd_to_tt_b1td(device, hidden)
    y_tt = tok.output_proj_forward(h_tt)
    ttnn.deallocate(h_tt)

    # ``[B,1,T_out,C_mel]`` → ``[B,T_out,C_mel]`` (same layout as ``ref_btc``; do not swap T/C).
    tt_btc = ttnn.to_torch(y_tt).squeeze(1).contiguous().float()
    assert ref_btc.shape == tt_btc.shape, f"shape ref={ref_btc.shape} tt={tt_btc.shape}"
    passing, msg = comp_pcc(ref_btc.float(), tt_btc, pcc=0.99)
    assert passing, f"output_proj PCC failed: {msg}"
