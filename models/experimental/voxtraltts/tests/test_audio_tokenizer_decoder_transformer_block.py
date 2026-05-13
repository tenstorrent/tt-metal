# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.voxtraltts.reference.audio_tokenizer_ops import (
    audio_tokenizer_sliding_window_attention_bias,
    decoder_transformer_block_reference_bf16,
)
from models.experimental.voxtraltts.reference.voxtral_config import load_voxtral_config
from models.experimental.voxtraltts.tests.common import resolve_voxtral_model_name_or_skip
from models.experimental.voxtraltts.tt.audio_tokenizer.model import extract_audio_tokenizer_state_dict
from models.experimental.voxtraltts.tt.audio_tokenizer.transformer import VoxtralTTAudioTokenizerDecoderTransformerBlock
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
@pytest.mark.parametrize("decoder_block_index", [1, 3, 5, 7])
@pytest.mark.parametrize("layer_index", [0, 1])
@pytest.mark.parametrize("time_len", [16, 32])
def test_audio_tokenizer_decoder_transformer_layer_pcc(device, reset_seeds, time_len, layer_index, decoder_block_index):
    """Each odd ``decoder_blocks.{1,3,5,7}`` transformer layer vs ``audio_tokenizer_ops`` golden."""
    model_name = resolve_voxtral_model_name_or_skip()
    try:
        full = _load_safetensors_state_dict(model_name)
    except Exception as exc:
        pytest.skip(f"No checkpoint available: {exc}")

    cfg = load_voxtral_config(model_name).audio_tokenizer_args
    sd = extract_audio_tokenizer_state_dict(full)
    try:
        tt_block = VoxtralTTAudioTokenizerDecoderTransformerBlock(
            device,
            state_dict=sd,
            tokenizer_cfg=cfg,
            block_index=decoder_block_index,
            layer_index=layer_index,
        )
    except KeyError as exc:
        pytest.skip(f"Missing decoder_blocks.{decoder_block_index} transformer weights: {exc}")

    x = torch.randn(1, time_len, cfg.dim, dtype=torch.bfloat16)
    ref = decoder_transformer_block_reference_bf16(x, sd, cfg, block_index=decoder_block_index, layer_index=layer_index)

    x_tt = _btd_to_tt_b1td(device, x)
    mask_tt = ttnn.from_torch(
        audio_tokenizer_sliding_window_attention_bias(cfg.n_heads, time_len, cfg.attn_sliding_window_size).to(
            torch.bfloat16
        ),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    y_tt = tt_block(x_tt, attn_mask=mask_tt)
    ttnn.deallocate(mask_tt)

    tt_out = ttnn.to_torch(y_tt).squeeze(1).float()
    passing, msg = comp_pcc(ref.float(), tt_out, pcc=0.99)
    assert passing, f"decoder_blocks.{decoder_block_index}.layers.{layer_index} PCC failed: {msg}"
