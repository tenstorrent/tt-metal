# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import create_position_ids_from_input_ids

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_seamless_m4t_v2_model import (
    forward_text_modality_logits,
    load_pretrained_seamless_m4t_v2_model,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_seamless_m4t_v2_model_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_seamless_m4t_v2_model import TTSeamlessM4Tv2Model

PCC_THRESHOLD = 0.99


def _create_bidirectional_additive_mask(attention_mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    bsz, seq = attention_mask.shape
    mask = 1.0 - attention_mask[:, None, None, :].to(dtype=dtype)
    mask = mask.expand(bsz, 1, seq, seq)
    mask = mask * torch.finfo(dtype).min
    return mask


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_seamless_m4t_v2_model_text_forward_pcc(device, reset_seeds):
    """PCC for full HF ``SeamlessM4Tv2Model`` text path vs ``TTSeamlessM4Tv2Model.forward_text``."""
    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    torch.manual_seed(1)
    model, cfg = load_pretrained_seamless_m4t_v2_model(weights_dir, dtype=torch.bfloat16)
    t2u_cfg = model.t2u_model.config
    dev = next(model.parameters()).device

    batch, enc_seq, dec_seq = 1, 32, 32
    input_ids = torch.randint(1, min(cfg.vocab_size - 1, 2**31 - 1), (batch, enc_seq), dtype=torch.int64, device=dev)
    enc_attn = torch.ones(batch, enc_seq, dtype=torch.long, device=dev)
    decoder_input_ids = torch.randint(
        1, min(cfg.vocab_size - 1, 2**31 - 1), (batch, dec_seq), dtype=torch.int64, device=dev
    )
    dec_attn = torch.ones(batch, dec_seq, dtype=torch.long, device=dev)

    enc_embeds = model.text_encoder.embed_tokens(input_ids)
    enc_bidir = _create_bidirectional_additive_mask(enc_attn, dtype=enc_embeds.dtype)
    enc_pos = create_position_ids_from_input_ids(input_ids, cfg.pad_token_id, past_key_values_length=0)

    dec_embeds = model.text_decoder.embed_tokens(decoder_input_ids)
    dec_causal = _prepare_4d_causal_attention_mask(dec_attn, (batch, dec_seq), dec_embeds, past_key_values_length=0)
    dec_cross = _prepare_4d_attention_mask(enc_attn, dec_embeds.dtype, tgt_len=dec_seq)
    dec_pos = create_position_ids_from_input_ids(decoder_input_ids, cfg.pad_token_id, past_key_values_length=0)

    ref_logits = forward_text_modality_logits(
        model,
        input_ids=input_ids,
        attention_mask=enc_attn,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=dec_attn,
    )
    ref_logits = ref_logits.to(torch.bfloat16).cpu()

    params = create_seamless_m4t_v2_model_parameters(model, device=device)
    tt_model = TTSeamlessM4Tv2Model(
        device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        encoder_layers=cfg.encoder_layers,
        encoder_attention_heads=cfg.encoder_attention_heads,
        decoder_layers=cfg.decoder_layers,
        decoder_attention_heads=cfg.decoder_attention_heads,
        hidden_size=cfg.hidden_size,
        feature_projection_input_dim=cfg.feature_projection_input_dim,
        speech_encoder_attention_heads=cfg.speech_encoder_attention_heads,
        speech_encoder_intermediate_size=cfg.speech_encoder_intermediate_size,
        speech_encoder_layers=cfg.speech_encoder_layers,
        speech_encoder_chunk_size=cfg.speech_encoder_chunk_size,
        speech_encoder_left_chunk_num=cfg.speech_encoder_left_chunk_num,
        t2u_layer_norm_eps=t2u_cfg.layer_norm_eps,
        t2u_encoder_layers=t2u_cfg.encoder_layers,
        t2u_encoder_attention_heads=t2u_cfg.encoder_attention_heads,
        t2u_decoder_layers=t2u_cfg.decoder_layers,
        t2u_decoder_attention_heads=t2u_cfg.decoder_attention_heads,
        t2u_pad_token_id=t2u_cfg.pad_token_id,
        variance_predictor_embed_dim=t2u_cfg.variance_predictor_embed_dim,
        variance_predictor_hidden_dim=t2u_cfg.variance_predictor_hidden_dim,
        variance_predictor_kernel_size=t2u_cfg.variance_predictor_kernel_size,
        vocoder_config=cfg,
    )

    enc_ids_tt = ttnn.from_torch(
        input_ids.to(torch.int32).cpu(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    enc_pos_tt = ttnn.from_torch(
        enc_pos.to(torch.int32).cpu(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    enc_mask_tt = ttnn.from_torch(
        enc_bidir.to(torch.bfloat16).cpu(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    dec_ids_tt = ttnn.from_torch(
        decoder_input_ids.to(torch.int32).cpu(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    dec_pos_tt = ttnn.from_torch(
        dec_pos.to(torch.int32).cpu(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    dec_causal_tt = ttnn.from_torch(
        dec_causal.to(torch.bfloat16).cpu(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    dec_cross_tt = ttnn.from_torch(
        dec_cross.to(torch.bfloat16).cpu(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logits_tt = tt_model.forward_text(
        enc_ids_tt,
        enc_pos_tt,
        enc_mask_tt,
        dec_ids_tt,
        dec_pos_tt,
        dec_causal_tt,
        dec_cross_tt,
    )

    V = int(ref_logits.shape[2])
    Sd = int(ref_logits.shape[1])
    flat = ttnn.to_torch(ttnn.from_device(logits_tt)).to(torch.bfloat16).contiguous().reshape(-1)
    Sp = flat.numel() // V
    tt_logits = flat.reshape(1, Sp, V)[:, :Sd, :V].contiguous()

    assert tt_logits.shape == ref_logits.shape, f"TT {tuple(tt_logits.shape)} vs HF {tuple(ref_logits.shape)}"

    ok, msg = check_with_pcc(ref_logits, tt_logits, pcc=PCC_THRESHOLD)
    logger.info(f"SeamlessM4Tv2 full model (text path) PCC: {msg} (threshold {PCC_THRESHOLD})")
    assert ok, msg
