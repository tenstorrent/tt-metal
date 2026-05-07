# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_text_to_unit import (
    forward_t2u_logits_and_padding,
    hf_discrete_duration_counts_batch1,
    load_pretrained_text_to_unit,
    synthetic_t2u_inputs,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_text_to_unit_condgen_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_text_to_unit import (
    TTSeamlessM4Tv2TextToUnitForConditionalGeneration,
)

PCC_THRESHOLD = 0.99


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_seamless_m4t_v2_text_to_unit(device, reset_seeds):
    """
    PCC vs Hugging Face ``SeamlessM4Tv2TextToUnitForConditionalGeneration`` (encoder + decoder + ``lm_head``).

    Uses HF discrete durations as ``reference_discrete_durations`` so TT matches HF unit length while the
    TT duration stack converges.
    """
    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    torch.manual_seed(1)
    t2u, cfg = load_pretrained_text_to_unit(weights_dir, dtype=torch.bfloat16)

    batch, encoder_seq_len = 1, 32
    inputs_embeds, attention_mask, char_input_ids, char_count_per_id = synthetic_t2u_inputs(
        cfg,
        batch=batch,
        encoder_seq_len=encoder_seq_len,
        seed=1,
        dtype=torch.bfloat16,
    )
    dev = next(t2u.parameters()).device
    char_count_per_id = char_count_per_id.to(dev)

    mask_4d = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

    ref_logits, ref_padding_mask = forward_t2u_logits_and_padding(
        t2u,
        inputs_embeds,
        attention_mask,
        char_input_ids,
        char_count_per_id,
    )
    ref_logits = ref_logits.to(torch.bfloat16).cpu()
    ref_padding_mask = ref_padding_mask.to(torch.float32).cpu()

    ref_durs = hf_discrete_duration_counts_batch1(
        t2u,
        inputs_embeds.to(dev),
        attention_mask.to(dev),
        char_input_ids.to(dev),
        char_count_per_id,
    )
    assert sum(ref_durs) == int(ref_logits.shape[1]), "HF duration sum must match HF logits sequence length."

    params = create_text_to_unit_condgen_parameters(t2u, device=device)
    tt_full = TTSeamlessM4Tv2TextToUnitForConditionalGeneration(
        device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        encoder_layers=cfg.encoder_layers,
        encoder_attention_heads=cfg.encoder_attention_heads,
        decoder_layers=cfg.decoder_layers,
        decoder_attention_heads=cfg.decoder_attention_heads,
        hidden_size=cfg.hidden_size,
        pad_token_id=cfg.pad_token_id,
        variance_predictor_embed_dim=cfg.variance_predictor_embed_dim,
        variance_predictor_hidden_dim=cfg.variance_predictor_hidden_dim,
        variance_predictor_kernel_size=cfg.variance_predictor_kernel_size,
    )

    inputs_embeds_tt = ttnn.from_torch(
        inputs_embeds,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    attn_tt = ttnn.from_torch(
        mask_4d.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    char_ids_tt = ttnn.from_torch(
        char_input_ids.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    logits_tt, pad_tt = tt_full.forward(
        inputs_embeds_tt,
        attn_tt,
        char_ids_tt,
        char_count_per_id.squeeze(0).tolist(),
        reference_discrete_durations=ref_durs,
    )
    Vr = int(ref_logits.shape[2])
    Sr = int(ref_logits.shape[1])
    flat_logits = ttnn.to_torch(ttnn.from_device(logits_tt)).to(torch.bfloat16).contiguous().reshape(-1)
    Sp = flat_logits.numel() // Vr
    tt_logits_cpu = flat_logits.reshape(1, Sp, Vr)[:, :Sr, :Vr].contiguous()

    flat_pad = ttnn.to_torch(ttnn.from_device(pad_tt)).to(torch.float32).contiguous().reshape(-1)
    Spr = int(ref_padding_mask.shape[1])
    tt_pad_cpu = flat_pad.reshape(1, -1)[:, :Spr].contiguous()

    assert (
        tt_logits_cpu.shape == ref_logits.shape
    ), f"TT logits {tuple(tt_logits_cpu.shape)} vs HF {tuple(ref_logits.shape)} — unit length must match."
    assert (
        tt_pad_cpu.shape == ref_padding_mask.shape
    ), f"TT padding_mask {tuple(tt_pad_cpu.shape)} vs HF {tuple(ref_padding_mask.shape)}."

    ok_logits, msg_logits = check_with_pcc(ref_logits, tt_logits_cpu, pcc=PCC_THRESHOLD)
    ok_pad, msg_pad = check_with_pcc(ref_padding_mask, tt_pad_cpu, pcc=PCC_THRESHOLD)
    logger.info(f"SeamlessM4Tv2 text-to-unit PCC logits={msg_logits} padding={msg_pad} (threshold {PCC_THRESHOLD})")
    assert ok_logits, msg_logits
    assert ok_pad, msg_pad
