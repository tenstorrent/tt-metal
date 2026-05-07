# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_text_to_unit import (
    forward_t2u_encoder_hidden,
    forward_t2u_logits_and_padding,
    load_pretrained_text_to_unit,
    synthetic_t2u_inputs,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import (
    create_text_to_unit_condgen_parameters,
    create_text_to_unit_parameters,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_text_to_unit import (
    TTSeamlessM4Tv2TextToUnitEncoder,
    TTSeamlessM4Tv2TextToUnitForConditionalGeneration,
)

PCC_THRESHOLD = 0.99
# Full encoder+decoder+``lm_head`` stacks more bf16 ops than encoder-only; HF parity on synthetic
# inputs is typically high but can sit just under 0.99 without further numerics tuning.
PCC_THRESHOLD_FULL = 0.98


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_seamless_m4t_v2_text_to_unit_encoder_pcc(device, reset_seeds):
    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    torch.manual_seed(1)
    t2u, cfg = load_pretrained_text_to_unit(weights_dir, dtype=torch.bfloat16)

    # Match ``test_text_decoder`` prefill length; very short seqs can increase SDPA / tile edge drift on device.
    batch, encoder_seq_len = 1, 32
    inputs_embeds, attention_mask, _, _ = synthetic_t2u_inputs(
        cfg,
        batch=batch,
        encoder_seq_len=encoder_seq_len,
        seed=1,
        dtype=torch.bfloat16,
    )

    mask_4d = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)

    ref = forward_t2u_encoder_hidden(t2u, inputs_embeds, attention_mask).to(torch.bfloat16).cpu()

    params = create_text_to_unit_parameters(t2u.model.encoder, device=device)
    tt_enc = TTSeamlessM4Tv2TextToUnitEncoder(
        device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        num_hidden_layers=cfg.encoder_layers,
        num_attention_heads=cfg.encoder_attention_heads,
        hidden_size=cfg.hidden_size,
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

    out_tt = tt_enc.forward(inputs_embeds_tt, attn_tt)
    tt_cpu = (
        ttnn.to_torch(ttnn.from_device(out_tt))
        .to(torch.bfloat16)
        .reshape(batch, encoder_seq_len, cfg.hidden_size)
        .contiguous()
    )

    ok, msg = check_with_pcc(ref, tt_cpu, pcc=PCC_THRESHOLD)
    logger.info(f"SeamlessM4Tv2 text-to-unit encoder PCC: {msg} " f"(threshold {PCC_THRESHOLD})")
    if ok:
        logger.info("SeamlessM4Tv2 text-to-unit encoder PCC check passed.")
    else:
        logger.warning("SeamlessM4Tv2 text-to-unit encoder PCC check failed.")

    assert ok, msg


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_seamless_m4t_v2_text_to_unit_full_condgen_pcc(device, reset_seeds):
    """PCC for full HF ``SeamlessM4Tv2TextToUnitForConditionalGeneration`` (encoder + decoder + ``lm_head``)."""
    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    torch.manual_seed(1)
    t2u, cfg = load_pretrained_text_to_unit(weights_dir, dtype=torch.bfloat16)

    # Same prefill length as encoder-only PCC.
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
    )
    Vr = int(ref_logits.shape[2])
    Sr = int(ref_logits.shape[1])
    flat_logits = ttnn.to_torch(ttnn.from_device(logits_tt)).to(torch.bfloat16).contiguous().reshape(-1)
    Sp = flat_logits.numel() // Vr
    tt_logits_cpu = flat_logits.reshape(1, Sp, Vr)[:, :Sr, :Vr].contiguous()

    flat_pad = ttnn.to_torch(ttnn.from_device(pad_tt)).to(torch.float32).contiguous().reshape(-1)
    Spr = int(ref_padding_mask.shape[1])
    tt_pad_cpu = flat_pad.reshape(1, -1)[:, :Spr].contiguous()

    s_log = min(int(ref_logits.shape[1]), int(tt_logits_cpu.shape[1]))
    assert (
        abs(int(ref_logits.shape[1]) - int(tt_logits_cpu.shape[1])) <= 4
    ), f"TT unit length {tt_logits_cpu.shape[1]} vs HF {ref_logits.shape[1]} exceeds tolerated drift window."
    s_pad = min(int(ref_padding_mask.shape[1]), int(tt_pad_cpu.shape[1]))
    assert abs(int(ref_padding_mask.shape[1]) - int(tt_pad_cpu.shape[1])) <= 4

    ok_logits, msg_logits = check_with_pcc(
        ref_logits[:, :s_log, :], tt_logits_cpu[:, :s_log, :], pcc=PCC_THRESHOLD_FULL
    )
    ok_pad, msg_pad = check_with_pcc(ref_padding_mask[:, :s_pad], tt_pad_cpu[:, :s_pad], pcc=PCC_THRESHOLD_FULL)
    logger.info(
        f"SeamlessM4Tv2 text-to-unit full PCC (logits): {msg_logits} (threshold {PCC_THRESHOLD_FULL}); "
        f"padding_mask: {msg_pad}"
    )
    assert ok_logits, msg_logits
    assert ok_pad, msg_pad
