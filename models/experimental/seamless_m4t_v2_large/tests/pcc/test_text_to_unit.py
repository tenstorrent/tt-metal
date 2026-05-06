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
    load_pretrained_text_to_unit,
    synthetic_t2u_inputs,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_text_to_unit_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_text_to_unit import TTSeamlessM4Tv2TextToUnitEncoder

PCC_THRESHOLD = 0.99


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
