# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.utils_for_testing import check_with_pcc

from models.experimental.seamless_m4t_v2_large.reference.torch_text_encoder import (
    forward_torch_reference,
    load_pretrained_text_encoder,
)
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_text_encoder_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_text_encoder import TTSeamlessM4Tv2Encoder

PCC_THRESHOLD = 0.99


def _create_bidirectional_additive_mask(attention_mask: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """
    Build HF-style additive encoder mask with shape [B, 1, S, S].
    Input mask convention: 1 = keep, 0 = masked.
    """
    bsz, seq = attention_mask.shape
    mask = 1.0 - attention_mask[:, None, None, :].to(dtype=dtype)
    mask = mask.expand(bsz, 1, seq, seq)
    mask = mask * torch.finfo(dtype).min
    return mask


def _create_position_ids_from_input_ids(
    input_ids: torch.Tensor, padding_idx: int, past_key_values_length: int = 0
) -> torch.Tensor:
    """
    Copy of HF helper used by SeamlessM4Tv2 positional embedding.
    """
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_seamless_m4t_v2_text_encoder_pcc(device, reset_seeds):
    try:
        weights_dir = ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")

    torch.manual_seed(0)
    encoder, cfg = load_pretrained_text_encoder(weights_dir, dtype=torch.bfloat16)

    batch, seq = 1, 32
    input_ids = torch.randint(1, min(cfg.vocab_size - 1, 2**31 - 1), (batch, seq), dtype=torch.int64)
    attn_mask = torch.ones(batch, seq, dtype=torch.long)

    with torch.no_grad():
        inputs_embeds = encoder.embed_tokens(input_ids)
        bidir_mask = _create_bidirectional_additive_mask(
            attn_mask,
            dtype=inputs_embeds.dtype,
        )
        position_ids = _create_position_ids_from_input_ids(input_ids, cfg.pad_token_id, past_key_values_length=0)

    ref = forward_torch_reference(encoder, input_ids, attn_mask).to(torch.bfloat16)

    params = create_text_encoder_parameters(encoder, device=device)
    tt_enc = TTSeamlessM4Tv2Encoder(
        device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        num_hidden_layers=cfg.encoder_layers,
        num_attention_heads=cfg.encoder_attention_heads,
        hidden_size=cfg.hidden_size,
    )

    input_ids_tt = ttnn.from_torch(
        input_ids.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    position_ids_tt = ttnn.from_torch(
        position_ids.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    bidir_tt = ttnn.from_torch(
        bidir_mask.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out_tt = tt_enc.forward(input_ids_tt, position_ids_tt, bidir_tt)
    tt_cpu = (
        ttnn.to_torch(ttnn.from_device(out_tt)).to(torch.bfloat16).reshape(batch, seq, cfg.hidden_size).contiguous()
    )

    ok, msg = check_with_pcc(ref, tt_cpu, pcc=PCC_THRESHOLD)
    logger.info(f"SeamlessM4Tv2 text encoder PCC: {msg} (threshold {PCC_THRESHOLD})")
    if ok:
        logger.info("SeamlessM4Tv2 text encoder PCC check passed.")
    else:
        logger.warning("SeamlessM4Tv2 text encoder PCC check failed.")

    assert ok, msg
