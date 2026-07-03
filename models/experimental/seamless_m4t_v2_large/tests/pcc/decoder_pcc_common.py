# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for layer-0 decode/prefill PCC tests (``test_decode.py``, ``test_prefill.py``).

Random hidden states in, layer block out, PCC ≥ 0.99 — same contract as
``models/tt_transformers/tests/test_decoder.py`` and ``test_decoder_prefill.py``.
"""

from __future__ import annotations

import pytest
import torch
import ttnn
from loguru import logger
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import (
    SeamlessM4Tv2Decoder,
    SeamlessM4Tv2DecoderLayer,
)

from models.common.utility_functions import comp_allclose, comp_pcc, nearest_32
from models.experimental.seamless_m4t_v2_large.scripts.download_weights import ensure_seamless_m4t_v2_large_weights
from models.experimental.seamless_m4t_v2_large.tests.pcc.decoder_pcc_fixtures import load_hf_model_and_processor
from models.experimental.seamless_m4t_v2_large.tt.common import (
    build_causal_with_padding_4d,
    build_cross_attn_mask_4d,
    to_torch_replicated_first_shard,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import (
    from_torch_bfloat16_tile,
    from_torch_uint32_rm,
    get_tp,
)
from models.experimental.seamless_m4t_v2_large.tt.model_preprocessing import create_text_decoder_parameters
from models.experimental.seamless_m4t_v2_large.tt.tt_text_decoder import (
    TTSeamlessM4Tv2Decoder,
    init_text_decoder_kv_cache,
)

PCC_REQUIRED = 0.99
LAYER_IDX = 0
PCC_BATCH_SIZE = 1
DECODE_GENERATION_LENGTH = 10
CROSS_ENC_SEQ = 128
PREFILL_LAYER_SEQ_LENGTHS = (32, 64, 128, 256, 512, 1024, 2048, 4096)


def weights_dir_or_skip() -> str:
    try:
        return ensure_seamless_m4t_v2_large_weights()
    except ImportError as e:
        pytest.skip(str(e))
    except Exception as e:
        pytest.skip(f"Could not prepare seamless-m4t-v2-large weights: {e}")


def load_hf_model_for_layer_pcc(weights_dir: str):
    hf_model, _, _ = load_hf_model_and_processor(weights_dir, dtype=torch.bfloat16)
    return hf_model


def _max_seq_len(dec_seq_len: int, *, decode_steps: int = 0) -> int:
    return max(64, dec_seq_len + decode_steps + 8)


def _build_layer_tt_decoder(
    mesh_device,
    hf_decoder: SeamlessM4Tv2Decoder,
    cfg,
    *,
    max_seq_len: int,
) -> TTSeamlessM4Tv2Decoder:
    params = create_text_decoder_parameters(
        hf_decoder,
        device=mesh_device,
        num_layers=1,
        include_token_embeddings=False,
        include_embed_positions=False,
    )
    return TTSeamlessM4Tv2Decoder(
        mesh_device,
        params,
        layer_norm_eps=cfg.layer_norm_eps,
        num_hidden_layers=1,
        num_attention_heads=cfg.decoder_attention_heads,
        hidden_size=cfg.hidden_size,
        max_batch_size=PCC_BATCH_SIZE,
        max_seq_len=max_seq_len,
    )


def _hf_causal_mask(batch: int, seq: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    finfo_min = torch.finfo(torch.float32).min
    mask = torch.triu(torch.full((seq, seq), finfo_min, device=device, dtype=torch.float32), diagonal=1)
    return mask.to(dtype).unsqueeze(0).unsqueeze(0).expand(batch, 1, seq, seq)


def _hf_cross_mask(batch: int, tgt: int, src: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(batch, 1, tgt, src, device=device, dtype=dtype)


def _hf_layer_cache() -> EncoderDecoderCache:
    return EncoderDecoderCache(DynamicCache(), DynamicCache())


def _hf_layer_out(out) -> torch.Tensor:
    if isinstance(out, tuple):
        out = out[0]
    return out.to(torch.bfloat16)


def _hf_prefill_forward(
    layer: SeamlessM4Tv2DecoderLayer,
    hidden: torch.Tensor,
    encoder_hidden: torch.Tensor,
    *,
    causal_mask: torch.Tensor,
    cache: EncoderDecoderCache,
) -> torch.Tensor:
    batch, tgt, src = int(hidden.shape[0]), int(hidden.shape[1]), int(encoder_hidden.shape[1])
    with torch.no_grad():
        out = layer(
            hidden_states=hidden,
            attention_mask=causal_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=_hf_cross_mask(batch, tgt, src, hidden.device, hidden.dtype),
            past_key_values=cache,
            use_cache=True,
        )
    return _hf_layer_out(out)


def _hf_decode_forward(
    layer: SeamlessM4Tv2DecoderLayer,
    hidden: torch.Tensor,
    encoder_hidden: torch.Tensor,
    *,
    cache: EncoderDecoderCache,
) -> torch.Tensor:
    batch, tgt, src = int(hidden.shape[0]), int(hidden.shape[1]), int(encoder_hidden.shape[1])
    with torch.no_grad():
        out = layer(
            hidden_states=hidden,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=_hf_cross_mask(batch, tgt, src, hidden.device, hidden.dtype),
            past_key_values=cache,
            use_cache=True,
        )
    out = _hf_layer_out(out)
    return out.unsqueeze(1) if out.dim() == 2 else out


def _dealloc_kv_caches(kv_cache: list[list[ttnn.Tensor]], cross_attn_cache: list[list[ttnn.Tensor]]) -> None:
    for layer in kv_cache:
        ttnn.deallocate(layer[0])
        ttnn.deallocate(layer[1])
    for layer in cross_attn_cache:
        ttnn.deallocate(layer[0])
        ttnn.deallocate(layer[1])


def run_prefill_layer_pcc(mesh_device, hf_model, *, seq_len: int) -> None:
    """One-shot prefill: random ``[1, seq_len, H]``, KV fill 0 … seq_len-1."""
    cfg = hf_model.config
    hidden_size = int(cfg.hidden_size)
    padded_seq = nearest_32(seq_len)
    padded_enc = nearest_32(CROSS_ENC_SEQ)

    hf_layer = hf_model.text_decoder.layers[LAYER_IDX]
    p0 = next(hf_layer.parameters())
    torch.manual_seed(seq_len)

    hidden_padded = torch.zeros(PCC_BATCH_SIZE, padded_seq, hidden_size, device=p0.device, dtype=p0.dtype)
    hidden_padded[:, :seq_len] = (torch.rand(PCC_BATCH_SIZE, seq_len, hidden_size) * 2 - 1).to(
        device=p0.device, dtype=p0.dtype
    )

    enc_hidden = torch.randn(PCC_BATCH_SIZE, CROSS_ENC_SEQ, hidden_size, dtype=torch.bfloat16, device=p0.device)
    enc_mask = torch.ones(PCC_BATCH_SIZE, CROSS_ENC_SEQ, dtype=torch.long, device=p0.device)

    ref_out = _hf_prefill_forward(
        hf_layer,
        hidden_padded,
        enc_hidden,
        causal_mask=_hf_causal_mask(PCC_BATCH_SIZE, padded_seq, p0.device, p0.dtype),
        cache=_hf_layer_cache(),
    )[:, :seq_len, :].contiguous()

    tt_dec = _build_layer_tt_decoder(mesh_device, hf_model.text_decoder, cfg, max_seq_len=_max_seq_len(seq_len))
    tp = get_tp(mesh_device)
    kv_cache, cross_attn_cache = init_text_decoder_kv_cache(
        mesh_device,
        num_hidden_layers=1,
        num_attention_heads=int(cfg.decoder_attention_heads),
        hidden_size=hidden_size,
        max_batch_size=PCC_BATCH_SIZE,
        max_seq_len=_max_seq_len(seq_len),
        encoder_seq_len=padded_enc,
        tp=tp,
    )

    hidden_tt = from_torch_bfloat16_tile(mesh_device, hidden_padded)
    enc_tt = from_torch_bfloat16_tile(mesh_device, enc_hidden.cpu())
    enc_mask_tt = from_torch_uint32_rm(mesh_device, enc_mask.cpu())
    causal_tt = build_causal_with_padding_4d(None, PCC_BATCH_SIZE, padded_seq, mesh_device)
    cross_tt = build_cross_attn_mask_4d(enc_mask_tt, tgt_seq=padded_seq, device=mesh_device)

    tt_out = tt_dec.forward_prefill_layer_hidden(
        hidden_tt,
        enc_tt,
        causal_tt,
        cross_tt,
        kv_cache=kv_cache,
        cross_attn_cache=cross_attn_cache,
        kv_cache_fill_len=seq_len,
    )
    tt_cpu = (
        to_torch_replicated_first_shard(tt_out)
        .to(torch.bfloat16)
        .reshape(PCC_BATCH_SIZE, padded_seq, hidden_size)[:, :seq_len, :]
        .contiguous()
    )

    for t in (hidden_tt, tt_out, enc_tt, enc_mask_tt, causal_tt, cross_tt):
        ttnn.deallocate(t)
    _dealloc_kv_caches(kv_cache, cross_attn_cache)

    passing, pcc_val = comp_pcc(ref_out, tt_cpu, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_cpu))
    assert passing, f"prefill seq_len={seq_len} PCC {pcc_val} < {PCC_REQUIRED}"


def run_decode_layer_pcc(mesh_device, hf_model) -> None:
    """Decode loop: random hidden per step, KV positions 0 … DECODE_GENERATION_LENGTH-1."""
    decode_steps = DECODE_GENERATION_LENGTH
    cfg = hf_model.config
    hidden_size = int(cfg.hidden_size)
    padded_enc = nearest_32(CROSS_ENC_SEQ)

    hf_layer = hf_model.text_decoder.layers[LAYER_IDX]
    p0 = next(hf_layer.parameters())
    torch.manual_seed(0)
    enc_hidden = torch.randn(PCC_BATCH_SIZE, CROSS_ENC_SEQ, hidden_size, dtype=torch.bfloat16, device=p0.device)
    enc_mask = torch.ones(PCC_BATCH_SIZE, CROSS_ENC_SEQ, dtype=torch.long, device=p0.device)

    tt_dec = _build_layer_tt_decoder(
        mesh_device, hf_model.text_decoder, cfg, max_seq_len=_max_seq_len(0, decode_steps=decode_steps)
    )
    tp = get_tp(mesh_device)
    kv_cache, cross_attn_cache = init_text_decoder_kv_cache(
        mesh_device,
        num_hidden_layers=1,
        num_attention_heads=int(cfg.decoder_attention_heads),
        hidden_size=hidden_size,
        max_batch_size=PCC_BATCH_SIZE,
        max_seq_len=_max_seq_len(0, decode_steps=decode_steps),
        encoder_seq_len=padded_enc,
        tp=tp,
    )

    enc_tt = from_torch_bfloat16_tile(mesh_device, enc_hidden.cpu())
    enc_mask_tt = from_torch_uint32_rm(mesh_device, enc_mask.cpu())
    cross_tt = build_cross_attn_mask_4d(enc_mask_tt, tgt_seq=1, device=mesh_device)
    cache = _hf_layer_cache()
    cross_valid = False

    for step in range(decode_steps):
        hidden = ((torch.rand(PCC_BATCH_SIZE, 1, hidden_size, dtype=torch.bfloat16) * 2) - 1).to(
            device=p0.device, dtype=p0.dtype
        )
        ref_out = _hf_decode_forward(hf_layer, hidden, enc_hidden, cache=cache)

        hidden_tt = from_torch_bfloat16_tile(mesh_device, hidden)
        tt_out = tt_dec.forward_decode_layer_hidden(
            hidden_tt,
            enc_tt,
            cross_tt,
            kv_cache=kv_cache,
            cross_attn_cache=cross_attn_cache,
            current_decode_pos=tt_dec.borrow_current_decode_pos_tensor(step, batch_size=PCC_BATCH_SIZE),
            cache_seq_len=step + 1,
            cross_attn_cache_valid=cross_valid,
        )
        cross_valid = True
        tt_cpu = (
            to_torch_replicated_first_shard(tt_out)
            .to(torch.bfloat16)
            .reshape(PCC_BATCH_SIZE, 1, hidden_size)
            .contiguous()
        )
        ttnn.deallocate(hidden_tt)
        ttnn.deallocate(tt_out)

        passing, pcc_val = comp_pcc(ref_out, tt_cpu, PCC_REQUIRED)
        logger.info(comp_allclose(ref_out, tt_cpu))
        assert passing, f"decode step={step} pos={step} PCC {pcc_val} < {PCC_REQUIRED}"

    ttnn.deallocate(enc_tt)
    ttnn.deallocate(enc_mask_tt)
    ttnn.deallocate(cross_tt)
    _dealloc_kv_caches(kv_cache, cross_attn_cache)
