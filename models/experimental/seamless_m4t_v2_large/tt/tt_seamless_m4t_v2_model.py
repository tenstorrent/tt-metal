# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2Model`]: text/speech encoders, text decoder, T2U, vocoder, ``lm_head``.

Inference parity with Hugging Face ``SeamlessM4Tv2Model``: ``forward`` and ``generate`` consume
and return ``ttnn.Tensor`` on ``self.device``. Position IDs, additive 4D attention masks, the
greedy ``argmax`` decode and all per-step tensor math run on device. Host-side work is limited to:

* ``generation_config`` dictionary lookups (subword/char tables, lang code IDs — string ops);
* a single scalar readback of the subsampled-encoder length (needed for ``ttnn.slice`` end index);
* a per-step scalar readback of the predicted greedy token (for the EOS early-stop check).

Out of scope: KV cache, attentions/hidden-state outputs, label loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import ttnn
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import format_speech_generation_kwargs

from models.experimental.seamless_m4t_v2_large.tt.tt_code_hifigan import TTSeamlessM4Tv2CodeHifiGan
from models.experimental.seamless_m4t_v2_large.tt.tt_speech_encoder import (
    SpeechEncoderTraceMasks,
    TTSeamlessM4Tv2SpeechEncoder,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_text_decoder import TTSeamlessM4Tv2Decoder
from models.experimental.seamless_m4t_v2_large.tt.tt_text_encoder import TTSeamlessM4Tv2Encoder
from models.experimental.seamless_m4t_v2_large.tt.tt_text_to_unit import (
    T2UTraceHardUpsampleCumsums,
    TTSeamlessM4Tv2TextToUnitForConditionalGeneration,
)

# ``torch.finfo(torch.bfloat16).min`` — the additive-mask "minus infinity" HF uses. Bf16-representable.
NEG_INF = -3.3895313892515355e38

# Tile alignment: TT SDPA must score against tile-aligned key sequences; we pad inputs to ``ceil(seq/32)*32``.
_TILE = 32


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TTSeamlessM4Tv2GreedySearchOutput:
    """``generate(generate_speech=False)`` output."""

    sequences: ttnn.Tensor


@dataclass
class TTSeamlessM4Tv2GenerationOutput:
    """``generate(generate_speech=True, return_intermediate_token_ids=True)`` output."""

    waveform: ttnn.Tensor
    waveform_lengths: ttnn.Tensor
    sequences: ttnn.Tensor
    unit_sequences: ttnn.Tensor


# ---------------------------------------------------------------------------
# TTNN helpers (tensor math runs on device only)
# ---------------------------------------------------------------------------


def _core_grid(device: ttnn.Device) -> ttnn.CoreGrid:
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=grid.x)


def _ensure_tile_bf16_sdpa_mask(x: ttnn.Tensor) -> ttnn.Tensor:
    """SDPA requires a TILE bf16 mask; ``expand``/``add`` paths often yield ROW_MAJOR."""
    if x.get_layout() == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16:
        return x
    out = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(x)
    return out


def _tile_align(seq: int) -> int:
    return ((seq + _TILE - 1) // _TILE) * _TILE


def _tt_position_ids(input_ids: ttnn.Tensor, pad_id: int) -> ttnn.Tensor:
    """HF ``create_position_ids_from_input_ids`` on device — ``cumsum`` of non-pad mask + offset."""
    ids_tile = (
        ttnn.to_layout(input_ids, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if input_ids.get_layout() != ttnn.TILE_LAYOUT
        else input_ids
    )
    mask = ttnn.ne(ids_tile, pad_id)
    if ids_tile is not input_ids:
        ttnn.deallocate(ids_tile)
    mask_i32 = ttnn.typecast(mask, ttnn.int32)
    ttnn.deallocate(mask)
    cumsum = ttnn.cumsum(mask_i32, dim=1, dtype=ttnn.int32)
    pos = ttnn.multiply(cumsum, mask_i32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(cumsum)
    ttnn.deallocate(mask_i32)
    pos = ttnn.add(pos, pad_id, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pos = ttnn.typecast(pos, ttnn.uint32)
    if pos.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        pos = ttnn.to_layout(pos, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return pos


def _tt_seq_position_ids(bsz: int, seq: int, pad_id: int, device: ttnn.Device) -> ttnn.Tensor:
    """HF ``create_position_ids_from_inputs_embeds`` on device — ``[pad+1, pad+2, …, pad+seq]``."""
    pos_1d = ttnn.arange(
        pad_id + 1,
        seq + pad_id + 1,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_2d = ttnn.reshape(pos_1d, [1, seq])
    if bsz <= 1:
        return pos_2d
    pos_out = ttnn.expand(pos_2d, [bsz, seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(pos_2d)
    return pos_out


def _key_padding_additive(mask_2d: ttnn.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    """``[B, S]`` 0/1 → ``[B, S]`` bf16 with ``0`` at real and ``NEG_INF`` at padded positions."""
    # ``ttnn.eq`` returns a bool/u8 mask of pad positions. typecast → bf16 (1.0 pad, 0.0 real),
    # then multiply by NEG_INF → additive mask.
    pad_bool = ttnn.eq(mask_2d, 0)
    pad_bf = ttnn.typecast(pad_bool, ttnn.bfloat16)
    ttnn.deallocate(pad_bool)
    additive = ttnn.multiply(pad_bf, NEG_INF, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(pad_bf)
    return additive


def _build_causal_mask_4d(batch: int, seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """HF ``_prepare_4d_causal_attention_mask`` (causal half only) on device.

    Returns ``[B, 1, S, S]`` bf16 with ``NEG_INF`` strictly above the diagonal, ``0`` on/below.
    """
    full_neg = ttnn.full(
        [seq, seq],
        NEG_INF,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    causal_2d = ttnn.triu(full_neg, diagonal=1)
    ttnn.deallocate(full_neg)
    causal_4d = ttnn.reshape(causal_2d, [1, 1, seq, seq])
    if batch <= 1:
        # ``reshape`` / ``triu`` can return non-TILE / non-bf16 storage; SDPA requires TILE bf16.
        return _ensure_tile_bf16_sdpa_mask(causal_4d)
    expanded = ttnn.expand(causal_4d, [batch, 1, seq, seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(causal_4d)
    return _ensure_tile_bf16_sdpa_mask(expanded)


def _build_causal_with_padding_4d(
    attention_mask_2d: Optional[ttnn.Tensor], batch: int, seq: int, device: ttnn.Device
) -> ttnn.Tensor:
    """HF ``_prepare_4d_causal_attention_mask`` (causal + key padding) on device → ``[B, 1, S, S]`` bf16."""
    causal_4d = _build_causal_mask_4d(batch, seq, device)
    if attention_mask_2d is None:
        # ``_build_causal_mask_4d`` already normalises layout/dtype, but go through the helper for
        # consistency with the cross-/encoder-mask builders below.
        return _ensure_tile_bf16_sdpa_mask(causal_4d)
    pad_add_2d = _key_padding_additive(attention_mask_2d, device=device)
    pad_add_4d = ttnn.reshape(pad_add_2d, [batch, 1, 1, seq])
    pad_add_expanded = ttnn.expand(pad_add_4d, [batch, 1, seq, seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(pad_add_4d)
    combined = ttnn.add(causal_4d, pad_add_expanded, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(causal_4d)
    ttnn.deallocate(pad_add_expanded)
    return _ensure_tile_bf16_sdpa_mask(combined)


def _build_cross_attn_mask_4d(encoder_pad_mask_2d: ttnn.Tensor, *, tgt_seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """HF ``_prepare_4d_attention_mask`` for cross-attn → ``[B, 1, tgt_seq, src_seq]`` bf16."""
    batch = int(encoder_pad_mask_2d.shape[0])
    src_seq = int(encoder_pad_mask_2d.shape[1])
    add_2d = _key_padding_additive(encoder_pad_mask_2d, device=device)
    add_4d = ttnn.reshape(add_2d, [batch, 1, 1, src_seq])
    expanded = ttnn.expand(add_4d, [batch, 1, tgt_seq, src_seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(add_4d)
    return _ensure_tile_bf16_sdpa_mask(expanded)


def _build_encoder_self_mask_4d(attention_mask_2d: ttnn.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    """HF ``_prepare_4d_attention_mask`` for encoder self-attn → ``[B, 1, S, S]`` bf16."""
    batch = int(attention_mask_2d.shape[0])
    seq = int(attention_mask_2d.shape[1])
    add_2d = _key_padding_additive(attention_mask_2d, device=device)
    add_4d = ttnn.reshape(add_2d, [batch, 1, 1, seq])
    expanded = ttnn.expand(add_4d, [batch, 1, seq, seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(add_4d)
    return _ensure_tile_bf16_sdpa_mask(expanded)


def _encoder_self_additive_mask_all_zeros_4d(batch: int, seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """Additive encoder self-attention mask when every position is valid (all keys visible).

    Numerically matches ``_build_encoder_self_mask_4d`` on an all-ones ``[B, seq]`` mask (zero padding
    contribution everywhere), but uses a single ``zeros`` tensor instead of ``eq`` / ``expand`` chains.
    """
    zeros = ttnn.zeros(
        [batch, 1, seq, seq],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return _ensure_tile_bf16_sdpa_mask(zeros)


def _pad_input_ids_to(input_ids: ttnn.Tensor, padded_seq: int, pad_id: int, device: ttnn.Device) -> ttnn.Tensor:
    """Right-pad ``[B, S]`` uint32 to ``[B, padded_seq]`` with ``pad_id`` (on device, ``ttnn.concat``)."""
    bsz = int(input_ids.shape[0])
    seq = int(input_ids.shape[1])
    if padded_seq == seq:
        return input_ids
    pad_tail = ttnn.full(
        [bsz, padded_seq - seq],
        float(pad_id),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    padded = ttnn.concat([input_ids, pad_tail], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(pad_tail)
    return padded


def _pad_mask_to(mask: ttnn.Tensor, padded_seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """Right-pad ``[B, S]`` uint32 attention mask to ``[B, padded_seq]`` with 0 (on device)."""
    bsz = int(mask.shape[0])
    seq = int(mask.shape[1])
    if padded_seq == seq:
        return mask
    zeros = ttnn.full(
        [bsz, padded_seq - seq],
        0.0,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    padded = ttnn.concat([mask, zeros], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(zeros)
    return padded


def _ones_mask(batch: int, seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """``[B, S]`` uint32 all-ones (real-position mask) — used when caller omits ``attention_mask``."""
    return ttnn.full(
        [batch, seq],
        1.0,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _subsampled_lens_dev(attention_mask_2d: ttnn.Tensor, kernel_size: int, stride: int) -> ttnn.Tensor:
    """HF ``_compute_sub_sample_lengths_from_attention_mask`` on device → ``[B]`` int32.

    Formula: ``floor((real_len + 2 * pad - kernel) / stride) + 1`` with ``pad = kernel // 2``.
    """
    pad = kernel_size // 2
    # NOTE: ``ttnn.typecast(uint32, bf16)`` is broken (returns 2^31 for value 1, with positions
    # mid-tile mis-converted). The reliable path is ``uint32 → int32 → bf16``. ``ttnn.sum`` also
    # needs TILE layout.
    tile_u = (
        ttnn.to_layout(attention_mask_2d, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if attention_mask_2d.get_layout() != ttnn.TILE_LAYOUT
        else attention_mask_2d
    )
    tile_i = ttnn.typecast(tile_u, ttnn.int32)
    if tile_u is not attention_mask_2d:
        ttnn.deallocate(tile_u)
    mask_f = ttnn.typecast(tile_i, ttnn.bfloat16)
    ttnn.deallocate(tile_i)
    real_count = ttnn.sum(mask_f, dim=1)  # [B] bf16 — count of 1s per row
    ttnn.deallocate(mask_f)
    adj = ttnn.add(real_count, float(2 * pad - kernel_size), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(real_count)
    div = ttnn.multiply(adj, float(1.0 / stride), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(adj)
    floored = ttnn.floor(div, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(div)
    plus1 = ttnn.add(floored, 1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(floored)
    out_i32 = ttnn.typecast(plus1, ttnn.int32)
    ttnn.deallocate(plus1)
    return out_i32  # [B] int32


def _tt_speech_enc_attn(sub_lens_tt: ttnn.Tensor, enc_seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """``[B, enc_seq]`` uint32 mask from per-row subsampled lengths — ``index < length`` row-wise."""
    bsz = int(sub_lens_tt.shape[0])
    sub_col = ttnn.reshape(sub_lens_tt, [bsz, 1])
    indices = ttnn.arange(
        0,
        enc_seq,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    indices_row = ttnn.reshape(indices, [1, enc_seq])
    ttnn.deallocate(indices)
    mask_bool = ttnn.lt(indices_row, sub_col, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(indices_row)
    mask_u32 = ttnn.typecast(mask_bool, ttnn.uint32)
    ttnn.deallocate(mask_bool)
    return mask_u32


# ---------------------------------------------------------------------------
# Host-only helpers (string operations on ``generation_config`` dictionaries)
# ---------------------------------------------------------------------------


def _ttnn_ids_from_list(rows: List[List[int]], device: ttnn.Device) -> ttnn.Tensor:
    """Build a small ``[B, S]`` uint32 ttnn tensor from a Python list of int rows."""
    bsz = len(rows)
    seq = len(rows[0])
    flat = [int(v) for r in rows for v in r]
    return ttnn.from_torch(
        _torch_int32_2d(flat, bsz, seq),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _torch_int32_2d(flat: List[int], bsz: int, seq: int):
    """Wrap ``ttnn.from_torch`` requirement in a tiny torch transport tensor (host construction only)."""
    import torch as _torch  # local import — torch is host transport, no math runs through it

    return _torch.tensor(flat, dtype=_torch.int32).reshape(bsz, seq)


def _read_int_row(scalars_tt: ttnn.Tensor) -> List[int]:
    """Read a ``[B]`` int32 device tensor as a Python ``list[int]`` (scalar readback only)."""
    import torch as _torch  # transport

    if scalars_tt.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        scalars_tt = ttnn.to_layout(scalars_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    host = ttnn.to_torch(ttnn.from_device(scalars_tt)).to(_torch.int64).reshape(-1)
    return [int(x) for x in host.tolist()]


def _read_int_scalar(scalar_tt: ttnn.Tensor) -> int:
    """Read a single int from a ``[1]`` or ``[1, 1]`` device tensor (scalar readback)."""
    import torch as _torch  # transport

    if scalar_tt.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        scalar_tt = ttnn.to_layout(scalar_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    host = ttnn.to_torch(ttnn.from_device(scalar_tt)).to(_torch.int64).reshape(-1)
    return int(host[0].item())


def _eos_id_set(value: Any) -> set:
    if value is None:
        return set()
    return {int(x) for x in value} if isinstance(value, (list, tuple)) else {int(value)}


def _indices_to_subwords(generation_config: Any, ids: List[int]) -> List[str]:
    """``generation_config.id_to_text`` per token id — pure string lookup."""
    if not hasattr(generation_config, "id_to_text"):
        raise ValueError("generation_config.id_to_text required for speech generation.")
    return [str(generation_config.id_to_text.get(str(int(i)))) for i in ids]


def _char_count_per_subword(
    ids: List[int], subwords: List[str], pad_token_id: int, unk_token_id: int = 1, space: str = "▁"
) -> List[int]:
    """HF ``_count_character_length_in_subword`` — pure Python string analysis."""
    n = len(ids)
    counts = [0] * n
    is_next_start_with_space = [
        len(subwords[i + 1]) > 1 and subwords[i + 1][0] == space if i < n - 1 else False for i in range(n)
    ]
    is_punc = [
        len(subwords[i]) == 1 and not subwords[i].isalpha() and not subwords[i].isnumeric() and subwords[i] != space
        for i in range(n)
    ]
    for i, (sid, sub) in enumerate(zip(ids, subwords)):
        if sid == pad_token_id:
            break
        if sid == unk_token_id:
            counts[i] = 1
            continue
        c = len(sub)
        if is_punc[i] and is_next_start_with_space[i]:
            c += 1
        elif i > 0 and is_punc[i - 1] and is_next_start_with_space[i - 1]:
            c -= 1
        counts[i] = c
    return counts


def _get_char_ids(
    generation_config: Any,
    ids: List[int],
    subwords: List[str],
    char_counts: List[int],
    pad_token_id: int,
    unk_token_id: int = 1,
) -> List[int]:
    """HF ``_get_char_input_ids`` — pure Python ``char_to_id`` dict lookup."""
    if not hasattr(generation_config, "char_to_id"):
        raise ValueError("generation_config.char_to_id required for speech generation.")
    total = int(sum(char_counts))
    out = [pad_token_id] * total
    cursor = 0
    for sid, sub in zip(ids, subwords):
        if sid == pad_token_id:
            break
        char_ids = (
            [unk_token_id]
            if sid == unk_token_id
            else [generation_config.char_to_id.get(ch, unk_token_id) for ch in list(sub)]
        )
        for c in char_ids:
            out[cursor] = c
            cursor += 1
    return out


def _t2u_attention_mask(real_len: int, padded_dec_seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """``[1, padded_dec_seq]`` uint32 mask: 1 where ``i < real_len`` else 0.

    Pure TTNN equivalent of HF ``_compute_new_attention_mask`` for the batch-1 path: an
    ``arange(padded_dec_seq) < real_len`` comparison broadcast into a 2-D ``[1, padded_dec_seq]`` mask.
    """
    indices = ttnn.arange(
        0,
        padded_dec_seq,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    indices_2d = ttnn.reshape(indices, [1, padded_dec_seq])
    bound = ttnn.full(
        [1, 1],
        float(real_len),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    less_bool = ttnn.lt(indices_2d, bound, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(indices_2d)
    ttnn.deallocate(bound)
    out_u32 = ttnn.typecast(less_bool, ttnn.uint32)
    ttnn.deallocate(less_bool)
    return out_u32


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------


class TTSeamlessM4Tv2Model:
    """TTNN port of HF ``SeamlessM4Tv2Model``. See module docstring for scope.

    ``forward`` and ``generate`` accept ``ttnn.Tensor`` inputs and return ``ttnn.Tensor`` outputs.
    All tensor math is on device; only Python control-flow + ``generation_config`` dictionary
    lookups (subword/char tables, lang codes) run on host.
    """

    def __init__(
        self,
        device: ttnn.Device,
        parameters: Any,
        *,
        layer_norm_eps: float,
        encoder_layers: int,
        encoder_attention_heads: int,
        decoder_layers: int,
        decoder_attention_heads: int,
        hidden_size: int,
        feature_projection_input_dim: int,
        speech_encoder_attention_heads: int,
        speech_encoder_intermediate_size: int,
        speech_encoder_layers: int,
        speech_encoder_chunk_size: Optional[int],
        speech_encoder_left_chunk_num: int,
        pad_token_id: int,
        decoder_start_token_id: int,
        vocab_size: int,
        adaptor_kernel_size: int,
        adaptor_stride: int,
        t2u_eos_token_id: int,
        t2u_pad_token_id: int,
        vocoder_offset: int,
        t2u_layer_norm_eps: float,
        t2u_encoder_layers: int,
        t2u_encoder_attention_heads: int,
        t2u_decoder_layers: int,
        t2u_decoder_attention_heads: int,
        variance_predictor_embed_dim: int,
        variance_predictor_hidden_dim: int,
        variance_predictor_kernel_size: int,
        vocoder_config: Any,
        generation_config: Optional[Any] = None,
        hf_config: Optional[Any] = None,  # accepted for API compatibility; not used
    ):
        del hf_config  # accepted by callers but no behaviour depends on it
        self.device = device
        self.parameters = parameters
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.vocab_size = vocab_size
        self.adaptor_kernel_size = adaptor_kernel_size
        self.adaptor_stride = adaptor_stride
        self.t2u_eos_token_id = t2u_eos_token_id
        self.t2u_pad_token_id = t2u_pad_token_id
        self.vocoder_offset = vocoder_offset
        self.generation_config = generation_config

        self.text_encoder = TTSeamlessM4Tv2Encoder(
            device,
            parameters.text_encoder,
            layer_norm_eps=layer_norm_eps,
            num_hidden_layers=encoder_layers,
            num_attention_heads=encoder_attention_heads,
            hidden_size=hidden_size,
        )
        self.text_decoder = TTSeamlessM4Tv2Decoder(
            device,
            parameters.text_decoder,
            layer_norm_eps=layer_norm_eps,
            num_hidden_layers=decoder_layers,
            num_attention_heads=decoder_attention_heads,
            hidden_size=hidden_size,
        )
        self.speech_encoder = TTSeamlessM4Tv2SpeechEncoder(
            device,
            parameters.speech_encoder,
            hidden_size=hidden_size,
            feature_projection_input_dim=feature_projection_input_dim,
            speech_encoder_attention_heads=speech_encoder_attention_heads,
            speech_encoder_intermediate_size=speech_encoder_intermediate_size,
            speech_encoder_layers=speech_encoder_layers,
            layer_norm_eps=layer_norm_eps,
            speech_encoder_chunk_size=speech_encoder_chunk_size,
            speech_encoder_left_chunk_num=speech_encoder_left_chunk_num,
        )
        self.t2u = TTSeamlessM4Tv2TextToUnitForConditionalGeneration(
            device,
            parameters.t2u,
            layer_norm_eps=t2u_layer_norm_eps,
            encoder_layers=t2u_encoder_layers,
            encoder_attention_heads=t2u_encoder_attention_heads,
            decoder_layers=t2u_decoder_layers,
            decoder_attention_heads=t2u_decoder_attention_heads,
            hidden_size=hidden_size,
            pad_token_id=t2u_pad_token_id,
            variance_predictor_embed_dim=variance_predictor_embed_dim,
            variance_predictor_hidden_dim=variance_predictor_hidden_dim,
            variance_predictor_kernel_size=variance_predictor_kernel_size,
        )
        self.vocoder = TTSeamlessM4Tv2CodeHifiGan(device, parameters.vocoder, vocoder_config)

        self._lm_head_compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ------------------------------------------------------------------
    # Internal pieces
    # ------------------------------------------------------------------

    def _lm_head(self, dec_out: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(
            dec_out,
            self.parameters.lm_head.weight,
            bias=None,
            core_grid=_core_grid(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._lm_head_compute,
        )

    def materialize_text_encoder_trace_tensors(
        self,
        input_ids: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor],
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, bool]:
        """Build text-encoder padded ids, positions, and 4D self mask **once** (outside ``begin_trace_capture``).

        Matches the tensor construction in ``_encode_text`` before ``text_encoder.forward``. Pair with
        ``forward_text_e2e_prefill_trace`` so trace capture avoids ``ttnn.full`` / ``concat`` padding ops.

        Returns:
            ``(ids_padded, position_ids, encoder_self_mask_4d, encoder_attn_2d_padded, attn_owned)`` —
            same ``attn_owned`` contract as ``_encode_text``.
        """
        batch = int(input_ids.shape[0])
        seq = int(input_ids.shape[1])
        padded_seq = _tile_align(seq)

        ids_padded = _pad_input_ids_to(input_ids, padded_seq, self.pad_token_id, self.device)
        if attention_mask is None:
            attn_padded = _ones_mask(batch, padded_seq, self.device)
            attn_owned = True
        else:
            attn_padded = _pad_mask_to(attention_mask, padded_seq, self.device)
            attn_owned = attn_padded is not attention_mask

        pos_tt = _tt_position_ids(ids_padded, self.pad_token_id)
        if attention_mask is None:
            enc_mask_4d = _encoder_self_additive_mask_all_zeros_4d(batch, padded_seq, self.device)
        else:
            enc_mask_4d = _build_encoder_self_mask_4d(attn_padded, device=self.device)
        return ids_padded, pos_tt, enc_mask_4d, attn_padded, attn_owned

    def _encode_text(
        self,
        input_ids: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor],
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, bool]:
        """Text encoder. Returns ``(encoder_out, encoder_attn_2d_padded, attn_owned)``.

        ``attn_owned=True`` means the caller must ``ttnn.deallocate`` the returned attn tensor;
        ``False`` means it aliases the input ``attention_mask`` (no padding was needed).
        """
        ids_padded, pos_tt, enc_mask_4d, attn_padded, attn_owned = self.materialize_text_encoder_trace_tensors(
            input_ids, attention_mask
        )
        enc_out = self.text_encoder.forward(ids_padded, pos_tt, enc_mask_4d)
        if ids_padded is not input_ids:
            ttnn.deallocate(ids_padded)
        ttnn.deallocate(pos_tt)
        ttnn.deallocate(enc_mask_4d)
        return enc_out, attn_padded, attn_owned

    def _speech_attention_uint_to_conv_bf16(self, mask_2d: ttnn.Tensor) -> ttnn.Tensor:
        # NOTE: ``ttnn.typecast(uint32, bf16)`` is broken (returns 2^31 for value 1 at most tile
        # positions). Reliable path is ``uint32 → int32 → bf16`` with TILE layout for both casts.
        mask_tile_u = ttnn.to_layout(mask_2d, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mask_tile_i = ttnn.typecast(mask_tile_u, ttnn.int32)
        ttnn.deallocate(mask_tile_u)
        mask_bf16_tile = ttnn.typecast(mask_tile_i, ttnn.bfloat16)
        ttnn.deallocate(mask_tile_i)
        return mask_bf16_tile

    def _speech_encoder_trim_pad_and_cross_attn(
        self, enc_raw: ttnn.Tensor, sub_lens_tt: ttnn.Tensor, batch: int
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Apply subsampled-length slice/pad and build ``[B, padded_src]`` encoder attention (deallocs ``sub_lens_tt``)."""
        sub_lens = _read_int_row(sub_lens_tt)[:batch]
        physical_len = int(enc_raw.shape[1])
        logical_len = max(1, min(min(sub_lens), physical_len))
        padded_len = _tile_align(logical_len)

        enc_out = enc_raw
        if physical_len > logical_len:
            sliced = ttnn.slice(enc_out, [0, 0, 0], [batch, logical_len, self.hidden_size], (1, 1, 1))
            ttnn.deallocate(enc_out)
            enc_out = sliced
        if logical_len < padded_len:
            pad_tail = ttnn.full(
                [batch, padded_len - logical_len, self.hidden_size],
                0.0,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            cat = ttnn.concat([enc_out, pad_tail], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(enc_out)
            ttnn.deallocate(pad_tail)
            enc_out = cat

        enc_attn_tt = _tt_speech_enc_attn(sub_lens_tt, padded_len, self.device)
        ttnn.deallocate(sub_lens_tt)
        return enc_out, enc_attn_tt

    def _encode_speech(
        self,
        input_features: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor],
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, bool]:
        """Speech encoder with adaptor subsampling.

        Returns ``(encoder_out, encoder_attn_2d, attn_owned=True)`` — the subsampled mask is always
        a fresh tensor that the caller must deallocate.
        """
        batch = int(input_features.shape[0])
        seq_in = int(input_features.shape[1])
        mask_2d = attention_mask if attention_mask is not None else _ones_mask(batch, seq_in, self.device)
        owned_input_mask = attention_mask is None

        mask_bf16_tile = self._speech_attention_uint_to_conv_bf16(mask_2d)
        enc_raw = self.speech_encoder.forward(input_features, conv_attention_mask_1d=mask_bf16_tile)
        ttnn.deallocate(mask_bf16_tile)

        sub_lens_tt = _subsampled_lens_dev(mask_2d, self.adaptor_kernel_size, self.adaptor_stride)
        if owned_input_mask:
            ttnn.deallocate(mask_2d)

        enc_out, enc_attn_tt = self._speech_encoder_trim_pad_and_cross_attn(enc_raw, sub_lens_tt, batch)
        return enc_out, enc_attn_tt, True

    def materialize_speech_encoder_trace_tensors(
        self,
        input_features: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor],
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor], int, int, ttnn.Tensor, SpeechEncoderTraceMasks]:
        """Prepare speech-encoder trace inputs **outside** ``begin_trace_capture``.

        Builds ``SpeechEncoderTraceMasks`` (conformer + adapter additive masks, depthwise left pads,
        and warmed relative-position index caches) so ``speech_encoder.forward(..., trace_masks=…)``
        matches ``_encode_speech`` math without mask-building or ``from_torch`` inside trace capture.

        Runs one probe ``speech_encoder.forward`` with those masks to learn ``physical_len``, performs
        subsampled-length **host readback** here, and preallocates a zero ``pad_tail`` when tile padding
        is required.

        Returns:
            ``(conv_mask_bf16, pad_tail_or_none, logical_len, physical_len, enc_attn_2d, trace_masks)``.
        """
        batch = int(input_features.shape[0])
        seq_in = int(input_features.shape[1])
        mask_2d = attention_mask if attention_mask is not None else _ones_mask(batch, seq_in, self.device)
        owned_input_mask = attention_mask is None

        conv_mask_bf16 = self._speech_attention_uint_to_conv_bf16(mask_2d)
        sub_lens_tt = _subsampled_lens_dev(mask_2d, self.adaptor_kernel_size, self.adaptor_stride)
        if owned_input_mask:
            ttnn.deallocate(mask_2d)

        trace_masks = self.speech_encoder.materialize_trace_attention_masks(conv_mask_bf16, batch=batch, seq=seq_in)
        probe_enc = self.speech_encoder.forward(
            input_features, conv_attention_mask_1d=conv_mask_bf16, trace_masks=trace_masks
        )
        physical_len = int(probe_enc.shape[1])
        ttnn.deallocate(probe_enc)

        sub_lens = _read_int_row(sub_lens_tt)[:batch]
        logical_len = max(1, min(min(sub_lens), physical_len))
        padded_len = _tile_align(logical_len)

        pad_tail: Optional[ttnn.Tensor] = None
        if logical_len < padded_len:
            pad_tail = ttnn.full(
                [batch, padded_len - logical_len, self.hidden_size],
                0.0,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        enc_attn_tt = _tt_speech_enc_attn(sub_lens_tt, padded_len, self.device)
        ttnn.deallocate(sub_lens_tt)

        return conv_mask_bf16, pad_tail, logical_len, physical_len, enc_attn_tt, trace_masks

    def _decode_and_lm_head(
        self,
        encoder_hidden: ttnn.Tensor,
        encoder_attn_2d: ttnn.Tensor,
        decoder_input_ids: ttnn.Tensor,
        decoder_attention_mask: Optional[ttnn.Tensor],
        *,
        prebuilt_causal_4d: Optional[ttnn.Tensor] = None,
        prebuilt_cross_4d: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """text-decoder → lm_head with on-device mask construction. Returns ``[B, padded_dec_seq, V]`` logits.

        When ``prebuilt_causal_4d`` and ``prebuilt_cross_4d`` are both set (greedy ``generate`` path),
        ``decoder_attention_mask`` must be ``None``; those tensors are **not** freed here so callers
        can reuse them across steps while tile-padded length is unchanged.
        """
        batch = int(decoder_input_ids.shape[0])
        dec_seq = int(decoder_input_ids.shape[1])
        padded_dec_seq = _tile_align(dec_seq)

        ids_padded = _pad_input_ids_to(decoder_input_ids, padded_dec_seq, self.pad_token_id, self.device)

        use_prebuilt = prebuilt_causal_4d is not None or prebuilt_cross_4d is not None
        if use_prebuilt:
            if prebuilt_causal_4d is None or prebuilt_cross_4d is None:
                raise ValueError("Pass both prebuilt_causal_4d and prebuilt_cross_4d, or neither.")
            if decoder_attention_mask is not None:
                raise ValueError("prebuilt_* masks are only valid with decoder_attention_mask=None.")
            causal_4d = prebuilt_causal_4d
            cross_4d = prebuilt_cross_4d
            own_masks = False
        else:
            own_masks = True
            # Decoder mask for ``_build_causal_with_padding_4d``: an all-ones 2-D mask adds **zero** padding
            # to the causal 4-D mask (HF semantics). Skip allocating ``_ones_mask`` + ``_key_padding_additive``
            # when the caller omits ``decoder_attention_mask`` — use the ``None`` fast path instead.
            if decoder_attention_mask is None:
                dec_attn_padded = None
                attn_owned = False
            else:
                dec_attn_padded = _pad_mask_to(decoder_attention_mask, padded_dec_seq, self.device)
                attn_owned = dec_attn_padded is not decoder_attention_mask

            causal_4d = _build_causal_with_padding_4d(dec_attn_padded, batch, padded_dec_seq, self.device)
            if attn_owned:
                ttnn.deallocate(dec_attn_padded)
            cross_4d = _build_cross_attn_mask_4d(encoder_attn_2d, tgt_seq=padded_dec_seq, device=self.device)

        pos_tt = _tt_position_ids(ids_padded, self.pad_token_id)

        dec_out = self.text_decoder.forward(ids_padded, pos_tt, encoder_hidden, causal_4d, cross_4d)
        if ids_padded is not decoder_input_ids:
            ttnn.deallocate(ids_padded)
        ttnn.deallocate(pos_tt)
        if own_masks:
            ttnn.deallocate(causal_4d)
            ttnn.deallocate(cross_4d)
        logits = self._lm_head(dec_out)
        ttnn.deallocate(dec_out)
        return logits

    def materialize_decoder_trace_tensors(
        self,
        encoder_attn_2d: ttnn.Tensor,
        decoder_input_ids: ttnn.Tensor,
        decoder_attention_mask: Optional[ttnn.Tensor],
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Build decoder masks and padded ids **once** on device (for use outside ``begin_trace_capture``).

        ``forward_decoder_and_lm_head_trace`` consumes these tensors and does **not** free them so
        the same buffers survive compile + trace capture + ``execute_trace`` replay.
        """
        batch = int(decoder_input_ids.shape[0])
        dec_seq = int(decoder_input_ids.shape[1])
        padded_dec_seq = _tile_align(dec_seq)

        ids_padded = _pad_input_ids_to(decoder_input_ids, padded_dec_seq, self.pad_token_id, self.device)

        if decoder_attention_mask is None:
            dec_attn_padded = None
            attn_owned = False
        else:
            dec_attn_padded = _pad_mask_to(decoder_attention_mask, padded_dec_seq, self.device)
            attn_owned = dec_attn_padded is not decoder_attention_mask

        pos_tt = _tt_position_ids(ids_padded, self.pad_token_id)
        causal_4d = _build_causal_with_padding_4d(dec_attn_padded, batch, padded_dec_seq, self.device)
        if attn_owned:
            ttnn.deallocate(dec_attn_padded)
        cross_4d = _build_cross_attn_mask_4d(encoder_attn_2d, tgt_seq=padded_dec_seq, device=self.device)

        return ids_padded, pos_tt, causal_4d, cross_4d

    def forward_decoder_and_lm_head_trace(
        self,
        encoder_hidden: ttnn.Tensor,
        ids_padded: ttnn.Tensor,
        pos_tt: ttnn.Tensor,
        causal_4d: ttnn.Tensor,
        cross_4d: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Decoder + ``lm_head`` only — **no** padding or mask construction. Caller owns ``encoder_hidden`` and mask tensors."""
        dec_out = self.text_decoder.forward(ids_padded, pos_tt, encoder_hidden, causal_4d, cross_4d)
        logits = self._lm_head(dec_out)
        ttnn.deallocate(dec_out)
        return logits

    def forward_text_e2e_prefill_trace(
        self,
        enc_ids_padded: ttnn.Tensor,
        enc_pos: ttnn.Tensor,
        enc_self_mask_4d: ttnn.Tensor,
        dec_ids_padded: ttnn.Tensor,
        dec_pos: ttnn.Tensor,
        dec_causal_4d: ttnn.Tensor,
        dec_cross_4d: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Text encoder → text decoder → ``lm_head`` with **no** padding or host-dependent mask ops.

        Intended for ``begin_trace_capture``: all masks and padded ids must be built via
        ``materialize_text_encoder_trace_tensors`` and ``materialize_decoder_trace_tensors`` beforehand.
        Does not deallocate ``enc_ids_padded``, positions, masks, or encoder hidden states — those
        buffers must remain stable for trace replay (same contract as ``forward_decoder_and_lm_head_trace``).
        """
        enc_out = self.text_encoder.forward(enc_ids_padded, enc_pos, enc_self_mask_4d)
        dec_out = self.text_decoder.forward(dec_ids_padded, dec_pos, enc_out, dec_causal_4d, dec_cross_4d)
        logits = self._lm_head(dec_out)
        ttnn.deallocate(dec_out)
        return logits

    def forward_speech_e2e_prefill_trace(
        self,
        input_features: ttnn.Tensor,
        conv_mask_bf16: ttnn.Tensor,
        speech_trace_masks: SpeechEncoderTraceMasks,
        pad_tail: Optional[ttnn.Tensor],
        logical_len: int,
        physical_len: int,
        dec_ids_padded: ttnn.Tensor,
        dec_pos: ttnn.Tensor,
        dec_causal_4d: ttnn.Tensor,
        dec_cross_4d: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Speech encoder (with prebuilt masks) → trim/pad → text decoder → ``lm_head``.

        ``materialize_speech_encoder_trace_tensors`` supplies ``conv_mask_bf16``, ``speech_trace_masks``,
        optional ``pad_tail``, and integer lengths. Trace capture should not rebuild additive masks,
        depthwise zero pads, or relative-position index tensors.
        """
        batch = int(input_features.shape[0])
        enc_raw = self.speech_encoder.forward(
            input_features,
            conv_attention_mask_1d=conv_mask_bf16,
            trace_masks=speech_trace_masks,
        )
        enc_mid = enc_raw
        if physical_len > logical_len:
            enc_mid = ttnn.slice(enc_raw, [0, 0, 0], [batch, logical_len, self.hidden_size], (1, 1, 1))
            ttnn.deallocate(enc_raw)
        if pad_tail is not None:
            enc_out = ttnn.concat([enc_mid, pad_tail], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(enc_mid)
        else:
            enc_out = enc_mid

        dec_out = self.text_decoder.forward(dec_ids_padded, dec_pos, enc_out, dec_causal_4d, dec_cross_4d)
        logits = self._lm_head(dec_out)
        ttnn.deallocate(dec_out)
        return logits

    def forward_text_e2e_plus_t2u_trace(
        self,
        enc_ids_padded: ttnn.Tensor,
        enc_pos: ttnn.Tensor,
        enc_self_mask_4d: ttnn.Tensor,
        dec_ids_padded: ttnn.Tensor,
        dec_pos: ttnn.Tensor,
        dec_causal_4d: ttnn.Tensor,
        dec_cross_4d: ttnn.Tensor,
        t2u_inputs_embeds: ttnn.Tensor,
        t2u_encoder_attn_4d: ttnn.Tensor,
        t2u_char_input_ids: ttnn.Tensor,
        t2u_char_count_per_id: List[int],
        t2u_reference_discrete_durations: List[int],
        t2u_hard_cums: T2UTraceHardUpsampleCumsums,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Text E2E prefill trace + T2U forward (T2U inputs and hard-upsample cums pre-materialized)."""
        text_logits = self.forward_text_e2e_prefill_trace(
            enc_ids_padded, enc_pos, enc_self_mask_4d, dec_ids_padded, dec_pos, dec_causal_4d, dec_cross_4d
        )
        t2u_logits, _ = self.t2u.forward(
            t2u_inputs_embeds,
            t2u_encoder_attn_4d,
            t2u_char_input_ids,
            t2u_char_count_per_id,
            reference_discrete_durations=t2u_reference_discrete_durations,
            hard_upsample_cums=t2u_hard_cums,
            trace_no_profiler=True,
        )
        return (text_logits, t2u_logits)

    def forward_speech_e2e_plus_t2u_trace(
        self,
        input_features: ttnn.Tensor,
        conv_mask_bf16: ttnn.Tensor,
        speech_trace_masks: SpeechEncoderTraceMasks,
        pad_tail: Optional[ttnn.Tensor],
        logical_len: int,
        physical_len: int,
        dec_ids_padded: ttnn.Tensor,
        dec_pos: ttnn.Tensor,
        dec_causal_4d: ttnn.Tensor,
        dec_cross_4d: ttnn.Tensor,
        t2u_inputs_embeds: ttnn.Tensor,
        t2u_encoder_attn_4d: ttnn.Tensor,
        t2u_char_input_ids: ttnn.Tensor,
        t2u_char_count_per_id: List[int],
        t2u_reference_discrete_durations: List[int],
        t2u_hard_cums: T2UTraceHardUpsampleCumsums,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Speech E2E prefill trace + T2U forward (T2U inputs and hard-upsample cums pre-materialized)."""
        text_logits = self.forward_speech_e2e_prefill_trace(
            input_features,
            conv_mask_bf16,
            speech_trace_masks,
            pad_tail,
            logical_len,
            physical_len,
            dec_ids_padded,
            dec_pos,
            dec_causal_4d,
            dec_cross_4d,
        )
        t2u_logits, _ = self.t2u.forward(
            t2u_inputs_embeds,
            t2u_encoder_attn_4d,
            t2u_char_input_ids,
            t2u_char_count_per_id,
            reference_discrete_durations=t2u_reference_discrete_durations,
            hard_upsample_cums=t2u_hard_cums,
            trace_no_profiler=True,
        )
        return (text_logits, t2u_logits)

    def _decoder_hidden(
        self,
        encoder_hidden: ttnn.Tensor,
        encoder_attn_2d: ttnn.Tensor,
        decoder_input_ids: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, int]:
        """text-decoder → ``(decoder_hidden_states [B, padded_dec_seq, H], padded_dec_seq)`` (no lm_head)."""
        batch = int(decoder_input_ids.shape[0])
        dec_seq = int(decoder_input_ids.shape[1])
        padded_dec_seq = _tile_align(dec_seq)

        ids_padded = _pad_input_ids_to(decoder_input_ids, padded_dec_seq, self.pad_token_id, self.device)

        # Build padded key mask: real positions (1) at non-pad tokens, 0 elsewhere.
        # ``ttnn.ne(ids, pad_id)`` gives a bool mask of real positions; typecast to uint32.
        ne_pad = ttnn.ne(ids_padded, self.pad_token_id)
        attn_2d = ttnn.typecast(ne_pad, ttnn.uint32)
        ttnn.deallocate(ne_pad)
        if attn_2d.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            attn_2d_rm = ttnn.to_layout(attn_2d, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn_2d)
            attn_2d = attn_2d_rm

        pos_tt = _tt_position_ids(ids_padded, self.pad_token_id)
        causal_4d = _build_causal_with_padding_4d(attn_2d, batch, padded_dec_seq, self.device)
        ttnn.deallocate(attn_2d)
        cross_4d = _build_cross_attn_mask_4d(encoder_attn_2d, tgt_seq=padded_dec_seq, device=self.device)

        dec_out = self.text_decoder.forward(ids_padded, pos_tt, encoder_hidden, causal_4d, cross_4d)
        if ids_padded is not decoder_input_ids:
            ttnn.deallocate(ids_padded)
        ttnn.deallocate(pos_tt)
        ttnn.deallocate(causal_4d)
        ttnn.deallocate(cross_4d)
        return dec_out, padded_dec_seq

    def _greedy_next_token(self, logits: ttnn.Tensor, dec_len: int) -> Tuple[ttnn.Tensor, int]:
        """Slice ``[B, dec_seq, V]`` → ``[B, 1, V]`` at position ``dec_len-1``, then ``ttnn.argmax`` → ``[B, 1]``.

        Returns ``(next_token_uint32 [B, 1], next_token_id_int)``. The scalar int is read for the EOS check.
        """
        batch = int(logits.shape[0])
        idx = dec_len - 1
        vocab_w = int(logits.shape[2])
        last = ttnn.slice(logits, [0, idx, 0], [batch, idx + 1, vocab_w], (1, 1, 1))
        argmax = ttnn.argmax(last, dim=-1)  # [B, 1] int32
        ttnn.deallocate(last)
        # Reshape if argmax dropped a dim
        if len(tuple(argmax.shape)) == 1:
            argmax = ttnn.reshape(argmax, [batch, 1])
        next_uint = ttnn.typecast(argmax, ttnn.uint32) if argmax.dtype != ttnn.uint32 else argmax
        if next_uint is not argmax:
            ttnn.deallocate(argmax)
        if next_uint.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            next_uint_rm = ttnn.to_layout(next_uint, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(next_uint)
            next_uint = next_uint_rm
        next_id_int = _read_int_scalar(next_uint)
        return next_uint, next_id_int

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Optional[ttnn.Tensor] = None,
        input_features: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        decoder_input_ids: Optional[ttnn.Tensor] = None,
        decoder_attention_mask: Optional[ttnn.Tensor] = None,
        **kwargs: Any,
    ) -> Seq2SeqLMOutput:
        """encoder (text or speech) → text-decoder → ``lm_head``.

        Always returns ``Seq2SeqLMOutput`` (``return_dict`` semantics in HF) with ``logits`` and
        ``encoder_last_hidden_state`` populated; the other fields are ``None`` (no KV cache, no
        attentions, no intermediate hidden states for this inference build). ``**kwargs`` is
        accepted for ``use_cache=False`` / ``return_dict=True`` etc. and silently ignored.
        """
        del kwargs  # ``use_cache``, ``return_dict``, etc. — TT stack is inference-only
        if input_ids is None and input_features is None:
            raise ValueError("Provide one of `input_ids` or `input_features`.")
        if decoder_input_ids is None:
            raise ValueError("`decoder_input_ids` is required.")

        if input_features is not None:
            enc_tt, enc_attn_tt, enc_attn_owned = self._encode_speech(input_features, attention_mask)
        else:
            enc_tt, enc_attn_tt, enc_attn_owned = self._encode_text(input_ids, attention_mask)  # type: ignore[arg-type]

        logits = self._decode_and_lm_head(enc_tt, enc_attn_tt, decoder_input_ids, decoder_attention_mask)
        if enc_attn_owned:
            ttnn.deallocate(enc_attn_tt)
        return Seq2SeqLMOutput(
            loss=None,
            logits=logits,
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=enc_tt,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )

    def generate(
        self,
        input_ids: Optional[ttnn.Tensor] = None,
        input_features: Optional[ttnn.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        speaker_id: int = 0,
        generate_speech: bool = True,
        **kwargs: Any,
    ) -> Union[TTSeamlessM4Tv2GreedySearchOutput, TTSeamlessM4Tv2GenerationOutput, Tuple[ttnn.Tensor, ttnn.Tensor]]:
        """Greedy (``num_beams=1``, ``do_sample=False``) analog of HF ``SeamlessM4Tv2Model.generate``.

        Returns ``ttnn.Tensor`` outputs only. For text modality, the first-pass encoder output is
        reused in the speech generation path, matching HF's ``text_generation_output.encoder_hidden_states[-1]``.
        """
        if input_ids is None and input_features is None:
            raise ValueError("Provide one of `input_ids` or `input_features`.")
        if generate_speech and tgt_lang is None:
            raise ValueError("`tgt_lang` is required when `generate_speech=True`.")
        if tgt_lang is not None:
            tgt_lang = tgt_lang.replace("__", "")
        kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)

        if kwargs_text.get("num_beams", 1) != 1:
            raise NotImplementedError("TT generate currently supports num_beams=1 only.")
        if kwargs_text.get("do_sample", False):
            raise NotImplementedError("TT generate currently supports do_sample=False (greedy) only.")

        max_new_tokens = int(kwargs_text.get("max_new_tokens", 20))
        eos_ids = _eos_id_set(kwargs_text.get("eos_token_id"))
        if self.generation_config is not None:
            eos_ids |= _eos_id_set(getattr(self.generation_config, "eos_token_id", None))
        attn_tt_text = kwargs_text.get("attention_mask")

        batch_size = int((input_features if input_features is not None else input_ids).shape[0])
        if batch_size != 1:
            raise NotImplementedError("TT generate supports batch_size=1.")

        if tgt_lang is not None:
            if self.generation_config is None:
                raise ValueError("`generation_config` must be set on the TT model when `tgt_lang` is used.")
            keys = (
                ["text_decoder_lang_to_code_id", "t2u_lang_code_to_id", "vocoder_lang_code_to_id"]
                if generate_speech
                else ["text_decoder_lang_to_code_id"]
            )
            for key in keys:
                lang_map = getattr(self.generation_config, key, None)
                if lang_map is None or tgt_lang not in lang_map:
                    raise ValueError(f"`tgt_lang={tgt_lang}` missing from generation_config.{key}.")

        # ---- First encode ----
        if input_features is not None:
            enc_tt, enc_attn_tt, enc_attn_owned = self._encode_speech(input_features, attn_tt_text)
        else:
            enc_tt, enc_attn_tt, enc_attn_owned = self._encode_text(input_ids, attn_tt_text)  # type: ignore[arg-type]

        # ---- Seed decoder sequence ----
        # ``tgt_lang`` overrides ``decoder_input_ids`` (HF semantics in ``SeamlessM4Tv2Model.generate``).
        # Either way, we end up owning ``seed_tt`` so the greedy loop's ``ttnn.deallocate`` is safe.
        user_seed = kwargs_text.get("decoder_input_ids")
        if tgt_lang is not None:
            tid = int(self.generation_config.text_decoder_lang_to_code_id[tgt_lang])
            ds = int(self.decoder_start_token_id)
            seed_tt = _ttnn_ids_from_list([[ds, tid]], self.device)
        elif user_seed is not None:
            seed_tt = ttnn.clone(user_seed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            raise ValueError("Provide `decoder_input_ids` or `tgt_lang` for TT generate.")

        # ---- Greedy decode loop (everything on device; one scalar readback per step for EOS) ----
        # Causal + cross 4-D masks depend only on ``(batch, tile_padded_dec_seq, encoder_attn_2d)`` when
        # ``decoder_attention_mask`` is omitted — rebuild only when that key changes (tile boundary).
        sequences_tt = seed_tt
        gen_causal: Optional[ttnn.Tensor] = None
        gen_cross: Optional[ttnn.Tensor] = None
        gen_mask_key: Optional[Tuple[int, int, int]] = None
        for _ in range(max_new_tokens):
            batch_i = int(sequences_tt.shape[0])
            padded_i = _tile_align(int(sequences_tt.shape[1]))
            mask_key = (batch_i, padded_i, id(enc_attn_tt))
            if mask_key != gen_mask_key:
                if gen_causal is not None:
                    ttnn.deallocate(gen_causal)
                    ttnn.deallocate(gen_cross)
                gen_causal = _build_causal_with_padding_4d(None, batch_i, padded_i, self.device)
                gen_cross = _build_cross_attn_mask_4d(enc_attn_tt, tgt_seq=padded_i, device=self.device)
                gen_mask_key = mask_key
            logits = self._decode_and_lm_head(
                enc_tt,
                enc_attn_tt,
                sequences_tt,
                None,
                prebuilt_causal_4d=gen_causal,
                prebuilt_cross_4d=gen_cross,
            )
            next_tt, next_id = self._greedy_next_token(logits, int(sequences_tt.shape[1]))
            ttnn.deallocate(logits)
            new_seq = ttnn.concat([sequences_tt, next_tt], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(sequences_tt)
            ttnn.deallocate(next_tt)
            sequences_tt = new_seq
            if eos_ids and next_id in eos_ids:
                break

        if gen_causal is not None:
            ttnn.deallocate(gen_causal)
            ttnn.deallocate(gen_cross)

        # ---- Text-only generation: return tokens ----
        if not generate_speech:
            ttnn.deallocate(enc_tt)
            if enc_attn_owned:
                ttnn.deallocate(enc_attn_tt)
            return TTSeamlessM4Tv2GreedySearchOutput(sequences=sequences_tt)

        # ---- Speech generation: re-encode for speech modality (HF parity), then T2U + vocoder ----
        gc = self.generation_config
        pad_token_id = int(gc.pad_token_id)
        attn_enc = kwargs_speech.get("attention_mask", attn_tt_text)
        if input_features is not None:
            ttnn.deallocate(enc_tt)
            if enc_attn_owned:
                ttnn.deallocate(enc_attn_tt)
            enc_tt2, enc_attn_tt2, enc_attn_owned2 = self._encode_speech(input_features, attn_enc)
        else:
            # Text path reuses the first-pass encoder output (HF parity).
            enc_tt2, enc_attn_tt2, enc_attn_owned2 = enc_tt, enc_attn_tt, enc_attn_owned

        # T2U decoder hidden states come from running text-decoder on ``sequences[:, :-1]`` (HF
        # trims the final EOS). Build a ttnn input slice (on device).
        seq_len_full = int(sequences_tt.shape[1])
        dec_in_tt = ttnn.slice(sequences_tt, [0, 0], [batch_size, seq_len_full - 1], (1, 1))
        dec_hidden_padded, padded_dec_seq = self._decoder_hidden(enc_tt2, enc_attn_tt2, dec_in_tt)
        ttnn.deallocate(dec_in_tt)
        ttnn.deallocate(enc_tt2)
        if enc_attn_owned2:
            ttnn.deallocate(enc_attn_tt2)

        # T2U prep: characters & char counts come from the generated text-token sequence.
        seq_full_ints = _read_int_row(sequences_tt)  # tiny: dec_seq tokens (single batch)
        dec_in_ints = seq_full_ints[:-1]
        real_dec_len = sum(1 for x in dec_in_ints if x != pad_token_id)
        # ``t2u_input_ids = sequences[:, 2:-1]`` with the lang/EOS positions stripped + EOS→pad replaced.
        eos_id = int(gc.eos_token_id)
        t2u_ids = [pad_token_id if t == eos_id else int(t) for t in seq_full_ints[2:-1]]
        subwords = _indices_to_subwords(gc, t2u_ids)
        cc_inner = _char_count_per_subword(t2u_ids, subwords, pad_token_id=pad_token_id)
        # Pad with one zero each side (for the stripped lang + EOS columns) then pad to padded_dec_seq.
        cc_list = [0] + cc_inner + [0]
        if len(cc_list) < padded_dec_seq:
            cc_list = cc_list + [0] * (padded_dec_seq - len(cc_list))
        char_ids = _get_char_ids(gc, t2u_ids, subwords, cc_inner, pad_token_id=pad_token_id)

        # On-device tensors for T2U: char_input_ids and the T2U attention mask.
        char_ids_tt = _ttnn_ids_from_list([char_ids], self.device)
        t2u_mask_2d = _t2u_attention_mask(real_dec_len, padded_dec_seq, self.device)
        t2u_mask_4d = _build_encoder_self_mask_4d(t2u_mask_2d, device=self.device)
        ttnn.deallocate(t2u_mask_2d)

        t2u_logits_tt, padding_tt = self.t2u.forward(
            dec_hidden_padded,
            t2u_mask_4d,
            char_ids_tt,
            cc_list,
            reference_discrete_durations=None,
        )
        ttnn.deallocate(dec_hidden_padded)
        ttnn.deallocate(t2u_mask_4d)
        ttnn.deallocate(char_ids_tt)

        # T2U linear may return ``[B, 1, unit_seq, V]`` (extra broadcast dim). Drop it before
        # ``argmax`` / vocoder so unit ids are ``[B, unit_seq]`` and the vocoder does not read
        # ``shape[1] == 1`` as the temporal length.
        _ls0 = tuple(t2u_logits_tt.shape)
        if len(_ls0) == 4 and int(_ls0[1]) == 1:
            _sq = ttnn.reshape(
                t2u_logits_tt,
                (_ls0[0], _ls0[2], _ls0[3]),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(t2u_logits_tt)
            t2u_logits_tt = _sq

        # T2U logits are sliced to logical ``unit_seq`` (see ``tt_text_to_unit.py`` end of forward),
        # while ``padding_tt`` stays at the tile-padded length. Slice ``padding_tt`` to the logical
        # ``unit_seq`` so the eq + logical_or below operate on matching shapes.
        unit_seq = int(t2u_logits_tt.shape[-2])
        pad_batch = int(padding_tt.shape[0])
        if int(padding_tt.shape[1]) != unit_seq:
            padding_logical = ttnn.slice(padding_tt, [0, 0], [pad_batch, unit_seq], (1, 1))
            ttnn.deallocate(padding_tt)
            padding_tt = padding_logical

        if t2u_logits_tt.get_layout() != ttnn.TILE_LAYOUT:
            t2u_tile = ttnn.to_layout(t2u_logits_tt, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(t2u_logits_tt)
            t2u_logits_tt = t2u_tile
        unit_ids_argmax = ttnn.argmax(t2u_logits_tt, dim=-1)  # [B, unit_seq] int32
        ttnn.deallocate(t2u_logits_tt)
        if unit_ids_argmax.dtype != ttnn.uint32:
            unit_u = ttnn.typecast(unit_ids_argmax, ttnn.uint32)
            ttnn.deallocate(unit_ids_argmax)
            unit_ids_argmax = unit_u
        if unit_ids_argmax.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            unit_rm = ttnn.to_layout(unit_ids_argmax, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(unit_ids_argmax)
            unit_ids_argmax = unit_rm
        # Preserve a copy for the ``unit_sequences`` output before vocoder remapping.
        output_unit_ids_tt = ttnn.clone(unit_ids_argmax, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # HF ``replace_mask = (unit_ids == t2u_eos) | (~padding_mask)`` with ``padding_mask`` = 1
        # at valid positions (``modeling_seamless_m4t_v2.py`` ~3577). Building that mask on device
        # with ``ttnn.logical_or`` across TILE bf16 padding vs ROW unit ids has produced wrong
        # booleans in the middle of the span, spuriously filling ``vocoder_input`` with T2U pads,
        # shrinking ``(input_ids != pad).sum()`` and thus HiFi-GAN ``t_audio``. Match HF on host.
        if padding_tt.dtype != ttnn.bfloat16:
            pad_bf = ttnn.typecast(padding_tt, ttnn.bfloat16)
            ttnn.deallocate(padding_tt)
        else:
            pad_bf = padding_tt
        if pad_bf.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            pad_rm = ttnn.to_layout(pad_bf, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(pad_bf)
            pad_bf = pad_rm

        b_m = int(unit_ids_argmax.shape[0])
        s_m = int(unit_ids_argmax.shape[1])
        if int(pad_bf.shape[1]) != s_m:
            pad_sl = ttnn.slice(pad_bf, [0, 0], [b_m, s_m], (1, 1))
            ttnn.deallocate(pad_bf)
            pad_bf = pad_sl

        # HF order (``modeling_seamless_m4t_v2.py`` ~3577-3584), applied on host below:
        #   1. ``unit_ids = unit_ids.masked_fill(replace_mask, t2u_pad_token_id)``
        #   2. ``unit_ids = where(unit_ids == t2u_pad_token_id, unit_ids, unit_ids - vocoder_offset)``
        # ``ttnn.where`` on uint32 TILE (required by the ternary op) does not match PyTorch
        # ``masked_fill`` / ``where`` for this remapping — it produced bogus indices (e.g. large
        # ``0xFFFF…`` values) and wrong HiFi-GAN lengths. HF-equivalent remap on host for the
        # small ``[B, S]`` unit grid (batch 1 in practice), then push back once as ``uint32`` ROW.
        import torch as _torch

        unit_host = ttnn.to_torch(ttnn.from_device(unit_ids_argmax)).to(_torch.long)
        pad_host = ttnn.to_torch(ttnn.from_device(pad_bf)).to(_torch.float32)
        ttnn.deallocate(unit_ids_argmax)
        ttnn.deallocate(pad_bf)

        rm_host = (unit_host == int(self.t2u_eos_token_id)) | (pad_host < 0.5)

        pad_id = int(self.t2u_pad_token_id)
        off = int(self.vocoder_offset)
        u = unit_host.clone()
        u[rm_host] = pad_id
        voc = _torch.where(u == pad_id, u, u - off).to(_torch.int32)
        vocoder_input = ttnn.from_torch(
            voc,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Vocoder language + speaker tensors built on device.
        voc_id = int(gc.vocoder_lang_code_to_id[tgt_lang])
        voc_tt = ttnn.full(
            [1, 1],
            float(voc_id),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        spk_tt = ttnn.full(
            [1, 1],
            float(int(speaker_id)),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        wav_tt, lengths_tt = self.vocoder.forward(vocoder_input, spk_tt, voc_tt)
        ttnn.deallocate(vocoder_input)
        ttnn.deallocate(voc_tt)
        ttnn.deallocate(spk_tt)

        # Vocoder lengths is 1D ``[B]``; standardise to ``[1, B]`` for callers / tests.
        if len(tuple(lengths_tt.shape)) == 1:
            lengths_tt = ttnn.reshape(lengths_tt, (1, int(lengths_tt.shape[0])))

        if return_intermediate_token_ids:
            return TTSeamlessM4Tv2GenerationOutput(
                waveform=wav_tt,
                waveform_lengths=lengths_tt,
                sequences=sequences_tt,
                unit_sequences=output_unit_ids_tt,
            )
        ttnn.deallocate(sequences_tt)
        ttnn.deallocate(output_unit_ids_tt)
        return wav_tt, lengths_tt
