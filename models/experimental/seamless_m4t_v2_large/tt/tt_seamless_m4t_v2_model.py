# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2Model`]: composes text/speech encoders, text decoder, T2U, vocoder, and ``lm_head``.

**Inference parity vs Hugging Face ``SeamlessM4Tv2Model.forward``**

Implemented on device (``ttnn.Tensor``): ``input_ids`` / ``input_features`` / ``encoder_outputs`` +
``encoder_modality``, ``attention_mask``, ``decoder_input_ids`` / ``decoder_inputs_embeds``,
``decoder_attention_mask``, ``return_dict``, ``use_cache=False``, ``inputs_embeds``, and ``**kwargs`` (ignored).

Decoder position IDs stay on device (``ttnn``). Encoder and decoder **4D additive attention masks** match
Hugging Face ``_prepare_4d_attention_mask`` / ``_prepare_4d_causal_attention_mask`` on CPU, then upload as
``bfloat16`` tile tensors. Small host readouts also include subsampled speech lengths for ``ttnn.slice``.

Explicitly **not** implemented: ``past_key_values`` / KV cache, ``labels`` / CE loss,
``output_attentions`` / ``output_hidden_states`` (must stay false).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import ttnn
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import (
    _compute_new_attention_mask,
    format_speech_generation_kwargs,
)

from models.experimental.seamless_m4t_v2_large.tt.tt_code_hifigan import TTSeamlessM4Tv2CodeHifiGan
from models.experimental.seamless_m4t_v2_large.tt.tt_speech_encoder import TTSeamlessM4Tv2SpeechEncoder
from models.experimental.seamless_m4t_v2_large.tt.tt_text_decoder import TTSeamlessM4Tv2Decoder
from models.experimental.seamless_m4t_v2_large.tt.tt_text_encoder import TTSeamlessM4Tv2Encoder
from models.experimental.seamless_m4t_v2_large.tt.tt_text_to_unit import (
    TTSeamlessM4Tv2TextToUnitForConditionalGeneration,
)

# ---------------------------------------------------------------------------
# Device helpers (position IDs, speech subsampled mask)
# ---------------------------------------------------------------------------


def _core_grid(device: ttnn.Device) -> ttnn.CoreGrid:
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=grid.x)


def _tt_position_ids(input_ids: ttnn.Tensor, pad_id: int) -> ttnn.Tensor:
    """Compute position IDs on device via cumsum — equivalent to HF ``create_position_ids_from_input_ids``.

    Positions start at ``pad_id + 1`` for the first non-padding token; padding positions receive ``pad_id``.
    No host download is performed; the operation runs entirely in ttnn.
    """
    ids_tile = ttnn.to_layout(input_ids, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mask = ttnn.ne(ids_tile, pad_id)
    ttnn.deallocate(ids_tile)
    mask_i32 = ttnn.typecast(mask, ttnn.int32)
    ttnn.deallocate(mask)
    cumsum = ttnn.cumsum(mask_i32, dim=1, dtype=ttnn.int32)
    pos = ttnn.multiply(cumsum, mask_i32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(cumsum)
    ttnn.deallocate(mask_i32)
    pos = ttnn.add(pos, pad_id, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pos = ttnn.typecast(pos, ttnn.uint32)
    pos = ttnn.to_layout(pos, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return pos


def _tt_seq_position_ids(bsz: int, seq: int, pad_id: int, device: ttnn.Device) -> ttnn.Tensor:
    """Sequential position IDs for the ``inputs_embeds`` path (no padding markers).

    Equivalent to HF ``create_position_ids_from_inputs_embeds``: positions are
    ``[pad_id+1, pad_id+2, …, pad_id+seq]``, broadcast to ``[bsz, seq]``.
    """
    pos_1d = ttnn.arange(
        pad_id + 1,
        seq + pad_id + 1,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_2d = ttnn.reshape(pos_1d, [1, seq])
    if bsz > 1:
        pos_out = ttnn.expand(pos_2d, [bsz, seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(pos_2d)
        return pos_out
    return pos_2d


def _tt_speech_enc_attn(sub_lens_list: List[float], enc_seq: int, bsz: int, device: ttnn.Device) -> ttnn.Tensor:
    """Build ``[bsz, enc_seq]`` uint32 attention mask from per-batch subsampled lengths.

    ``sub_lens_list`` is a tiny Python list (one float per batch) already on host — the minimum scalar
    readout that must happen to determine the slice endpoint after adaptor subsampling.
    """
    sub_int = torch.tensor([int(v) for v in sub_lens_list], dtype=torch.int32).reshape(bsz, 1)
    sub_tt = ttnn.from_torch(
        sub_int, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    indices = ttnn.arange(
        0, enc_seq, dtype=ttnn.int32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    indices_2d = ttnn.reshape(indices, [1, enc_seq])
    ttnn.deallocate(indices)
    # broadcast [1, enc_seq] < [bsz, 1] → [bsz, enc_seq]
    mask_bool = ttnn.lt(indices_2d, sub_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(indices_2d)
    ttnn.deallocate(sub_tt)
    mask_u32 = ttnn.typecast(mask_bool, ttnn.uint32)
    ttnn.deallocate(mask_bool)
    return mask_u32


def _tile_align(seq: int) -> int:
    """Round up to the next multiple of 32 (tile alignment for ttnn TILE_LAYOUT)."""
    return ((seq + 31) // 32) * 32


# ---------------------------------------------------------------------------
# CPU-side helpers (scalars / tiny tensors that do NOT flow through the model)
# ---------------------------------------------------------------------------


def _subsample_lengths_from_frame_mask(attention_mask: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
    """Compute subsampled sequence lengths (one scalar per batch) on CPU."""
    pad = kernel_size // 2
    seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)
    seq_lens = ((seq_lens + 2 * pad - kernel_size) / stride) + 1
    return seq_lens.floor()


def _indices_to_subwords(generation_config: Any, input_ids: torch.Tensor) -> List[List[str]]:
    if not hasattr(generation_config, "id_to_text"):
        raise ValueError(
            "generation_config must define id_to_text (token id string -> subword) for speech generation prep."
        )
    batch_size, sequence_len = input_ids.shape
    subwords_batch: List[List[str]] = []
    for batch_id in range(batch_size):
        subwords: List[str] = []
        for i in range(sequence_len):
            subword = generation_config.id_to_text.get(str(input_ids[batch_id, i].item()))
            subwords.append(str(subword))
        subwords_batch.append(subwords)
    return subwords_batch


def _count_character_length_in_subword(
    input_ids: torch.Tensor,
    subwords_batch: List[List[str]],
    *,
    pad_token_id: int = 0,
    unk_token_id: int = 1,
    space: str = "▁",
) -> torch.Tensor:
    batch_size, _ = input_ids.shape
    char_count_per_id = input_ids.new_zeros(input_ids.size())
    subword_lens = input_ids.ne(pad_token_id).sum(1)
    for batch_id in range(batch_size):
        subword_indices = input_ids[batch_id, : subword_lens[batch_id]]
        subwords = subwords_batch[batch_id][: subword_lens[batch_id]]
        is_next_start_with_space = [
            len(subwords[i + 1]) > 1 and subwords[i + 1][0] == space if i < len(subwords) - 1 else False
            for i in range(len(subwords))
        ]
        is_punc = [
            len(subwords[i]) == 1 and not subwords[i].isalpha() and not subwords[i].isnumeric() and subwords[i] != space
            for i in range(len(subwords))
        ]
        for i, (subword_idx, subword) in enumerate(zip(subword_indices, subwords)):
            if subword_idx == pad_token_id:
                break
            if subword_idx == unk_token_id:
                char_len = 1
            else:
                char_len = len(subword)
                if is_punc[i] and is_next_start_with_space[i]:
                    char_len += 1
                elif i > 0 and is_punc[i - 1] and is_next_start_with_space[i - 1]:
                    char_len -= 1
            char_count_per_id[batch_id, i] = char_len
    return char_count_per_id


def _get_char_input_ids(
    generation_config: Any,
    input_ids: torch.Tensor,
    subwords_batch: List[List[str]],
    char_count_per_id: torch.Tensor,
    *,
    pad_token_id: int = 0,
    unk_token_id: int = 1,
) -> torch.Tensor:
    if not hasattr(generation_config, "char_to_id"):
        raise ValueError("generation_config must define char_to_id for speech generation prep.")
    batch_size = input_ids.shape[0]
    max_len = int(char_count_per_id.sum(1).max().item())
    char_seqs = input_ids.new_zeros((batch_size, max_len)).fill_(pad_token_id)
    subword_lens = input_ids.ne(pad_token_id).sum(1)
    for batch_id in range(batch_size):
        total = 0
        subword_indices = input_ids[batch_id, : subword_lens[batch_id]]
        subwords = subwords_batch[batch_id][: subword_lens[batch_id]]
        for subword_idx, subword in zip(subword_indices, subwords):
            if subword_idx == unk_token_id:
                char_ids = [unk_token_id]
            else:
                char_ids = [generation_config.char_to_id.get(ch, unk_token_id) for ch in list(subword)]
            char_seq_len = len(char_ids)
            char_seqs[batch_id, total : total + char_seq_len] = torch.tensor(char_ids).to(char_seqs)
            total += char_seq_len
    return char_seqs


def _eos_token_id_set(value: Any) -> set:
    if value is None:
        return set()
    if isinstance(value, (list, tuple)):
        return {int(x) for x in value}
    return {int(value)}


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TTSeamlessM4Tv2GreedySearchOutput:
    """Minimal ``generate`` output when ``generate_speech=False`` (greedy, ``num_beams=1``)."""

    sequences: ttnn.Tensor


@dataclass
class TTSeamlessM4Tv2GenerationOutput:
    """``generate`` output when ``generate_speech=True`` and ``return_intermediate_token_ids``."""

    waveform: ttnn.Tensor
    waveform_lengths: ttnn.Tensor
    sequences: ttnn.Tensor
    unit_sequences: ttnn.Tensor


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------


class TTSeamlessM4Tv2Model:
    """
    TTNN port of Hugging Face ``SeamlessM4Tv2Model``.

    ``forward`` / ``generate`` take and return ``ttnn.Tensor`` on ``self.device``.
    Decoder position IDs are computed on device. Encoder/decoder 4D additive masks follow Hugging Face
    ``_prepare_4d_*`` on CPU then upload; speech still uses a tiny scalar readout for adaptor subsampling.
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
        hf_config: Optional[Any] = None,
    ):
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
        self.hf_config = hf_config
        self._default_use_cache = bool(getattr(hf_config, "use_cache", False)) if hf_config is not None else False
        self._default_output_attentions = (
            bool(getattr(hf_config, "output_attentions", False)) if hf_config is not None else False
        )
        self._default_output_hidden_states = (
            bool(getattr(hf_config, "output_hidden_states", False)) if hf_config is not None else False
        )
        self._default_use_return_dict = (
            bool(getattr(hf_config, "use_return_dict", True)) if hf_config is not None else True
        )

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

        self._linear_ln_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ------------------------------------------------------------------
    # Internal forward helpers
    # ------------------------------------------------------------------

    def _lm_head(self, dec_out: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(
            dec_out,
            self.parameters.lm_head.weight,
            bias=None,
            core_grid=_core_grid(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )

    def _decoder_lm_logits(
        self,
        encoder_hidden_states: ttnn.Tensor,
        decoder_input_ids: Optional[ttnn.Tensor],
        decoder_position_ids: ttnn.Tensor,
        decoder_causal_mask: ttnn.Tensor,
        decoder_cross_mask: Optional[ttnn.Tensor],
        *,
        decoder_inputs_embeds: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        dec_out = self.text_decoder.forward(
            decoder_input_ids,
            decoder_position_ids,
            encoder_hidden_states,
            decoder_causal_mask,
            decoder_cross_mask,
            inputs_embeds=decoder_inputs_embeds,
        )
        logits = self._lm_head(dec_out)
        ttnn.deallocate(dec_out)
        return logits

    def _torch_logits_last_token(self, logits_tt: ttnn.Tensor, batch: int, dec_len: int) -> torch.Tensor:
        if dec_len < 1:
            raise ValueError("dec_len must be >= 1 for last-token logits.")
        v = int(self.vocab_size)
        flat = ttnn.to_torch(ttnn.from_device(logits_tt)).to(torch.float32).contiguous().reshape(-1)
        if flat.numel() % v != 0:
            raise ValueError(f"logits numel {flat.numel()} not divisible by vocab_size {v}")
        sp = flat.numel() // v
        logits_2d = flat.reshape(int(batch), sp, v)
        idx = int(dec_len) - 1
        if idx >= sp:
            raise ValueError(f"decoder length index {idx} out of bounds for padded seq dim {sp}")
        return logits_2d[:, idx, :].contiguous()

    def _torch_additive_4d_to_tt(self, mask_4d: torch.Tensor) -> ttnn.Tensor:
        """Upload HF-style additive 4D mask ``[bsz, 1, q, k]`` as bfloat16 TILE on device."""
        return ttnn.from_torch(
            mask_4d.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ------------------------------------------------------------------
    # Encoder helpers — return (hidden_states_tt, enc_attn_tt)
    # enc_attn_tt is [bsz, enc_seq] uint32 (1=real, 0=pad); None → all real
    # ------------------------------------------------------------------

    def _encode_text_from_ttnn(
        self,
        input_ids: Optional[ttnn.Tensor],
        attention_mask: Optional[ttnn.Tensor],
        *,
        inputs_embeds: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
        """Text encoder: position IDs on device; encoder self-attn mask matches HF ``_prepare_4d_attention_mask``.

        Inputs are zero-padded on host to a multiple of 32 (tile-aligned) before encoding so SDPA does
        not score real queries against tile-padded garbage keys (parity collapse for non-aligned seq).
        The returned attention mask reflects the padded length: 1 for real positions, 0 for padding.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify only one of input_ids or inputs_embeds for the text encoder.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("One of input_ids or inputs_embeds is required for the text encoder.")

        if input_ids is not None:
            bsz, seq = int(input_ids.shape[0]), int(input_ids.shape[1])
        else:
            assert inputs_embeds is not None
            bsz, seq = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])

        padded_seq = _tile_align(seq)
        pad_n = padded_seq - seq

        if attention_mask is not None:
            attn_cpu = ttnn.to_torch(ttnn.from_device(attention_mask)).to(torch.long).cpu().contiguous()
            if attn_cpu.ndim != 2:
                raise ValueError("attention_mask must decode to a 2D [batch, seq] long tensor.")
            if attn_cpu.shape != (bsz, seq):
                raise ValueError(f"attention_mask shape {tuple(attn_cpu.shape)} does not match inputs ({bsz}, {seq}).")
        else:
            attn_cpu = torch.ones(bsz, seq, dtype=torch.long)

        if pad_n > 0:
            attn_cpu_padded = torch.cat([attn_cpu, torch.zeros(bsz, pad_n, dtype=torch.long)], dim=1)
        else:
            attn_cpu_padded = attn_cpu

        if input_ids is not None:
            if pad_n > 0:
                id_cpu = ttnn.to_torch(ttnn.from_device(input_ids)).to(torch.int64).cpu().contiguous()
                pad_ids = torch.full((bsz, pad_n), self.pad_token_id, dtype=id_cpu.dtype)
                id_cpu_padded = torch.cat([id_cpu, pad_ids], dim=1)
                input_ids_for_enc = ttnn.from_torch(
                    id_cpu_padded.to(torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                input_ids_for_enc = input_ids
            enc_pos_tt = _tt_position_ids(input_ids_for_enc, self.pad_token_id)
            inputs_embeds_for_enc = None
        else:
            if pad_n > 0:
                emb_cpu = ttnn.to_torch(ttnn.from_device(inputs_embeds)).to(torch.bfloat16).cpu().contiguous()
                emb_cpu = emb_cpu.reshape(bsz, seq, self.hidden_size)
                emb_pad = torch.zeros(bsz, pad_n, self.hidden_size, dtype=torch.bfloat16)
                emb_padded = torch.cat([emb_cpu, emb_pad], dim=1)
                inputs_embeds_for_enc = ttnn.from_torch(
                    emb_padded,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                inputs_embeds_for_enc = inputs_embeds
            enc_pos_tt = _tt_seq_position_ids(bsz, padded_seq, self.pad_token_id, self.device)
            input_ids_for_enc = None

        enc_mask_4d = _prepare_4d_attention_mask(attn_cpu_padded, torch.bfloat16)
        enc_mask_tt = self._torch_additive_4d_to_tt(enc_mask_4d)

        if input_ids_for_enc is not None:
            enc_out = self.text_encoder.forward(input_ids_for_enc, enc_pos_tt, enc_mask_tt)
        else:
            enc_out = self.text_encoder.forward(None, enc_pos_tt, enc_mask_tt, inputs_embeds=inputs_embeds_for_enc)
        ttnn.deallocate(enc_pos_tt)
        ttnn.deallocate(enc_mask_tt)
        if pad_n > 0:
            if input_ids_for_enc is not None and input_ids_for_enc is not input_ids:
                ttnn.deallocate(input_ids_for_enc)
            if inputs_embeds_for_enc is not None and inputs_embeds_for_enc is not inputs_embeds:
                ttnn.deallocate(inputs_embeds_for_enc)

        # Return a padded 2D attention mask (uint32) that reflects the padded encoder length so that
        # downstream cross-attn builds an additive mask masking out the tile padding.
        if pad_n > 0:
            enc_attn_out = ttnn.from_torch(
                attn_cpu_padded.to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            enc_attn_out = attention_mask
        return enc_out, enc_attn_out

    def _encode_speech_from_ttnn(
        self,
        input_features: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor],
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor]]:
        """Speech encoder + compute subsampled 2D attention mask (ttnn).

        The subsampled *scalar* lengths are read to host once (unavoidable for ``ttnn.slice``).
        The resulting ``enc_attn_tt`` is ``[bsz, enc_seq]`` uint32 built in ttnn.
        """
        batch, seq_in = int(input_features.shape[0]), int(input_features.shape[1])

        if attention_mask is None:
            attn_cpu = torch.ones(batch, seq_in, dtype=torch.long)
            attn_tt_for_enc = None
        else:
            # Download only to obtain scalar subsampled lengths for slicing.
            attn_cpu = ttnn.to_torch(ttnn.from_device(attention_mask)).to(torch.long).contiguous()
            if attn_cpu.ndim != 2:
                raise ValueError("attention_mask must decode to a 2D [batch, seq] long tensor.")
            if attn_cpu.shape[1] < seq_in:
                raise ValueError(f"attention_mask seq {attn_cpu.shape[1]} is shorter than input_features seq {seq_in}.")
            attn_cpu = attn_cpu[:, :seq_in].contiguous()
            # Pass the (trimmed) mask to the speech encoder as bfloat16 tile tensor.
            attn_tt_for_enc = ttnn.from_torch(
                attn_cpu.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        enc_out = self.speech_encoder.forward(input_features, conv_attention_mask_1d=attn_tt_for_enc)
        if attn_tt_for_enc is not None:
            ttnn.deallocate(attn_tt_for_enc)

        # Compute subsampled lengths on CPU (scalar readout — one integer per batch element).
        sub = _subsample_lengths_from_frame_mask(attn_cpu, self.adaptor_kernel_size, self.adaptor_stride)
        logical_len = int(sub.min().item())
        # Pad/slice the encoder output to a tile-aligned length so the downstream decoder cross-attn
        # does not score real queries against tile-padded encoder keys (parity collapse for non-aligned).
        padded_len = _tile_align(logical_len)
        physical_len = int(enc_out.shape[1])
        if physical_len != padded_len:
            enc_cpu_out = ttnn.to_torch(ttnn.from_device(enc_out)).to(torch.bfloat16).cpu().contiguous()
            enc_cpu_out = enc_cpu_out.reshape(batch, physical_len, self.hidden_size)
            enc_cpu_out = enc_cpu_out[:, :logical_len, :].contiguous()
            if padded_len > logical_len:
                pad = torch.zeros(batch, padded_len - logical_len, self.hidden_size, dtype=torch.bfloat16)
                enc_cpu_out = torch.cat([enc_cpu_out, pad], dim=1)
            ttnn.deallocate(enc_out)
            enc_out = ttnn.from_torch(
                enc_cpu_out,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        enc_seq = int(enc_out.shape[1])
        sub_list = sub.tolist()  # tiny list of Python floats (batch_size elements)
        enc_attn_tt = _tt_speech_enc_attn(sub_list, enc_seq, batch, self.device)
        return enc_out, enc_attn_tt

    def _encoder_hidden_from_outputs_ttnn(self, encoder_outputs: Any) -> ttnn.Tensor:
        if isinstance(encoder_outputs, ttnn.Tensor):
            return encoder_outputs
        if isinstance(encoder_outputs, (tuple, list)):
            t0 = encoder_outputs[0]
            if not isinstance(t0, ttnn.Tensor):
                raise TypeError("encoder_outputs[0] must be a ttnn.Tensor when encoder_outputs is a sequence.")
            return t0
        raise TypeError("encoder_outputs must be a ttnn.Tensor (or tuple of one ttnn.Tensor).")

    # ------------------------------------------------------------------
    # Decoder mask + logit helpers (HF 4D masks on CPU → device)
    # ------------------------------------------------------------------

    def _lm_logits_from_encoder_tt(
        self,
        encoder_hidden_tt: ttnn.Tensor,
        encoder_attn_tt: Optional[ttnn.Tensor],
        decoder_input_ids_tt: Optional[ttnn.Tensor],
        decoder_attention_mask_tt: Optional[ttnn.Tensor],
        *,
        deallocate_encoder: bool,
        decoder_inputs_embeds_tt: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Build decoder position IDs on device; causal + cross masks match HF prepare helpers.

        Decoder input is zero-padded on host to a multiple of 32 (tile-aligned) so SDPA does not score
        real queries against tile-padded garbage keys. Logits are returned at the padded length; callers
        slice down to the logical decoder sequence (e.g. ``[:, :sd, :]``).
        """
        if decoder_input_ids_tt is not None and decoder_inputs_embeds_tt is not None:
            raise ValueError("Specify only one of decoder_input_ids or decoder_inputs_embeds.")
        if decoder_input_ids_tt is None and decoder_inputs_embeds_tt is None:
            raise ValueError("One of decoder_input_ids or decoder_inputs_embeds is required.")

        if decoder_input_ids_tt is not None:
            bsz = int(decoder_input_ids_tt.shape[0])
            dec_seq = int(decoder_input_ids_tt.shape[1])
        else:
            bsz = int(decoder_inputs_embeds_tt.shape[0])  # type: ignore[union-attr]
            dec_seq = int(decoder_inputs_embeds_tt.shape[1])  # type: ignore[union-attr]

        padded_dec_seq = _tile_align(dec_seq)
        pad_n = padded_dec_seq - dec_seq

        if decoder_input_ids_tt is not None:
            if pad_n > 0:
                id_cpu = ttnn.to_torch(ttnn.from_device(decoder_input_ids_tt)).to(torch.int64).cpu().contiguous()
                pad_ids = torch.full((bsz, pad_n), self.pad_token_id, dtype=id_cpu.dtype)
                id_cpu_padded = torch.cat([id_cpu, pad_ids], dim=1)
                decoder_input_ids_for_dec = ttnn.from_torch(
                    id_cpu_padded.to(torch.int32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                decoder_input_ids_for_dec = decoder_input_ids_tt
            dec_pos_tt = _tt_position_ids(decoder_input_ids_for_dec, self.pad_token_id)
            decoder_inputs_embeds_for_dec = None
        else:
            if pad_n > 0:
                emb_cpu = (
                    ttnn.to_torch(ttnn.from_device(decoder_inputs_embeds_tt)).to(torch.bfloat16).cpu().contiguous()
                )
                emb_cpu = emb_cpu.reshape(bsz, dec_seq, self.hidden_size)
                emb_pad = torch.zeros(bsz, pad_n, self.hidden_size, dtype=torch.bfloat16)
                emb_padded = torch.cat([emb_cpu, emb_pad], dim=1)
                decoder_inputs_embeds_for_dec = ttnn.from_torch(
                    emb_padded,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                decoder_inputs_embeds_for_dec = decoder_inputs_embeds_tt
            dec_pos_tt = _tt_seq_position_ids(bsz, padded_dec_seq, self.pad_token_id, self.device)
            decoder_input_ids_for_dec = None

        enc_seq = int(encoder_hidden_tt.shape[1])

        if decoder_attention_mask_tt is None:
            dec_attn_cpu: Optional[torch.Tensor] = None
        else:
            dec_attn_cpu = ttnn.to_torch(ttnn.from_device(decoder_attention_mask_tt)).to(torch.long).cpu().contiguous()
            if dec_attn_cpu.shape != (bsz, dec_seq):
                raise ValueError(f"decoder_attention_mask shape {tuple(dec_attn_cpu.shape)} != ({bsz}, {dec_seq}).")

        if pad_n > 0:
            if dec_attn_cpu is None:
                dec_attn_cpu = torch.ones(bsz, dec_seq, dtype=torch.long)
            dec_attn_cpu = torch.cat([dec_attn_cpu, torch.zeros(bsz, pad_n, dtype=torch.long)], dim=1)

        dummy_dec = torch.zeros(bsz, padded_dec_seq, self.hidden_size, dtype=torch.bfloat16)
        dec_causal_torch = _prepare_4d_causal_attention_mask(
            dec_attn_cpu, (bsz, padded_dec_seq), dummy_dec, past_key_values_length=0
        )

        if encoder_attn_tt is None:
            dec_cross_torch = None
        else:
            enc_cpu = ttnn.to_torch(ttnn.from_device(encoder_attn_tt)).float().cpu().contiguous()
            if enc_cpu.ndim != 2:
                raise ValueError("encoder attention mask must be 2D [batch, enc_seq].")
            if int(enc_cpu.shape[0]) != bsz:
                raise ValueError(f"encoder attention batch {int(enc_cpu.shape[0])} != decoder batch {bsz}.")
            enc_cpu = enc_cpu[:, :enc_seq].contiguous()
            dec_cross_torch = _prepare_4d_attention_mask(enc_cpu, torch.bfloat16, tgt_len=padded_dec_seq)

        dec_causal_tt = self._torch_additive_4d_to_tt(dec_causal_torch)
        dec_cross_tt = self._torch_additive_4d_to_tt(dec_cross_torch) if dec_cross_torch is not None else None

        logits = self._decoder_lm_logits(
            encoder_hidden_tt,
            decoder_input_ids_for_dec,
            dec_pos_tt,
            dec_causal_tt,
            dec_cross_tt,
            decoder_inputs_embeds=decoder_inputs_embeds_for_dec,
        )
        ttnn.deallocate(dec_pos_tt)
        ttnn.deallocate(dec_causal_tt)
        if dec_cross_tt is not None:
            ttnn.deallocate(dec_cross_tt)
        if pad_n > 0:
            if decoder_input_ids_for_dec is not None and decoder_input_ids_for_dec is not decoder_input_ids_tt:
                ttnn.deallocate(decoder_input_ids_for_dec)
            if (
                decoder_inputs_embeds_for_dec is not None
                and decoder_inputs_embeds_for_dec is not decoder_inputs_embeds_tt
            ):
                ttnn.deallocate(decoder_inputs_embeds_for_dec)
        if deallocate_encoder:
            ttnn.deallocate(encoder_hidden_tt)
        return logits

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
        encoder_outputs: Optional[Any] = None,
        past_key_values: Any = None,
        inputs_embeds: Optional[ttnn.Tensor] = None,
        decoder_inputs_embeds: Optional[ttnn.Tensor] = None,
        labels: Any = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *,
        encoder_modality: Literal["text", "speech"] = "text",
        **kwargs: Any,
    ) -> Union[Seq2SeqLMOutput, Tuple[Any, ...]]:
        """
        Equivalent to HF ``SeamlessM4Tv2Model.forward`` for inference.

        Encoder (text, speech, or pre-computed ``encoder_outputs``) → text decoder → ``lm_head``.
        Position IDs on device; 4D masks match HF prepare helpers on CPU.  ``past_key_values``, ``labels``,
        ``output_attentions``, ``output_hidden_states``, and ``use_cache=True`` are not implemented.
        """
        _ = kwargs

        _ = labels  # ignored — inference only; no LM loss computed
        if past_key_values is not None:
            raise NotImplementedError(
                "TT text decoder has no KV-cache path; pass `past_key_values=None` and `use_cache=False`."
            )

        output_attentions = output_attentions if output_attentions is not None else self._default_output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self._default_output_hidden_states
        )
        if output_attentions or output_hidden_states:
            raise NotImplementedError(
                "TT stack does not return encoder/decoder attentions or intermediate hidden states."
            )

        use_cache = use_cache if use_cache is not None else self._default_use_cache
        if use_cache:
            raise NotImplementedError("TT text decoder does not implement KV cache; pass `use_cache=False`.")

        return_dict = return_dict if return_dict is not None else self._default_use_return_dict

        # ---- Input validation (mirrors HF) ----
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both `input_ids` and `inputs_embeds` for the text encoder.")
        if input_features is not None and inputs_embeds is not None:
            inputs_embeds = None  # input_features wins (HF-aligned warning omitted)
        if decoder_input_ids is not None and decoder_inputs_embeds is not None:
            raise ValueError("You cannot specify both `decoder_input_ids` and `decoder_inputs_embeds`.")
        if input_ids is None and input_features is None and inputs_embeds is None and encoder_outputs is None:
            raise ValueError("`input_ids`, `input_features`, `inputs_embeds`, and `encoder_outputs` are all empty.")
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            raise ValueError("Provide `decoder_input_ids` or `decoder_inputs_embeds`.")

        for name, val in [
            ("input_ids", input_ids),
            ("input_features", input_features),
            ("inputs_embeds", inputs_embeds),
            ("decoder_input_ids", decoder_input_ids),
            ("decoder_inputs_embeds", decoder_inputs_embeds),
            ("attention_mask", attention_mask),
            ("decoder_attention_mask", decoder_attention_mask),
        ]:
            if val is not None and not isinstance(val, ttnn.Tensor):
                raise TypeError(f"{name} must be a ttnn.Tensor on device when provided.")

        # ---- Encode ----
        enc_tt: ttnn.Tensor
        enc_attn_tt: Optional[ttnn.Tensor]

        if input_features is not None:
            # Speech modality: _encode_speech_from_ttnn handles subsampling + builds enc_attn_tt in ttnn.
            enc_tt, enc_attn_tt = self._encode_speech_from_ttnn(input_features, attention_mask)
        elif input_ids is not None:
            enc_tt, enc_attn_tt = self._encode_text_from_ttnn(input_ids, attention_mask)
        elif inputs_embeds is not None:
            enc_tt, enc_attn_tt = self._encode_text_from_ttnn(None, attention_mask, inputs_embeds=inputs_embeds)
        else:
            # Pre-computed encoder_outputs — caller must signal modality via encoder_modality.
            if encoder_modality not in ("text", "speech"):
                raise ValueError('encoder_modality must be "text" or "speech".')
            enc_tt = self._encoder_hidden_from_outputs_ttnn(encoder_outputs)
            if encoder_modality == "speech":
                # Replicate HF forward(): subsample the frame-level attention_mask to match
                # the encoder output sequence (post-adaptor striding).
                attn_cpu = ttnn.to_torch(ttnn.from_device(attention_mask)).to(torch.long).contiguous()
                seq_in = int(attention_mask.shape[1])
                attn_cpu = attn_cpu[:, :seq_in].contiguous()
                sub = _subsample_lengths_from_frame_mask(attn_cpu, self.adaptor_kernel_size, self.adaptor_stride)
                enc_seq = int(enc_tt.shape[1])
                batch = int(enc_tt.shape[0])
                enc_attn_tt = _tt_speech_enc_attn(sub.tolist(), enc_seq, batch, self.device)
            else:
                enc_attn_tt = attention_mask

        # ---- Decode + lm_head ----
        logits_tt = self._lm_logits_from_encoder_tt(
            enc_tt,
            enc_attn_tt,
            decoder_input_ids,
            decoder_attention_mask,
            deallocate_encoder=False,
            decoder_inputs_embeds_tt=decoder_inputs_embeds,
        )

        if not return_dict:
            ttnn.deallocate(enc_tt)
            return (logits_tt,)

        return Seq2SeqLMOutput(
            loss=None,
            logits=logits_tt,
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=enc_tt,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )

    @torch.no_grad()
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
        """Greedy ``num_beams=1`` analog of HF ``SeamlessM4Tv2Model.generate``.

        Accepts ``inputs_embeds`` (in ``kwargs``) like HF. For text modality, the first-pass encoder output
        is reused in the speech generation path — mirrors HF, which pulls it from
        ``text_generation_output.encoder_hidden_states[-1]``.
        """
        inputs_embeds = kwargs.pop("inputs_embeds", None)

        if input_ids is None and input_features is None and inputs_embeds is None:
            raise ValueError("Provide one of `input_ids`, `input_features`, or `inputs_embeds`.")
        if input_ids is not None and not isinstance(input_ids, ttnn.Tensor):
            raise TypeError("input_ids must be a ttnn.Tensor on device when provided.")
        if input_features is not None and not isinstance(input_features, ttnn.Tensor):
            raise TypeError("input_features must be a ttnn.Tensor on device when provided.")
        if inputs_embeds is not None and not isinstance(inputs_embeds, ttnn.Tensor):
            raise TypeError("inputs_embeds must be a ttnn.Tensor on device when provided.")
        if generate_speech and tgt_lang is None:
            raise ValueError("tgt_lang is required when generate_speech=True.")
        if tgt_lang is not None:
            tgt_lang = tgt_lang.replace("__", "")
        kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)

        attn_tt_text = kwargs_text.get("attention_mask")
        if attn_tt_text is not None and not isinstance(attn_tt_text, ttnn.Tensor):
            raise TypeError("attention_mask must be a ttnn.Tensor on device when provided.")

        if kwargs_text.get("num_beams", 1) != 1:
            raise NotImplementedError("TT generate currently supports num_beams=1 only.")
        if kwargs_text.get("do_sample", False):
            raise NotImplementedError("TT generate currently supports do_sample=False (greedy) only.")

        max_new_tokens = int(kwargs_text.get("max_new_tokens", 20))

        eos_ids: set = set()
        eos_ids |= _eos_token_id_set(kwargs_text.get("eos_token_id"))
        if self.generation_config is not None:
            eos_ids |= _eos_token_id_set(getattr(self.generation_config, "eos_token_id", None))

        if input_features is not None:
            batch_size = int(input_features.shape[0])
        elif input_ids is not None:
            batch_size = int(input_ids.shape[0])
        else:
            batch_size = int(inputs_embeds.shape[0])  # type: ignore[union-attr]
        if batch_size != 1:
            raise NotImplementedError("TT generate supports batch_size=1.")

        if tgt_lang is not None and self.generation_config is None:
            raise ValueError("generation_config must be set on the TT model when tgt_lang is used.")
        if tgt_lang is not None:
            gc = self.generation_config
            keys = (
                ["text_decoder_lang_to_code_id", "t2u_lang_code_to_id", "vocoder_lang_code_to_id"]
                if generate_speech
                else ["text_decoder_lang_to_code_id"]
            )
            for key in keys:
                lang_map = getattr(gc, key, None)
                if lang_map is None or tgt_lang not in lang_map:
                    raise ValueError(f"tgt_lang={tgt_lang} missing from generation_config.{key}.")

        # --- First encode ---
        if input_features is not None:
            enc_tt, enc_attn_tt = self._encode_speech_from_ttnn(input_features, attn_tt_text)
        elif inputs_embeds is not None:
            enc_tt, enc_attn_tt = self._encode_text_from_ttnn(None, attn_tt_text, inputs_embeds=inputs_embeds)
        else:
            enc_tt, enc_attn_tt = self._encode_text_from_ttnn(input_ids, attn_tt_text)  # type: ignore[arg-type]

        # --- Seed decoder sequence ---
        text_decoder_input_ids_tt = kwargs_text.get("decoder_input_ids")
        if text_decoder_input_ids_tt is not None and not isinstance(text_decoder_input_ids_tt, ttnn.Tensor):
            raise TypeError("decoder_input_ids must be a ttnn.Tensor on device when provided.")
        if tgt_lang is not None:
            tid = int(self.generation_config.text_decoder_lang_to_code_id[tgt_lang])
            ds = int(self.decoder_start_token_id)
            seed = torch.tensor([[ds, tid]], dtype=torch.int32, device="cpu")
            text_decoder_input_ids_tt = ttnn.from_torch(
                seed,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if text_decoder_input_ids_tt is None:
            raise ValueError("decoder_input_ids or tgt_lang must be provided for TT generate.")

        # --- Greedy decode loop (causal/cross masks via _lm_logits_from_encoder_tt / HF prepare) ---
        sequences_tt = text_decoder_input_ids_tt
        finished = torch.zeros(batch_size, dtype=torch.bool)
        for _ in range(max_new_tokens):
            logits_tt = self._lm_logits_from_encoder_tt(
                enc_tt, enc_attn_tt, sequences_tt, None, deallocate_encoder=False
            )
            dec_len = int(sequences_tt.shape[1])
            next_scores = self._torch_logits_last_token(logits_tt, batch_size, dec_len)
            ttnn.deallocate(logits_tt)
            next_token = torch.argmax(next_scores, dim=-1, keepdim=True).to(torch.int32)
            next_tt = ttnn.from_torch(
                next_token.cpu(),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            new_seq = ttnn.concat([sequences_tt, next_tt], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(sequences_tt)
            ttnn.deallocate(next_tt)
            sequences_tt = new_seq
            if eos_ids:
                last = int(next_token.view(-1)[0].item())
                finished |= torch.tensor([last in eos_ids], dtype=torch.bool)
                if bool(finished.all()):
                    break

        # For text modality (HF reuses ``text_generation_output.encoder_hidden_states[-1]``), keep the
        # first-pass encoder outputs alive — they will be reused below in the speech generation path.
        # For speech modality, HF re-runs the speech encoder; we mirror that and free the first-pass copies.
        reuse_text_encoder = generate_speech and input_features is None
        if not reuse_text_encoder:
            ttnn.deallocate(enc_tt)
            if enc_attn_tt is not None and enc_attn_tt is not attn_tt_text:
                ttnn.deallocate(enc_attn_tt)

        if not generate_speech:
            return TTSeamlessM4Tv2GreedySearchOutput(sequences=sequences_tt)

        # --- Speech generation: run text decoder for T2U embeddings (reuse text encoder, re-run speech encoder) ---
        gc = self.generation_config
        pad_token_id = int(gc.pad_token_id)

        attn_enc = kwargs_speech.get("attention_mask", kwargs_text.get("attention_mask", None))
        if attn_enc is not None and not isinstance(attn_enc, ttnn.Tensor):
            raise TypeError("speech/text attention_mask for second encode must be ttnn.Tensor on device.")
        if input_features is not None:
            # HF: speech path re-runs ``self.speech_encoder`` to refresh the subsampled attention mask.
            enc_tt2, enc_attn_tt2 = self._encode_speech_from_ttnn(input_features, attn_enc)
        else:
            # HF: text path reuses the first-pass encoder output (no second encode).
            enc_tt2, enc_attn_tt2 = enc_tt, enc_attn_tt

        # Download sequences to CPU for T2U char/subword preparation (unavoidable for string ops).
        seq_cpu = ttnn.to_torch(ttnn.from_device(sequences_tt)).to(torch.int64).contiguous()
        dec_in = seq_cpu[:, :-1].contiguous()

        bsz = batch_size
        dec_seq = dec_in.shape[1]
        enc_seq2 = int(enc_tt2.shape[1])

        # Pad dec_in to tile-aligned for decoder SDPA correctness (same reason as the forward path).
        padded_dec_seq = _tile_align(dec_seq)
        pad_n = padded_dec_seq - dec_seq
        if pad_n > 0:
            dec_in_padded = torch.cat([dec_in, torch.full((bsz, pad_n), self.pad_token_id, dtype=dec_in.dtype)], dim=1)
            dec_attn_padded = torch.cat(
                [(dec_in != pad_token_id).long(), torch.zeros(bsz, pad_n, dtype=torch.long)], dim=1
            )
        else:
            dec_in_padded = dec_in
            dec_attn_padded = (dec_in != pad_token_id).long()

        # Build decoder inputs for T2U hidden-state extraction (HF-aligned 4D masks).
        dec_ids_tt = ttnn.from_torch(
            dec_in_padded.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        dec_pos_tt = _tt_position_ids(dec_ids_tt, self.pad_token_id)
        dummy_dec = torch.zeros(bsz, padded_dec_seq, self.hidden_size, dtype=torch.bfloat16)
        dec_causal_torch = _prepare_4d_causal_attention_mask(
            dec_attn_padded.cpu().contiguous(), (bsz, padded_dec_seq), dummy_dec, past_key_values_length=0
        )
        if enc_attn_tt2 is None:
            dec_cross_torch = None
        else:
            enc_cpu = ttnn.to_torch(ttnn.from_device(enc_attn_tt2)).float().cpu().contiguous()
            enc_cpu = enc_cpu[:, :enc_seq2].contiguous()
            dec_cross_torch = _prepare_4d_attention_mask(enc_cpu, torch.bfloat16, tgt_len=padded_dec_seq)
        dec_causal_tt = self._torch_additive_4d_to_tt(dec_causal_torch)
        dec_cross_tt = self._torch_additive_4d_to_tt(dec_cross_torch) if dec_cross_torch is not None else None

        dec_hidden_padded = self.text_decoder.forward(dec_ids_tt, dec_pos_tt, enc_tt2, dec_causal_tt, dec_cross_tt)
        ttnn.deallocate(dec_ids_tt)
        ttnn.deallocate(dec_pos_tt)
        ttnn.deallocate(dec_causal_tt)
        if dec_cross_tt is not None:
            ttnn.deallocate(dec_cross_tt)
        # Keep the decoder hidden state at the padded sequence length: the T2U encoder runs its own
        # SDPA over this sequence and would hit the same tile-padding bug for ``dec_seq < 32``. Pad
        # ``cc_list`` / attention mask below to match ``padded_dec_seq``.
        dec_hidden = dec_hidden_padded
        dec_seq_for_t2u = padded_dec_seq
        # ``enc_tt2`` is either freshly allocated (speech re-encode) or the same object as ``enc_tt``
        # (text reuse). Either way, this is the final use; deallocate once.
        ttnn.deallocate(enc_tt2)
        # ``enc_attn_tt2`` may be (a) freshly allocated by speech subsampling, (b) the user-supplied
        # ``attn_enc`` / ``attn_tt_text`` for text modality, or (c) None. Only free freshly-allocated.
        if enc_attn_tt2 is not None and enc_attn_tt2 is not attn_enc and enc_attn_tt2 is not attn_tt_text:
            ttnn.deallocate(enc_attn_tt2)

        # --- T2U preparation (requires CPU string ops — unavoidable) ---
        t2u_input_embeds = dec_hidden
        seq_lens = (dec_in != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(
            torch.zeros(1, int(t2u_input_embeds.shape[1]), self.hidden_size, dtype=torch.bfloat16),
            seq_lens,
        )
        t2u_input_ids = seq_cpu[:, 2:-1].contiguous()
        t2u_input_ids = torch.masked_fill(t2u_input_ids, t2u_input_ids == int(gc.eos_token_id), pad_token_id)
        t2u_subwords = _indices_to_subwords(gc, t2u_input_ids)
        t2u_char_count_per_id = _count_character_length_in_subword(
            t2u_input_ids, t2u_subwords, pad_token_id=pad_token_id
        )
        pad_zero = t2u_char_count_per_id.new_zeros((t2u_char_count_per_id.shape[0], 1))
        t2u_char_count_per_id = torch.cat([pad_zero, t2u_char_count_per_id, pad_zero], dim=1)
        t2u_char_input_ids = _get_char_input_ids(
            gc, t2u_input_ids, t2u_subwords, t2u_char_count_per_id, pad_token_id=pad_token_id
        )
        # Pad ``t2u_char_count_per_id`` to ``dec_seq_for_t2u`` so its length matches the (padded)
        # T2U encoder sequence length; padded positions contribute zero characters (no-op).
        if t2u_char_count_per_id.shape[1] < dec_seq_for_t2u:
            extra = dec_seq_for_t2u - t2u_char_count_per_id.shape[1]
            zero_pad = t2u_char_count_per_id.new_zeros((t2u_char_count_per_id.shape[0], extra))
            t2u_char_count_per_id = torch.cat([t2u_char_count_per_id, zero_pad], dim=1)

        mask_4d = _prepare_4d_attention_mask(t2u_model_attention_mask, torch.bfloat16)
        enc_seq_t2u = int(t2u_input_embeds.shape[1])
        attn_tt = ttnn.from_torch(
            mask_4d.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        char_ids_tt = ttnn.from_torch(
            t2u_char_input_ids.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cc_list = [int(x) for x in t2u_char_count_per_id[0].tolist()]
        if len(cc_list) != enc_seq_t2u:
            ttnn.deallocate(t2u_input_embeds)
            ttnn.deallocate(attn_tt)
            ttnn.deallocate(char_ids_tt)
            raise RuntimeError(f"T2U char_count length {len(cc_list)} != encoder seq {enc_seq_t2u}.")

        temperature = kwargs_speech.get("temperature", None)
        do_sample = bool(kwargs_speech.get("do_sample", False))

        t2u_logits_tt, padding_tt = self.t2u.forward(
            t2u_input_embeds,
            attn_tt,
            char_ids_tt,
            cc_list,
            reference_discrete_durations=None,
        )
        ttnn.deallocate(t2u_input_embeds)
        ttnn.deallocate(attn_tt)
        ttnn.deallocate(char_ids_tt)

        # Read the logical unit-sequence and vocab size from the ttnn tensor's logical shape — the
        # ``ttnn.to_torch`` readback may unfold a ``[1, unit_seq, vocab]`` tile into 4D and report
        # tile-padded widths. Force ROW_MAJOR before readback so the torch tensor strips internal
        # tile padding and matches the logical shape exactly.
        t2u_shape = tuple(t2u_logits_tt.shape)
        if len(t2u_shape) < 2:
            raise RuntimeError(f"t2u logits shape {t2u_shape} has rank < 2.")
        vb = int(t2u_shape[-1])
        ulen = int(t2u_shape[-2])
        t2u_logits_rm = ttnn.to_layout(t2u_logits_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(t2u_logits_tt)
        t2u_logits = ttnn.to_torch(ttnn.from_device(t2u_logits_rm)).to(torch.float32)
        ttnn.deallocate(t2u_logits_rm)
        pad_mask = ttnn.to_torch(ttnn.from_device(padding_tt)).to(torch.bool)
        ttnn.deallocate(padding_tt)
        t2u_logits = t2u_logits.reshape(1, ulen, vb).contiguous()

        if temperature is None or float(temperature) == 1.0 or not do_sample:
            unit_ids = t2u_logits.argmax(dim=-1).to(torch.long)
        else:
            logits_scaled = t2u_logits / float(temperature)
            probs = F.softmax(logits_scaled, dim=-1).reshape(-1, vb)
            unit_ids = torch.multinomial(probs, num_samples=1).view(1, -1)

        # ``unit_ids`` is ``[1, ulen]``; flatten ``pad_mask`` to the same logical shape because
        # ``ttnn.to_torch`` may unfold the tile-padded buffer into an extra leading dim.
        pad_mask = pad_mask.reshape(-1)[:ulen].reshape(1, ulen)
        output_unit_ids = unit_ids.detach().clone()
        replace_mask = (unit_ids == self.t2u_eos_token_id) | (~pad_mask)
        unit_ids = unit_ids.masked_fill(replace_mask, self.t2u_pad_token_id)
        unit_ids = torch.where(unit_ids == self.t2u_pad_token_id, unit_ids, unit_ids - self.vocoder_offset).to(
            torch.long
        )

        voc_id = int(gc.vocoder_lang_code_to_id[tgt_lang])
        voc_tensor = torch.tensor([[voc_id]], dtype=torch.int32)
        spk_tensor = torch.tensor([[speaker_id]], dtype=torch.int32)
        unit_ids_tt = ttnn.from_torch(
            unit_ids.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        voc_tt = ttnn.from_torch(
            voc_tensor,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        spk_tt = ttnn.from_torch(
            spk_tensor,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        wav_tt, lengths_tt = self.vocoder.forward(unit_ids_tt, spk_tt, voc_tt, input_ids_torch=unit_ids)
        ttnn.deallocate(unit_ids_tt)
        ttnn.deallocate(voc_tt)
        ttnn.deallocate(spk_tt)

        # Vocoder returns ``lengths`` as a 1D device int32 ``[B]`` tensor; downstream consumers
        # expect a 2D ``[1, B]`` shape, mirroring the HF ``waveform_lengths`` layout in tests.
        if len(tuple(lengths_tt.shape)) == 1:
            lengths_tt = ttnn.reshape(lengths_tt, (1, int(lengths_tt.shape[0])))
        unit_ids_out_tt = ttnn.from_torch(
            output_unit_ids.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if return_intermediate_token_ids:
            return TTSeamlessM4Tv2GenerationOutput(
                waveform=wav_tt,
                waveform_lengths=lengths_tt,
                sequences=sequences_tt,
                unit_sequences=unit_ids_out_tt,
            )
        return wav_tt, lengths_tt
