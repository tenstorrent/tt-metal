# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2Model`]: text/speech encoders, text decoder, T2U, vocoder, ``lm_head``.

Inference parity with Hugging Face ``SeamlessM4Tv2Model``: ``forward`` and ``generate`` consume
and return ``ttnn.Tensor`` on ``self.device``. Position IDs, additive 4D attention masks, the
greedy ``argmax`` decode and all per-step tensor math run on device. When on-device sampling
is enabled (default on multi-device meshes), both greedy and ``do_sample=True`` token selection
run via ``TTSampling`` on width-sharded ``lm_head`` logits — including post-trace sampling
with ``use_decode_trace=True``. Host-side work is limited to:

* ``generation_config`` dictionary lookups (subword/char tables, lang code IDs — string ops);
* a single scalar readback of the subsampled-encoder length (needed for ``ttnn.slice`` end index);
* a per-step scalar readback of the predicted token (for the EOS early-stop check).

Out of scope: attentions/hidden-state outputs, label loss. Text-decoder KV cache is used in ``generate`` when ``use_kv_cache=True``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import ttnn
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import format_speech_generation_kwargs

from models.experimental.seamless_m4t_v2_large.tt.t2u_char_lookup import (
    T2UCharLookupTables,
    build_t2u_char_lookup_tables,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_code_hifigan import (
    TTSeamlessM4Tv2CodeHifiGan,
    _vocoder_timeline_bucket,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_speech_encoder import TTSeamlessM4Tv2SpeechEncoder
from models.experimental.seamless_m4t_v2_large.tt.tt_text_decoder import (
    TTSeamlessM4Tv2Decoder,
    init_text_decoder_kv_cache,
    warm_text_decoder_kv_cache_prefill,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_text_encoder import TTSeamlessM4Tv2Encoder
from models.experimental.seamless_m4t_v2_large.tt.common import (
    apply_hf_generation_defaults,
    build_causal_with_padding_4d,
    build_cross_attn_mask_4d,
    build_encoder_self_mask_4d,
    core_grid,
    ones_mask,
    pad_input_ids_to,
    pad_mask_to,
    tile_align,
    to_torch_replicated_first_shard,
    tt_position_ids,
    tt_position_ids_decode_step,
)
from models.experimental.seamless_m4t_v2_large.tt.tt_text_to_unit import (
    TTSeamlessM4Tv2TextToUnitForConditionalGeneration,
    make_t2u_trace_prealloc_tensors,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import get_tp, mesh_cluster_axis
from models.experimental.seamless_m4t_v2_large.tt.on_device_sampling import (
    prepare_row_logits_for_sampling,
    sample_first_token_from_prefill_logits,
    sample_in_decode_trace,
    try_create_on_device_sampler,
)
from models.tt_transformers.tt.ccl import TT_CCL

# Bucket the speech-encoder mel-sequence length to this multiple so run-to-run length jitter reuses
# the shape-specialized (JIT-compiled) kernels instead of recompiling cold (~7-20 s). 256 keeps the
# extra (masked) frames — and the encoder's O(S^2) attention overhead — small (≤256, ~2-5% typical).
_SPEECH_ENC_SEQ_BUCKET = 256
# Dummy ``_encode_speech`` prewarm poisons persistent L1 only on this JIT bucket band (2048 mel
# collapse). Longer buckets (e.g. 4096) still need the dummy forward for S2ST L1 layout.
_SPEECH_ENC_PREWARM_SKIP_DUMMY_ENCODE_MIN = 1920
_SPEECH_ENC_PREWARM_SKIP_DUMMY_ENCODE_MAX = 2560

# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SeamlessGenerateTimings:
    """Phase-separated ``generate()`` timings (TT catalog style: TTFT + steady decode t/s/u)."""

    input_is_speech: bool = False
    mel_frames: int = 0
    encoder_ms: float = 0.0
    prefill_ms: float = 0.0
    ttft_ms: float = 0.0
    ttft_audio_ms: float = 0.0
    decode_step_ms: List[float] = field(default_factory=list)
    t2u_ms: float = 0.0
    t2u_decoder_hidden_ms: float = 0.0
    t2u_char_prep_ms: float = 0.0
    t2u_forward_ms: float = 0.0
    vocoder_ms: float = 0.0
    output_tokens: int = 0
    output_samples: int = 0
    e2e_ms: float = 0.0
    per_step_tracking: bool = True

    @property
    def steady_decode_ms_per_tok(self) -> float:
        """Mean decode-step latency excluding the first step (trace/compile outlier), Qwen3-TTS style."""
        steps = self.decode_step_ms[1:] if len(self.decode_step_ms) > 1 else self.decode_step_ms
        if not steps:
            return 0.0
        return sum(steps) / len(steps)

    @property
    def decode_tok_s_u(self) -> float:
        ms = self.steady_decode_ms_per_tok
        return 1000.0 / ms if ms > 0 else 0.0

    @property
    def rtf(self) -> float:
        """Real-time factor: ``e2e_s / audio_duration_s`` (<1 is faster than real time)."""
        if self.output_samples <= 0:
            return 0.0
        sample_rate = 16000
        audio_dur_s = self.output_samples / float(sample_rate)
        if audio_dur_s <= 0:
            return 0.0
        return (self.e2e_ms / 1000.0) / audio_dur_s

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary for demo logs and tracy side-channel files."""
        return {
            "input_is_speech": self.input_is_speech,
            "mel_frames": self.mel_frames,
            "encoder_ms": self.encoder_ms,
            "prefill_ms": self.prefill_ms,
            "ttft_ms": self.ttft_ms,
            "ttft_audio_ms": self.ttft_audio_ms,
            "decode_step_ms": list(self.decode_step_ms),
            "steady_decode_ms_per_tok": self.steady_decode_ms_per_tok,
            "decode_tok_s_u": self.decode_tok_s_u,
            "t2u_ms": self.t2u_ms,
            "t2u_decoder_hidden_ms": self.t2u_decoder_hidden_ms,
            "t2u_char_prep_ms": self.t2u_char_prep_ms,
            "t2u_forward_ms": self.t2u_forward_ms,
            "vocoder_ms": self.vocoder_ms,
            "output_tokens": self.output_tokens,
            "output_samples": self.output_samples,
            "e2e_ms": self.e2e_ms,
            "rtf": self.rtf,
            "per_step_tracking": self.per_step_tracking,
        }


@dataclass
class TTSeamlessM4Tv2GreedySearchOutput:
    """``generate(generate_speech=False)`` output."""

    sequences: ttnn.Tensor
    timings: Optional[SeamlessGenerateTimings] = None


@dataclass
class TTSeamlessM4Tv2GenerationOutput:
    """``generate(generate_speech=True, return_intermediate_token_ids=True)`` output."""

    waveform: ttnn.Tensor
    waveform_lengths: ttnn.Tensor
    sequences: ttnn.Tensor
    unit_sequences: ttnn.Tensor
    timings: Optional[SeamlessGenerateTimings] = None


class _GeneratePhaseTimer:
    """Collects synced device phase times inside ``generate()`` when ``return_timings=True``."""

    def __init__(self, device: ttnn.Device, enabled: bool, *, input_is_speech: bool) -> None:
        self.device = device
        self.enabled = enabled
        self.input_is_speech = input_is_speech
        self.mel_frames = 0
        self.encoder_ms = 0.0
        self.prefill_ms = 0.0
        self.ttft_ms = 0.0
        self.ttft_audio_ms = 0.0
        self.decode_step_ms: List[float] = []
        self.t2u_ms = 0.0
        self.t2u_decoder_hidden_ms = 0.0
        self.t2u_char_prep_ms = 0.0
        self.t2u_forward_ms = 0.0
        self.vocoder_ms = 0.0
        self.per_step_tracking = True
        self._t0: Optional[float] = None
        self._ttft_recorded = False

    def start(self) -> None:
        if not self.enabled:
            return
        ttnn.synchronize_device(self.device)
        self._t0 = time.perf_counter()

    def sync_now(self) -> float:
        if self.enabled:
            ttnn.synchronize_device(self.device)
        return time.perf_counter()

    def elapsed_ms(self, t_start: float) -> float:
        return (self.sync_now() - t_start) * 1000.0

    def begin_step(self) -> Optional[float]:
        if not self.enabled or not self.per_step_tracking:
            return None
        return self.sync_now()

    def end_step(self, t_step: Optional[float]) -> None:
        if not self.enabled or t_step is None or not self.per_step_tracking:
            return
        self.decode_step_ms.append(self.elapsed_ms(t_step))
        if not self._ttft_recorded and self._t0 is not None:
            self.ttft_ms = (self.sync_now() - self._t0) * 1000.0
            self._ttft_recorded = True

    def finish(
        self,
        *,
        seed_len: int,
        seq_host: List[int],
        output_samples: int,
        generate_speech: bool,
    ) -> SeamlessGenerateTimings:
        e2e_ms = self.elapsed_ms(self._t0) if self.enabled and self._t0 is not None else 0.0
        return SeamlessGenerateTimings(
            input_is_speech=self.input_is_speech,
            mel_frames=self.mel_frames,
            encoder_ms=self.encoder_ms,
            prefill_ms=self.prefill_ms,
            ttft_ms=self.ttft_ms,
            ttft_audio_ms=self.ttft_audio_ms if generate_speech else 0.0,
            decode_step_ms=list(self.decode_step_ms),
            t2u_ms=self.t2u_ms,
            t2u_decoder_hidden_ms=self.t2u_decoder_hidden_ms,
            t2u_char_prep_ms=self.t2u_char_prep_ms,
            t2u_forward_ms=self.t2u_forward_ms,
            vocoder_ms=self.vocoder_ms,
            output_tokens=max(0, len(seq_host) - seed_len),
            output_samples=output_samples,
            e2e_ms=e2e_ms,
            per_step_tracking=self.per_step_tracking,
        )


@dataclass
class TextDecoderKvDecodeRuntime:
    """Pre-allocated device tensors for single-token KV decode (+ optional Metal trace)."""

    batch_size: int
    token_tt: ttnn.Tensor
    pos_tt: ttnn.Tensor
    cur_pos_tt: ttnn.Tensor
    logits_tt: Optional[ttnn.Tensor] = None
    logits_tt_by_cache_seq_len: dict[int, ttnn.Tensor] = field(default_factory=dict)
    # On-device greedy argmax fused into the decode trace: a ``(local_idx, chunk_max)`` pair, each
    # ``[1, 1, 32]`` (32 vocab chunks). The loop reads back 64 scalars and combines on host instead of the
    # full 256k-vocab logits row. ``logits_tt`` is still kept for the rep-penalty fallback (host recompute
    # only when the winner is a prior token).
    tok_tt: Optional[tuple[ttnn.Tensor, ttnn.Tensor]] = None
    tok_tt_by_cache_seq_len: dict[int, tuple[ttnn.Tensor, ttnn.Tensor]] = field(default_factory=dict)
    tok_li_host: Optional[ttnn.Tensor] = None
    tok_cm_host: Optional[ttnn.Tensor] = None
    trace_cache_seq_len: Optional[int] = None
    trace_id: Optional[int] = None
    trace_ids_by_cache_seq_len: dict[int, int] = field(default_factory=dict)
    last_dec_out_tt_by_cache_seq_len: dict[int, ttnn.Tensor] = field(default_factory=dict)


def _subsampled_lens_dev(attention_mask_2d: ttnn.Tensor, kernel_size: int, stride: int) -> ttnn.Tensor:
    """HF ``_compute_sub_sample_lengths_from_attention_mask`` on device → ``[B]`` int32.

    Formula: ``floor((real_len + 2 * pad - kernel) / stride) + 1`` with ``pad = kernel // 2``.
    """
    pad = kernel_size // 2
    # Use uint32 → int32 → bf16 (direct uint32→bf16 typecast is incorrect on device).
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
    """Read a ``[B]`` or ``[B, S]`` int32 device tensor as a Python ``list[int]`` (scalar readback only)."""
    import torch as _torch  # transport

    if scalars_tt.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        scalars_tt = ttnn.to_layout(scalars_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    host = to_torch_replicated_first_shard(scalars_tt).to(_torch.int64)
    if host.dim() >= 2:
        host = host.reshape(host.shape[0], -1)[0]
    else:
        host = host.reshape(-1)
    return [int(x) for x in host.tolist()]


def _trim_seq_host_for_speech(
    seq_host: List[int],
    *,
    pad_token_id: int,
    eos_id: int,
    seed_len: int,
) -> List[int]:
    """Drop trailing pad and cut at first EOS after the lang seed (speech / T2U path only)."""
    end = len(seq_host)
    while end > seed_len and seq_host[end - 1] == pad_token_id:
        end -= 1
    seq_host = seq_host[:end]
    tail = seq_host[seed_len:]
    if eos_id in tail:
        end = seed_len + tail.index(eos_id) + 1
        seq_host = seq_host[:end]
    return seq_host


def _read_int_scalar(scalar_tt: ttnn.Tensor) -> int:
    """Read a single int from a ``[1]`` or ``[1, 1]`` device tensor (scalar readback)."""
    import torch as _torch  # transport

    if scalar_tt.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        scalar_tt = ttnn.to_layout(scalar_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    host = to_torch_replicated_first_shard(scalar_tt).reshape(-1)
    if host.dtype == _torch.uint32:
        return int(host[0].item())
    val = int(host.to(_torch.int64)[0].item())
    if val < 0:
        val &= 0xFFFFFFFF
    return val


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


# T2U encoder sequence-length bucket. The T2U encoder runs at ``enc_seq == padded_dec_seq`` (the
# text-decoder hidden length, only tile-aligned), which jitters run-to-run with S2ST's variable target
# text → the encoder (and char-domain) recompile every task. Rounding the T2U encoder input to a coarse
# grid collapses the jitter onto a fixed program set (disk-warm rebuild), matching the committed
# ``_T2U_UNIT_SEQ_BUCKET`` decoder fix. Padded rows are masked in self-attn and get char-count 0, so
# they vanish in the char upsample → output unchanged.
_T2U_ENC_SEQ_BUCKET = 256


def _t2u_padded_enc_seq(seq: int) -> int:
    return ((int(seq) + _T2U_ENC_SEQ_BUCKET - 1) // _T2U_ENC_SEQ_BUCKET) * _T2U_ENC_SEQ_BUCKET


def _t2u_attention_mask_uncached(real_len: int, padded_dec_seq: int, device: ttnn.Device) -> ttnn.Tensor:
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
    lookups (lang codes) run on host. T2U subword→char tables are precomputed once at init.
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
        max_text_seq_len: int = 4096,
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
        # TP degree: number of devices in the tensor-parallel group (4 on BH QB 1×4 mesh).
        self._tp = get_tp(device)
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.vocab_size = vocab_size
        self.feature_projection_input_dim = feature_projection_input_dim
        # Speech-encoder mel buckets already JIT-warmed this process (see prewarm_speech_encoder).
        self._speech_prewarmed_buckets: set[int] = set()
        self.adaptor_kernel_size = adaptor_kernel_size
        self.adaptor_stride = adaptor_stride
        self.t2u_eos_token_id = t2u_eos_token_id
        self.t2u_pad_token_id = t2u_pad_token_id
        self.vocoder_offset = vocoder_offset
        self.generation_config = generation_config
        self._t2u_char_tables: Optional[T2UCharLookupTables] = None
        if generation_config is not None:
            self._t2u_char_tables = build_t2u_char_lookup_tables(
                generation_config,
                vocab_size=int(vocab_size),
                pad_token_id=int(pad_token_id),
            )
            self._t2u_char_tables.bind_device(device)
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
            max_batch_size=1,
            max_seq_len=max_text_seq_len,
        )
        self.max_text_seq_len = max_text_seq_len
        self.decoder_attention_heads = decoder_attention_heads
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
            matmul_token_rows=64,
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
        # Width-sharded lm_head metadata (see model_preprocessing): the traced decode step computes a
        # ``local_vocab_size``-wide logits slice per device; the host combine maps (device, chunk, idx)
        # back to a global token id.
        self._lm_local_vocab = int(getattr(parameters.lm_head, "local_vocab_size", 0) or 0)
        self._lm_num_devices = max(1, self.device.get_num_devices() if hasattr(self.device, "get_num_devices") else 1)
        self._lm_cluster_axis = mesh_cluster_axis(self.device)
        self._argmax_off_sharded: Optional[ttnn.Tensor] = None  # cached [nd,1,nch] global-offset table
        self._argmax_tok_mesh_composer: Optional[ttnn.MeshComposer] = None
        self._argmax_tie_break_bias_cache: dict[int, ttnn.Tensor] = {}
        self._kv_decode_rt: Optional[TextDecoderKvDecodeRuntime] = None
        self._decode_h2d_cache: dict = {}  # per-batch reusable host staging buffers for per-step uploads
        self._decode_trace_kernels_warmed = False
        self._t2u_attn_mask_cache: dict[tuple[int, int], ttnn.Tensor] = {}
        self._speech_hidden_parts: list[ttnn.Tensor] = []
        self._speech_hidden_cache_active = False
        self._t2u_trace_key: Optional[tuple[Any, ...]] = None
        self._vocoder_prewarm_keys: set[tuple[int, int]] = set()
        self._vocoder_prewarm_input: Optional[ttnn.Tensor] = None
        self._vocoder_prewarm_spk: Optional[ttnn.Tensor] = None
        self._vocoder_prewarm_lang: Optional[ttnn.Tensor] = None
        self._tt_ccl = TT_CCL(device) if self._tp > 1 else None
        self.sampling = None
        self._on_device_sampler = try_create_on_device_sampler(
            device,
            vocab_size=int(vocab_size),
            lm_head=getattr(parameters, "lm_head", None),
            tt_ccl=self._tt_ccl,
        )
        self._supports_on_device_sampling = self._on_device_sampler is not None
        if self._supports_on_device_sampling:
            self.sampling = self._on_device_sampler.generator

    def _use_on_device_sampling(self) -> bool:
        """On-device TTSampling for greedy and stochastic when the mesh supports it."""
        return self._on_device_sampler is not None and not getattr(self, "_decode_hf_exact_greedy", False)

    @property
    def _fold_sampling_in_trace(self) -> bool:
        return self._use_on_device_sampling() and sample_in_decode_trace(self.device)

    def _configure_on_device_sampling(
        self,
        *,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        seed: Optional[int],
    ) -> None:
        if self._on_device_sampler is None:
            return
        self._on_device_sampler.configure(
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

    def _read_token_from_traced_logits(self, logits: ttnn.Tensor, *, dec_len: int = 1) -> int:
        """Post-trace or in-trace token read via ``OnDeviceSampler``."""
        sampler = self._on_device_sampler
        if sampler is None:
            raise RuntimeError("On-device sampler required for traced sampling path.")
        if self._fold_sampling_in_trace:
            return sampler.tt_read_token(0)
        row_logits = prepare_row_logits_for_sampling(logits, dec_len=dec_len)
        sampler.advance_seed()
        sampler.sample_into_buffer(row_logits)
        if row_logits is not logits:
            ttnn.deallocate(row_logits)
        return sampler.tt_read_token(0)

    def _reset_speech_hidden_cache(self) -> None:
        for t in self._speech_hidden_parts:
            ttnn.deallocate(t)
        self._speech_hidden_parts = []
        self._speech_hidden_cache_active = False

    def _stash_speech_hidden_block(self, hidden: ttnn.Tensor, *, start: int, end: int) -> None:
        """Append decoder hidden rows ``[start:end)`` from a prefill ``[B,S,H]`` tensor (device-only)."""
        if not self._speech_hidden_cache_active or end <= start:
            return
        h = int(self.hidden_size)
        block = ttnn.slice(hidden, [0, start, 0], [1, end, h], (1, 1, 1))
        self._speech_hidden_parts.append(block)

    def _stash_speech_hidden_step(self, dec_out: ttnn.Tensor) -> None:
        """Append one decode-step hidden ``[B,1,H]`` (cloned before ``dec_out`` deallocate)."""
        if not self._speech_hidden_cache_active:
            return
        self._speech_hidden_parts.append(ttnn.clone(dec_out, memory_config=ttnn.DRAM_MEMORY_CONFIG))

    def _speech_hidden_for_t2u(self, logical_rows: int) -> Optional[Tuple[ttnn.Tensor, int]]:
        """Concat cached decode hiddens to ``[1, padded_dec_seq, H]`` when row count matches."""
        if not self._speech_hidden_cache_active or not self._speech_hidden_parts:
            return None
        if logical_rows <= 0:
            return None
        hidden = (
            self._speech_hidden_parts[0]
            if len(self._speech_hidden_parts) == 1
            else ttnn.concat(self._speech_hidden_parts, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        got = int(hidden.shape[1])
        if got != logical_rows:
            ttnn.deallocate(hidden)
            return None
        padded_dec_seq = tile_align(logical_rows)
        if padded_dec_seq != logical_rows:
            hidden = ttnn.pad(
                hidden,
                [(0, 0), (0, padded_dec_seq - logical_rows), (0, 0)],
                value=0.0,
            )
        return hidden, padded_dec_seq

    def prewarm_vocoder_shape_buckets(self, *, unit_seq: int, t_audio: int, batch: int = 1) -> None:
        """Warm vocoder conv programs for bucketed ``(unit_seq, t_audio)``."""
        u_bucket = _vocoder_timeline_bucket(max(1, int(unit_seq)))
        t_bucket = _vocoder_timeline_bucket(max(1, int(t_audio)))
        self.prewarm_vocoder_conv1d_weights(unit_seq=u_bucket, t_audio=t_bucket, batch=batch)
        ttnn.synchronize_device(self.device)

    def _release_vocoder_prewarm_stash(self) -> None:
        for attr in ("_vocoder_prewarm_input", "_vocoder_prewarm_spk", "_vocoder_prewarm_lang"):
            t = getattr(self, attr, None)
            if t is not None:
                ttnn.deallocate(t)
                setattr(self, attr, None)

    def _stash_vocoder_prewarm_inputs(
        self,
        vocoder_input: ttnn.Tensor,
        speaker_tt: ttnn.Tensor,
        lang_tt: ttnn.Tensor,
    ) -> None:
        """Keep cloned vocoder inputs from the last speech path for program prewarm replay."""
        self._release_vocoder_prewarm_stash()
        self._vocoder_prewarm_input = ttnn.clone(vocoder_input, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._vocoder_prewarm_spk = ttnn.clone(speaker_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._vocoder_prewarm_lang = ttnn.clone(lang_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def prewarm_vocoder_programs(self) -> None:
        """Warm vocoder conv weights and JIT-compile programs for the last speech ``generate()`` shapes."""
        voc = self.vocoder
        unit_seq = getattr(voc, "_last_unit_seq", None)
        t_audio = getattr(voc, "_last_t_audio", None)
        if unit_seq is None or t_audio is None or int(t_audio) < 1:
            return
        u_bucket = _vocoder_timeline_bucket(max(1, int(unit_seq)))
        t_bucket = _vocoder_timeline_bucket(max(1, int(t_audio)))
        key = (u_bucket, t_bucket)
        if key in self._vocoder_prewarm_keys:
            return
        self.prewarm_vocoder_conv1d_weights(unit_seq=u_bucket, t_audio=t_bucket)
        if (
            self._vocoder_prewarm_input is None
            or self._vocoder_prewarm_spk is None
            or self._vocoder_prewarm_lang is None
        ):
            self._vocoder_prewarm_keys.add(key)
            return
        inp = ttnn.clone(self._vocoder_prewarm_input, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        spk = ttnn.clone(self._vocoder_prewarm_spk, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        lang = ttnn.clone(self._vocoder_prewarm_lang, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        wav, lens = self.vocoder.forward(inp, spk, lang, trace_no_profiler=True)
        ttnn.deallocate(wav)
        ttnn.deallocate(lens)
        ttnn.deallocate(inp)
        ttnn.deallocate(spk)
        ttnn.deallocate(lang)
        self._vocoder_prewarm_keys.add(key)
        ttnn.synchronize_device(self.device)

    def _t2u_trace_signature(
        self,
        dec_hidden: ttnn.Tensor,
        char_ids: ttnn.Tensor,
        cc_list: Sequence[int],
    ) -> tuple[Any, ...]:
        return (
            tuple(int(x) for x in dec_hidden.shape),
            tuple(int(x) for x in char_ids.shape),
            tuple(int(x) for x in cc_list),
        )

    def _run_t2u_speech_forward(
        self,
        dec_hidden: ttnn.Tensor,
        t2u_mask_4d: ttnn.Tensor,
        char_ids_tt: ttnn.Tensor,
        cc_list: Sequence[int],
        *,
        use_t2u_trace: bool,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        sig = self._t2u_trace_signature(dec_hidden, char_ids_tt, cc_list)
        if use_t2u_trace and self._t2u_trace_key is not None and self._t2u_trace_key != sig:
            self.t2u.release_forward_trace()
            self._t2u_trace_key = None

        if use_t2u_trace and self._t2u_trace_key == sig and self.t2u._forward_trace_rt is not None:
            return self.t2u.execute_forward_trace()

        t2u_logits_tt, padding_tt = self.t2u.forward(
            dec_hidden,
            t2u_mask_4d,
            char_ids_tt,
            cc_list,
            reference_discrete_durations=None,
        )
        durs = self.t2u._last_dur_list
        if use_t2u_trace and durs is not None and self.t2u._forward_trace_rt is None:
            char_inc, char_prev = self.t2u._cached_repeat_cumsums(list(cc_list))
            unit_inc, unit_prev = self.t2u._cached_repeat_cumsums(durs)
            hard = make_t2u_trace_prealloc_tensors(
                self.device,
                pad_token_id=self.t2u_pad_token_id,
                hidden_size=self.hidden_size,
                char_w=int(char_ids_tt.shape[1]),
                cc_list=cc_list,
                ref_durs=durs,
                char_inc=char_inc,
                char_prev=char_prev,
                unit_inc=unit_inc,
                unit_prev=unit_prev,
            )
            self.t2u.capture_forward_trace(
                dec_hidden,
                t2u_mask_4d,
                char_ids_tt,
                cc_list,
                reference_discrete_durations=durs,
                hard_upsample_cums=hard,
            )
            self._t2u_trace_key = sig
        return t2u_logits_tt, padding_tt

    # ------------------------------------------------------------------
    # Speech-path conv prewarm (T2U + vocoder)
    # ------------------------------------------------------------------

    def prewarm_t2u_conv1d_weights(self, *, char_len: int, padded_unit_seq: int) -> None:
        """Prepare T2U duration + decoder conv weights (host upload only, no forward)."""
        self.t2u.prewarm_conv1d_weights(char_len=int(char_len), padded_unit_seq=int(padded_unit_seq))
        ttnn.synchronize_device(self.device)

    def prewarm_vocoder_conv1d_weights(self, *, unit_seq: int, t_audio: int, batch: int = 1) -> None:
        """Prepare vocoder conv weights for ``(unit_seq, t_audio)`` (host upload only, no forward)."""
        self.vocoder.prewarm_conv1d_weights(batch=batch, seq=int(unit_seq), t_audio=int(t_audio))
        ttnn.synchronize_device(self.device)

    def prewarm_speech_encoder(self, mel_seq_lens: List[int]) -> None:
        """Warm speech-encoder JIT for bucketed mel lengths.

        Short buckets and long buckets (e.g. 4096) use a dummy ``_encode_speech`` forward for
        kernel JIT. Only the 1920–2560 mel JIT band skips the dummy forward (cache-only
        ``pre_warm``): a dummy forward there left bad persistent L1 / LN state and collapsed
        S2TT/S2ST/ASR at 2048 mel to 1–2 tokens.
        """
        n_mels = self.feature_projection_input_dim
        for sl in mel_seq_lens:
            sl = int(sl)
            if sl <= 0:
                continue
            bucket = ((sl + _SPEECH_ENC_SEQ_BUCKET - 1) // _SPEECH_ENC_SEQ_BUCKET) * _SPEECH_ENC_SEQ_BUCKET
            if bucket in self._speech_prewarmed_buckets:
                continue
            self._speech_prewarmed_buckets.add(bucket)
            self.speech_encoder.pre_warm(batch=1, seq_len=bucket)
            if _SPEECH_ENC_PREWARM_SKIP_DUMMY_ENCODE_MIN <= bucket <= _SPEECH_ENC_PREWARM_SKIP_DUMMY_ENCODE_MAX:
                continue
            import torch as _torch

            feats = ttnn.from_torch(
                _torch.zeros(1, bucket, n_mels, dtype=_torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
            mask_host = _torch.zeros(1, bucket, dtype=_torch.int32)
            mask_host[0, :sl] = 1
            mask = ttnn.from_torch(
                mask_host,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
            enc, enc_attn, owned = self._encode_speech(feats, mask)
            ttnn.deallocate(enc)
            if owned:
                ttnn.deallocate(enc_attn)
            ttnn.deallocate(feats)
            ttnn.deallocate(mask)
        ttnn.synchronize_device(self.device)

    def release_generation_runtime(self) -> None:
        """Release decode trace, host readback buffers, and KV-decode runtime.

        Call before ``close_mesh_device`` (e.g. demo shutdown) so ``allocate_tensor_on_host``
        mirrors do not trip ``PinnedMemoryCache::release_for_device`` on mesh teardown.
        """
        self._release_vocoder_prewarm_stash()
        self._vocoder_prewarm_keys.clear()
        self._release_kv_decode_runtime()
        ttnn.synchronize_device(self.device)

    def clear_runtime_program_cache(self) -> None:
        """Release decode trace, per-module prep caches, and reset the device program cache."""
        # A captured decode trace hard-codes this modality's programs *and* KV/encoder buffer
        # addresses; release it (before clearing programs) so a later ``generate()`` re-captures
        # against its own buffers instead of replaying a stale trace (hangs in ``execute_trace``).
        self.release_text_decoder_decode_trace()
        self.device.clear_program_cache()
        self.vocoder._conv1d_prepared_cache.clear()
        self.vocoder._matmul_pc_cache.clear()
        self.speech_encoder._conv1d_prepared_cache.clear()
        self.speech_encoder._matmul_pc_cache.clear()
        self.t2u._conv1d_prepared_cache.clear()
        self.t2u._matmul_pc_cache.clear()
        self.text_decoder._matmul_pc_cache.clear()
        self.text_encoder._dram_matmul_pc_cache.clear()

    def _evict_t2u_prep_for_vocoder(self) -> None:
        """Drop T2U prep caches before vocoder without flushing the global Metal program cache."""
        self.t2u._conv1d_prepared_cache.clear()
        self.t2u._matmul_pc_cache.clear()

    def _clear_decode_and_t2u_programs(self, *, preserve_vocoder: bool = True) -> None:
        """Evict decode/T2U/speech-encoder programs for L1 headroom; keep vocoder prep when requested.

        Warm repeated ``generate(generate_speech=True)`` calls reuse vocoder conv1d programs when
        ``preserve_vocoder=True`` — avoids ~8–25 s vocoder cold-compile on every warmup/timed iter.
        """
        self.release_text_decoder_decode_trace()
        if not preserve_vocoder:
            self.device.clear_program_cache()
        self.t2u._conv1d_prepared_cache.clear()
        self.t2u._matmul_pc_cache.clear()
        self.text_decoder._matmul_pc_cache.clear()
        self.text_encoder._dram_matmul_pc_cache.clear()
        self.speech_encoder._conv1d_prepared_cache.clear()
        self.speech_encoder._matmul_pc_cache.clear()
        if not preserve_vocoder:
            self.vocoder._conv1d_prepared_cache.clear()
            self.vocoder._matmul_pc_cache.clear()
            self._vocoder_prewarm_keys.clear()

    def _release_speech_generate_runtime(self) -> None:
        """Release traces/KV after speech ``generate()`` without flushing vocoder program cache."""
        self._reset_speech_hidden_cache()
        self._t2u_trace_key = None
        self.t2u.release_forward_trace()
        self.vocoder.release_forward_trace()
        self.release_generation_runtime()

    # ------------------------------------------------------------------
    # Internal pieces
    # ------------------------------------------------------------------

    def _lm_head(self, dec_out: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.linear(
            dec_out,
            self.parameters.lm_head.weight,
            bias=None,
            core_grid=core_grid(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._lm_head_compute,
        )

    def _lm_head_sharded(self, dec_out: ttnn.Tensor) -> ttnn.Tensor:
        """Width-sharded ``lm_head`` for the traced decode step → ``[1, 1, V/tp]`` *per device* (each
        device holds a different vocab slice). The ``-1e9`` bias masks the padding columns. Feeds the
        per-shard chunked argmax; the host combine (``_greedy_next_token_id``) reduces across shards."""
        return ttnn.linear(
            dec_out,
            self.parameters.lm_head.weight_sharded,
            bias=self.parameters.lm_head.bias_sharded,
            core_grid=core_grid(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._lm_head_compute,
        )

    # Number of vocab chunks for the fused decode argmax (tile-aligned so multicore argmax is valid).
    _ARGMAX_CHUNKS = 32
    # Bias subtracted along argmax dims so ``ttnn.argmax`` picks the lowest index on ties (HF ``torch.argmax``).
    _ARGMAX_TIE_BREAK_EPS = 1e-9

    def _argmax_tie_break_bias_1d(self, length: int) -> ttnn.Tensor:
        """Cached float32 ``[1,1,1,L]`` bias: ``idx * eps`` (lowest index wins on equal scores)."""
        cached = self._argmax_tie_break_bias_cache.get(length)
        if cached is not None:
            return cached
        idx = torch.arange(length, dtype=torch.float32) * self._ARGMAX_TIE_BREAK_EPS
        bias = ttnn.from_torch(
            idx.reshape(1, 1, 1, length),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._argmax_tie_break_bias_cache[length] = bias
        return bias

    @staticmethod
    def _apply_tie_break_subtract(scores_f32: ttnn.Tensor, bias_1d: ttnn.Tensor) -> ttnn.Tensor:
        """Subtract ``bias_1d`` broadcast along the last dim of ``scores_f32`` (caller owns inputs)."""
        rank = len(tuple(scores_f32.shape))
        if rank == 3:
            bias = ttnn.reshape(bias_1d, [1, 1, int(bias_1d.shape[-1])])
        elif rank == 4:
            bias = bias_1d
        else:
            bias = ttnn.reshape(bias_1d, [1] * (rank - 1) + [int(bias_1d.shape[-1])])
        return ttnn.sub(scores_f32, bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _decode_argmax_token(self, logits: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """On-device greedy argmax for a single-token decode step ``logits`` ``[1, 1, V]``.

        Returns ``(local_idx, chunk_max)``, each ``[1, 1, 32]``: the vocab is reshaped into 32 contiguous
        chunks, and per chunk we keep the in-chunk argmax index and the in-chunk max value. The host
        combine (cheap, 64 scalars) picks the winning chunk ``c = argmax(chunk_max)`` then the global token
        ``c * chunk_width + local_idx[c]`` (see ``_greedy_next_token_id``).

        ``ttnn.argmax(use_multicore=True)`` is far faster than the 256k-vocab host readback but has two
        alignment traps that both return *silent garbage*: (1) the dim before the reduced vocab dim must be
        tile-aligned (32); (2) the multicore kernel reduces over the *physical* (tile-padded) width, so the
        reduced dim must also be a multiple of 32. The natural ``[1, 1, 1, V]`` shape pads the height to 32
        physically, so untilize/argmax churn ``32 × V`` elements for one real row. Reshaping to
        ``[1, 1, 32, chunk_width]`` instead makes the 32 the *chunk count* (free tile alignment) and keeps
        the physical work at ``32 × chunk_width ≈ V`` — ~3× faster than padding the row dim to 32. Pad V up
        to ``32 * chunk_width`` with ``-1e9`` so the padding columns can never win. ``ttnn.max`` runs on the
        TILE tensor directly; argmax needs ROW_MAJOR so untilize first. Both are used in the
        compile-outside-trace step and inside the trace capture so the trace itself outputs the token.
        """
        v = int(logits.shape[-1])
        nch = self._ARGMAX_CHUNKS
        chunk_w = (((v + nch - 1) // nch) + 31) // 32 * 32  # tile-aligned chunk width
        v_pad = nch * chunk_w
        lp = logits
        if v_pad != v:
            lp = ttnn.pad(logits, [(0, 0), (0, 0), (0, v_pad - v)], value=-1e9)
        cr = ttnn.reshape(lp, [1, 1, nch, chunk_w])
        chunk_max = ttnn.max(cr, dim=-1, keepdim=False)
        cu = ttnn.untilize(cr, use_multicore=True)
        local_idx = ttnn.argmax(cu, dim=-1, keepdim=False, use_multicore=True)
        return local_idx, chunk_max

    def _argmax_chunk_width(self, v_loc: int) -> int:
        nch = self._ARGMAX_CHUNKS
        return (((v_loc + nch - 1) // nch) + 31) // 32 * 32

    def _argmax_off_table(self, v_loc: int) -> ttnn.Tensor:
        """Cached ``[nd,1,nch]`` int32 table sharded along dim0: device ``d`` holds
        ``off[c] = d*v_loc + c*chunk_w`` — the global-token base for each (device, chunk). Pre-adding it
        to the per-device argmax index makes the gathered per-chunk values *global* token candidates, so
        the cross-shard combine is a single ``argmax`` (winning chunk) + ``gather`` (its global token)."""
        if self._argmax_off_sharded is not None:
            return self._argmax_off_sharded
        nch = self._ARGMAX_CHUNKS
        nd = self._lm_num_devices
        cw = self._argmax_chunk_width(v_loc)
        d = torch.arange(nd, dtype=torch.int32).reshape(nd, 1)
        c = torch.arange(nch, dtype=torch.int32).reshape(1, nch)
        off = (d * v_loc + c * cw).reshape(nd, 1, nch)
        self._argmax_off_sharded = ttnn.from_torch(
            off,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.device, dim=0),
        )
        return self._argmax_off_sharded

    def _ondevice_global_argmax_from_chunks(
        self,
        local_idx: ttnn.Tensor,
        chunk_max: ttnn.Tensor,
        v_loc: int,
        *,
        dealloc_inputs: bool = False,
    ) -> ttnn.Tensor:
        """Cross-shard greedy combine from trace-fused ``(local_idx, chunk_max)`` → global token id."""
        nd = self._lm_num_devices
        n_flat = nd * self._ARGMAX_CHUNKS
        off = self._argmax_off_table(v_loc)
        li = ttnn.to_layout(ttnn.typecast(local_idx, ttnn.int32), ttnn.TILE_LAYOUT)
        if dealloc_inputs:
            ttnn.deallocate(local_idx)
        cand = ttnn.add(li, off)
        ttnn.deallocate(li)
        if nd > 1:
            cm_g = ttnn.all_gather(chunk_max, dim=-1, cluster_axis=self._lm_cluster_axis, num_links=1)
            cand_g = ttnn.all_gather(cand, dim=-1, cluster_axis=self._lm_cluster_axis, num_links=1)
            if dealloc_inputs:
                ttnn.deallocate(chunk_max)
            ttnn.deallocate(cand)
        else:
            cm_g, cand_g = chunk_max, cand
            if dealloc_inputs:
                ttnn.deallocate(chunk_max)
        cm_u = ttnn.untilize(cm_g, use_multicore=True)
        ttnn.deallocate(cm_g)
        cm_f = ttnn.typecast(cm_u, ttnn.float32)
        ttnn.deallocate(cm_u)
        cm_b = self._apply_tie_break_subtract(
            ttnn.reshape(cm_f, [1, 1, n_flat]),
            self._argmax_tie_break_bias_1d(n_flat),
        )
        ttnn.deallocate(cm_f)
        flat = ttnn.argmax(cm_b, dim=-1, keepdim=True, use_multicore=True)
        ttnn.deallocate(cm_b)
        flat_t = ttnn.to_layout(flat, ttnn.TILE_LAYOUT)
        ttnn.deallocate(flat)
        token = ttnn.gather(cand_g, dim=-1, index=flat_t)
        ttnn.deallocate(cand_g)
        ttnn.deallocate(flat_t)
        token_u = ttnn.typecast(ttnn.reshape(token, [1, 1]), ttnn.uint32)
        ttnn.deallocate(token)
        token_rm = ttnn.to_layout(token_u, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(token_u)
        return token_rm

    def _ondevice_global_argmax_token(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        """Device-side greedy argmax over width-sharded ``[1,1,V/tp]`` logits → global token id."""
        local_idx, chunk_max = self._decode_argmax_token(logits)
        return self._ondevice_global_argmax_from_chunks(
            local_idx,
            chunk_max,
            int(logits.shape[-1]),
            dealloc_inputs=True,
        )

    def _device_greedy_token_from_trace_chunks(
        self,
        tok_tt: tuple[ttnn.Tensor, ttnn.Tensor],
        v_loc: int,
    ) -> int:
        """Read global greedy token from trace-owned chunk argmax buffers (single scalar D2H)."""
        li, cm = tok_tt
        # ``all_gather`` must not consume trace-owned ``tok_tt`` buffers — clone before combine.
        li_copy = ttnn.clone(li, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cm_copy = ttnn.clone(cm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        token_tt = self._ondevice_global_argmax_from_chunks(li_copy, cm_copy, v_loc, dealloc_inputs=True)
        token_id = _read_int_scalar(token_tt)
        ttnn.deallocate(token_tt)
        return token_id

    @staticmethod
    def _global_greedy_token_from_chunk_tensors(
        chunk_max: "torch.Tensor",
        local_idx: "torch.Tensor",
        *,
        v_loc: int,
        nch: int = 32,
        eps: float = _ARGMAX_TIE_BREAK_EPS,
    ) -> int:
        """Host combine matching ``_ondevice_global_argmax_from_chunks`` (gather + HF tie-break)."""
        nd, nch_act = int(chunk_max.shape[0]), int(chunk_max.shape[1])
        assert nch_act == nch
        chunk_w = (((v_loc + nch - 1) // nch) + 31) // 32 * 32
        d = torch.arange(nd, dtype=torch.int64).reshape(nd, 1)
        c = torch.arange(nch, dtype=torch.int64).reshape(1, nch)
        off = d * v_loc + c * chunk_w
        cand = local_idx.to(torch.int64) + off
        n_flat = nd * nch
        cm_flat = chunk_max.reshape(n_flat).to(torch.float32)
        cm_flat = cm_flat - torch.arange(n_flat, dtype=torch.float32) * eps
        flat = int(cm_flat.argmax())
        return int(cand.reshape(n_flat)[flat].item())

    @staticmethod
    def _free_kv_host_readback_buffers(rt: TextDecoderKvDecodeRuntime) -> None:
        """Release host mirrors for chunk-argmax D2H (avoids PinnedMemoryCache errors on device close)."""
        for attr in ("tok_li_host", "tok_cm_host"):
            host_t = getattr(rt, attr, None)
            if host_t is not None:
                ttnn.deallocate(host_t)
            setattr(rt, attr, None)

    def _ensure_tok_host_readback_buffers(
        self, rt: TextDecoderKvDecodeRuntime, tok_tt: tuple[ttnn.Tensor, ttnn.Tensor]
    ) -> None:
        if rt.tok_li_host is not None and rt.tok_cm_host is not None:
            return
        li_tt, cm_tt = tok_tt
        # ``from_device`` pins host memory tied to the mesh; must ``deallocate`` on trace release.
        rt.tok_li_host = ttnn.allocate_tensor_on_host(li_tt.spec, self.device)
        rt.tok_cm_host = ttnn.allocate_tensor_on_host(cm_tt.spec, self.device)

    def _start_chunk_argmax_d2h(
        self,
        tok_tt: tuple[ttnn.Tensor, ttnn.Tensor],
        *,
        cq_id: int = 0,
        blocking: bool = True,
    ) -> Optional[object]:
        """Issue D2H for trace-fused ``(local_idx, chunk_max)``; return a CQ event when ``blocking=False``."""
        rt = self._kv_decode_rt
        if rt is None:
            raise RuntimeError("KV decode runtime missing for tok_tt readback.")
        self._ensure_tok_host_readback_buffers(rt, tok_tt)
        li_tt, cm_tt = tok_tt
        ttnn.copy_device_to_host_tensor(li_tt, rt.tok_li_host, blocking=blocking, cq_id=cq_id)
        ttnn.copy_device_to_host_tensor(cm_tt, rt.tok_cm_host, blocking=blocking, cq_id=cq_id)
        if blocking:
            return None
        return ttnn.record_event(self.device, cq_id)

    def _ttnn_greedy_token_id(
        self,
        logits: ttnn.Tensor,
        dec_len: int,
        *,
        repetition_penalty: float = 1.0,
        prev_token_ids: Optional["Collection[int]"] = None,
    ) -> int:
        """Greedy next token via device ``ttnn`` ops; scalar readback only (no host ``argmax``)."""
        hf_exact = getattr(self, "_decode_hf_exact_greedy", False)
        if repetition_penalty > 1.0 or hf_exact:
            return self._host_argmax_from_logits_row(
                logits,
                dec_len,
                repetition_penalty=repetition_penalty,
                prev_token_ids=prev_token_ids,
                sharded=self._tp > 1,
            )
        rank = len(tuple(logits.shape))
        if rank == 3 and int(logits.shape[1]) == 1 and self._tp > 1:
            rt = self._kv_decode_rt
            # Never allocate fresh argmax ops on trace-backed logits while a trace is active.
            if rt is not None and rt.trace_id is not None and rt.tok_tt is not None:
                self._start_chunk_argmax_d2h(rt.tok_tt, blocking=True)
                return self._chunk_argmax_from_host_buffers(
                    logits,
                    dec_len,
                    repetition_penalty=repetition_penalty,
                    prev_token_ids=prev_token_ids,
                )
            token_tt = self._ondevice_global_argmax_token(logits)
            token_id = _read_int_scalar(token_tt)
            ttnn.deallocate(token_tt)
            return token_id
        _, next_id = self._greedy_next_token(
            logits,
            dec_len,
            repetition_penalty=repetition_penalty,
            prev_token_ids=list(prev_token_ids) if prev_token_ids else None,
        )
        return next_id

    def _host_argmax_from_logits_row(
        self,
        logits: ttnn.Tensor,
        dec_len: int,
        *,
        repetition_penalty: float = 1.0,
        prev_token_ids: Optional["Collection[int]"] = None,
        sharded: bool = False,
    ) -> int:
        """HF ``argmax`` on the full last-step logits row (gathered on host)."""
        import torch as _torch

        host = self._logits_row_to_host(logits, dec_len, sharded=sharded)
        if repetition_penalty > 1.0 and prev_token_ids:
            ids = _torch.as_tensor(list(prev_token_ids), dtype=_torch.int64)
            vocab_w = int(host.shape[-1])
            ids = ids[(ids >= 0) & (ids < vocab_w)]
            if ids.numel() > 0:
                scores = host[0, ids]
                penalized = _torch.where(
                    scores < 0, scores * float(repetition_penalty), scores / float(repetition_penalty)
                )
                host[0, ids] = penalized
        return int(host[0].argmax().item())

    def _chunk_argmax_from_host_buffers(
        self,
        logits: ttnn.Tensor,
        dec_len: int,
        *,
        repetition_penalty: float = 1.0,
        prev_token_ids: Optional["Collection[int]"] = None,
    ) -> int:
        """HF-exact combine from cached host mirrors (after :meth:`_start_chunk_argmax_d2h`)."""
        import torch as _torch

        rt = self._kv_decode_rt
        if rt is None or rt.tok_li_host is None or rt.tok_cm_host is None:
            raise RuntimeError("Chunk argmax host buffers not initialized.")
        if self._argmax_tok_mesh_composer is None:
            self._argmax_tok_mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
        nch = self._ARGMAX_CHUNKS
        nd = self._lm_num_devices
        v_loc = int(logits.shape[-1])
        local_idx = (
            ttnn.to_torch(rt.tok_li_host, mesh_composer=self._argmax_tok_mesh_composer)
            .reshape(nd, nch)
            .to(_torch.int64)
        )
        chunk_max = (
            ttnn.to_torch(rt.tok_cm_host, mesh_composer=self._argmax_tok_mesh_composer)
            .reshape(nd, nch)
            .to(_torch.float32)
        )
        token = self._global_greedy_token_from_chunk_tensors(chunk_max, local_idx, v_loc=v_loc, nch=nch)
        if not (repetition_penalty > 1.0 and prev_token_ids and token in prev_token_ids):
            return token
        host = self._logits_row_to_host(logits, dec_len, sharded=True)
        if repetition_penalty > 1.0 and prev_token_ids:
            ids = _torch.as_tensor(list(prev_token_ids), dtype=_torch.int64)
            vocab_w = int(host.shape[-1])
            ids = ids[(ids >= 0) & (ids < vocab_w)]
            if ids.numel() > 0:
                scores = host[0, ids]
                penalized = _torch.where(
                    scores < 0, scores * float(repetition_penalty), scores / float(repetition_penalty)
                )
                host[0, ids] = penalized
        return int(host[0].argmax().item())

    def _greedy_token_after_traced_step(
        self,
        logits: ttnn.Tensor,
        dec_len: int,
        *,
        repetition_penalty: float = 1.0,
        prev_token_ids: Optional["Collection[int]"] = None,
        read_cq_id: int = 0,
        d2h_event: Optional[object] = None,
    ) -> int:
        """Greedy next token after a traced step (host chunk combine or rep-penalty fallback)."""
        rt = self._kv_decode_rt
        tok_tt = rt.tok_tt if rt is not None else None
        if tok_tt is not None:
            if d2h_event is not None:
                ttnn.event_synchronize(d2h_event)
            elif read_cq_id:
                self._start_chunk_argmax_d2h(tok_tt, cq_id=read_cq_id, blocking=True)
            else:
                self._start_chunk_argmax_d2h(tok_tt, blocking=True)
            return self._chunk_argmax_from_host_buffers(
                logits,
                dec_len,
                repetition_penalty=repetition_penalty,
                prev_token_ids=prev_token_ids,
            )
        return self._ttnn_greedy_token_id(
            logits,
            dec_len,
            repetition_penalty=repetition_penalty,
            prev_token_ids=prev_token_ids,
        )

    def _encode_text(
        self,
        input_ids: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor],
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, bool]:
        """Text encoder. Returns ``(encoder_out, encoder_attn_2d_padded, attn_owned)``.

        ``attn_owned=True`` means the caller must ``ttnn.deallocate`` the returned attn tensor;
        ``False`` means it aliases the input ``attention_mask`` (no padding was needed).
        """
        batch = int(input_ids.shape[0])
        seq = int(input_ids.shape[1])
        padded_seq = tile_align(seq)

        ids_padded = pad_input_ids_to(input_ids, padded_seq, self.pad_token_id, self.device)
        if attention_mask is None:
            attn_padded = ones_mask(batch, padded_seq, self.device)
            attn_owned = True
            enc_mask_4d: Optional[ttnn.Tensor] = None
        else:
            attn_padded = pad_mask_to(attention_mask, padded_seq, self.device)
            attn_owned = attn_padded is not attention_mask
            enc_mask_4d = build_encoder_self_mask_4d(attn_padded, device=self.device)
        pos_tt = tt_position_ids(ids_padded, self.pad_token_id)

        enc_out = self.text_encoder.forward(ids_padded, pos_tt, enc_mask_4d)
        if ids_padded is not input_ids:
            ttnn.deallocate(ids_padded)
        ttnn.deallocate(pos_tt)
        if enc_mask_4d is not None:
            ttnn.deallocate(enc_mask_4d)
        return enc_out, attn_padded, attn_owned

    def _speech_attention_uint_to_conv_bf16(self, mask_2d: ttnn.Tensor) -> ttnn.Tensor:
        # Use uint32 → int32 → bf16 (direct uint32→bf16 typecast is incorrect on device).
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
        padded_len = tile_align(logical_len)

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

        # Bucket the mel-sequence length so the (shape-specialized) speech-encoder kernels are reused
        # across calls whose real length only jitters (e.g. EOS-terminated audio that varies a few
        # frames run-to-run). Without this, every distinct length triggers a full cold JIT recompile
        # (~7-20 s); padding to a coarse grid makes nearby lengths share one compiled program (warm
        # disk cache → ~1-2 s). Padded frames are masked (mask=0) so ``_subsampled_lens_dev`` (counts
        # real 1s) and the output trim in ``_speech_encoder_trim_pad_and_cross_attn`` ignore them —
        # the encoder output for the real frames is unchanged.
        real_mask = attention_mask if attention_mask is not None else ones_mask(batch, seq_in, self.device)
        bucketed = ((seq_in + _SPEECH_ENC_SEQ_BUCKET - 1) // _SPEECH_ENC_SEQ_BUCKET) * _SPEECH_ENC_SEQ_BUCKET
        owned_feats = bucketed != seq_in
        if owned_feats:
            input_features = ttnn.pad(input_features, [(0, 0), (0, bucketed - seq_in), (0, 0)], value=0.0)
            mask_2d = ttnn.pad(real_mask, [(0, 0), (0, bucketed - seq_in)], value=0)
            if attention_mask is None:
                ttnn.deallocate(real_mask)
            owned_input_mask = True
        else:
            mask_2d = real_mask
            owned_input_mask = attention_mask is None

        mask_bf16_tile = self._speech_attention_uint_to_conv_bf16(mask_2d)
        enc_raw = self.speech_encoder.forward(input_features, conv_attention_mask_1d=mask_bf16_tile)
        ttnn.deallocate(mask_bf16_tile)
        if owned_feats:
            ttnn.deallocate(input_features)

        sub_lens_tt = _subsampled_lens_dev(mask_2d, self.adaptor_kernel_size, self.adaptor_stride)
        if owned_input_mask:
            ttnn.deallocate(mask_2d)

        enc_out, enc_attn_tt = self._speech_encoder_trim_pad_and_cross_attn(enc_raw, sub_lens_tt, batch)
        return enc_out, enc_attn_tt, True

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
        padded_dec_seq = tile_align(dec_seq)

        ids_padded = pad_input_ids_to(decoder_input_ids, padded_dec_seq, self.pad_token_id, self.device)

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
            # Decoder mask for ``build_causal_with_padding_4d``: an all-ones 2-D mask adds **zero** padding
            # to the causal 4-D mask (HF semantics). Skip allocating ``ones_mask`` + ``key_padding_additive``
            # when the caller omits ``decoder_attention_mask`` — use the ``None`` fast path instead.
            if decoder_attention_mask is None:
                dec_attn_padded = None
                attn_owned = False
            else:
                dec_attn_padded = pad_mask_to(decoder_attention_mask, padded_dec_seq, self.device)
                attn_owned = dec_attn_padded is not decoder_attention_mask

            causal_4d = build_causal_with_padding_4d(dec_attn_padded, batch, padded_dec_seq, self.device)
            if attn_owned:
                ttnn.deallocate(dec_attn_padded)
            cross_4d = build_cross_attn_mask_4d(encoder_attn_2d, tgt_seq=padded_dec_seq, device=self.device)

        pos_tt = tt_position_ids(ids_padded, self.pad_token_id)

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

    def _fixed_decode_trace_sdpa_len(self) -> int:
        """Single SDPA bucket for Metal decode trace (max 256; see ``_next_power_of_2_cap256``)."""
        return self.text_decoder.decode_trace_cache_seq_len(self.max_text_seq_len)

    def _release_all_active_decode_traces(self) -> None:
        """Release every captured decode trace (required before re-capture on BH)."""
        if self._kv_decode_rt is None:
            return
        rt = self._kv_decode_rt
        for bucket in list(rt.trace_ids_by_cache_seq_len.keys()):
            self._release_decode_trace_bucket(bucket)
        ttnn.synchronize_device(self.device)

    def _release_decode_trace_bucket(self, cache_seq_len: int) -> None:
        """Release one SDPA bucket trace and its logits output tensor."""
        if self._kv_decode_rt is None:
            return
        rt = self._kv_decode_rt
        bucket = int(cache_seq_len)
        trace_id = rt.trace_ids_by_cache_seq_len.pop(bucket, None)
        if trace_id is not None:
            ttnn.release_trace(self.device, trace_id)
        old_logits = rt.logits_tt_by_cache_seq_len.pop(bucket, None)
        if old_logits is not None:
            ttnn.deallocate(old_logits)
        old_tok = rt.tok_tt_by_cache_seq_len.pop(bucket, None)
        if old_tok is not None:
            for _t in old_tok:
                ttnn.deallocate(_t)
        old_dec = rt.last_dec_out_tt_by_cache_seq_len.pop(bucket, None)
        if old_dec is not None:
            ttnn.deallocate(old_dec)
        if rt.trace_cache_seq_len == bucket:
            rt.trace_id = None
            rt.trace_cache_seq_len = None
        if rt.logits_tt is old_logits:
            rt.logits_tt = None
        if rt.tok_tt is old_tok:
            rt.tok_tt = None

    def release_text_decoder_decode_trace(self) -> None:
        """Release captured decode traces and trace output tensors (keeps H2D staging buffers)."""
        if self._kv_decode_rt is None:
            return
        rt = self._kv_decode_rt
        for trace_id in list(rt.trace_ids_by_cache_seq_len.values()):
            ttnn.release_trace(self.device, trace_id)
        rt.trace_ids_by_cache_seq_len.clear()
        for logits in rt.logits_tt_by_cache_seq_len.values():
            ttnn.deallocate(logits)
        rt.logits_tt_by_cache_seq_len.clear()
        rt.logits_tt = None
        for tok in rt.tok_tt_by_cache_seq_len.values():
            for _t in tok:
                ttnn.deallocate(_t)
        rt.tok_tt_by_cache_seq_len.clear()
        rt.tok_tt = None
        for dec_out in rt.last_dec_out_tt_by_cache_seq_len.values():
            ttnn.deallocate(dec_out)
        rt.last_dec_out_tt_by_cache_seq_len.clear()
        self._free_kv_host_readback_buffers(rt)
        rt.trace_id = None
        rt.trace_cache_seq_len = None

    def _release_kv_decode_runtime(self) -> None:
        if self._kv_decode_rt is not None:
            self._free_kv_host_readback_buffers(self._kv_decode_rt)
        self.release_text_decoder_decode_trace()
        self._kv_decode_rt = None
        self._decode_trace_kernels_warmed = False

    def _ensure_kv_decode_runtime(self, batch_size: int) -> TextDecoderKvDecodeRuntime:
        if self._kv_decode_rt is not None and self._kv_decode_rt.batch_size == batch_size:
            return self._kv_decode_rt
        self._release_kv_decode_runtime()

        token_tt = ttnn.from_torch(
            torch.zeros((batch_size, 1), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pos_tt = ttnn.from_torch(
            torch.zeros((batch_size, 1), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cur_pos_tt = ttnn.from_torch(
            torch.zeros((batch_size,), dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._kv_decode_rt = TextDecoderKvDecodeRuntime(
            batch_size=batch_size,
            token_tt=token_tt,
            pos_tt=pos_tt,
            cur_pos_tt=cur_pos_tt,
        )
        return self._kv_decode_rt

    def _decode_h2d_staging(self, batch_size: int) -> dict:
        """Reusable host staging buffers for the per-step decode H2D uploads — filled in place each step
        (``fill_``) instead of re-allocating ``torch.tensor`` lists every step."""
        st = self._decode_h2d_cache.get(batch_size)
        if st is None:
            st = {
                "tok_cpu": torch.zeros((batch_size, 1), dtype=torch.int32),
                "pos_cpu": torch.zeros((batch_size, 1), dtype=torch.int32),
                "curpos_cpu": torch.zeros((batch_size,), dtype=torch.int32),
            }
            self._decode_h2d_cache[batch_size] = st
        return st

    def _reset_kv_decode_cur_pos(self, position: int, batch_size: int, *, cq_id: int = 0) -> None:
        rt = self._ensure_kv_decode_runtime(batch_size)
        st = self._decode_h2d_staging(batch_size)
        st["curpos_cpu"].fill_(position)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(st["curpos_cpu"], dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
            rt.cur_pos_tt,
            cq_id=cq_id,
        )

    def _upload_single_token_to_decode_rt(self, token_id: int, batch_size: int, *, cq_id: int = 0) -> None:
        """Upload one greedy-decode token id into the pre-allocated ``[B, 1]`` decode buffer."""
        rt = self._ensure_kv_decode_runtime(batch_size)
        st = self._decode_h2d_staging(batch_size)
        st["tok_cpu"].fill_(int(token_id))
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(st["tok_cpu"], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            rt.token_tt,
            cq_id=cq_id,
        )

    def _upload_kv_decode_step_inputs(
        self,
        token_id: int,
        position: int,
        batch_size: int,
        *,
        cq_id: int = 0,
    ) -> None:
        """Upload decode token + position (host-only; safe while a decode trace is active)."""
        rt = self._ensure_kv_decode_runtime(batch_size)
        st = self._decode_h2d_staging(batch_size)
        st["tok_cpu"].fill_(int(token_id))
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(st["tok_cpu"], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            rt.token_tt,
            cq_id=cq_id,
        )
        # HF ``create_position_ids_from_input_ids`` for a non-pad ``[B,1]`` decode token:
        # ``(cumsum(mask) + position) * mask + pad_id`` → ``1 + position + pad_id``.
        pos_val = 1 + position + self.pad_token_id
        st["pos_cpu"].fill_(pos_val)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(st["pos_cpu"], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            rt.pos_tt,
            cq_id=cq_id,
        )

    def _upload_kv_decode_step_inputs_cq1(
        self,
        token_id: int,
        position: int,
        batch_size: int,
    ) -> None:
        """Upload decode token + position on CQ1 (for 2CQ pipelined decode)."""
        self._upload_kv_decode_step_inputs(token_id, position, batch_size, cq_id=1)

    def _upload_pos_only(self, position: int, batch_size: int, *, cq_id: int = 0) -> None:
        """Upload only ``pos_tt`` (self-feed decode: the token is written on-device by the trace, but the
        position counter is token-independent and stays host-driven). ``cur_pos_tt`` via ``_reset_kv_decode_cur_pos``.
        """
        rt = self._ensure_kv_decode_runtime(batch_size)
        st = self._decode_h2d_staging(batch_size)
        pos_val = 1 + position + self.pad_token_id
        st["pos_cpu"].fill_(pos_val)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(st["pos_cpu"], dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
            rt.pos_tt,
            cq_id=cq_id,
        )

    def _cached_t2u_attention_mask(self, real_len: int, padded_dec_seq: int) -> ttnn.Tensor:
        key = (int(real_len), int(padded_dec_seq))
        cached = self._t2u_attn_mask_cache.get(key)
        if cached is not None:
            return cached
        mask = _t2u_attention_mask_uncached(real_len, padded_dec_seq, self.device)
        self._t2u_attn_mask_cache[key] = mask
        return mask

    def _forward_decode_kv_lm(
        self,
        rt: TextDecoderKvDecodeRuntime,
        encoder_hidden: ttnn.Tensor,
        cross_4d: ttnn.Tensor,
        kv_cache: list,
        cross_attn_cache: list,
        *,
        cache_seq_len: int,
    ) -> ttnn.Tensor:
        """Decoder forward + ``lm_head`` on pre-allocated decode tensors."""
        dec_out = self.text_decoder.forward(
            rt.token_tt,
            rt.pos_tt,
            encoder_hidden,
            None,
            cross_4d,
            kv_cache=kv_cache,
            cross_attn_cache=cross_attn_cache,
            cross_attn_cache_valid=True,
            current_decode_pos=rt.cur_pos_tt,
            cache_seq_len=cache_seq_len,
            trace_no_profiler=True,
        )
        logits = self._lm_head(dec_out)
        ttnn.deallocate(dec_out)
        return logits

    def _decode_trace_bucket(self, position: int) -> int:
        """SDPA bucket for decode trace at ``position`` (matches eager ``cache_seq_len=position+1``)."""
        return self.text_decoder.decode_trace_cache_seq_len(int(position) + 1)

    def _select_decode_trace_for_position(self, position: int, batch_size: int) -> Optional[int]:
        """Return trace id for the SDPA bucket at ``position``, or None if not captured yet."""
        rt = self._ensure_kv_decode_runtime(batch_size)
        cache_len = self._decode_trace_bucket(position)
        trace_id = rt.trace_ids_by_cache_seq_len.get(cache_len)
        if trace_id is None:
            return None
        rt.trace_id = trace_id
        rt.trace_cache_seq_len = cache_len
        rt.logits_tt = rt.logits_tt_by_cache_seq_len.get(cache_len)
        rt.tok_tt = rt.tok_tt_by_cache_seq_len.get(cache_len)
        return trace_id

    def _decode_step_kv_lm(
        self,
        rt: TextDecoderKvDecodeRuntime,
        encoder_hidden: ttnn.Tensor,
        cross_4d: ttnn.Tensor,
        kv_cache: list,
        cross_attn_cache: list,
        *,
        cache_seq_len: int,
        stash_speech_hidden: bool = True,
        retain_dec_out: bool = False,
    ) -> ttnn.Tensor:
        """One KV decode step + width-sharded ``lm_head`` → ``[1, 1, V/tp]`` logits *per device*."""
        dec_out = self.text_decoder.forward(
            rt.token_tt,
            rt.pos_tt,
            encoder_hidden,
            None,
            cross_4d,
            kv_cache=kv_cache,
            cross_attn_cache=cross_attn_cache,
            cross_attn_cache_valid=True,
            current_decode_pos=rt.cur_pos_tt,
            cache_seq_len=cache_seq_len,
            trace_no_profiler=True,
        )
        if stash_speech_hidden:
            self._stash_speech_hidden_step(dec_out)
        logits = self._lm_head_sharded(dec_out)
        if retain_dec_out:
            rt.last_dec_out_tt_by_cache_seq_len[int(cache_seq_len)] = dec_out
        else:
            ttnn.deallocate(dec_out)
        return logits

    def capture_text_decoder_decode_trace(
        self,
        token_id: int,
        position: int,
        encoder_hidden: ttnn.Tensor,
        cross_4d: ttnn.Tensor,
        kv_cache: list,
        cross_attn_cache: list,
        *,
        batch_size: int = 1,
        cache_seq_len: Optional[int] = None,
    ) -> None:
        """Capture Metal trace for one KV decode step (decoder + lm_head [+ optional sampling])."""
        rt = self._ensure_kv_decode_runtime(batch_size)
        config_cache_len = int(cache_seq_len) if cache_seq_len is not None else self._fixed_decode_trace_sdpa_len()
        if config_cache_len in rt.trace_ids_by_cache_seq_len:
            return
        use_sampling = self._use_on_device_sampling()
        fold_sampling = self._fold_sampling_in_trace
        self._release_all_active_decode_traces()
        ttnn.synchronize_device(self.device)

        self._upload_kv_decode_step_inputs(token_id, position, batch_size)
        self._reset_kv_decode_cur_pos(position, batch_size)
        ttnn.synchronize_device(self.device)

        # Compile outside trace (``tt_transformers`` / Llama pattern). JIT during capture
        # issues host→device writes forbidden while ``trace_id`` is active.
        compile_logits = self._decode_step_kv_lm(
            rt,
            encoder_hidden,
            cross_4d,
            kv_cache,
            cross_attn_cache,
            cache_seq_len=config_cache_len,
            stash_speech_hidden=False,
        )
        if use_sampling:
            row = prepare_row_logits_for_sampling(compile_logits, dec_len=1)
            self._on_device_sampler.sample_into_buffer(row)
            if row is not compile_logits:
                ttnn.deallocate(row)
        else:
            compile_tok = self._decode_argmax_token(compile_logits)
            ttnn.synchronize_device(self.device)
            for _t in compile_tok:
                ttnn.deallocate(_t)
        ttnn.deallocate(compile_logits)
        ttnn.synchronize_device(self.device)

        self._upload_kv_decode_step_inputs(token_id, position, batch_size)
        self._reset_kv_decode_cur_pos(position, batch_size)
        ttnn.synchronize_device(self.device)
        self._decode_trace_kernels_warmed = True

        capture_logits: Optional[ttnn.Tensor] = None
        capture_tok: Optional[tuple[ttnn.Tensor, ttnn.Tensor]] = None

        def traced_step_capture() -> None:
            nonlocal capture_logits, capture_tok
            capture_logits = self._decode_step_kv_lm(
                rt,
                encoder_hidden,
                cross_4d,
                kv_cache,
                cross_attn_cache,
                cache_seq_len=config_cache_len,
                stash_speech_hidden=False,
                retain_dec_out=True,
            )
            if use_sampling:
                if fold_sampling:
                    row = prepare_row_logits_for_sampling(capture_logits, dec_len=1)
                    self._on_device_sampler.sample_into_buffer(row)
            else:
                capture_tok = self._decode_argmax_token(capture_logits)

        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        traced_step_capture()
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        if capture_logits is None:
            raise RuntimeError("Decode trace capture produced no logits tensor.")
        if not use_sampling and capture_tok is None:
            raise RuntimeError("Decode trace capture produced no logits/token tensor.")
        rt.logits_tt_by_cache_seq_len[config_cache_len] = capture_logits
        rt.logits_tt = capture_logits
        if use_sampling:
            rt.tok_tt_by_cache_seq_len.pop(config_cache_len, None)
            rt.tok_tt = None
        else:
            rt.tok_tt_by_cache_seq_len[config_cache_len] = capture_tok
            rt.tok_tt = capture_tok
            self._free_kv_host_readback_buffers(rt)
            self._ensure_tok_host_readback_buffers(rt, capture_tok)
        rt.trace_ids_by_cache_seq_len[config_cache_len] = trace_id
        rt.trace_id = trace_id
        rt.trace_cache_seq_len = config_cache_len
        ttnn.synchronize_device(self.device)

    def execute_text_decoder_decode_trace(self, *, cache_seq_len: Optional[int] = None) -> ttnn.Tensor:
        """Replay a captured decode trace; caller must upload inputs and reset ``cur_pos_tt`` first."""
        if self._kv_decode_rt is None:
            raise RuntimeError("Decode trace not captured; call capture_text_decoder_decode_trace first.")
        rt = self._kv_decode_rt
        bucket = int(cache_seq_len) if cache_seq_len is not None else rt.trace_cache_seq_len
        if bucket is None:
            raise RuntimeError("Decode trace bucket unknown; call capture_text_decoder_decode_trace first.")
        trace_id = rt.trace_ids_by_cache_seq_len.get(bucket)
        if trace_id is None:
            raise RuntimeError(f"Decode trace not captured for bucket {bucket}.")
        rt.trace_id = trace_id
        rt.trace_cache_seq_len = bucket
        logits_tt = rt.logits_tt_by_cache_seq_len.get(bucket)
        if logits_tt is None:
            raise RuntimeError("Decode trace logits buffer missing; re-capture the trace.")
        rt.logits_tt = logits_tt
        rt.tok_tt = rt.tok_tt_by_cache_seq_len.get(bucket)
        ttnn.execute_trace(self.device, trace_id, cq_id=0, blocking=True)
        return logits_tt

    def _decode_token_with_kv_cache_traced(
        self,
        token_id: int,
        position: int,
        encoder_hidden: ttnn.Tensor,
        cross_4d: ttnn.Tensor,
        kv_cache: list,
        cross_attn_cache: list,
        *,
        batch_size: int = 1,
    ) -> ttnn.Tensor:
        ttnn.synchronize_device(self.device)
        cache_seq_len = self._decode_trace_bucket(position)
        rt = self._ensure_kv_decode_runtime(batch_size)
        need_capture = cache_seq_len not in rt.trace_ids_by_cache_seq_len
        if need_capture:
            self.capture_text_decoder_decode_trace(
                token_id,
                position,
                encoder_hidden,
                cross_4d,
                kv_cache,
                cross_attn_cache,
                batch_size=batch_size,
                cache_seq_len=cache_seq_len,
            )
            # Capture's traced_step already ran one decode + lm_head (single KV update).
            return rt.logits_tt
        self._upload_kv_decode_step_inputs(token_id, position, batch_size)
        self._reset_kv_decode_cur_pos(position, batch_size)
        return self.execute_text_decoder_decode_trace(cache_seq_len=cache_seq_len)

    def _decode_loop_selffed(
        self,
        enc_tt: ttnn.Tensor,
        cross_4d: ttnn.Tensor,
        kv_cache: list,
        cross_attn_cache: list,
        seq_host: list,
        cur_pos: int,
        max_steps: int,
        eos_ids: set,
        batch_size: int,
        *,
        use_2cq: bool = False,
        repetition_penalty: float = 1.0,
        prev_token_ids: Optional["Collection[int]"] = None,
        phase_timer: Optional[_GeneratePhaseTimer] = None,
    ) -> list:
        """Traced greedy decode on TP>1: trace replay, chunk D2H, host argmax combine, token upload."""
        rt = self._ensure_kv_decode_runtime(batch_size)
        dev = self.device
        prev_set = prev_token_ids if prev_token_ids is not None else set(seq_host)
        h2d_cq = 1 if use_2cq else 0

        cur_tok = int(seq_host[cur_pos])
        t_step = phase_timer.begin_step() if phase_timer is not None else None
        cache_seq_len = self._decode_trace_bucket(cur_pos)
        if cache_seq_len not in rt.trace_ids_by_cache_seq_len:
            self.capture_text_decoder_decode_trace(
                cur_tok,
                cur_pos,
                enc_tt,
                cross_4d,
                kv_cache,
                cross_attn_cache,
                batch_size=batch_size,
                cache_seq_len=cache_seq_len,
            )
        else:
            rt.trace_id = rt.trace_ids_by_cache_seq_len[cache_seq_len]
            rt.trace_cache_seq_len = cache_seq_len
            rt.logits_tt = rt.logits_tt_by_cache_seq_len.get(cache_seq_len)
            rt.tok_tt = rt.tok_tt_by_cache_seq_len.get(cache_seq_len)
            self._upload_kv_decode_step_inputs(cur_tok, cur_pos, batch_size, cq_id=h2d_cq)
            self._reset_kv_decode_cur_pos(cur_pos, batch_size, cq_id=h2d_cq)
            if use_2cq:
                write_event = ttnn.record_event(dev, 1)
                ttnn.wait_for_event(0, write_event)
            ttnn.execute_trace(dev, rt.trace_id, cq_id=0, blocking=True)

        if self._use_on_device_sampling():
            next_id = self._read_token_from_traced_logits(rt.logits_tt, dec_len=1)
        else:
            next_id = self._greedy_token_after_traced_step(
                rt.logits_tt,
                1,
                repetition_penalty=repetition_penalty,
                prev_token_ids=prev_set,
                read_cq_id=0,
            )
        if phase_timer is not None:
            phase_timer.end_step(t_step)
        prev_set.add(next_id)
        seq_host.append(next_id)
        cur_pos += 1
        if eos_ids and next_id in eos_ids:
            return seq_host

        self._upload_single_token_to_decode_rt(next_id, batch_size, cq_id=h2d_cq)
        cq0_idle = ttnn.record_event(dev, 0) if use_2cq else None

        for _ in range(max_steps - 1):
            d2h_event = None
            t_step = phase_timer.begin_step() if phase_timer is not None else None
            cache_seq_len = self._decode_trace_bucket(cur_pos)
            if use_2cq:
                if cq0_idle is not None:
                    ttnn.wait_for_event(1, cq0_idle)
                self._upload_pos_only(cur_pos, batch_size, cq_id=1)
                self._reset_kv_decode_cur_pos(cur_pos, batch_size, cq_id=1)
                write_event = ttnn.record_event(dev, 1)
                ttnn.wait_for_event(0, write_event)
                if cache_seq_len not in rt.trace_ids_by_cache_seq_len:
                    self.capture_text_decoder_decode_trace(
                        int(seq_host[cur_pos]),
                        cur_pos,
                        enc_tt,
                        cross_4d,
                        kv_cache,
                        cross_attn_cache,
                        batch_size=batch_size,
                        cache_seq_len=cache_seq_len,
                    )
                else:
                    rt.trace_id = rt.trace_ids_by_cache_seq_len[cache_seq_len]
                    rt.trace_cache_seq_len = cache_seq_len
                    rt.logits_tt = rt.logits_tt_by_cache_seq_len.get(cache_seq_len)
                    rt.tok_tt = rt.tok_tt_by_cache_seq_len.get(cache_seq_len)
                ttnn.execute_trace(dev, rt.trace_id, cq_id=0, blocking=True)
                cq0_idle = ttnn.record_event(dev, 0)
                if not self._use_on_device_sampling():
                    d2h_event = self._start_chunk_argmax_d2h(rt.tok_tt, cq_id=1, blocking=False)
            else:
                self._upload_pos_only(cur_pos, batch_size)
                self._reset_kv_decode_cur_pos(cur_pos, batch_size)
                if cache_seq_len not in rt.trace_ids_by_cache_seq_len:
                    self.capture_text_decoder_decode_trace(
                        int(seq_host[cur_pos]),
                        cur_pos,
                        enc_tt,
                        cross_4d,
                        kv_cache,
                        cross_attn_cache,
                        batch_size=batch_size,
                        cache_seq_len=cache_seq_len,
                    )
                else:
                    rt.trace_id = rt.trace_ids_by_cache_seq_len[cache_seq_len]
                    rt.trace_cache_seq_len = cache_seq_len
                    rt.logits_tt = rt.logits_tt_by_cache_seq_len.get(cache_seq_len)
                    rt.tok_tt = rt.tok_tt_by_cache_seq_len.get(cache_seq_len)
                ttnn.execute_trace(dev, rt.trace_id, cq_id=0, blocking=True)

            if self._use_on_device_sampling():
                next_id = self._read_token_from_traced_logits(rt.logits_tt, dec_len=1)
            else:
                next_id = self._greedy_token_after_traced_step(
                    rt.logits_tt,
                    1,
                    repetition_penalty=repetition_penalty,
                    prev_token_ids=prev_set,
                    read_cq_id=0,
                    d2h_event=d2h_event if use_2cq else None,
                )
            if phase_timer is not None:
                phase_timer.end_step(t_step)
            prev_set.add(next_id)
            seq_host.append(next_id)
            cur_pos += 1
            if eos_ids and next_id in eos_ids:
                break
            self._upload_single_token_to_decode_rt(next_id, batch_size, cq_id=h2d_cq)

        return seq_host

    def _prefill_text_decoder_kv_cache(
        self,
        input_ids_tt: ttnn.Tensor,
        encoder_hidden: ttnn.Tensor,
        encoder_attn_2d: ttnn.Tensor,
        kv_cache: list,
        cross_attn_cache: list,
        cross_4d: Optional[ttnn.Tensor] = None,
    ) -> Optional[ttnn.Tensor]:
        """Populate KV caches with one batched prefill forward (``slice_write`` self + cross copy).

        Returns decoder hidden states ``[B, S, H]`` (caller must deallocate), or ``None`` if ``seq==0``.
        """
        seq = int(input_ids_tt.shape[1])
        if seq == 0:
            return None
        batch_size = int(input_ids_tt.shape[0])
        padded_seq = tile_align(seq)
        ids_padded = pad_input_ids_to(input_ids_tt, padded_seq, self.pad_token_id, self.device)
        attn_2d = ones_mask(batch_size, padded_seq, self.device)
        pos_tt = tt_position_ids(ids_padded, self.pad_token_id)
        causal_4d = build_causal_with_padding_4d(attn_2d, batch_size, padded_seq, self.device)
        ttnn.deallocate(attn_2d)

        owns_cross_4d = cross_4d is None
        if cross_4d is None:
            cross_4d = build_cross_attn_mask_4d(encoder_attn_2d, tgt_seq=padded_seq, device=self.device)

        dec_out = warm_text_decoder_kv_cache_prefill(
            self.text_decoder,
            ids_padded,
            pos_tt,
            encoder_hidden,
            causal_4d,
            cross_4d,
            kv_cache,
            cross_attn_cache,
            kv_cache_fill_len=seq,
            trace_no_profiler=True,
        )

        if ids_padded is not input_ids_tt:
            ttnn.deallocate(ids_padded)
        ttnn.deallocate(pos_tt)
        ttnn.deallocate(causal_4d)
        if owns_cross_4d:
            ttnn.deallocate(cross_4d)
        return dec_out

    def _single_token_tt(self, token_id: int, batch_size: int) -> ttnn.Tensor:
        """``[B, 1]`` uint32 on device for one greedy-decode step (tiny H2D)."""
        tok_cpu = torch.tensor([[int(token_id)]] * batch_size, dtype=torch.int32)
        return ttnn.from_torch(
            tok_cpu,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _decode_token_with_kv_cache(
        self,
        token_id: int,
        position: int,
        encoder_hidden: ttnn.Tensor,
        encoder_attn_2d: ttnn.Tensor,
        kv_cache: list,
        cross_attn_cache: list,
        cross_attn_cache_valid: bool,
        cross_4d: Optional[ttnn.Tensor] = None,
        *,
        batch_size: int = 1,
        sharded_logits: bool = False,
    ) -> ttnn.Tensor:
        """Single decoder step with KV cache → ``[B, 1, V]`` or sharded ``[B, 1, V/tp]`` logits."""
        token_ids = self._single_token_tt(token_id, batch_size)
        pos_tt = tt_position_ids_decode_step(token_ids, self.pad_token_id, position)
        owns_cross_4d = cross_4d is None
        if cross_4d is None:
            cross_4d = build_cross_attn_mask_4d(encoder_attn_2d, tgt_seq=1, device=self.device)
        cur_pos = self.text_decoder.borrow_current_decode_pos_tensor(position, batch_size=batch_size)
        dec_out = self.text_decoder.forward(
            token_ids,
            pos_tt,
            encoder_hidden,
            None,
            cross_4d,
            kv_cache=kv_cache,
            cross_attn_cache=cross_attn_cache,
            cross_attn_cache_valid=cross_attn_cache_valid,
            current_decode_pos=cur_pos,
            cache_seq_len=position + 1,
            trace_no_profiler=True,
        )
        self._stash_speech_hidden_step(dec_out)
        if sharded_logits:
            logits = self._lm_head_sharded(dec_out)
        else:
            logits = self._lm_head(dec_out)
        ttnn.deallocate(dec_out)
        ttnn.deallocate(pos_tt)
        ttnn.deallocate(token_ids)
        if owns_cross_4d:
            ttnn.deallocate(cross_4d)
        return logits

    def _decoder_hidden(
        self,
        encoder_hidden: ttnn.Tensor,
        encoder_attn_2d: ttnn.Tensor,
        decoder_input_ids: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, int]:
        """text-decoder → ``(decoder_hidden_states [B, padded_dec_seq, H], padded_dec_seq)`` (no lm_head)."""
        batch = int(decoder_input_ids.shape[0])
        dec_seq = int(decoder_input_ids.shape[1])
        padded_dec_seq = tile_align(dec_seq)

        ids_padded = pad_input_ids_to(decoder_input_ids, padded_dec_seq, self.pad_token_id, self.device)

        # Build padded key mask: real positions (1) at non-pad tokens, 0 elsewhere.
        # ``ttnn.ne(ids, pad_id)`` gives a bool mask of real positions; typecast to uint32.
        ne_pad = ttnn.ne(ids_padded, self.pad_token_id)
        attn_2d = ttnn.typecast(ne_pad, ttnn.uint32)
        ttnn.deallocate(ne_pad)
        if attn_2d.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            attn_2d_rm = ttnn.to_layout(attn_2d, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn_2d)
            attn_2d = attn_2d_rm

        pos_tt = tt_position_ids(ids_padded, self.pad_token_id)
        causal_4d = build_causal_with_padding_4d(attn_2d, batch, padded_dec_seq, self.device)
        ttnn.deallocate(attn_2d)
        cross_4d = build_cross_attn_mask_4d(encoder_attn_2d, tgt_seq=padded_dec_seq, device=self.device)

        dec_out = self.text_decoder.forward(ids_padded, pos_tt, encoder_hidden, causal_4d, cross_4d)
        if ids_padded is not decoder_input_ids:
            ttnn.deallocate(ids_padded)
        ttnn.deallocate(pos_tt)
        ttnn.deallocate(causal_4d)
        ttnn.deallocate(cross_4d)
        return dec_out, padded_dec_seq

    def _logits_to_host(self, logits: ttnn.Tensor, dec_idx: int) -> "torch.Tensor":
        """Slice ``[B, S, V]`` at ``dec_idx`` and read back as fp32 torch tensor ``[V]``.

        Used by beam search to compute log-softmax + scores on host.
        """
        import torch as _torch

        batch = int(logits.shape[0])
        vocab_w = int(logits.shape[2])
        last = ttnn.slice(logits, [0, dec_idx, 0], [batch, dec_idx + 1, vocab_w], (1, 1, 1))
        host = to_torch_replicated_first_shard(last).to(_torch.float32).reshape(batch, vocab_w)
        ttnn.deallocate(last)
        return host[0]

    def _generate_beam_kv(
        self,
        enc_tt: ttnn.Tensor,
        enc_attn_tt: ttnn.Tensor,
        seed_tokens: List[int],
        *,
        num_beams: int,
        max_new_tokens: int,
        eos_ids: set,
        repetition_penalty: float,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
    ) -> List[int]:
        """K-beam search over the text decoder + KV cache.

        Allocates ``num_beams`` independent KV caches, runs decoder forward once per beam per
        step, scores candidates host-side, and re-orders caches via ``ttnn.copy`` when a beam's
        parent changes. Returns the best beam's full token sequence (seed + decoded tokens),
        ranked by ``score / length^length_penalty``. ``early_stopping=True`` matches HF.

        Memory: ``num_beams * (24 layers × 2 × bf16 K-cache + cross cache)`` of DRAM.
        Compute: ``num_beams ×`` the greedy decode cost (sequential per-beam forwards).
        """
        import torch as _torch

        device = self.device
        K = int(num_beams)
        if K < 1:
            raise ValueError(f"num_beams must be >= 1 (got {num_beams})")
        seed_len = len(seed_tokens)
        if seed_len + max_new_tokens > self.max_text_seq_len:
            raise ValueError(
                f"seed_len ({seed_len}) + max_new_tokens ({max_new_tokens}) exceeds "
                f"max_text_seq_len ({self.max_text_seq_len})"
            )
        enc_seq = int(enc_tt.shape[1])

        # Allocate K independent KV caches. Each beam runs the full decoder graph against its
        # own cache; per-step we'll ``ttnn.copy`` parent K/V into child slots when beams shuffle.
        beam_kv: List[list] = []
        beam_cross: List[list] = []
        for _ in range(K):
            kv, cross = init_text_decoder_kv_cache(
                device,
                num_hidden_layers=self.text_decoder.num_hidden_layers,
                num_attention_heads=self.decoder_attention_heads,
                hidden_size=self.hidden_size,
                max_batch_size=1,
                max_seq_len=self.max_text_seq_len,
                encoder_seq_len=enc_seq,
                tp=self._tp,
            )
            beam_kv.append(kv)
            beam_cross.append(cross)
        decode_cross_4d = build_cross_attn_mask_4d(enc_attn_tt, tgt_seq=1, device=device)

        # Prefill once on beam 0, then ``ttnn.copy`` its KV / cross caches into beams 1..K-1.
        # Same input would produce the same K/V; copy is much cheaper than redundant prefills.
        warm_tt0 = _ttnn_ids_from_list([seed_tokens], device)
        warm_out0 = self._prefill_text_decoder_kv_cache(warm_tt0, enc_tt, enc_attn_tt, beam_kv[0], beam_cross[0])
        ttnn.deallocate(warm_tt0)
        for k in range(1, K):
            for layer in range(self.text_decoder.num_hidden_layers):
                ttnn.copy(beam_kv[0][layer][0], beam_kv[k][layer][0])
                ttnn.copy(beam_kv[0][layer][1], beam_kv[k][layer][1])
                ttnn.copy(beam_cross[0][layer][0], beam_cross[k][layer][0])
                ttnn.copy(beam_cross[0][layer][1], beam_cross[k][layer][1])

        # Score the seed prefix → top-K initial tokens so beams diverge from step 0.
        if warm_out0 is None:
            seed_logits = None
        else:
            seed_logits_tt = self._lm_head(warm_out0)
            ttnn.deallocate(warm_out0)
            seed_logits = self._logits_to_host(seed_logits_tt, seed_len - 1)
            ttnn.deallocate(seed_logits_tt)

        # ``_prefill_text_decoder_kv_cache`` may or may not return the prefill hidden state — fall
        # back to a no-op token if it returned ``None`` (this matches the greedy path).
        if seed_logits is None:
            seed_logp = _torch.zeros(K)
            init_tokens = [seed_tokens[-1]] * K
        else:
            log_probs0 = _torch.log_softmax(seed_logits, dim=-1)
            if repetition_penalty > 1.0:
                ids = _torch.as_tensor([t for t in seed_tokens if 0 <= t < log_probs0.numel()], dtype=_torch.int64)
                if ids.numel() > 0:
                    scores = log_probs0[ids]
                    log_probs0[ids] = _torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
            topk = _torch.topk(log_probs0, k=K)
            seed_logp = topk.values
            init_tokens = [int(t) for t in topk.indices.tolist()]

        seqs: List[List[int]] = [list(seed_tokens) + [t] for t in init_tokens]
        scores: List[float] = [float(s) for s in seed_logp.tolist()]
        finished: List[bool] = [tok in eos_ids for tok in init_tokens]
        cur_pos = seed_len  # next decode writes at cur_pos+1

        for _step in range(max_new_tokens - 1):
            if all(finished):
                break

            # Per-beam decode at cur_pos (the token we just emitted).
            beam_log_probs: List["torch.Tensor"] = []
            for k in range(K):
                if finished[k]:
                    beam_log_probs.append(None)  # type: ignore[arg-type]
                    continue
                logits = self._decode_token_with_kv_cache(
                    int(seqs[k][cur_pos]),
                    cur_pos,
                    enc_tt,
                    enc_attn_tt,
                    beam_kv[k],
                    beam_cross[k],
                    True,
                    cross_4d=decode_cross_4d,
                    batch_size=1,
                )
                lp_host = self._logits_to_host(logits, 0)
                ttnn.deallocate(logits)
                lp = _torch.log_softmax(lp_host, dim=-1)
                if repetition_penalty > 1.0:
                    ids = _torch.as_tensor([t for t in seqs[k] if 0 <= t < lp.numel()], dtype=_torch.int64)
                    if ids.numel() > 0:
                        s_at = lp[ids]
                        lp[ids] = _torch.where(s_at < 0, s_at * repetition_penalty, s_at / repetition_penalty)
                beam_log_probs.append(lp)

            # Build candidates: for each live beam, score = beam_score + log_p(token).
            # Pick top-K tokens per beam (K*K total), then global top-K of those.
            candidates: List[Tuple[float, int, int]] = []  # (score, parent_beam, token_id)
            for k in range(K):
                if finished[k]:
                    candidates.append((scores[k], k, -1))  # frozen
                    continue
                topk = _torch.topk(beam_log_probs[k], k=K)
                for j in range(K):
                    tok = int(topk.indices[j].item())
                    candidates.append((scores[k] + float(topk.values[j].item()), k, tok))
            candidates.sort(key=lambda c: c[0], reverse=True)
            new_beams = candidates[:K]

            # Apply the chosen beams. If a beam's parent != its slot index, copy the parent's KV
            # caches (self-attn + cross-attn) into this slot so the next decode sees the right K/V.
            new_seqs: List[List[int]] = [None] * K  # type: ignore[list-item]
            new_scores: List[float] = [0.0] * K
            new_finished: List[bool] = [False] * K
            copy_plan: List[Tuple[int, int]] = []  # (src_parent, dst_slot)
            for slot, (sc, parent, tok) in enumerate(new_beams):
                new_scores[slot] = sc
                if tok < 0:
                    # Frozen finished beam — keep its sequence.
                    new_seqs[slot] = list(seqs[parent])
                    new_finished[slot] = True
                else:
                    new_seqs[slot] = list(seqs[parent]) + [tok]
                    new_finished[slot] = tok in eos_ids
                if parent != slot:
                    copy_plan.append((parent, slot))

            # KV cache re-order: copy parent → slot. Two-step (stash → write) to avoid overwriting
            # a parent before another child reads it. For batch=1 the practical case is K<=4 with
            # at most a few crossings per step, so simple per-layer ``ttnn.copy`` is OK.
            for parent, slot in copy_plan:
                for layer in range(self.text_decoder.num_hidden_layers):
                    ttnn.copy(beam_kv[parent][layer][0], beam_kv[slot][layer][0])
                    ttnn.copy(beam_kv[parent][layer][1], beam_kv[slot][layer][1])

            seqs = new_seqs
            scores = new_scores
            finished = new_finished
            cur_pos += 1

        ttnn.deallocate(decode_cross_4d)

        # Length-normalize and pick the best beam.
        def _norm_score(s: float, length: int) -> float:
            l = max(1, length - seed_len)
            return s / (l**length_penalty)

        ranked = sorted(range(K), key=lambda k: _norm_score(scores[k], len(seqs[k])), reverse=True)
        return seqs[ranked[0]]

    def _logits_row_to_host(
        self,
        logits: ttnn.Tensor,
        dec_len: int,
        *,
        sharded: bool = False,
    ) -> "torch.Tensor":
        """Read one logits row ``[B, V]`` on host (no device allocations; safe under active trace).

        ``sharded`` (the width-sharded ``lm_head`` decode path): gather the per-device vocab slices via
        ``ConcatMeshToTensor`` into the full ``[B, dec_seq, V_padded]`` row (decode is B=dec_seq=1)."""
        import torch as _torch

        idx = dec_len - 1
        if sharded:
            host = (
                ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=-1))
                .to(_torch.float32)
                .contiguous()
            )
            batch = int(host.shape[0])
            if host.dim() == 3:
                return host[:, idx, :].reshape(batch, -1)
            return host.reshape(batch, -1)

        batch = int(logits.shape[0])
        dec_seq = int(logits.shape[1])
        vocab_w = int(logits.shape[2])
        host = to_torch_replicated_first_shard(logits).to(_torch.float32).contiguous()
        if host.dim() == 3:
            return host[:, idx, :].reshape(batch, vocab_w)
        if host.dim() == 2 and int(host.shape[0]) == batch and int(host.shape[1]) == vocab_w:
            return host
        if host.dim() == 2 and int(host.shape[0]) == batch:
            return host.reshape(batch, dec_seq, vocab_w)[:, idx, :]
        return host.reshape(batch, dec_seq, vocab_w)[:, idx, :]

    def _greedy_next_token_id(
        self,
        logits: ttnn.Tensor,
        dec_len: int,
        *,
        repetition_penalty: float = 1.0,
        prev_token_ids: Optional["Collection[int]"] = None,
        tok_tt: Optional[tuple[ttnn.Tensor, ttnn.Tensor]] = None,
    ) -> int:
        """Greedy next-token id for traced KV decode (chunk D2H when trace outputs ``tok_tt``)."""
        rt = self._kv_decode_rt
        if tok_tt is not None or (rt is not None and rt.trace_id is not None and rt.tok_tt is not None):
            return self._greedy_token_after_traced_step(
                logits,
                dec_len,
                repetition_penalty=repetition_penalty,
                prev_token_ids=prev_token_ids,
            )
        return self._ttnn_greedy_token_id(
            logits,
            dec_len,
            repetition_penalty=repetition_penalty,
            prev_token_ids=prev_token_ids,
        )

    @staticmethod
    def _read_kv_decode_token_tt(token_tt: ttnn.Tensor) -> int:
        return _read_int_scalar(token_tt)

    def _greedy_next_token(
        self,
        logits: ttnn.Tensor,
        dec_len: int,
        *,
        repetition_penalty: float = 1.0,
        prev_token_ids: Optional[List[int]] = None,
    ) -> Tuple[ttnn.Tensor, int]:
        """Slice ``[B, dec_seq, V]`` → ``[B, 1, V]`` at position ``dec_len-1``, then ``argmax`` → ``[B, 1]``.

        Returns ``(next_token_uint32 [B, 1], next_token_id_int)``. The scalar int is read for the
        EOS check. When ``repetition_penalty > 1.0`` and ``prev_token_ids`` is non-empty, the HF
        ``RepetitionPenaltyLogitsProcessor`` rule is applied host-side before the ``argmax`` (this
        matches HF semantics: positive logits at already-emitted token ids are divided by
        ``penalty``, negative logits are multiplied — both reduce the probability of repeats).
        Host-side ``argmax`` over a ``[V=256k]`` row is sub-ms on CPU so the per-step roundtrip is
        cheap relative to the decoder forward.
        """
        import torch as _torch

        batch = int(logits.shape[0])
        vocab_w = int(logits.shape[2])
        dec_seq = int(logits.shape[1])
        if dec_seq == 1:
            last = logits
            own_last = False
        else:
            # Prefill lm_head returns ``[B, seed_len, V]``. Device ``slice`` JIT clashes with the
            # decode-trace L1 reservation; gather the row on host and re-upload for device argmax.
            host_row = self._logits_row_to_host(logits, dec_len, sharded=False)
            last = ttnn.from_torch(
                host_row.to(_torch.bfloat16).reshape(batch, 1, vocab_w),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            own_last = True

        apply_penalty = repetition_penalty > 1.0 and prev_token_ids
        if apply_penalty:
            host = to_torch_replicated_first_shard(last).to(_torch.float32).reshape(batch, vocab_w)
            if own_last:
                ttnn.deallocate(last)
            # HF RepetitionPenaltyLogitsProcessor: ``score < 0 -> *penalty, else /penalty``.
            ids = _torch.as_tensor(list(prev_token_ids), dtype=_torch.int64)
            ids = ids[(ids >= 0) & (ids < vocab_w)]
            if ids.numel() > 0:
                scores = host[:, ids]
                penalized = _torch.where(
                    scores < 0, scores * float(repetition_penalty), scores / float(repetition_penalty)
                )
                host[:, ids] = penalized
            next_id_int = int(host[0].argmax().item())
            next_uint = ttnn.from_torch(
                _torch.tensor([[next_id_int]] * batch, dtype=_torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return next_uint, next_id_int

        argmax = ttnn.argmax(last, dim=-1)  # [B, 1] int32
        if own_last:
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

    def _sample_next_token(
        self,
        logits: ttnn.Tensor,
        dec_len: int,
        *,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> Tuple[ttnn.Tensor, int]:
        """Temperature/top-p/top-k sampling from ``[B, dec_seq, V]`` logits at position ``dec_len-1``.

        Returns ``(next_token_uint32 [B, 1], next_token_id_int)`` — same interface as
        ``_greedy_next_token``.  All sampling math runs host-side on torch after a single
        device readback of the ``[V]`` logit row.  Not traced (trace replay is greedy-only).
        """
        import torch as _torch

        batch = int(logits.shape[0])
        vocab_w = int(logits.shape[2])
        host = self._logits_row_to_host(logits, dec_len, sharded=False)

        logits_h = host[0]  # [V] for batch=1
        temp = max(float(temperature), 1e-8)
        logits_h = logits_h / temp

        if top_k > 0:
            top_vals, _ = _torch.topk(logits_h, min(top_k, vocab_w))
            min_top = top_vals[-1]
            logits_h[logits_h < min_top] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_idx = _torch.sort(logits_h, descending=True)
            cumprobs = _torch.cumsum(_torch.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative prob above threshold, keeping the first one above.
            remove = cumprobs - _torch.softmax(sorted_logits, dim=-1) > top_p
            logits_h[sorted_idx[remove]] = float("-inf")

        probs = _torch.softmax(logits_h, dim=-1)
        next_id_int = int(_torch.multinomial(probs, num_samples=1).item())

        next_uint = ttnn.from_torch(
            _torch.tensor([[next_id_int]] * batch, dtype=_torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return next_uint, next_id_int

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: Optional[ttnn.Tensor] = None,
        input_features: Optional[ttnn.Tensor] = None,
        return_intermediate_token_ids: Optional[bool] = None,
        tgt_lang: Optional[str] = None,
        speaker_id: int = 0,
        generate_speech: bool = True,
        text_sequences: Optional[ttnn.Tensor] = None,
        **kwargs: Any,
    ) -> Union[TTSeamlessM4Tv2GreedySearchOutput, TTSeamlessM4Tv2GenerationOutput, Tuple[ttnn.Tensor, ttnn.Tensor]]:
        """Greedy / sampling analog of HF ``SeamlessM4Tv2Model.generate``.

        Supports:
        - Greedy search (``do_sample=False``, default)
        - Beam search (``num_beams > 1``)
        - Temperature / top-p / top-k sampling (``do_sample=True``; on device when enabled)
        - On-device greedy via ``TTSampling`` when ``SEAMLESS_ONDEVICE_SAMPLING`` is enabled (default)
        - Single-capture Metal decode trace (``use_decode_trace=True``; compatible with on-device sampling)
        - 2CQ decode pipeline (``use_2cq=True``; requires ``use_decode_trace=True``)

        Returns ``ttnn.Tensor`` outputs only. For text modality, the first-pass encoder output is
        reused in the speech generation path, matching HF's ``text_generation_output.encoder_hidden_states[-1]``.
        """
        if input_ids is None and input_features is None:
            raise ValueError("Provide one of `input_ids` or `input_features`.")
        if generate_speech and tgt_lang is None:
            raise ValueError("`tgt_lang` is required when `generate_speech=True`.")
        if tgt_lang is not None:
            tgt_lang = tgt_lang.replace("__", "")
        if text_sequences is not None and not generate_speech:
            raise ValueError("`text_sequences` is only valid when `generate_speech=True`.")
        return_timings = bool(kwargs.pop("return_timings", False))
        kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)
        if self.generation_config is not None:
            kwargs_text = apply_hf_generation_defaults(kwargs_text, self.generation_config)

        num_beams = int(kwargs_text.get("num_beams", 1) or 1)
        if num_beams < 1:
            raise ValueError(f"num_beams must be >= 1 (got {num_beams})")
        do_sample = bool(kwargs_text.get("do_sample", False))
        temperature = float(kwargs_text.get("temperature", 1.0) or 1.0)
        top_p = float(kwargs_text.get("top_p", 1.0) or 1.0)
        top_k = int(kwargs_text.get("top_k", 0) or 0)
        sampling_seed = kwargs_text.get("seed")
        if sampling_seed is not None:
            sampling_seed = int(sampling_seed)
        use_2cq = bool(kwargs_text.get("use_2cq", False))
        use_t2u_trace = bool(kwargs_text.pop("use_t2u_trace", False))
        if do_sample and num_beams > 1:
            raise ValueError("do_sample=True is not compatible with num_beams > 1.")
        length_penalty = float(kwargs_text.get("length_penalty", 1.0) or 1.0)
        early_stopping = bool(kwargs_text.get("early_stopping", True))

        max_new_tokens = int(kwargs_text.get("max_new_tokens", 256))
        repetition_penalty = float(kwargs_text.get("repetition_penalty", 1.0) or 1.0)
        if repetition_penalty < 1.0:
            raise ValueError(f"repetition_penalty must be >= 1.0 (got {repetition_penalty})")
        self._decode_hf_exact_greedy = bool(kwargs_text.get("hf_exact_greedy", False))
        use_on_device_sampling = self._use_on_device_sampling()
        if use_on_device_sampling:
            self._configure_on_device_sampling(
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=sampling_seed,
            )
        if use_2cq and use_on_device_sampling and self._fold_sampling_in_trace:
            raise ValueError(
                "use_2cq=True is incompatible with SEAMLESS_SAMPLE_IN_TRACE=1 (in-trace sampling uses single CQ)."
            )
        eos_ids = _eos_id_set(kwargs_text.get("eos_token_id"))
        if self.generation_config is not None:
            eos_ids |= _eos_id_set(getattr(self.generation_config, "eos_token_id", None))
        attn_tt_text = kwargs_text.get("attention_mask")

        input_is_speech = input_features is not None
        phase_timer = _GeneratePhaseTimer(self.device, return_timings, input_is_speech=input_is_speech)
        phase_timer.start()
        t_phase = phase_timer.sync_now()
        self._reset_speech_hidden_cache()

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
            # Speech encoder conv programs do not coexist with vocoder/T2U JIT in the same L1 budget.
            # Back-to-back ``generate()`` (demo warmups) may leave decode-trace / T2U / vocoder state
            # that collides with speech-encoder JIT on the timed run.
            self.t2u.release_forward_trace()
            self.vocoder.release_forward_trace()
            self.release_generation_runtime()
            self.clear_runtime_program_cache()
            ttnn.synchronize_device(self.device)
            enc_tt, enc_attn_tt, enc_attn_owned = self._encode_speech(input_features, attn_tt_text)
            if return_timings and attn_tt_text is not None:
                phase_timer.mel_frames = int(
                    to_torch_replicated_first_shard(attn_tt_text).long().reshape(-1).sum().item()
                )
        else:
            enc_tt, enc_attn_tt, enc_attn_owned = self._encode_text(input_ids, attn_tt_text)  # type: ignore[arg-type]
        phase_timer.encoder_ms = phase_timer.elapsed_ms(t_phase)

        reuse_text_sequences = text_sequences is not None
        use_decode_trace = bool(kwargs_text.get("use_decode_trace", False))
        disable_hidden_cache = os.environ.get("SEAMLESS_USE_SPEECH_HIDDEN_CACHE", "1").lower() in (
            "0",
            "false",
            "no",
        )
        if (
            generate_speech
            and not reuse_text_sequences
            and num_beams == 1
            and not do_sample
            and not use_decode_trace
            and not disable_hidden_cache
        ):
            self._speech_hidden_cache_active = True

        # ---- Seed decoder sequence (skipped when reusing T2TT ``text_sequences``) ----
        seed_tt: Optional[ttnn.Tensor] = None
        if not reuse_text_sequences:
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

        # ---- Greedy decode loop (device decode; one scalar readback per step for EOS) ----
        # Track token ids on host to avoid per-step ``ttnn.concat`` reallocations; materialize
        # ``sequences_tt`` once after the loop (or for T2U / return).
        if reuse_text_sequences:
            sequences_tt = ttnn.clone(text_sequences, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            seq_host = _read_int_row(sequences_tt)
            seed_len = 2
        else:
            seq_host = _read_int_row(seed_tt)
            ttnn.deallocate(seed_tt)
            seed_len = len(seq_host)
        use_kv_cache = bool(kwargs_text.get("use_kv_cache", True))
        # Metal trace replay for KV decode (requires ``trace_region_size`` in device params).
        # ``prewarm_conv1d_weights`` is accepted for API compat but ignored here: in-generate
        # prewarm poisoned ``_conv1d_prepared_cache`` (wrong vocoder ``t_audio``, T2U bounds).
        # Call ``prewarm_t2u_conv1d_weights`` / ``prewarm_vocoder_conv1d_weights`` explicitly
        # with trace-known shapes (``test_e2e_perf_2cq.py``) before ``generate()`` or trace capture.
        gen_causal: Optional[ttnn.Tensor] = None
        gen_cross: Optional[ttnn.Tensor] = None
        gen_mask_key: Optional[Tuple[int, int, int]] = None

        if reuse_text_sequences:
            phase_timer.per_step_tracking = False
        elif num_beams > 1 and use_kv_cache:
            phase_timer.per_step_tracking = False
            # Multi-beam search path. Allocates K independent KV caches and runs per-beam decoder
            # forwards with host-side top-K scoring + length-normalized beam selection.
            seq_host = self._generate_beam_kv(
                enc_tt,
                enc_attn_tt,
                seq_host,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                eos_ids=eos_ids,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )
            sequences_tt = _ttnn_ids_from_list([seq_host], self.device)
        elif use_kv_cache:
            if seed_len + max_new_tokens > self.max_text_seq_len:
                raise ValueError(
                    f"seed_len ({seed_len}) + max_new_tokens ({max_new_tokens}) exceeds "
                    f"max_text_seq_len ({self.max_text_seq_len})"
                )
            enc_seq = int(enc_tt.shape[1])
            kv_cache, cross_attn_cache = init_text_decoder_kv_cache(
                self.device,
                num_hidden_layers=self.text_decoder.num_hidden_layers,
                num_attention_heads=self.decoder_attention_heads,
                hidden_size=self.hidden_size,
                max_batch_size=1,
                max_seq_len=self.max_text_seq_len,
                encoder_seq_len=enc_seq,
                tp=self._tp,
            )
            decode_cross_4d = build_cross_attn_mask_4d(enc_attn_tt, tgt_seq=1, device=self.device)
            # Prefill the full seed (matches ``test_text_decoder`` KV PCC). First new token comes from
            # ``lm_head`` on the last seed hidden state; decode steps start at ``cur_pos == seed_len``.
            cross_valid = False
            cur_pos = seed_len
            # Trace from the first decode-loop step (one SDPA bucket; no mid-loop re-capture).
            # Trace when greedy, or when on-device sampling handles token pick post-trace.
            decode_trace_ready = use_decode_trace and (not do_sample or use_on_device_sampling)
            # A decode trace captured by a previous ``generate()`` call references that call's
            # ``kv_cache`` / encoder buffers, which were just reallocated above. Release it so this
            # call re-captures against its own buffers (guards back-to-back generate without a
            # ``clear_runtime_program_cache`` in between).
            if decode_trace_ready:
                self.release_text_decoder_decode_trace()
            decode_steps_remaining = max_new_tokens

            if seed_len > 0:
                t_prefill = phase_timer.sync_now()
                warm_tt = _ttnn_ids_from_list([seq_host], self.device)
                warm_out = self._prefill_text_decoder_kv_cache(
                    warm_tt,
                    enc_tt,
                    enc_attn_tt,
                    kv_cache,
                    cross_attn_cache,
                )
                self._stash_speech_hidden_block(warm_out, start=0, end=seed_len)
                phase_timer.prefill_ms = phase_timer.elapsed_ms(t_prefill)
                ttnn.deallocate(warm_tt)
                cross_valid = True
                t_first = phase_timer.begin_step()
                if use_on_device_sampling:
                    logits_sh = self._lm_head_sharded(warm_out)
                    ttnn.deallocate(warm_out)
                    next_id = sample_first_token_from_prefill_logits(
                        self._on_device_sampler,
                        logits_sh,
                        dec_len=seed_len,
                    )
                    ttnn.deallocate(logits_sh)
                elif do_sample:
                    logits = self._lm_head(warm_out)
                    ttnn.deallocate(warm_out)
                    next_tt, next_id = self._sample_next_token(
                        logits, seed_len, temperature=temperature, top_k=top_k, top_p=top_p
                    )
                    ttnn.deallocate(logits)
                    ttnn.deallocate(next_tt)
                else:
                    logits = self._lm_head(warm_out)
                    ttnn.deallocate(warm_out)
                    next_tt, next_id = self._greedy_next_token(
                        logits, seed_len, repetition_penalty=repetition_penalty, prev_token_ids=seq_host
                    )
                    ttnn.deallocate(logits)
                    ttnn.deallocate(next_tt)
                phase_timer.end_step(t_first)
                seq_host.append(next_id)
                if eos_ids and next_id in eos_ids:
                    decode_steps_remaining = 0
                else:
                    decode_steps_remaining = max_new_tokens - 1

            # 2CQ event state — initialized once before the decode loop so the CQ0-done
            # signal persists across iterations.  ``_2cq_op_event`` represents "CQ0 has
            # finished the previous trace; CQ1 is safe to overwrite the decode buffers."
            # After each execute_trace we record a fresh op_event so the next iteration
            # waits for the correct execution boundary.
            _2cq_op_event = None
            if use_decode_trace and use_2cq and decode_trace_ready:
                # CQ0 is idle after prefill; record the initial "done" event.
                _2cq_op_event = ttnn.record_event(self.device, 0)

            seq_host_set = set(seq_host)

            if decode_trace_ready and self._tp > 1:
                seq_host = self._decode_loop_selffed(
                    enc_tt,
                    decode_cross_4d,
                    kv_cache,
                    cross_attn_cache,
                    seq_host,
                    cur_pos,
                    decode_steps_remaining,
                    eos_ids,
                    batch_size,
                    use_2cq=use_2cq,
                    repetition_penalty=repetition_penalty,
                    prev_token_ids=seq_host_set,
                    phase_timer=phase_timer if return_timings else None,
                )
                decode_steps_remaining = 0

            for _decode_step in range(decode_steps_remaining):
                cur_tok = int(seq_host[cur_pos])
                t_step = phase_timer.begin_step()
                traced_step = use_decode_trace and decode_trace_ready
                if traced_step:
                    if use_2cq:
                        # 2CQ pipelined decode: CQ1 uploads H2D while CQ0 executes trace.
                        trace_id = self._select_decode_trace_for_position(cur_pos, batch_size)
                        if trace_id is None:
                            logits = self._decode_token_with_kv_cache_traced(
                                cur_tok,
                                cur_pos,
                                enc_tt,
                                decode_cross_4d,
                                kv_cache,
                                cross_attn_cache,
                                batch_size=batch_size,
                            )
                            _2cq_op_event = ttnn.record_event(self.device, 0)
                        else:
                            rt = self._kv_decode_rt
                            ttnn.wait_for_event(1, _2cq_op_event)
                            self._upload_kv_decode_step_inputs_cq1(cur_tok, cur_pos, batch_size)
                            self._reset_kv_decode_cur_pos(cur_pos, batch_size, cq_id=1)
                            write_event = ttnn.record_event(self.device, 1)
                            ttnn.wait_for_event(0, write_event)
                            ttnn.execute_trace(self.device, trace_id, cq_id=0, blocking=True)
                            _2cq_op_event = ttnn.record_event(self.device, 0)
                            logits = rt.logits_tt
                    else:
                        logits = self._decode_token_with_kv_cache_traced(
                            cur_tok,
                            cur_pos,
                            enc_tt,
                            decode_cross_4d,
                            kv_cache,
                            cross_attn_cache,
                            batch_size=batch_size,
                        )
                else:
                    logits = self._decode_token_with_kv_cache(
                        cur_tok,
                        cur_pos,
                        enc_tt,
                        enc_attn_tt,
                        kv_cache,
                        cross_attn_cache,
                        cross_valid,
                        cross_4d=decode_cross_4d,
                        batch_size=batch_size,
                        sharded_logits=use_on_device_sampling,
                    )
                # Guard against occasional invalid traced logits buffer; fall back to eager decode.
                if (
                    use_decode_trace
                    and decode_trace_ready
                    and hasattr(ttnn, "is_allocated")
                    and not ttnn.is_allocated(logits)
                ):
                    logits = self._decode_token_with_kv_cache(
                        cur_tok,
                        cur_pos,
                        enc_tt,
                        enc_attn_tt,
                        kv_cache,
                        cross_attn_cache,
                        cross_valid,
                        cross_4d=decode_cross_4d,
                        batch_size=batch_size,
                        sharded_logits=use_on_device_sampling,
                    )
                    use_decode_trace = False
                    decode_trace_ready = False
                if use_on_device_sampling:
                    if traced_step:
                        next_id = self._read_token_from_traced_logits(logits, dec_len=1)
                    else:
                        next_id = self._on_device_sampler.sample_and_read(logits, dec_len=1)
                elif do_sample:
                    next_tt, next_id = self._sample_next_token(
                        logits, 1, temperature=temperature, top_k=top_k, top_p=top_p
                    )
                    ttnn.deallocate(next_tt)
                elif use_decode_trace and decode_trace_ready:
                    # Fast path reads the trace-fused argmax token (``rt.tok_tt``); only the rep-penalty
                    # fallback touches the logits on host. The argmax is captured inside the trace (no
                    # per-step device alloc, which would corrupt the active trace).
                    next_id = self._greedy_next_token_id(
                        logits,
                        1,
                        repetition_penalty=repetition_penalty,
                        prev_token_ids=seq_host_set,
                        tok_tt=self._kv_decode_rt.tok_tt if self._kv_decode_rt is not None else None,
                    )
                else:
                    next_tt, next_id = self._greedy_next_token(
                        logits, 1, repetition_penalty=repetition_penalty, prev_token_ids=seq_host
                    )
                    ttnn.deallocate(next_tt)
                phase_timer.end_step(t_step)
                if not (use_decode_trace and decode_trace_ready):
                    ttnn.deallocate(logits)
                seq_host.append(next_id)
                seq_host_set.add(next_id)
                cur_pos += 1
                if not cross_valid:
                    cross_valid = True
                if eos_ids and next_id in eos_ids:
                    break
            ttnn.deallocate(decode_cross_4d)
            sequences_tt = _ttnn_ids_from_list([seq_host], self.device)
        else:
            sequences_tt = _ttnn_ids_from_list([seq_host], self.device)
            for _ in range(max_new_tokens):
                t_step = phase_timer.begin_step()
                batch_i = int(sequences_tt.shape[0])
                dec_len = len(seq_host)
                padded_i = tile_align(dec_len)
                mask_key = (batch_i, padded_i, id(enc_attn_tt))
                if mask_key != gen_mask_key:
                    if gen_causal is not None:
                        ttnn.deallocate(gen_causal)
                        ttnn.deallocate(gen_cross)
                    gen_causal = build_causal_with_padding_4d(None, batch_i, padded_i, self.device)
                    gen_cross = build_cross_attn_mask_4d(enc_attn_tt, tgt_seq=padded_i, device=self.device)
                    gen_mask_key = mask_key
                logits = self._decode_and_lm_head(
                    enc_tt,
                    enc_attn_tt,
                    sequences_tt,
                    None,
                    prebuilt_causal_4d=gen_causal,
                    prebuilt_cross_4d=gen_cross,
                )
                next_tt, next_id = self._greedy_next_token(
                    logits, dec_len, repetition_penalty=repetition_penalty, prev_token_ids=seq_host
                )
                phase_timer.end_step(t_step)
                ttnn.deallocate(logits)
                ttnn.deallocate(sequences_tt)
                ttnn.deallocate(next_tt)
                seq_host.append(next_id)
                sequences_tt = _ttnn_ids_from_list([seq_host], self.device)
                if eos_ids and next_id in eos_ids:
                    break

        if gen_causal is not None:
            ttnn.deallocate(gen_causal)
            ttnn.deallocate(gen_cross)

        # ---- Text-only generation: return tokens ----
        if not generate_speech:
            self.release_text_decoder_decode_trace()
            ttnn.deallocate(enc_tt)
            if enc_attn_owned:
                ttnn.deallocate(enc_attn_tt)
            timings = (
                phase_timer.finish(
                    seed_len=seed_len,
                    seq_host=seq_host,
                    output_samples=0,
                    generate_speech=False,
                )
                if return_timings
                else None
            )
            return TTSeamlessM4Tv2GreedySearchOutput(sequences=sequences_tt, timings=timings)

        # Text decode may leave a captured Metal trace active; T2U / vocoder must not JIT new
        # programs while it is live (L1 CB clash with trace region, corrupt trace buffers).
        self._clear_decode_and_t2u_programs(preserve_vocoder=True)
        ttnn.synchronize_device(self.device)

        # ---- Speech generation: re-encode for speech modality (HF parity), then T2U + vocoder ----
        gc = self.generation_config
        pad_token_id = int(gc.pad_token_id)
        eos_id = int(getattr(gc, "eos_token_id", 3))
        seq_host = _trim_seq_host_for_speech(
            seq_host,
            pad_token_id=pad_token_id,
            eos_id=eos_id,
            seed_len=seed_len,
        )
        if int(sequences_tt.shape[1]) != len(seq_host):
            ttnn.deallocate(sequences_tt)
            sequences_tt = _ttnn_ids_from_list([seq_host], self.device)
        attn_enc = kwargs_speech.get("attention_mask", attn_tt_text)
        # Reuse the first-pass encoder output for the T2U decoder-hidden pass — HF parity
        # (``text_generation_output.encoder_hidden_states[-1]``). The speech encoder is
        # deterministic, so re-encoding the same features+mask just reproduces ``enc_tt`` bit for
        # bit; skipping it saves a full speech-encoder forward (~8 s on the demo's S2ST). Only
        # re-encode if the speech path was handed a *different* attention mask than the first pass.
        if input_features is not None and attn_enc is not attn_tt_text:
            ttnn.deallocate(enc_tt)
            if enc_attn_owned:
                ttnn.deallocate(enc_attn_tt)
            enc_tt2, enc_attn_tt2, enc_attn_owned2 = self._encode_speech(input_features, attn_enc)
        else:
            enc_tt2, enc_attn_tt2, enc_attn_owned2 = enc_tt, enc_attn_tt, enc_attn_owned

        # T2U decoder hidden states come from running text-decoder on ``sequences[:, :-1]`` (HF
        # trims the final EOS). Use logical ``seq_host`` length (not tile-padded tensor width).
        t_t2u = phase_timer.sync_now()
        logical_seq_len = len(seq_host)
        logical_hidden_rows = logical_seq_len - 1
        cached_hidden = self._speech_hidden_for_t2u(logical_hidden_rows)
        if cached_hidden is not None:
            dec_hidden_padded, padded_dec_seq = cached_hidden
            self._reset_speech_hidden_cache()
        else:
            dec_in_tt = ttnn.slice(sequences_tt, [0, 0], [batch_size, logical_hidden_rows], (1, 1))
            dec_hidden_padded, padded_dec_seq = self._decoder_hidden(enc_tt2, enc_attn_tt2, dec_in_tt)
            ttnn.deallocate(dec_in_tt)
        t_char_prep = phase_timer.sync_now()
        phase_timer.t2u_decoder_hidden_ms = phase_timer.elapsed_ms(t_t2u)
        ttnn.deallocate(enc_tt2)
        if enc_attn_owned2:
            ttnn.deallocate(enc_attn_tt2)

        # T2U prep: characters & char counts come from the generated text-token sequence.
        real_dec_len = sum(1 for x in seq_host[:-1] if x != pad_token_id)
        if self._t2u_char_tables is None:
            seq_full_ints = list(seq_host)
            t2u_ids = [pad_token_id if t == eos_id else int(t) for t in seq_full_ints[2:-1]]
            subwords = _indices_to_subwords(gc, t2u_ids)
            cc_inner = _char_count_per_subword(t2u_ids, subwords, pad_token_id=pad_token_id)
            char_ids = _get_char_ids(gc, t2u_ids, subwords, cc_inner, pad_token_id=pad_token_id)
            char_ids_tt = _ttnn_ids_from_list([char_ids], self.device)
        else:
            char_ids_tt, cc_inner_t = self._t2u_char_tables.prepare_speech_from_seq(
                seq_host,
                eos_id=eos_id,
                device=self.device,
            )
            cc_inner = cc_inner_t.tolist()
        t_t2u_fwd = phase_timer.sync_now()
        phase_timer.t2u_char_prep_ms = phase_timer.elapsed_ms(t_char_prep)
        # Bucket the T2U encoder sequence length so its programs don't recompile on S2ST text jitter.
        t2u_enc_seq = _t2u_padded_enc_seq(padded_dec_seq)
        # Pad with one zero each side (for the stripped lang + EOS columns) then pad to the bucket.
        cc_list = [0] + cc_inner + [0]
        if len(cc_list) < t2u_enc_seq:
            cc_list = cc_list + [0] * (t2u_enc_seq - len(cc_list))

        # On-device tensors for T2U: char_input_ids and the T2U attention mask (built at the bucket;
        # positions >= real_dec_len are masked, covering both the real padding and the bucket padding).
        t2u_mask_2d = self._cached_t2u_attention_mask(real_dec_len, t2u_enc_seq)
        t2u_mask_4d = build_encoder_self_mask_4d(t2u_mask_2d, device=self.device)
        # ``t2u_mask_2d`` is owned by ``_t2u_attn_mask_cache``; do not deallocate here.
        # Pad the T2U encoder input (text-decoder hidden) to the bucket; padded rows are masked above
        # and have char-count 0 in ``cc_list`` so they vanish in the char upsample (output unchanged).
        if t2u_enc_seq != padded_dec_seq:
            dec_hidden_bucketed = ttnn.pad(
                dec_hidden_padded,
                [(0, 0), (0, t2u_enc_seq - padded_dec_seq), (0, 0)],
                value=0.0,
            )
            ttnn.deallocate(dec_hidden_padded)
            dec_hidden_padded = dec_hidden_bucketed

        # ``_decoder_hidden`` compiles long text-decoder programs; drop them before T2U so
        # chunked lm_head matmuls do not clash with the decode-trace L1 reservation.
        self._clear_decode_and_t2u_programs(preserve_vocoder=True)
        ttnn.synchronize_device(self.device)

        t2u_logits_tt, padding_tt = self._run_t2u_speech_forward(
            dec_hidden_padded,
            t2u_mask_4d,
            char_ids_tt,
            cc_list,
            use_t2u_trace=use_t2u_trace,
        )
        phase_timer.t2u_forward_ms = phase_timer.elapsed_ms(t_t2u_fwd)
        phase_timer.t2u_ms = (
            phase_timer.t2u_decoder_hidden_ms + phase_timer.t2u_char_prep_ms + phase_timer.t2u_forward_ms
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
        # ``unit_seq`` so padding slice matches T2U logits before host vocoder remap.
        unit_seq = int(t2u_logits_tt.shape[-2])
        pad_batch = int(padding_tt.shape[0])
        if int(padding_tt.shape[1]) != unit_seq:
            padding_tt = ttnn.slice(padding_tt, [0, 0], [pad_batch, unit_seq], (1, 1))

        if t2u_logits_tt.get_layout() != ttnn.TILE_LAYOUT:
            t2u_tile = ttnn.to_layout(t2u_logits_tt, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(t2u_logits_tt)
            t2u_logits_tt = t2u_tile
        if unit_seq < int(t2u_logits_tt.shape[-2]):
            t2u_logits_tt = ttnn.slice(
                t2u_logits_tt,
                [0, 0, 0],
                [batch_size, unit_seq, int(t2u_logits_tt.shape[-1])],
                (1, 1, 1),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
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

        # Unit id remap (EOS/pad mask + vocoder offset) on host — int32 TILE ``where`` still diverges.
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

        import torch as _torch

        unit_host = to_torch_replicated_first_shard(unit_ids_argmax).to(_torch.long)
        pad_host = to_torch_replicated_first_shard(pad_bf).to(_torch.float32)
        ttnn.deallocate(unit_ids_argmax)
        ttnn.deallocate(pad_bf)

        rm_host = (unit_host == int(self.t2u_eos_token_id)) | (pad_host < 0.5)
        pad_id = int(self.t2u_pad_token_id)
        off = int(self.vocoder_offset)
        u = unit_host.clone()
        u[rm_host] = pad_id
        voc = _torch.where(u == pad_id, u, u - off).to(_torch.int32)
        output_unit_ids_tt = ttnn.from_torch(
            unit_host.to(_torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
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
        # Drop T2U prep before vocoder for L1 headroom; keep global program cache so vocoder
        # Metal programs compiled on prior warmups/timed iters stay resident.
        self._evict_t2u_prep_for_vocoder()
        ttnn.synchronize_device(self.device)
        self._stash_vocoder_prewarm_inputs(vocoder_input, spk_tt, voc_tt)
        t_voc = phase_timer.sync_now()
        wav_tt, lengths_tt = self.vocoder.forward(vocoder_input, spk_tt, voc_tt)
        phase_timer.vocoder_ms = phase_timer.elapsed_ms(t_voc)
        if return_timings and phase_timer._t0 is not None:
            phase_timer.ttft_audio_ms = (phase_timer.sync_now() - phase_timer._t0) * 1000.0
        ttnn.deallocate(vocoder_input)
        ttnn.deallocate(voc_tt)
        ttnn.deallocate(spk_tt)

        # Vocoder lengths is 1D ``[B]``; standardise to ``[1, B]`` for callers / tests.
        if len(tuple(lengths_tt.shape)) == 1:
            lengths_tt = ttnn.reshape(lengths_tt, (1, int(lengths_tt.shape[0])))

        output_samples = 0
        if return_timings:
            output_samples = int(to_torch_replicated_first_shard(lengths_tt).long().reshape(-1)[0].item())
        timings = (
            phase_timer.finish(
                seed_len=seed_len,
                seq_host=seq_host,
                output_samples=output_samples,
                generate_speech=True,
            )
            if return_timings
            else None
        )

        self._release_speech_generate_runtime()
        ttnn.synchronize_device(self.device)
        if return_intermediate_token_ids:
            return TTSeamlessM4Tv2GenerationOutput(
                waveform=wav_tt,
                waveform_lengths=lengths_tt,
                sequences=sequences_tt,
                unit_sequences=output_unit_ids_tt,
                timings=timings,
            )
        ttnn.deallocate(sequences_tt)
        ttnn.deallocate(output_unit_ids_tt)
        return wav_tt, lengths_tt
