# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""High-level S2ST (speech-to-speech translation) wrapper for SeamlessM4T-v2.

Pipeline mirroring HF ``SeamlessM4Tv2ForSpeechToSpeech.generate``:

    WAV
        -> processor.feature_extractor (HF: 80-mel + stride-2 stacking -> 160-d)
        -> [TTNN]  SpeechEncoder (24 Conformer layers + adapter)   -> encoder_hidden
        -> [TTNN]  TextGenerator.generate(...)                     -> token sequences
        -> [HF]    text_decoder(sequences[:, :-1])                 -> t2u_input_embeds
                   (re-run on host; cross-attn uses sub-sampled speech mask)
        -> [HF]    _indices_to_subwords + _count_char + _get_char_input_ids
        -> [TTNN]  T2UGenerator.synthesize_units                   -> unit_token_ids
        -> [TTNN]  CodeHifiGanVocoder                              -> waveform

This is the "speech-encoder side of Phase 6" composed with the
"T2U/vocoder side of Phase 7"; only the host hybrid boundary's
``encoder_attention_mask`` and ``encoder_hidden_states`` arguments
change (sub-sampled mask + tile-padded speech-encoder hidden instead
of the text tokenizer's mask + text-encoder hidden).

Example::

    from transformers import AutoProcessor
    import ttnn
    from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
    from models.demos.facebook_seamless_m4t_v2_large.tt.speech_to_speech_model import (
        SpeechToSpeechModel,
    )

    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    hf_sd = wl.load_hf_state_dict()
    proc = AutoProcessor.from_pretrained(wl.HF_PATH)
    model = SpeechToSpeechModel(device=device, hf_state_dict=hf_sd, processor=proc)
    audio_np = model.synthesize("hello.wav", src_lang="eng", tgt_lang="fra")
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
from models.demos.facebook_seamless_m4t_v2_large.tt.code_hifigan_vocoder import CodeHifiGanVocoder
from models.demos.facebook_seamless_m4t_v2_large.tt.speech_encoder import SpeechEncoder
from models.demos.facebook_seamless_m4t_v2_large.tt.speech_to_text_model import (
    ADAPTOR_KERNEL_SIZE,
    ADAPTOR_STRIDE,
    DEFAULT_AUDIO_SEQ_LEN,
    FEATURE_SIZE,
    _compute_sub_seq_len,
    _extract_features,
    _load_wav_to_16k_mono,
)
from models.demos.facebook_seamless_m4t_v2_large.tt.t2u_generator import T2UGenerator
from models.demos.facebook_seamless_m4t_v2_large.tt.text_generator import TextGenerator

# ---------------------------------------------------------------------------
# Model config (matches SeamlessM4T-v2-Large defaults)
# ---------------------------------------------------------------------------
EMBED_DIM = 1024
NUM_HEADS = 16
HEAD_DIM = 64
NUM_ENCODER_LAYERS = 24
NUM_DECODER_LAYERS = 24
NUM_ADAPTER_LAYERS = 1
EPS = 1e-5
DECODER_PADDING_IDX = 0
PAD_TOKEN_ID = 0
T2U_PAD_TOKEN_ID = 1
DEFAULT_MAX_DECODE_SEQ_LEN = 128

_TILE = 32


def _pad_to_tile(n: int) -> int:
    return (_TILE - n % _TILE) % _TILE


def _load_generation_config(hf_path: str) -> Dict:
    cfg_path = Path(hf_path) / "generation_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"generation_config.json not found under {hf_path}")
    with open(cfg_path) as f:
        return json.load(f)


class SpeechToSpeechModel:
    """SeamlessM4T-v2 speech-to-speech (S2ST) synthesiser (TTNN + HF hybrid).

    Args:
        device: opened ttnn device.
        hf_state_dict: result of :func:`weight_loader.load_hf_state_dict`.
        processor: HuggingFace ``SeamlessM4Tv2Processor`` (or any
            ``AutoProcessor`` over the same checkpoint).
        hf_path: path to local checkpoint snapshot. Defaults to
            ``weight_loader.HF_PATH``.
        audio_seq_len: post-feature-extraction time-axis budget. Must
            be a multiple of 32. Default 256 (~5 s of 16 kHz audio).
        max_decode_seq_len: KV-cache slot count for the text decoder
            (must be a multiple of 32).
        weight_dtype: TTNN storage dtype for all sub-module weights.
    """

    def __init__(
        self,
        device,
        hf_state_dict: Dict[str, torch.Tensor],
        processor,
        hf_path: Optional[str] = None,
        audio_seq_len: int = DEFAULT_AUDIO_SEQ_LEN,
        max_decode_seq_len: int = DEFAULT_MAX_DECODE_SEQ_LEN,
        weight_dtype=ttnn.bfloat16,
    ):
        if audio_seq_len % _TILE != 0:
            raise ValueError(f"audio_seq_len({audio_seq_len}) must be a multiple of {_TILE}")
        if max_decode_seq_len % _TILE != 0:
            raise ValueError(f"max_decode_seq_len({max_decode_seq_len}) must be a multiple of {_TILE}")

        self.device = device
        self.processor = processor
        self.hf_path = hf_path or wl.HF_PATH
        self.audio_seq_len = int(audio_seq_len)
        self.max_decode_seq_len = int(max_decode_seq_len)
        self.weight_dtype = weight_dtype
        self._hf_state_dict = hf_state_dict

        self.gen_cfg = _load_generation_config(self.hf_path)
        self.decoder_start_token_id = int(self.gen_cfg.get("decoder_start_token_id", 3))
        self.eos_token_id = int(self.gen_cfg.get("eos_token_id", 3))
        self.pad_token_id = int(self.gen_cfg.get("pad_token_id", PAD_TOKEN_ID))
        self.lang_to_code_id: Dict[str, int] = self.gen_cfg["text_decoder_lang_to_code_id"]
        self.vocoder_lang_code_to_id: Dict[str, int] = self.gen_cfg["vocoder_lang_code_to_id"]

        # ----- Speech encoder (TTNN) ----------------------------------------
        enc_sd = wl.speech_encoder_weights(
            hf_state_dict,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_adapter_layers=NUM_ADAPTER_LAYERS,
        )
        self.speech_encoder = SpeechEncoder(
            device=device,
            state_dict=enc_sd,
            feature_size=FEATURE_SIZE,
            hidden=EMBED_DIM,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            seq_len=self.audio_seq_len,
            batch_size=1,
            eps=EPS,
            adaptor_kernel_size=ADAPTOR_KERNEL_SIZE,
            adaptor_stride=ADAPTOR_STRIDE,
            add_adapter=True,
            weight_dtype=weight_dtype,
        )
        self.sub_seq_len = _compute_sub_seq_len(self.audio_seq_len)
        self.encoder_seq_len_padded = self.sub_seq_len + _pad_to_tile(self.sub_seq_len)

        # ----- Text decoder side (LM head + AR text generator, TTNN) --------
        self._text_decoder_sd = wl.text_decoder_weights(
            hf_state_dict,
            num_layers=NUM_DECODER_LAYERS,
            padding_idx=DECODER_PADDING_IDX,
        )
        self._lm_head_sd = wl.lm_head_weights(hf_state_dict)
        self.text_generator = TextGenerator(
            device=device,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            num_layers=NUM_DECODER_LAYERS,
            text_decoder_state_dict=self._text_decoder_sd,
            lm_head_state_dict=self._lm_head_sd,
            max_decode_seq_len=self.max_decode_seq_len,
            encoder_seq_len=self.encoder_seq_len_padded,
            eps=EPS,
            padding_idx=DECODER_PADDING_IDX,
            embed_scale=math.sqrt(EMBED_DIM),
            weight_dtype=weight_dtype,
        )

        # ----- T2U NAR generator (TTNN) -------------------------------------
        self.t2u_generator = T2UGenerator(
            device=device,
            hf_state_dict=hf_state_dict,
            weight_dtype=weight_dtype,
        )

        # ----- Code HiFi-GAN vocoder (TTNN) ---------------------------------
        vocoder_sd = wl.code_hifigan_vocoder_weights(hf_state_dict)
        self.vocoder = CodeHifiGanVocoder(
            device=device,
            state_dict=vocoder_sd,
            pad_token_id=T2U_PAD_TOKEN_ID,
            weight_dtype=weight_dtype,
        )

        # ----- HF host helpers (text_decoder rerun + char helpers) -----------
        self._hf_s2s_model = None

        self.last_waveform_length: int = 0

    # ------------------------------------------------------------------ helpers

    def _resolve_text_tgt_lang_id(self, tgt_lang: str) -> int:
        if tgt_lang not in self.lang_to_code_id:
            raise ValueError(
                f"tgt_lang={tgt_lang!r} not in text_decoder_lang_to_code_id; valid examples: fra, spa, deu, eng."
            )
        return int(self.lang_to_code_id[tgt_lang])

    def _resolve_vocoder_lang_id(self, tgt_lang: str) -> int:
        if tgt_lang not in self.vocoder_lang_code_to_id:
            raise ValueError(
                f"tgt_lang={tgt_lang!r} not in vocoder_lang_code_to_id; valid examples: "
                f"{','.join(list(self.vocoder_lang_code_to_id.keys())[:8])}."
            )
        return int(self.vocoder_lang_code_to_id[tgt_lang])

    def _load_hf_helper_model(self):
        """Lazy-load HF ``SeamlessM4Tv2ForSpeechToSpeech``.

        We only need ``self.text_decoder(...)`` and the char helpers
        (``_indices_to_subwords``, ``_count_character_length_in_subword``,
        ``_get_char_input_ids``). Keeping it lazy avoids paying the
        ~8 GB host RAM cost until the first ``synthesize`` call.
        """
        if self._hf_s2s_model is None:
            from transformers import SeamlessM4Tv2ForSpeechToSpeech

            model = SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(self.hf_path, torch_dtype=torch.float32)
            model.eval()
            self._hf_s2s_model = model
        return self._hf_s2s_model

    def _to_tt_features(self, input_features: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            input_features,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _post_adapter_attention_mask(
        self,
        attention_mask_2d: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Compute the post-adapter ``[B, sub_seq_len]`` 0/1 keep mask.

        Mirrors HF ``_compute_new_attention_mask`` exactly. The mask
        has the same downstream usage as in :class:`SpeechToTextModel`.
        """
        seq_lens = attention_mask_2d.size(1) - (1 - attention_mask_2d.int()).sum(1)
        pad = ADAPTOR_STRIDE // 2
        sub_lens = (((seq_lens + 2 * pad - ADAPTOR_KERNEL_SIZE) / ADAPTOR_STRIDE) + 1).floor().long()
        sub_lens = sub_lens.clamp(min=0, max=self.sub_seq_len)
        batch = attention_mask_2d.shape[0]
        sub_mask = torch.zeros((batch, self.sub_seq_len), dtype=torch.long)
        for b in range(batch):
            n = int(sub_lens[b].item())
            sub_mask[b, :n] = 1
        logical_sub = int(sub_lens[0].item())
        return sub_mask, logical_sub

    @staticmethod
    def _compute_new_attention_mask(seq_lens: torch.Tensor, batch: int, mask_seq_len: int) -> torch.Tensor:
        indices = torch.arange(mask_seq_len).expand(batch, -1)
        bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)
        mask = torch.ones((batch, mask_seq_len), dtype=torch.float32)
        return mask.masked_fill(bool_mask, 0.0)

    # ------------------------------------------------------------------ public API

    @torch.no_grad()
    def synthesize(
        self,
        audio_path: str,
        src_lang: str,
        tgt_lang: str,
        speaker_id: int = 0,
        max_new_tokens: int = 128,
        max_audio_seconds: Optional[float] = 5.0,
    ) -> np.ndarray:
        """Synthesize speech for the spoken ``audio_path`` translated into ``tgt_lang``.

        Args:
            audio_path: path to a WAV (any sample rate, mono/stereo).
            src_lang: source language code (API symmetry; the speech
                encoder is language-agnostic).
            tgt_lang: target language code (e.g. ``fra``). Must appear in
                both ``text_decoder_lang_to_code_id`` and
                ``vocoder_lang_code_to_id``.
            speaker_id: vocoder speaker id (default 0).
            max_new_tokens: AR text-generation budget (incl. 2-token prefix).
            max_audio_seconds: truncate the input audio to this many
                seconds before feature extraction.

        Returns:
            ``np.ndarray`` of shape ``(T_samples,)`` float32 at 16 kHz in
            roughly ``[-1, 1]``. Truncated to the vocoder's reported
            ``waveform_length`` (also stored on ``self.last_waveform_length``).
        """
        # ----------------------------------------------------------------
        # 1. Load + resample + feature-extract on host.
        # ----------------------------------------------------------------
        audio = _load_wav_to_16k_mono(audio_path)
        input_features, attention_mask_2d, _logical_len = _extract_features(
            audio,
            processor=self.processor,
            target_seq_len=self.audio_seq_len,
            max_audio_seconds=max_audio_seconds,
        )

        # ----------------------------------------------------------------
        # 2. Speech encoder forward (TTNN).
        # ----------------------------------------------------------------
        feat_tt = self._to_tt_features(input_features)
        enc_hidden_tt = self.speech_encoder(feat_tt, attention_mask_2d=attention_mask_2d)
        enc_hidden_torch = ttnn.to_torch(enc_hidden_tt).to(torch.float32)
        ttnn.deallocate(enc_hidden_tt)
        while enc_hidden_torch.dim() > 3:
            enc_hidden_torch = enc_hidden_torch.squeeze(0)
        enc_hidden_torch = enc_hidden_torch.reshape(1, self.sub_seq_len, EMBED_DIM)

        # ----------------------------------------------------------------
        # 3. Build post-adapter cross-attention mask + tile-pad enc hidden.
        # ----------------------------------------------------------------
        sub_mask_2d, logical_sub = self._post_adapter_attention_mask(attention_mask_2d)
        pad_needed = _pad_to_tile(self.sub_seq_len)
        if pad_needed > 0:
            zeros = torch.zeros((1, pad_needed, EMBED_DIM), dtype=enc_hidden_torch.dtype)
            enc_hidden_padded = torch.cat([enc_hidden_torch, zeros], dim=1)
            mask_pad = torch.zeros((1, pad_needed), dtype=sub_mask_2d.dtype)
            sub_mask_2d_padded = torch.cat([sub_mask_2d, mask_pad], dim=1)
        else:
            enc_hidden_padded = enc_hidden_torch
            sub_mask_2d_padded = sub_mask_2d

        # ----------------------------------------------------------------
        # 4. AR text decoder + LM head (TTNN, greedy).
        # ----------------------------------------------------------------
        text_tgt_lang_id = self._resolve_text_tgt_lang_id(tgt_lang)
        max_total = min(int(max_new_tokens), self.max_decode_seq_len)
        text_tokens = self.text_generator.generate(
            encoder_hidden_states=enc_hidden_padded,
            encoder_attention_mask=sub_mask_2d_padded,
            decoder_start_token_id=self.decoder_start_token_id,
            tgt_lang_id=text_tgt_lang_id,
            eos_token_id=self.eos_token_id,
            max_new_tokens=max_total,
            do_sample=False,
        )
        if isinstance(text_tokens, torch.Tensor):
            seq = text_tokens.to(torch.int64).view(1, -1)
        else:
            seq = torch.tensor(text_tokens, dtype=torch.int64).view(1, -1)
        # Ensure trailing EOS so the [:-1] trim matches HF.
        if int(seq[0, -1].item()) != self.eos_token_id:
            seq = torch.cat([seq, torch.tensor([[self.eos_token_id]], dtype=torch.int64)], dim=1)

        # ----------------------------------------------------------------
        # 5. Hybrid boundary: HF re-runs the text_decoder over sequences[:, :-1]
        #    to recover a full-sequence last_hidden_state, then computes the
        #    T2U char inputs. Cross-attn here uses the SUB-SAMPLED speech
        #    mask -- matching HF S2ST exactly.
        # ----------------------------------------------------------------
        hf_model = self._load_hf_helper_model()

        # For the host text_decoder rerun, pass the logical (un-tile-padded)
        # speech-encoder hidden + matching sub-sampled keep mask. HF will
        # build its own 4-D additive mask internally.
        enc_hidden_logical = enc_hidden_torch[:, :logical_sub, :].to(torch.float32)
        encoder_attention_mask = sub_mask_2d[:, :logical_sub].to(torch.long)

        text_dec_out = hf_model.text_decoder(
            input_ids=seq[:, :-1],
            encoder_hidden_states=enc_hidden_logical,
            encoder_attention_mask=encoder_attention_mask,
        )
        t2u_input_embeds = text_dec_out.last_hidden_state  # [1, T_text, H]
        T_text = int(t2u_input_embeds.shape[1])

        # T2U attention mask via _compute_new_attention_mask.
        seq_lens = (seq[:, :-1] != self.pad_token_id).int().sum(1)
        t2u_attention_mask = self._compute_new_attention_mask(seq_lens, batch=1, mask_seq_len=T_text)

        # Strip lang_id + decoder_start prefix + trailing EOS to get
        # t2u_input_ids; replace any other EOS with pad (per HF).
        t2u_input_ids = seq[:, 2:-1].clone()
        t2u_input_ids = torch.masked_fill(t2u_input_ids, t2u_input_ids == self.eos_token_id, self.pad_token_id)

        # Compute t2u char inputs (HF helpers).
        t2u_subwords = hf_model._indices_to_subwords(t2u_input_ids)
        t2u_char_count_per_id = hf_model._count_character_length_in_subword(
            t2u_input_ids, t2u_subwords, pad_token_id=self.pad_token_id
        )
        pad_zero = t2u_char_count_per_id.new_zeros((t2u_char_count_per_id.shape[0], 1))
        t2u_char_count_per_id = torch.cat([pad_zero, t2u_char_count_per_id, pad_zero], dim=1)
        t2u_char_input_ids = hf_model._get_char_input_ids(
            t2u_input_ids,
            t2u_subwords,
            t2u_char_count_per_id,
            pad_token_id=self.pad_token_id,
        )

        # ----------------------------------------------------------------
        # 6. T2U generator (TTNN).
        # ----------------------------------------------------------------
        t2u_out = self.t2u_generator.synthesize_units(
            text_decoder_hidden=t2u_input_embeds,
            char_input_ids=t2u_char_input_ids,
            char_count_per_id=t2u_char_count_per_id,
            t2u_attention_mask=t2u_attention_mask,
        )
        unit_token_ids: torch.Tensor = t2u_out["unit_token_ids"]

        # ----------------------------------------------------------------
        # 7. Code HiFi-GAN vocoder (TTNN).
        # ----------------------------------------------------------------
        vocoder_lang_id = self._resolve_vocoder_lang_id(tgt_lang)
        speaker_t = torch.tensor([[int(speaker_id)]] * unit_token_ids.shape[0], dtype=torch.int64)
        lang_t = torch.tensor([[vocoder_lang_id]] * unit_token_ids.shape[0], dtype=torch.int64)
        waveform_tt = self.vocoder(
            input_ids=unit_token_ids,
            speaker_id=speaker_t,
            lang_id=lang_t,
        )
        waveform_torch = ttnn.to_torch(waveform_tt).to(torch.float32)
        ttnn.deallocate(waveform_tt)
        while waveform_torch.dim() > 1 and waveform_torch.shape[0] == 1:
            waveform_torch = waveform_torch.squeeze(0)
        last_lengths = self.vocoder.last_lengths
        if last_lengths is not None:
            try:
                valid = int(last_lengths.item() if last_lengths.dim() == 0 else last_lengths.view(-1)[0].item())
                valid = max(0, min(valid, int(waveform_torch.shape[-1])))
                waveform_torch = waveform_torch[..., :valid].contiguous()
            except Exception:
                pass
        self.last_waveform_length = int(waveform_torch.shape[-1])
        return waveform_torch.detach().cpu().numpy().astype(np.float32)
