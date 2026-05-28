# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""High-level T2ST (text-to-speech translation) wrapper for SeamlessM4T-v2.

Pipeline mirroring HF ``SeamlessM4Tv2ForTextToSpeech.generate``:

    src text
        -> processor (HF tokenizer)
        -> [TTNN]  TextEncoder.encode(input_ids)     -> encoder_hidden
        -> [TTNN]  TextGenerator.generate(...)       -> token sequences
        -> [HF]    text_decoder(sequences[:, :-1])   -> t2u_input_embeds  (re-run on host)
        -> [HF]    _indices_to_subwords + _count_char + _get_char_input_ids
        -> [TTNN]  T2UGenerator.synthesize_units      -> unit_token_ids
        -> [TTNN]  CodeHifiGanVocoder                -> waveform

Hybrid boundary -- the host (HF) handles:
    1. Re-running ``text_decoder`` to recover its ``last_hidden_state`` over
       the generated token sequence (our TTNN AR text_decoder exposes
       per-step hidden states but not a single full-sequence tensor in one
       pass with cross-attention; re-using HF here is dramatically simpler
       and the cost is just one host CPU forward).
    2. Character-input prep (``_indices_to_subwords``, ``_count_character_length_in_subword``,
       ``_get_char_input_ids``) which read ``generation_config.id_to_text``
       and ``generation_config.char_to_id`` and are tokeniser-bound rather
       than model-bound. Re-implementing them is not worth the bit-exact
       reproduction risk.

Everything else (text encoder, text AR generator, t2u_encoder, t2u_decoder
with duration upsample, t2u lm_head + argmax, code HiFi-GAN vocoder) runs
in TTNN.

Example::

    from transformers import AutoProcessor
    import ttnn
    from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
    from models.demos.facebook_seamless_m4t_v2_large.tt.text_to_speech_model import (
        TextToSpeechModel,
    )

    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    hf_sd = wl.load_hf_state_dict()
    proc = AutoProcessor.from_pretrained(wl.HF_PATH)
    model = TextToSpeechModel(device=device, hf_state_dict=hf_sd, processor=proc)
    audio_np = model.synthesize("Hello world.", src_lang="eng", tgt_lang="fra")
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
from models.demos.facebook_seamless_m4t_v2_large.tt.code_hifigan_vocoder import CodeHifiGanVocoder
from models.demos.facebook_seamless_m4t_v2_large.tt.t2u_generator import T2UGenerator
from models.demos.facebook_seamless_m4t_v2_large.tt.text_encoder import TextEncoder
from models.demos.facebook_seamless_m4t_v2_large.tt.text_generator import TextGenerator

# ---------------------------------------------------------------------------
# Model config (matches SeamlessM4T-v2-Large defaults)
# ---------------------------------------------------------------------------
EMBED_DIM = 1024
NUM_HEADS = 16
HEAD_DIM = 64
NUM_TEXT_LAYERS = 24
EPS = 1e-5
ENCODER_PADDING_IDX = 0
DECODER_PADDING_IDX = 0
DEFAULT_MAX_DECODE_SEQ_LEN = 128
PAD_TOKEN_ID = 0
T2U_PAD_TOKEN_ID = 1  # the t2u-prefix-stripped pad_token_id
SAMPLING_RATE = 16000

_TILE = 32


def _pad_to_tile(n: int) -> int:
    return (_TILE - n % _TILE) % _TILE


def _tile_pad_encoder_hidden(enc_hidden: torch.Tensor) -> torch.Tensor:
    s = int(enc_hidden.shape[1])
    pad = _pad_to_tile(s)
    if pad == 0:
        return enc_hidden
    z = torch.zeros((enc_hidden.shape[0], pad, enc_hidden.shape[2]), dtype=enc_hidden.dtype)
    return torch.cat([enc_hidden, z], dim=1)


def _load_generation_config(hf_path: str) -> Dict:
    cfg_path = Path(hf_path) / "generation_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"generation_config.json not found under {hf_path}")
    with open(cfg_path) as f:
        return json.load(f)


class TextToSpeechModel:
    """SeamlessM4T-v2 text-to-speech (T2ST) synthesiser (TTNN + HF hybrid).

    Args:
        device: opened ttnn device.
        hf_state_dict: result of :func:`weight_loader.load_hf_state_dict`.
        processor: HuggingFace ``SeamlessM4Tv2Processor`` (or any
            ``AutoProcessor`` over the same checkpoint).
        hf_path: path to local checkpoint snapshot. Defaults to
            ``weight_loader.HF_PATH``.
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
        max_decode_seq_len: int = DEFAULT_MAX_DECODE_SEQ_LEN,
        weight_dtype=ttnn.bfloat16,
    ):
        if max_decode_seq_len % _TILE != 0:
            raise ValueError(f"max_decode_seq_len({max_decode_seq_len}) must be a multiple of {_TILE}")

        self.device = device
        self.processor = processor
        self.hf_path = hf_path or wl.HF_PATH
        self.max_decode_seq_len = int(max_decode_seq_len)
        self.weight_dtype = weight_dtype
        self._hf_state_dict = hf_state_dict

        self.gen_cfg = _load_generation_config(self.hf_path)
        self.decoder_start_token_id = int(self.gen_cfg.get("decoder_start_token_id", 3))
        self.eos_token_id = int(self.gen_cfg.get("eos_token_id", 3))
        self.pad_token_id = int(self.gen_cfg.get("pad_token_id", PAD_TOKEN_ID))
        self.lang_to_code_id: Dict[str, int] = self.gen_cfg["text_decoder_lang_to_code_id"]
        self.vocoder_lang_code_to_id: Dict[str, int] = self.gen_cfg["vocoder_lang_code_to_id"]

        # ----- Text encoder + AR text generator (TTNN) ----------------------
        enc_sd = wl.text_encoder_weights(hf_state_dict, num_layers=NUM_TEXT_LAYERS, padding_idx=ENCODER_PADDING_IDX)
        self.text_encoder = TextEncoder(
            device=device,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            embed_tokens_weight=enc_sd["embed_tokens"]["weight"],
            embed_positions_weights=enc_sd["embed_positions_weights"],
            layers_state_dict=enc_sd["layers"],
            final_layer_norm_state_dict=enc_sd["final_layer_norm"],
            eps=EPS,
            padding_idx=ENCODER_PADDING_IDX,
            embed_scale=math.sqrt(EMBED_DIM),
            weight_dtype=weight_dtype,
        )

        self._text_decoder_sd = wl.text_decoder_weights(
            hf_state_dict, num_layers=NUM_TEXT_LAYERS, padding_idx=DECODER_PADDING_IDX
        )
        self._lm_head_sd = wl.lm_head_weights(hf_state_dict)
        self._text_generators: Dict[int, TextGenerator] = {}

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
        # Lazy-loaded to avoid the ~8 GB host alloc until first synthesize().
        self._hf_t2s_model = None

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

    def _get_or_build_text_generator(self, encoder_seq_len_padded: int) -> TextGenerator:
        gen = self._text_generators.get(encoder_seq_len_padded)
        if gen is not None:
            return gen
        gen = TextGenerator(
            device=self.device,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            num_layers=NUM_TEXT_LAYERS,
            text_decoder_state_dict=self._text_decoder_sd,
            lm_head_state_dict=self._lm_head_sd,
            max_decode_seq_len=self.max_decode_seq_len,
            encoder_seq_len=encoder_seq_len_padded,
            eps=EPS,
            padding_idx=DECODER_PADDING_IDX,
            embed_scale=math.sqrt(EMBED_DIM),
            weight_dtype=self.weight_dtype,
        )
        self._text_generators[encoder_seq_len_padded] = gen
        return gen

    def _run_text_encoder(self, input_ids: torch.Tensor, attn_mask_2d: torch.Tensor) -> torch.Tensor:
        from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

        tgt_len = int(input_ids.shape[-1])
        mask_4d = _prepare_4d_attention_mask(attn_mask_2d, torch.float32, tgt_len=tgt_len)
        enc_hidden_tt = self.text_encoder(input_ids, attention_mask_torch=mask_4d)
        enc_hidden_torch = ttnn.to_torch(enc_hidden_tt).to(torch.float32)
        ttnn.deallocate(enc_hidden_tt)
        if enc_hidden_torch.dim() == 4 and enc_hidden_torch.shape[0] == 1:
            enc_hidden_torch = enc_hidden_torch.squeeze(0)
        return enc_hidden_torch

    def _load_hf_helper_model(self):
        """Lazy-load HF SeamlessM4Tv2ForTextToSpeech.

        We only call ``self.text_decoder(...)`` and the char helpers from
        this instance; the rest is unused. Keeping it lazy avoids paying the
        ~8 GB host RAM cost until the first synthesize() call.
        """
        if self._hf_t2s_model is None:
            from transformers import SeamlessM4Tv2ForTextToSpeech

            model = SeamlessM4Tv2ForTextToSpeech.from_pretrained(self.hf_path, torch_dtype=torch.float32)
            model.eval()
            self._hf_t2s_model = model
        return self._hf_t2s_model

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
        src_text: str,
        src_lang: str,
        tgt_lang: str,
        speaker_id: int = 0,
        max_new_tokens: int = 128,
        use_trace: bool = False,
    ) -> np.ndarray:
        """Synthesize speech for ``src_text`` translated into ``tgt_lang``.

        Args:
            use_trace: when True, the AR text-decoder runs under a
                metal-trace replay that is **captured fresh inside this
                synthesize() call and explicitly released before T2U +
                vocoder run**. The cross-call trace reuse pattern used
                by T2TT/S2TT does not apply here: T2U + vocoder allocate
                fresh device buffers, and that is unsafe while a trace
                is armed (the new buffers can be corrupted by trace
                replay). This single-call trace pattern pays a per-call
                recapture cost (~30-40 ms) in exchange for trace-pace
                AR steps -- a small net win as soon as the AR loop
                runs >~20 tokens.

        Returns:
            ``np.ndarray`` of shape ``(T_samples,)`` float32 at 16 kHz in
            roughly ``[-1, 1]``. Trim to ``waveform_length`` if you want only
            the valid prefix (also exposed via ``self.last_waveform_length``).
        """
        # ----------------------------------------------------------------
        # 1. Tokenise source.
        # ----------------------------------------------------------------
        toks = self.processor(text=src_text, src_lang=src_lang, return_tensors="pt")
        input_ids: torch.Tensor = toks["input_ids"]
        attn_mask: torch.Tensor = toks["attention_mask"]
        src_logical = int(attn_mask.sum().item())
        input_ids = input_ids[:, :src_logical]
        attn_mask = attn_mask[:, :src_logical]

        # ----------------------------------------------------------------
        # 2. Text encoder forward (TTNN).
        # ----------------------------------------------------------------
        enc_hidden_logical = self._run_text_encoder(input_ids, attn_mask)
        enc_hidden_padded = _tile_pad_encoder_hidden(enc_hidden_logical[:, :src_logical, :])
        encoder_seq_len_padded = int(enc_hidden_padded.shape[1])

        # ----------------------------------------------------------------
        # 3. AR text decoder + LM head (TTNN, greedy).
        # ----------------------------------------------------------------
        gen = self._get_or_build_text_generator(encoder_seq_len_padded)
        text_tgt_lang_id = self._resolve_text_tgt_lang_id(tgt_lang)
        max_total = min(int(max_new_tokens), self.max_decode_seq_len)
        text_tokens = gen.generate(
            encoder_hidden_states=enc_hidden_padded,
            encoder_attention_mask=attn_mask,
            decoder_start_token_id=self.decoder_start_token_id,
            tgt_lang_id=text_tgt_lang_id,
            eos_token_id=self.eos_token_id,
            max_new_tokens=max_total,
            do_sample=False,
            use_trace=use_trace,
        )
        # CRITICAL: release the AR trace before T2U+vocoder allocate
        # fresh device buffers. With an armed trace, those allocations
        # would be unsafe (ttnn warns + the buffers may be corrupted
        # the next time the trace replays). The next synthesize() call
        # re-captures cleanly.
        if use_trace:
            gen.release_trace()
        # text_tokens is a 1-D list/tensor of token ids that ends with EOS.
        # Normalise to [1, T] long.
        if isinstance(text_tokens, torch.Tensor):
            seq = text_tokens.to(torch.int64).view(1, -1)
        else:
            seq = torch.tensor(text_tokens, dtype=torch.int64).view(1, -1)

        # Make sure sequence ends with EOS so the [:-1] trim below matches HF.
        if int(seq[0, -1].item()) != self.eos_token_id:
            seq = torch.cat([seq, torch.tensor([[self.eos_token_id]], dtype=torch.int64)], dim=1)

        # ----------------------------------------------------------------
        # 4. Hybrid boundary: HF re-runs the text_decoder over sequences[:, :-1]
        #    to get a full-sequence last_hidden_state, then computes the T2U
        #    char inputs and attention mask using the same generation_config.
        # ----------------------------------------------------------------
        hf_model = self._load_hf_helper_model()
        # 4a. Re-run text_decoder for hidden states.
        encoder_attention_mask = attn_mask
        text_dec_out = hf_model.text_decoder(
            input_ids=seq[:, :-1],
            encoder_hidden_states=enc_hidden_logical.to(torch.float32)[:, :src_logical, :],
            encoder_attention_mask=encoder_attention_mask,
        )
        t2u_input_embeds = text_dec_out.last_hidden_state  # [1, T_text, H]
        T_text = int(t2u_input_embeds.shape[1])

        # 4b. T2U attention mask via _compute_new_attention_mask.
        seq_lens = (seq[:, :-1] != self.pad_token_id).int().sum(1)  # [B]
        t2u_attention_mask = self._compute_new_attention_mask(seq_lens, batch=1, mask_seq_len=T_text)

        # 4c. Strip lang_id + decoder_start prefix + trailing EOS to get t2u_input_ids
        # and replace any subsequent EOS with t2u-prefix-stripped pad_token_id (per HF).
        t2u_input_ids = seq[:, 2:-1].clone()
        t2u_input_ids = torch.masked_fill(t2u_input_ids, t2u_input_ids == self.eos_token_id, self.pad_token_id)

        # 4d. Compute t2u char inputs (HF helpers).
        t2u_subwords = hf_model._indices_to_subwords(t2u_input_ids)
        t2u_char_count_per_id = hf_model._count_character_length_in_subword(
            t2u_input_ids, t2u_subwords, pad_token_id=self.pad_token_id
        )
        # Add zero pads for lang_id and EOS (per HF).
        pad_zero = t2u_char_count_per_id.new_zeros((t2u_char_count_per_id.shape[0], 1))
        t2u_char_count_per_id = torch.cat([pad_zero, t2u_char_count_per_id, pad_zero], dim=1)
        t2u_char_input_ids = hf_model._get_char_input_ids(
            t2u_input_ids,
            t2u_subwords,
            t2u_char_count_per_id,
            pad_token_id=self.pad_token_id,
        )

        # ----------------------------------------------------------------
        # 5. T2U generator (TTNN).
        # ----------------------------------------------------------------
        t2u_out = self.t2u_generator.synthesize_units(
            text_decoder_hidden=t2u_input_embeds,
            char_input_ids=t2u_char_input_ids,
            char_count_per_id=t2u_char_count_per_id,
            t2u_attention_mask=t2u_attention_mask,
        )
        unit_token_ids: torch.Tensor = t2u_out["unit_token_ids"]  # [B, T_u]

        # ----------------------------------------------------------------
        # 6. Code HiFi-GAN vocoder (TTNN).
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
        # Strip extra dims; expect [1, T_samples] or [T_samples].
        while waveform_torch.dim() > 1 and waveform_torch.shape[0] == 1:
            waveform_torch = waveform_torch.squeeze(0)
        # Trim to valid sample count.
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
