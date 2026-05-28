# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""High-level S2TT / ASR wrapper for SeamlessM4T-v2.

Composes the verified TTNN sub-models:

    - :class:`SpeechEncoder`   (W2v-BERT-2.0: 24 Conformer layers + 1 adapter)
    - :class:`TextGenerator`   (24-layer NLLB text decoder + LM head + AR loop)

into a single ``translate(audio_path, src_lang, tgt_lang) -> str`` API,
mirroring HuggingFace's ``SeamlessM4Tv2ForSpeechToText.generate`` flow.

ASR is exactly the same path with ``tgt_lang == src_lang``.

Audio is preprocessed on the host with HuggingFace's
``SeamlessM4TFeatureExtractor`` (80 mel bins, stride=2 -> 160-dim
features) and resampled to 16 kHz before extraction.

The TTNN ``SpeechEncoder`` is constructed for a fixed ``seq_len``
(the post-feature-extraction time axis, tile-aligned). On ``translate``
we pad / truncate the extracted features and the corresponding
attention mask to that ``seq_len`` so the encoder's relative-key
positional bias and chunked-attention masks remain valid for every
input length below the budget.

Example::

    import ttnn
    from transformers import AutoProcessor
    from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
    from models.demos.facebook_seamless_m4t_v2_large.tt.speech_to_text_model import (
        SpeechToTextModel,
    )

    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    hf_sd = wl.load_hf_state_dict()
    processor = AutoProcessor.from_pretrained(wl.HF_PATH)

    model = SpeechToTextModel(device=device, hf_state_dict=hf_sd, processor=processor)
    text = model.translate("hello.wav", src_lang="eng", tgt_lang="fra")
    print(text)  # -> French translation of the spoken input
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
from models.demos.facebook_seamless_m4t_v2_large.tt.speech_encoder import SpeechEncoder
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
FEATURE_SIZE = 160
SAMPLING_RATE = 16000
ADAPTOR_KERNEL_SIZE = 8
ADAPTOR_STRIDE = 8
EPS = 1e-5
DECODER_PADDING_IDX = 0
DEFAULT_MAX_DECODE_SEQ_LEN = 128
DEFAULT_AUDIO_SEQ_LEN = 256  # post-feature-extraction time axis budget (tile-aligned, ~5.1 s)

_TILE = 32


def _pad_to_tile(n: int) -> int:
    return (_TILE - n % _TILE) % _TILE


def _load_generation_config(hf_path: str) -> Dict:
    cfg_path = Path(hf_path) / "generation_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"generation_config.json not found under {hf_path}")
    with open(cfg_path) as f:
        return json.load(f)


def _load_wav_to_16k_mono(audio_path: str) -> np.ndarray:
    """Read a WAV from disk and return ``float32`` mono samples at 16 kHz.

    Uses ``scipy.io.wavfile`` to load, then ``torchaudio.functional.resample``
    when needed. ``int16`` PCM is normalised to ``[-1, 1)``.
    """
    import scipy.io.wavfile as wav

    sr, data = wav.read(audio_path)
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        audio = data
    elif data.dtype == np.float64:
        audio = data.astype(np.float32)
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV dtype: {data.dtype}")

    # Downmix to mono.
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != SAMPLING_RATE:
        import torchaudio

        t = torch.from_numpy(audio).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr, SAMPLING_RATE)
        audio = t.squeeze(0).numpy()
    return np.ascontiguousarray(audio, dtype=np.float32)


def _extract_features(
    audio: np.ndarray,
    processor,
    target_seq_len: int,
    max_audio_seconds: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Run the HF feature extractor and pad / truncate to ``target_seq_len``.

    Returns:
        input_features: ``[1, target_seq_len, FEATURE_SIZE]`` float32.
        attention_mask: ``[1, target_seq_len]`` int64 with 1=keep, 0=pad.
        logical_len: the unpadded feature length (``min(T, target_seq_len)``).
    """
    if max_audio_seconds is not None:
        max_samples = int(max_audio_seconds * SAMPLING_RATE)
        audio = audio[:max_samples]

    fe = processor.feature_extractor
    out = fe(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    input_features: torch.Tensor = out["input_features"].to(torch.float32)  # [1, T, 160]
    attention_mask: torch.Tensor = out["attention_mask"].to(torch.long)  # [1, T]
    if input_features.dim() != 3 or input_features.shape[-1] != FEATURE_SIZE:
        raise ValueError(
            f"Unexpected feature extractor output shape: {tuple(input_features.shape)}; "
            f"expected [1, T, {FEATURE_SIZE}]"
        )

    t = int(input_features.shape[1])
    if t > target_seq_len:
        input_features = input_features[:, :target_seq_len, :]
        attention_mask = attention_mask[:, :target_seq_len]
        logical_len = target_seq_len
    elif t < target_seq_len:
        pad = target_seq_len - t
        feat_pad = torch.zeros((1, pad, FEATURE_SIZE), dtype=input_features.dtype)
        input_features = torch.cat([input_features, feat_pad], dim=1)
        mask_pad = torch.zeros((1, pad), dtype=attention_mask.dtype)
        attention_mask = torch.cat([attention_mask, mask_pad], dim=1)
        logical_len = t
    else:
        logical_len = t
    return input_features, attention_mask, logical_len


def _compute_sub_seq_len(seq_len: int, kernel_size: int = ADAPTOR_KERNEL_SIZE, stride: int = ADAPTOR_STRIDE) -> int:
    """floor((T + 2*pad - kernel)/stride) + 1, with pad = stride // 2."""
    pad = stride // 2
    return ((seq_len + 2 * pad - kernel_size) // stride) + 1


class SpeechToTextModel:
    """SeamlessM4T-v2 speech-to-text (S2TT / ASR) translator (TTNN).

    Args:
        device: opened ttnn device.
        hf_state_dict: result of :func:`weight_loader.load_hf_state_dict`.
        processor: HuggingFace ``SeamlessM4TProcessor`` (or any
            ``AutoProcessor`` over the same checkpoint). The feature
            extractor (``processor.feature_extractor``) does log-mel +
            stride-2 stacking on the host.
        hf_path: path to the local checkpoint snapshot — used to read
            ``generation_config.json``.
        audio_seq_len: post-feature-extraction time axis budget. Must
            be a multiple of 32. Defaults to 256 (~5 s of 16 kHz audio).
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

        self.gen_cfg = _load_generation_config(self.hf_path)
        self.decoder_start_token_id = int(self.gen_cfg.get("decoder_start_token_id", 3))
        self.eos_token_id = int(self.gen_cfg.get("eos_token_id", 3))
        self.lang_to_code_id: Dict[str, int] = self.gen_cfg["text_decoder_lang_to_code_id"]

        # ----- Speech encoder (built once, reused for all translate() calls) -----
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

        # ----- Text decoder side state dicts (kept on host until TextGenerator wraps them) -----
        self._text_decoder_sd = wl.text_decoder_weights(
            hf_state_dict,
            num_layers=NUM_DECODER_LAYERS,
            padding_idx=DECODER_PADDING_IDX,
        )
        self._lm_head_sd = wl.lm_head_weights(hf_state_dict)

        # Build the TextGenerator. The cross-attn cache is sized to the
        # tile-padded post-adapter encoder length.
        self.encoder_seq_len_padded = self.sub_seq_len + _pad_to_tile(self.sub_seq_len)
        self.generator = TextGenerator(
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

    # ------------------------------------------------------------------ helpers
    def _resolve_tgt_lang_id(self, tgt_lang: str) -> int:
        if tgt_lang not in self.lang_to_code_id:
            raise ValueError(
                f"tgt_lang={tgt_lang!r} not in generation_config text_decoder_lang_to_code_id; "
                f"valid examples: fra, spa, deu, eng."
            )
        return int(self.lang_to_code_id[tgt_lang])

    def _to_tt_features(self, input_features: torch.Tensor) -> ttnn.Tensor:
        """Move ``[1, T, 160]`` float features to TILE_LAYOUT DRAM."""
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
        """Compute the post-adapter 2-D padding mask + logical length.

        HF's S2TT decoder cross-attn uses the down-sampled mask
        ``[B, T_sub]`` (1=keep, 0=pad). This mirrors
        ``_compute_new_attention_mask`` exactly.
        """
        # seq_lens: pre-stride non-pad lengths per batch.
        seq_lens = attention_mask_2d.size(1) - (1 - attention_mask_2d.int()).sum(1)
        pad = ADAPTOR_STRIDE // 2
        sub_lens = (((seq_lens + 2 * pad - ADAPTOR_KERNEL_SIZE) / ADAPTOR_STRIDE) + 1).floor().long()
        sub_lens = sub_lens.clamp(min=0, max=self.sub_seq_len)
        # Build [B, sub_seq_len] keep-mask from per-batch lengths.
        batch = attention_mask_2d.shape[0]
        sub_mask = torch.zeros((batch, self.sub_seq_len), dtype=torch.long)
        for b in range(batch):
            n = int(sub_lens[b].item())
            sub_mask[b, :n] = 1
        # Treat batch 1 only (we don't batch).
        logical_sub = int(sub_lens[0].item())
        return sub_mask, logical_sub

    # ------------------------------------------------------------------ public API
    def translate(
        self,
        audio_path: str,
        src_lang: str,
        tgt_lang: str,
        max_new_tokens: int = 128,
        max_audio_seconds: Optional[float] = 5.0,
    ) -> str:
        """Translate ``audio_path`` from ``src_lang`` to ``tgt_lang``.

        ``src_lang`` is accepted only for API symmetry with the T2TT
        wrapper / HF's ``generate(src_lang=...)`` for input tokenisation;
        the speech encoder is language-agnostic, so the value is not
        used to gate any path. For ASR pass ``tgt_lang == src_lang``.

        Returns the decoded transcription / translation string.
        """
        # 1. Load + resample + feature-extract on host.
        audio = _load_wav_to_16k_mono(audio_path)
        input_features, attention_mask_2d, logical_len = _extract_features(
            audio,
            processor=self.processor,
            target_seq_len=self.audio_seq_len,
            max_audio_seconds=max_audio_seconds,
        )

        # 2. Speech encoder forward (TTNN).
        feat_tt = self._to_tt_features(input_features)
        enc_hidden_tt = self.speech_encoder(feat_tt, attention_mask_2d=attention_mask_2d)
        enc_hidden_torch = ttnn.to_torch(enc_hidden_tt).to(torch.float32)
        ttnn.deallocate(enc_hidden_tt)
        # Expected shape: [1, sub_seq_len, hidden]. Some leaves return [B, S, H] via reshape;
        # strip extra dims if present.
        while enc_hidden_torch.dim() > 3:
            enc_hidden_torch = enc_hidden_torch.squeeze(0)
        enc_hidden_torch = enc_hidden_torch.reshape(1, self.sub_seq_len, EMBED_DIM)

        # 3. Build post-adapter cross-attention mask and tile-pad encoder hidden.
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

        # 4. Text decoder + LM head + greedy AR loop.
        tgt_lang_id = self._resolve_tgt_lang_id(tgt_lang)
        max_total = min(int(max_new_tokens), self.max_decode_seq_len)
        tokens = self.generator.generate(
            encoder_hidden_states=enc_hidden_padded,
            encoder_attention_mask=sub_mask_2d_padded,
            decoder_start_token_id=self.decoder_start_token_id,
            tgt_lang_id=tgt_lang_id,
            eos_token_id=self.eos_token_id,
            max_new_tokens=max_total,
            do_sample=False,
        )
        text = self.processor.decode(tokens, skip_special_tokens=True)
        return text

    # Alias for ASR (src_lang == tgt_lang).
    def transcribe(
        self,
        audio_path: str,
        lang: str,
        max_new_tokens: int = 128,
        max_audio_seconds: Optional[float] = 5.0,
    ) -> str:
        """ASR alias: ``translate(audio, lang, lang)``."""
        return self.translate(
            audio_path=audio_path,
            src_lang=lang,
            tgt_lang=lang,
            max_new_tokens=max_new_tokens,
            max_audio_seconds=max_audio_seconds,
        )
