# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end SeamlessM4Tv2 S2TT generator (Phase 5).

Pipeline: audio --(CPU fbank)--> TtSpeechEncoder --> autoregressive greedy
TtTextDecoder --> detokenize. Mirrors HF `SeamlessM4Tv2ForSpeechToText.generate`:
the decoder is seeded with the target-language code id
(`generation_config.text_decoder_lang_to_code_id[tgt_lang]`) and greedily extended
until `eos_token_id`.

v1 recomputes the full decoder sequence each step (no KV cache) — correct but
O(n^2); KV-cache decode is Phase 6. Encoder hidden states stay resident on device
across decode steps.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.seamless_m4t_v2.tt.conformer_encoder import TtSpeechEncoder
from models.demos.seamless_m4t_v2.tt.model_config import SeamlessS2TTConfig, DEFAULT_MODEL_ID
from models.demos.seamless_m4t_v2.tt.text_decoder import TtTextDecoder


def _apply_repetition_penalty(row: torch.Tensor, tokens, penalty: float):
    """In-place CTRL-style repetition penalty (Keskar et al.) on a (vocab,) logit row."""
    if penalty == 1.0 or not tokens:
        return
    idx = torch.tensor(sorted(set(tokens)), dtype=torch.long)
    vals = row[idx]
    row[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)


def _block_repeat_ngrams(row: torch.Tensor, tokens, n: int):
    """Mask tokens that would complete a previously-seen n-gram (no_repeat_ngram_size)."""
    if n <= 0 or len(tokens) < n:
        return
    prefix = tuple(tokens[-(n - 1):]) if n > 1 else ()
    for i in range(len(tokens) - n + 1):
        if tuple(tokens[i:i + n - 1]) == prefix:
            row[tokens[i + n - 1]] = float("-inf")


class SeamlessS2TTGenerator:
    def __init__(self, tt_encoder, tt_decoder, processor, config, gen_config, dtype=ttnn.bfloat16):
        self.encoder = tt_encoder
        self.decoder = tt_decoder
        self.processor = processor
        self.config = config
        self.gen_config = gen_config
        self.dtype = dtype

    @classmethod
    def build(cls, device, model_id: str = DEFAULT_MODEL_ID, dtype=ttnn.bfloat16):
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        cfg = SeamlessS2TTConfig.from_hf(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        hf = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_id).eval().float()

        tt_encoder = TtSpeechEncoder(hf.speech_encoder.state_dict(), cfg, device, dtype=dtype)
        tt_decoder = TtTextDecoder(hf.text_decoder.state_dict(), cfg, device, dtype=dtype)
        gen_config = hf.generation_config
        del hf
        return cls(tt_encoder, tt_decoder, processor, cfg, gen_config, dtype)

    def _tgt_lang_id(self, tgt_lang: str) -> int:
        tgt_lang = tgt_lang.replace("__", "")
        mapping = self.gen_config.text_decoder_lang_to_code_id
        if tgt_lang not in mapping:
            raise ValueError(f"tgt_lang={tgt_lang} unsupported; choose from {list(mapping)[:10]}...")
        return int(mapping[tgt_lang])

    def encode(self, audio, device, sampling_rate: int = 16000):
        """audio -> (encoder_hidden_states ttnn, feature seq len)."""
        try:  # transformers >= 4.57 uses `audio`; 4.53 uses `audios`
            inputs = self.processor(audios=audio, sampling_rate=sampling_rate, return_tensors="pt")
        except (TypeError, ValueError):
            inputs = self.processor(audio=audio, sampling_rate=sampling_rate, return_tensors="pt")
        feats = inputs["input_features"].float()  # (1, seq, 160)
        feats_tt = ttnn.from_torch(feats, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=device)
        return self.encoder(feats_tt)

    def _next_token(self, row, tokens, repetition_penalty, no_repeat_ngram_size):
        """row: (vocab,) torch logits. Apply penalties, return greedy argmax."""
        # don't penalise the fixed seed tokens (decoder_start, lang)
        history = tokens[2:]
        _apply_repetition_penalty(row, history, repetition_penalty)
        _block_repeat_ngrams(row, history, no_repeat_ngram_size)
        return int(row.argmax(-1))

    @torch.no_grad()
    def generate(self, audio, device, tgt_lang: str = "jpn", max_new_tokens: int = 256,
                 sampling_rate: int = 16000, bucket_len: int | None = None,
                 repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0):
        """Greedy S2TT.

        bucket_len=None: variable-length decode (recompiles kernels each step — slow).
        bucket_len=M: constant-shape decode — pad decoder input to M every step so the
        kernel shapes stay fixed (compile once). Only one lm_head row is transferred
        per step. Generation stops at eos or when len reaches M.

        repetition_penalty > 1 and no_repeat_ngram_size > 0 suppress greedy loops
        (defaults off so greedy output is bit-identical to HF greedy).
        """
        enc = self.encode(audio, device, sampling_rate)
        eos = self.config.eos_token_id
        pad = self.config.pad_token_id
        # HF prepends decoder_start_token_id before the forced target-language token.
        tokens = [self.config.decoder_start_token_id, self._tgt_lang_id(tgt_lang)]

        if bucket_len is None:
            for _ in range(max_new_tokens):
                ids = torch.tensor([tokens], dtype=torch.long)
                logits = ttnn.to_torch(self.decoder(ids, enc))
                nxt = self._next_token(logits[0, -1].float(), tokens, repetition_penalty, no_repeat_ngram_size)
                tokens.append(nxt)
                if nxt == eos:
                    break
        else:
            M = bucket_len
            limit = min(max_new_tokens + len(tokens), M)
            while len(tokens) < limit:
                cur = len(tokens)
                padded = tokens + [pad] * (M - cur)
                ids = torch.tensor([padded], dtype=torch.long)
                logits = self.decoder(ids, enc)  # (1, M, vocab), constant shape
                row = ttnn.to_torch(logits[:, cur - 1:cur, :])[0, 0].float()  # needed position
                nxt = self._next_token(row, tokens, repetition_penalty, no_repeat_ngram_size)
                tokens.append(nxt)
                if nxt == eos:
                    break
        ttnn.deallocate(enc)
        return self.processor.decode(tokens, skip_special_tokens=True), tokens
