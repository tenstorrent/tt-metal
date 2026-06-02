# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference harness for SeamlessM4Tv2 speech-to-text translation (S2TT).

Wraps HuggingFace `SeamlessM4Tv2ForSpeechToText` and exposes:
  - the CPU feature extractor (fbank) via `SeamlessM4TProcessor`,
  - the full speech encoder with optional capture of every intermediate tensor
    (feature_projection / per-conformer-layer / encoder-norm / intermediate_ffn /
    adapter / inner_layer_norm), used as PCC golden in the TT bring-up tests,
  - a single decoder step (logits) given a golden encoder output,
  - end-to-end `generate()` for en->ja.

The model is loaded lazily so importing this module is cheap and does not trigger
the ~9GB checkpoint download. Only `S2TTReference.load()` pulls weights.

Submodule map (transformers 4.57.3, modeling_seamless_m4t_v2.py):
  model.speech_encoder.feature_projection          (LN + Linear 160->1024)
  model.speech_encoder.encoder.layers[i]           (24 conformer layers)
  model.speech_encoder.encoder.layer_norm          (post-stack LN)
  model.speech_encoder.intermediate_ffn            (relu FFN, h + 0.5*ffn(h))
  model.speech_encoder.adapter.layers[0]           (stride-8 downsample adapter)
  model.speech_encoder.inner_layer_norm            (final encoder LN)
  model.text_decoder                               (24-layer NLLB decoder)
  model.lm_head                                    (1024 -> 256102, tied to shared)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

DEFAULT_MODEL_ID = "facebook/seamless-m4t-v2-large"


@dataclass
class S2TTReference:
    """Lazily-loaded HF reference for the S2TT path."""

    model_id: str = DEFAULT_MODEL_ID
    dtype: torch.dtype = torch.float32
    model: Optional[torch.nn.Module] = None
    processor: object = None
    config: object = None
    _captures: dict = field(default_factory=dict)
    _hook_handles: list = field(default_factory=list)

    @classmethod
    def load(cls, model_id: str = DEFAULT_MODEL_ID, dtype: torch.dtype = torch.float32) -> "S2TTReference":
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        self = cls(model_id=model_id, dtype=dtype)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(model_id, dtype=dtype)
        self.model.eval()
        self.config = self.model.config
        return self

    # ------------------------------------------------------------------ #
    # Feature extraction (CPU)                                           #
    # ------------------------------------------------------------------ #
    def extract_features(self, audio, sampling_rate: int = 16000):
        """Run the fbank feature extractor. Returns (input_features, attention_mask)."""
        inputs = self.processor(audio=audio, sampling_rate=sampling_rate, return_tensors="pt")
        feats = inputs["input_features"].to(self.dtype)
        mask = inputs.get("attention_mask", None)
        return feats, mask

    # ------------------------------------------------------------------ #
    # Speech encoder with intermediate capture                          #
    # ------------------------------------------------------------------ #
    def _register_encoder_hooks(self):
        self._remove_hooks()
        self._captures = {}
        enc = self.model.speech_encoder

        def save(name):
            def hook(_module, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                self._captures[name] = t.detach().clone()

            return hook

        h = self._hook_handles
        h.append(enc.feature_projection.register_forward_hook(save("feature_projection")))
        for i, layer in enumerate(enc.encoder.layers):
            h.append(layer.register_forward_hook(save(f"conformer_layer_{i}")))
        h.append(enc.encoder.layer_norm.register_forward_hook(save("encoder_layer_norm")))
        h.append(enc.intermediate_ffn.register_forward_hook(save("intermediate_ffn")))
        if enc.adapter is not None:
            h.append(enc.adapter.layers[0].register_forward_hook(save("adapter_layer_0")))
        h.append(enc.inner_layer_norm.register_forward_hook(save("inner_layer_norm")))

    def _remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    @torch.no_grad()
    def run_speech_encoder(self, input_features, attention_mask=None, capture: bool = False):
        """Return encoder last_hidden_state; if capture, also fill `self._captures`."""
        if capture:
            self._register_encoder_hooks()
        out = self.model.speech_encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=True,
        )
        if capture:
            self._remove_hooks()
        return out.last_hidden_state

    @property
    def captures(self) -> dict:
        return self._captures

    # ------------------------------------------------------------------ #
    # Single decoder step (logits) given golden encoder output          #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decoder_logits(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask=None):
        """One forward of the text decoder + lm_head (no cache). Returns logits."""
        dec_out = self.model.text_decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return self.model.lm_head(dec_out.last_hidden_state)

    # ------------------------------------------------------------------ #
    # End-to-end generation                                             #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_text(self, input_features, attention_mask=None, tgt_lang: str = "jpn", **gen_kwargs):
        """Run greedy (default) S2TT generation and return decoded text."""
        tokens = self.model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            tgt_lang=tgt_lang,
            **gen_kwargs,
        )
        ids = tokens[0].tolist() if hasattr(tokens[0], "tolist") else tokens[0]
        return self.processor.decode(ids, skip_special_tokens=True)
