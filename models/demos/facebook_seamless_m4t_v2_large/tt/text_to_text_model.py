# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""High-level T2TT (text-to-text translation) wrapper for SeamlessM4T-v2.

Composes the verified TTNN sub-models:

    - :class:`TextEncoder`     (24-layer NLLB-style text encoder)
    - :class:`TextGenerator`   (24-layer text decoder + LM head + AR loop)

into a single ``translate(src_text, src_lang, tgt_lang) -> str`` API,
mirroring HuggingFace's ``SeamlessM4Tv2ForTextToText.generate`` flow.

Construction loads weights ONCE; subsequent ``translate()`` calls reuse
the same encoder + decoder weights on device. The encoder is re-run per
call (different source length and tokens) and the cross-attention KV
cache is reset+repopulated each time. The text-decoder's self-attention
cache is also reset each call.

The current implementation rebuilds the ``TextGenerator`` instance
whenever the tile-padded source length changes (the generator
pre-allocates the cross-attention cache to a fixed S). Cached
TextGenerator instances are keyed by the padded encoder length so
repeated short prompts at the same padded S stay fast.

Example::

    import ttnn
    from transformers import AutoProcessor
    from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
    from models.demos.facebook_seamless_m4t_v2_large.tt.text_to_text_model import (
        TextToTextModel,
    )

    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    hf_sd = wl.load_hf_state_dict()
    proc = AutoProcessor.from_pretrained(wl.HF_PATH)

    model = TextToTextModel(device=device, hf_state_dict=hf_sd, processor=proc)
    out = model.translate("Hello world.", "eng", "fra")
    print(out)  # -> "Salut à vous, monde." (or similar greedy output)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt import weight_loader as wl
from models.demos.facebook_seamless_m4t_v2_large.tt.text_encoder import TextEncoder
from models.demos.facebook_seamless_m4t_v2_large.tt.text_generator import TextGenerator

# ---------------------------------------------------------------------------
# Model config (matches SeamlessM4T-v2-Large defaults)
# ---------------------------------------------------------------------------
EMBED_DIM = 1024
NUM_HEADS = 16
HEAD_DIM = 64
NUM_LAYERS = 24
EPS = 1e-5
ENCODER_PADDING_IDX = 0
DECODER_PADDING_IDX = 0
DEFAULT_MAX_DECODE_SEQ_LEN = 128  # tile-aligned cache capacity (incl. 2 prefix tokens)

_TILE = 32


def _pad_to_tile(n: int) -> int:
    return (_TILE - n % _TILE) % _TILE


def _tile_pad_encoder_hidden(enc_hidden: torch.Tensor) -> torch.Tensor:
    """Right-pad encoder hidden states along S to a tile multiple with zeros."""
    s = int(enc_hidden.shape[1])
    pad = _pad_to_tile(s)
    if pad == 0:
        return enc_hidden
    z = torch.zeros((enc_hidden.shape[0], pad, enc_hidden.shape[2]), dtype=enc_hidden.dtype)
    return torch.cat([enc_hidden, z], dim=1)


def _load_generation_config(hf_path: str) -> Dict:
    """Parse the HF ``generation_config.json`` next to the checkpoint."""
    cfg_path = Path(hf_path) / "generation_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"generation_config.json not found under {hf_path}")
    with open(cfg_path) as f:
        return json.load(f)


class TextToTextModel:
    """SeamlessM4T-v2 text-to-text translator (TTNN).

    Args:
        device: opened ttnn device.
        hf_state_dict: result of :func:`weight_loader.load_hf_state_dict`. Loaded
            once on host; partitioned by ``weight_loader.text_encoder_weights`` /
            ``text_decoder_weights`` / ``lm_head_weights`` internally.
        processor: HuggingFace ``SeamlessM4Tv2Processor`` (or any
            ``AutoProcessor`` over the same checkpoint). Used for both
            input tokenization and output detokenization.
        hf_path: path to the local checkpoint snapshot — used to read
            ``generation_config.json`` (decoder start, EOS, lang code IDs).
            Defaults to ``weight_loader.HF_PATH``.
        max_decode_seq_len: KV-cache slot count for the decoder side
            (must be a multiple of 32). Limits total generation length
            (including the 2-token prefix).
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

        self.gen_cfg = _load_generation_config(self.hf_path)
        self.decoder_start_token_id = int(self.gen_cfg.get("decoder_start_token_id", 3))
        self.eos_token_id = int(self.gen_cfg.get("eos_token_id", 3))
        self.lang_to_code_id: Dict[str, int] = self.gen_cfg["text_decoder_lang_to_code_id"]

        # ----- Encoder (built once, reused for all translate() calls) -----
        enc_sd = wl.text_encoder_weights(hf_state_dict, num_layers=NUM_LAYERS, padding_idx=ENCODER_PADDING_IDX)
        self.encoder = TextEncoder(
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

        # ----- Decoder state dicts (kept on host; TextGenerator copies to device) -----
        self._text_decoder_sd = wl.text_decoder_weights(
            hf_state_dict, num_layers=NUM_LAYERS, padding_idx=DECODER_PADDING_IDX
        )
        self._lm_head_sd = wl.lm_head_weights(hf_state_dict)

        # Cache TextGenerator instances by tile-padded encoder length so
        # repeated translate() calls with the same padded source length
        # do not pay the cost of rebuilding the 24-layer decoder.
        self._generators: Dict[int, TextGenerator] = {}

    # ------------------------------------------------------------------ helpers
    def _resolve_tgt_lang_id(self, tgt_lang: str) -> int:
        if tgt_lang not in self.lang_to_code_id:
            raise ValueError(
                f"tgt_lang={tgt_lang!r} not in generation_config text_decoder_lang_to_code_id; "
                f"valid examples: fra, spa, deu, eng."
            )
        return int(self.lang_to_code_id[tgt_lang])

    def _get_or_build_generator(self, encoder_seq_len_padded: int) -> TextGenerator:
        gen = self._generators.get(encoder_seq_len_padded)
        if gen is not None:
            return gen
        gen = TextGenerator(
            device=self.device,
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            num_layers=NUM_LAYERS,
            text_decoder_state_dict=self._text_decoder_sd,
            lm_head_state_dict=self._lm_head_sd,
            max_decode_seq_len=self.max_decode_seq_len,
            encoder_seq_len=encoder_seq_len_padded,
            eps=EPS,
            padding_idx=DECODER_PADDING_IDX,
            embed_scale=math.sqrt(EMBED_DIM),
            weight_dtype=self.weight_dtype,
        )
        self._generators[encoder_seq_len_padded] = gen
        return gen

    def _run_encoder(self, input_ids: torch.Tensor, attn_mask_2d: torch.Tensor) -> torch.Tensor:
        """Run the TTNN encoder and return host hidden states ``[1, S_logical, H]``."""
        from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

        tgt_len = int(input_ids.shape[-1])
        mask_4d = _prepare_4d_attention_mask(attn_mask_2d, torch.float32, tgt_len=tgt_len)
        enc_hidden_tt = self.encoder(input_ids, attention_mask_torch=mask_4d)
        enc_hidden_torch = ttnn.to_torch(enc_hidden_tt).to(torch.float32)
        ttnn.deallocate(enc_hidden_tt)
        if enc_hidden_torch.dim() == 4 and enc_hidden_torch.shape[0] == 1:
            enc_hidden_torch = enc_hidden_torch.squeeze(0)
        return enc_hidden_torch

    # ------------------------------------------------------------------ public API
    def translate(
        self,
        src_text: str,
        src_lang: str,
        tgt_lang: str,
        max_new_tokens: int = 128,
    ) -> str:
        """Translate ``src_text`` from ``src_lang`` to ``tgt_lang``.

        Returns the decoded translation as a string (special tokens stripped).
        """
        # 1. Tokenize source.
        toks = self.processor(text=src_text, src_lang=src_lang, return_tensors="pt")
        input_ids = toks["input_ids"]
        attn_mask = toks["attention_mask"]
        # Trim trailing padding to the logical length so the encoder/decoder
        # tile-padding math is well-defined.
        src_logical = int(attn_mask.sum().item())
        input_ids = input_ids[:, :src_logical]
        attn_mask = attn_mask[:, :src_logical]

        # 2. Encoder forward (TTNN).
        enc_hidden_logical = self._run_encoder(input_ids, attn_mask)
        enc_hidden_padded = _tile_pad_encoder_hidden(enc_hidden_logical[:, :src_logical, :])
        encoder_seq_len_padded = int(enc_hidden_padded.shape[1])

        # 3. Decoder + LM head + AR loop (TTNN).
        gen = self._get_or_build_generator(encoder_seq_len_padded)
        tgt_lang_id = self._resolve_tgt_lang_id(tgt_lang)

        # Bound max_new_tokens by the cache capacity (the generator does this
        # internally too, but doing it here keeps the API contract explicit).
        max_total = min(int(max_new_tokens), self.max_decode_seq_len)
        tokens = gen.generate(
            encoder_hidden_states=enc_hidden_padded,
            encoder_attention_mask=attn_mask,
            decoder_start_token_id=self.decoder_start_token_id,
            tgt_lang_id=tgt_lang_id,
            eos_token_id=self.eos_token_id,
            max_new_tokens=max_total,
            do_sample=False,
        )
        # The two prefix tokens (decoder_start, tgt_lang_id) are special;
        # the processor.decode call below strips them via skip_special_tokens.
        text = self.processor.decode(tokens, skip_special_tokens=True)
        return text
