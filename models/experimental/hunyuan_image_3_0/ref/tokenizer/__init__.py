# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Public exports for the HunyuanImage-3.0 host-side tokenizer package.
#
# Module layout
# -------------
#   hunyuan_tokenizer.py  — HunyuanTokenizer (main entry point)
#   chat_template.py      — T2I token sequence builder (gen_image)
#   gen_image_inputs.py   — prepare_gen_image_inputs host bundle
#   special_tokens.py     — multimodal token ID map
#   image_info.py         — ImageInfo latent grid metadata
#   resolution.py         — aspect-ratio / pixel-size helpers
#   assets/               — bundled config + tokenizer.json (download from HF)
#
# Typical usage
# -------------
#   tok = HunyuanTokenizer.from_pretrained()
#   bundle = prepare_gen_image_inputs(tok, "a cat on a mat", image_size=1024)
#
# References
# ----------
#   README.md                          — tokenizer.json download instructions
#   tests/tokenizer/                   — CFG + host preprocess tests

from .chat_template import ChatTemplateEncoder, TokenizerEncodeOutput
from .gen_image_inputs import GenImageHostInputs, build_rope_image_info, prepare_gen_image_inputs
from .hunyuan_tokenizer import (
    ASSETS_DIR,
    CONFIG_PATH,
    TOKENIZER_DIR,
    HunyuanConfig,
    HunyuanTokenizer,
    load_config,
    load_tokenizer,
)
from .image_info import ImageInfo, build_gen_image_info
from .resolution import Resolution, ResolutionGroup
from .special_tokens import SpecialTokens, build_special_tokens, validate_special_tokens

__all__ = [
    "ASSETS_DIR",
    "CONFIG_PATH",
    "TOKENIZER_DIR",
    "ChatTemplateEncoder",
    "GenImageHostInputs",
    "HunyuanConfig",
    "HunyuanTokenizer",
    "ImageInfo",
    "Resolution",
    "ResolutionGroup",
    "SpecialTokens",
    "TokenizerEncodeOutput",
    "build_gen_image_info",
    "build_rope_image_info",
    "build_special_tokens",
    "load_config",
    "load_tokenizer",
    "prepare_gen_image_inputs",
    "validate_special_tokens",
]
