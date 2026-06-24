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
from .gen_image_inputs import (
    GenImageHostInputs,
    build_attention_mask_for_bundle,
    build_full_attn_slices,
    build_i2i_cfg_conds,
    build_i2i_inputs_embeds,
    build_rope_image_info,
    bundle_to_denoise_cond,
    enrich_bundle_attention,
    get_gen_image_grid,
    get_gen_image_slice,
    prepare_gen_image_inputs,
    prepare_i2i_denoise_bundle,
    prepare_i2i_inputs,
    prepare_recaption_inputs,
    print_recaption_inputs_report,
    scatter_gen_timestep_embeds,
    tokenizer_output_from_bundle,
)
from .hunyuan_tokenizer import (
    ASSETS_DIR,
    CONFIG_PATH,
    TOKENIZER_DIR,
    HunyuanConfig,
    HunyuanTokenizer,
    load_config,
    load_tokenizer,
)
from .image_info import CondImage, ImageInfo, ImageTensor, JointImageInfo, build_gen_image_info
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
    "CondImage",
    "ImageInfo",
    "ImageTensor",
    "JointImageInfo",
    "Resolution",
    "ResolutionGroup",
    "SpecialTokens",
    "TokenizerEncodeOutput",
    "build_attention_mask_for_bundle",
    "build_full_attn_slices",
    "build_gen_image_info",
    "build_i2i_inputs_embeds",
    "build_i2i_cfg_conds",
    "build_rope_image_info",
    "bundle_to_denoise_cond",
    "build_special_tokens",
    "enrich_bundle_attention",
    "get_gen_image_grid",
    "get_gen_image_slice",
    "load_config",
    "load_tokenizer",
    "prepare_gen_image_inputs",
    "prepare_i2i_denoise_bundle",
    "prepare_i2i_inputs",
    "prepare_recaption_inputs",
    "print_recaption_inputs_report",
    "scatter_gen_timestep_embeds",
    "tokenizer_output_from_bundle",
    "validate_special_tokens",
]
