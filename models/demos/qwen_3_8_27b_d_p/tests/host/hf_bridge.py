# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Build an upstream HuggingFace ``Qwen3_5TextModel`` from this package's config class.

Test-only: it is the bridge that lets the vendored reference be compared against the code it was
trimmed from. It lives under ``tests/`` rather than ``reference/`` on purpose — the reference's
purity contract is "torch and nothing else", and ``transformers`` is not torch.
"""

from __future__ import annotations

import torch

from models.demos.qwen_3_8_27b_d_p.reference.config import Qwen35TextConfig
from models.demos.qwen_3_8_27b_d_p.reference.modeling import REF_DTYPE

# A reduced config that keeps every ratio and every tile-relevant width of the real model, so a
# reduced-config comparison still exercises the code paths the full model takes:
#   * partial rotary 64 of head_dim 256, with the real mrope_section
#   * GQA output gate on (q_proj is 2x wide)
#   * the 3:1 GDN value:key head ratio at the real 128-wide heads
#   * the real hybrid period (3 linear + 1 full)
# Only hidden_size / intermediate_size / vocab / depth shrink, and those are the dimensions a CPU
# comparison cannot afford at 27B scale.
REDUCED_OVERRIDES = dict(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=8,
    vocab_size=1024,
    num_attention_heads=2,
    num_key_value_heads=1,
    linear_num_key_heads=2,
    linear_num_value_heads=6,
    max_position_embeddings=4096,
)


def reduced_config(**overrides) -> Qwen35TextConfig:
    """The real config with :data:`REDUCED_OVERRIDES` applied — a DIAGNOSTIC shape (recipe section 4)."""
    base = Qwen35TextConfig.from_json()
    fields = {k: getattr(base, k) for k in base.__dataclass_fields__ if base.__dataclass_fields__[k].init}
    fields.update(REDUCED_OVERRIDES)
    fields.update(overrides)
    n = fields["num_hidden_layers"]
    interval = fields["full_attention_interval"]
    fields["layer_types"] = tuple("full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n))
    return Qwen35TextConfig(**fields)


def to_hf_text_config(cfg: Qwen35TextConfig):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        vocab_size=cfg.vocab_size,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        attention_bias=cfg.attention_bias,
        attn_output_gate=cfg.attn_output_gate,
        linear_key_head_dim=cfg.linear_key_head_dim,
        linear_value_head_dim=cfg.linear_value_head_dim,
        linear_num_key_heads=cfg.linear_num_key_heads,
        linear_num_value_heads=cfg.linear_num_value_heads,
        linear_conv_kernel_dim=cfg.linear_conv_kernel_dim,
        full_attention_interval=cfg.full_attention_interval,
        rms_norm_eps=cfg.rms_norm_eps,
        hidden_act=cfg.hidden_act,
        tie_word_embeddings=cfg.tie_word_embeddings,
        max_position_embeddings=cfg.max_position_embeddings,
        rope_parameters={
            "rope_type": cfg.rope_type,
            "rope_theta": cfg.rope_theta,
            "partial_rotary_factor": cfg.partial_rotary_factor,
            "mrope_section": list(cfg.mrope_section),
            "mrope_interleaved": cfg.mrope_interleaved,
        },
        dtype=REF_DTYPE,
    )


def build_hf_model(cfg: Qwen35TextConfig, seed: int = 0):
    """An fp16 HF ``Qwen3_5TextModel`` with reproducible random weights, ready to eval.

    ``_init_weights`` gives ``A_log`` its ``log(U(0,16])`` draw and ``dt_bias`` its ones, which is
    the decay regime the delta-rule scan is actually designed for — the reason this builds the HF
    model and copies weights OUT of it rather than initialising both sides independently.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    hf_cfg = to_hf_text_config(cfg)
    hf_cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = Qwen3_5TextModel(hf_cfg).to(REF_DTYPE).eval()
    return model
