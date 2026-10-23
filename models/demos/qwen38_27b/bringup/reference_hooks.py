# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU-side bring-up hooks for qwen38_27b: the reference stage owns this file; implement jobs may not edit it (the goldens
and every test are computed from it).

reference(spec, layers, dtype), and optionally tokenizer(spec), hf_model(spec, num_layers), hf_layers(model).
The block-graph contract is in reference/_bringup_interface.py (vendored, do not edit). This package never imports
the framework (bringup_fw).
"""

import torch

from models.demos.qwen38_27b.reference._bringup_interface import weight_dtype


def reference(spec, layers=None, dtype=None):
    from models.demos.qwen38_27b.reference.model import Reference

    return Reference(spec.hf_dir, layers=layers, dtype=dtype or torch.float32, wdtype=weight_dtype(spec))


def hf_model(spec, num_layers):
    """HF's text decoder (Qwen3_5ForCausalLM). The checkpoint's config is the multimodal Qwen3_5Config, whose
    top-level num_hidden_layers does not reach the text model: truncate text_config (and its layer_types) instead.
    num_layers given (parity): fp32, eager attention; None (the whole model): bf16. Runs with golden.trace load no
    HF model at all."""
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(spec.hf_dir)
    if num_layers is None:
        return AutoModelForCausalLM.from_pretrained(spec.hf_dir, config=cfg, dtype=torch.bfloat16).eval()
    tc = cfg.text_config
    tc.num_hidden_layers = num_layers
    tc.layer_types = tc.layer_types[:num_layers]
    return AutoModelForCausalLM.from_pretrained(
        spec.hf_dir, config=cfg, dtype=torch.float32, attn_implementation="eager"
    ).eval()
