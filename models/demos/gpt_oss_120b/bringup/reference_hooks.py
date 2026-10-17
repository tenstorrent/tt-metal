# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU-side bring-up hooks for gpt_oss_120b: the reference stage owns this file; implement jobs may not edit it (the goldens
and every test are computed from it).

reference(spec, layers, dtype), and optionally tokenizer(spec), hf_model(spec, num_layers), hf_layers(model).
The block-graph contract is in reference/_bringup_interface.py (vendored, do not edit). This package never imports
the framework (bringup_fw).

The HF oracle is transformers' GptOssForCausalLM, which dequantizes the mxfp4 experts to bf16 on CPU. hf_model
fixes how it computes them: the stock CPU path (grouped_mm) casts the expert activations to the weights' bf16, so
an "fp32" load is bf16 inside every MoE (rel 3e-3 per layer vs fp32). The truncated parity model (num_layers set)
therefore gets fp32 expert weights (bf16 -> fp32 is exact for mxfp4) and the eager per-expert loop; the full model
(num_layers None, the HF sanity) stays the stock bf16 load, as the framework's default.
"""

import torch


def reference(spec, layers=None, dtype=None):
    from models.demos.gpt_oss_120b.reference.gpt_oss_ref import GptOssReference

    return GptOssReference(spec.hf_dir, layers=layers, dtype=dtype or torch.float32)


def hf_model(spec, num_layers):
    from transformers import AutoConfig, AutoModelForCausalLM

    path = str(spec.hf_dir)
    if num_layers is None:
        return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()
    cfg = AutoConfig.from_pretrained(path)
    cfg.num_hidden_layers = num_layers
    cfg.layer_types = cfg.layer_types[:num_layers]
    model = AutoModelForCausalLM.from_pretrained(
        path, config=cfg, dtype=torch.float32, attn_implementation="eager", experts_implementation="eager"
    ).eval()
    for layer in model.model.layers:
        layer.mlp.experts.float()
    return model
