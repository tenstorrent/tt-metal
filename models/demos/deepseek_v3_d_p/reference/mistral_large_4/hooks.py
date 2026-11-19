# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU-side hooks for mistral_large_4: reference(spec, layers, dtype), hf_model(spec, num_layers), swap_context.

The block-graph interface is in reference/mistral_large_4/_bringup_interface.py.
"""

import torch


def reference(spec, layers=None, dtype=None):
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Reference

    return Reference(spec, layers=layers, dtype=dtype or torch.float32)


def hf_model(spec, num_layers):
    """The HF oracle (reference/mistral_large_4/hf_oracle.py) with the checkpoint's weights, FP8 dequantized exactly,
    routed experts read from the checkpoint when they run. num_layers=None builds the full 72-layer model in bf16
    (router weight and RoPE stay fp32); a layer prefix is built in fp32."""
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.hf_oracle import build_hf_model

    return build_hf_model(spec.hf_dir, num_layers, torch.bfloat16 if num_layers is None else torch.float32)


def swap_context(spec, ref, layer, golden, chunk, rctx, dctx):
    """Inputs one cross-layer block cannot compute alone, taken from the golden, into both contexts: a shared-index
    layer's top-k is its source full layer's ``topk`` of the same chunk (``shared_topk``); a shared-KV layer attends
    over its source layer's cache rows [0, chunk end) (``shared_kv``: kv_latent, k_rope). The golden's layer subset
    must hold both sources (16 for 17, 35 for 39)."""
    cfg = ref.cfg
    extra = {}
    if not cfg.is_full(layer):
        extra["shared_topk"] = golden.layer(chunk, cfg.topk_source(layer))["topk"]
    if not cfg.owns_kv(layer):
        st = golden.state(cfg.kv_source(layer))
        extra["shared_kv"] = {n: st[n].float() for n in ("kv_latent", "k_rope")}
    for ctx in (rctx, dctx):
        ctx.extra.update(extra)
