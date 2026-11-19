# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU-side bring-up hooks for mistral_large4_1t: the reference stage owns this file; implement jobs may not edit it (the goldens
and every test are computed from it).

reference(spec, layers, dtype), and optionally tokenizer(spec), hf_model(spec, num_layers), hf_layers(model).
The block-graph contract is in reference/_bringup_interface.py (vendored, do not edit). This package never imports
the framework (bringup_fw).
"""

import torch


def reference(spec, layers=None, dtype=None):
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.model import Reference

    return Reference(spec, layers=layers, dtype=dtype or torch.float32)


def hf_model(spec, num_layers):
    """The HF oracle: upstream transformers' mistral_large4 text model (vendored in reference/hf_oracle.py: this
    workspace's transformers 5.12 predates it) with the checkpoint's weights, FP8 dequantized exactly, routed experts
    read from the checkpoint when they run. num_layers=None is the full 72-layer model in bf16 (HF sanity: next-token
    accuracy; the router weight and RoPE stay fp32 as upstream keeps them); a layer prefix is built in fp32 (check_hf
    parity)."""
    from models.demos.deepseek_v3_d_p.reference.mistral_large_4.hf_oracle import build_hf_model

    return build_hf_model(spec.hf_dir, num_layers, torch.bfloat16 if num_layers is None else torch.float32)


def swap_context(spec, ref, layer, golden, chunk, rctx, dctx):
    """What one block of a cross-layer type cannot compute alone, from the golden, in both contexts (the component and
    swap harness hook, F49): a shared-index layer's top-k is its source full layer's ``topk`` of the same chunk
    (``shared_topk``); a shared-KV layer attends over its source layer's cache rows [0, chunk end) (``shared_kv``:
    kv_latent, k_rope). The golden's layer subset holds both sources (16 for 17, 35 for 39).

    The framework takes this hook from here while bringup/hooks.py defines none (core/spec.py
    REFERENCE_FALLBACK_HOOKS)."""
    cfg = ref.cfg
    extra = {}
    if not cfg.is_full(layer):
        extra["shared_topk"] = golden.layer(chunk, cfg.topk_source(layer))["topk"]
    if not cfg.owns_kv(layer):
        st = golden.state(cfg.kv_source(layer))
        extra["shared_kv"] = {n: st[n].float() for n in ("kv_latent", "k_rope")}
    for ctx in (rctx, dctx):
        ctx.extra.update(extra)
