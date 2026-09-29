# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The HF oracle for Hy4 Preview: transformers 5.17's ``hy_v4`` modeling code (vendored in ``hf_hy_v4/``) on this
host's transformers 5.12.1, with the checkpoint's weights.

The stock loader cannot run this checkpoint here: 5.12.1 has no ``hy_v4``, and the full model is 1.56 TB in bf16
against 503 GB of host RAM. So:

* ``HYV4ForCausalLM`` is built on the meta device from the vendored class (eager attention, eager experts), text
  decoder only (``model.mtp_layers.*`` is never read), optionally truncated to its first ``num_layers`` layers;
* every weight except the routed experts is loaded as stored and cast to ``dtype``, except the modules HF keeps in
  fp32 (``_keep_in_fp32_modules_strict``: hc_*, sinks, router bias, indexer k_norm / weights_proj, lm_head), with the
  checkpoint -> module renames of transformers' ``conversion_mapping.py`` (hy_v4 entry, rename-only);
* the routed experts keep HF's own ``HYV4Experts.forward`` (the loop over hit experts, clamped SwiGLU), but its
  ``gate_up_proj`` / ``down_proj`` are ``ExpertSlab`` views of the checkpoint: ``[e]`` reads that expert's bytes from
  disk and casts to ``dtype``. Only the experts some token selected are read, and they are prefetched per layer;
* compatibility shim: the 5.17 code calls ``create_causal_mask(..., allow_is_causal_skip=False)``; 5.12.1 has no such
  argument and never skips the mask for eager attention, so the wrapper drops it and checks a mask came back.
"""

from __future__ import annotations

import re

import torch

from models.demos.hy4_preview_d_p.reference.weights import ExpertSlab, WeightLoader

# transformers conversion_mapping.py "hy_v4" (checkpoint name -> module name), applied in order.
_RENAMES = [
    (r"\.hc_pre\.hc_", ".hc_"),
    (r"\.learnable_sink_param$", ".sinks"),
    (r"\.linear_gate", ".gate_proj"),
    (r"\.hc_attn_layer\.hc_fn", ".attn_hc.fn"),
    (r"\.hc_attn_layer\.hc_base", ".attn_hc.base"),
    (r"\.hc_attn_layer\.hc_scale", ".attn_hc.scale"),
    (r"\.hc_mlp_layer\.hc_fn", ".ffn_hc.fn"),
    (r"\.hc_mlp_layer\.hc_base", ".ffn_hc.base"),
    (r"\.hc_mlp_layer\.hc_scale", ".ffn_hc.scale"),
    (r"\.hc_head_fn", ".hc_fn"),
    (r"\.hc_head_base", ".hc_base"),
    (r"\.hc_head_scale", ".hc_scale"),
]


def module_name(ckpt_name: str) -> str:
    for pat, rep in _RENAMES:
        ckpt_name = re.sub(pat, rep, ckpt_name)
    return ckpt_name


def _mask_shim(fn):
    def wrapped(*args, allow_is_causal_skip=None, **kwargs):
        mask = fn(*args, **kwargs)
        assert mask is not None, "create_causal_mask skipped the mask; the indexer needs it"
        return mask

    wrapped._hy4_shim = True
    return wrapped


class _PrefetchingExperts:
    """Mixin for HYV4Experts: prefetch the hit experts' bytes, then run HF's own forward."""

    def forward(self, hidden_states, top_k_index, top_k_weights):
        hit = torch.unique(top_k_index).tolist()
        self.gate_up_proj.prefetch(hit)
        self.down_proj.prefetch(hit)
        return super().forward(hidden_states, top_k_index, top_k_weights)


def build_hf_model(model_path: str, num_layers: int | None = None, dtype=torch.float32):
    from transformers import GenerationConfig

    from models.demos.hy4_preview_d_p.reference.hf_hy_v4 import modeling_hy_v4 as mod
    from models.demos.hy4_preview_d_p.reference.hf_hy_v4.configuration_hy_v4 import HYV4Config

    if not getattr(mod.create_causal_mask, "_hy4_shim", False):
        mod.create_causal_mask = _mask_shim(mod.create_causal_mask)

    cfg = HYV4Config.from_pretrained(model_path)
    if num_layers:
        cfg.num_hidden_layers = num_layers
        for k in ("layer_types", "mlp_layer_types", "indexer_types"):
            setattr(cfg, k, list(getattr(cfg, k))[:num_layers])
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    cfg.dtype = dtype

    with torch.device("meta"):
        model = mod.HYV4ForCausalLM(cfg)

    keep = mod.HYV4ForCausalLM._keep_in_fp32_modules_strict
    keep_re = re.compile("|".join(rf"((^|\.){m}($|\.))" for m in keep))
    loader = WeightLoader(model_path)

    lazy_cls = type("HYV4LazyExperts", (_PrefetchingExperts, mod.HYV4Experts), {})
    n_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, mod.HYV4MoE):
            ex = layer.mlp.experts
            ex.__class__ = lazy_cls
            del ex.gate_up_proj, ex.down_proj
            p = f"model.layers.{i}.mlp.experts."
            ex.gate_up_proj = ExpertSlab(loader, p + "gate_up_proj", dtype)
            ex.down_proj = ExpertSlab(loader, p + "down_proj", dtype)

    want = dict(model.state_dict(keep_vars=True))
    sd = {}
    for name in loader.weight_map:
        if name.startswith("model.mtp_layers.") or ".mlp.experts." in name:
            continue
        m = re.match(r"model\.layers\.(\d+)\.", name)
        if m and int(m.group(1)) >= n_layers:
            continue
        target = module_name(name)
        assert target in want, f"checkpoint tensor {name} -> {target} has no module"
        t = loader.get(name)
        sd[target] = t.float() if keep_re.search(target) else t.to(dtype)
    missing = sorted(set(want) - set(sd))
    missing = [m for m in missing if not m.endswith(("rotary_emb.inv_freq", "rotary_emb.original_inv_freq"))]
    assert not missing, f"no checkpoint tensor for {missing}"
    model.load_state_dict(sd, strict=False, assign=True)

    model.model.rotary_emb = mod.HYV4RotaryEmbedding(config=cfg)
    for n, t in list(model.named_parameters()) + list(model.named_buffers()):
        assert t.device.type != "meta", f"{n} still on meta"
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except OSError:
        pass
    return model.eval()
