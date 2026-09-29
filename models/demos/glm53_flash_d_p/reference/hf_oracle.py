# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The HF oracle for GLM-5.3-Flash: the vendored transformers glm5_next modeling code with the checkpoint's weights.

The stock loader cannot run this checkpoint here: transformers 5.12.1 has no glm5_next (the code is vendored from
5.17 into reference/hf), and the FP8 checkpoint (328 GB) exceeds host RAM (249 GB). So:

* only the text model is built (``Glm5NextTextModel`` + an untied lm_head, ``Glm5NextTextForCausalLM`` below): no
  vision tower, no MTP layer; built on the meta device, then every weight is assigned;
* routed experts are ``PackedExperts``: the FP8 blocks stay on disk (safetensors, via the index) and each expert is
  read and dequantized when a token routes to it, same math as ``Glm5NextTextExperts.forward``. Everything else
  (about 20 GB in bf16) is dequantized at load. The dtypes follow ``from_pretrained(dtype=...)``: A_log, dt_bias,
  e_score_correction_bias and conv1d stay fp32 (``_keep_in_fp32_modules_strict``);
* checkpoint -> module names: ``hc_{attn,ffn}_{fn,base,scale}`` -> ``{attn,ffn}_hc.{fn,base,scale}``;
  ``{q,k,v}_conv1d.weight`` -> ``conv1d.weight`` (channels concatenated q|k|v, as the module concatenates the
  projections); ``f_a_proj``/``f_b_proj``/``dt_bias``/``A_log`` -> ``forget_gate.*``; experts per id -> packed;
* config.json names the DSA layers ``deepseek_sparse_attention``; the modeling code's mask table uses
  ``indexed_attention`` (and only compares against ``linear_attention`` otherwise), so layer_types are renamed;
* transformers 5.12.1's cache has different linear-attention semantics (``update_conv_state`` returns the state,
  not the concatenated input) and a 5.17 DynamicCache is not available, so ``generate`` is a plain greedy loop over
  ``GlmCache``, a minimal cache with the interface the modeling code calls. ``forward`` defaults to no cache.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.demos.glm53_flash_d_p.reference.weights import PREFIX, PackedExpert, WeightLoader

KEEP_FP32 = ("conv1d.weight", "dt_bias", "A_log", "e_score_correction_bias")


class PackedExperts(nn.Module):
    """Glm5NextTextExperts with the expert weights kept in FP8 on disk; forward is the HF forward."""

    def __init__(self, loader: WeightLoader, layer: int, num_experts: int, swiglu_limit: float, dtype):
        super().__init__()
        self.experts = [PackedExpert(loader, layer, e) for e in range(num_experts)]
        self.num_experts = num_experts
        self.swiglu_limit = swiglu_limit
        self.dtype = dtype

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            gate, up, down = self.experts[int(expert_idx)].weights(self.dtype)
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self._apply_gate(F.linear(hidden_states[token_idx], torch.cat([gate, up])))
            current = F.linear(current, down) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final

    def _apply_gate(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return F.silu(gate) * up


class _KdaCacheLayer:
    def __init__(self):
        self.conv_states = None  # [tensor [B, C, K]]
        self.recurrent_states = None  # [tensor [B, H, K, V]]


class _DsaCacheLayer:
    def __init__(self):
        self.keys = self.values = self.indexer = None

    def get_seq_length(self):
        return 0 if self.keys is None else self.keys.shape[-2]


def _cat(old, new, dim):
    return new if old is None else torch.cat([old, new], dim=dim)


class GlmCache:
    """The cache interface modeling_glm5_next calls, with transformers 5.17 semantics (single sequence)."""

    def __init__(self, config):
        self.layers = [_KdaCacheLayer() if t == "linear_attention" else _DsaCacheLayer() for t in config.layer_types]
        self.seen = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seen

    def has_previous_state(self, layer_idx: int) -> bool:
        return self.layers[layer_idx].recurrent_states is not None

    def update_conv_state(self, mixed, layer_idx: int, conv_kernel_size: int, **_):
        """Returns [previous conv tail | new inputs] for the conv; keeps the last conv_kernel_size inputs."""
        lay = self.layers[layer_idx]
        full = mixed if lay.conv_states is None else torch.cat([lay.conv_states[0].to(mixed.dtype), mixed], dim=-1)
        tail = full[..., -conv_kernel_size:]
        if tail.shape[-1] < conv_kernel_size:
            tail = F.pad(tail, (conv_kernel_size - tail.shape[-1], 0))
        lay.conv_states = [tail.clone()]
        return full

    def update_recurrent_state(self, state, layer_idx: int, **_):
        self.layers[layer_idx].recurrent_states = [state]
        return state

    def update(self, key_states, value_states, layer_idx: int, *_, **__):
        lay = self.layers[layer_idx]
        lay.keys, lay.values = _cat(lay.keys, key_states, -2), _cat(lay.values, value_states, -2)
        return lay.keys, lay.values

    def update_indexer(self, packed, layer_idx: int):
        lay = self.layers[layer_idx]
        lay.indexer = _cat(lay.indexer, packed, 1)
        return lay.indexer


class Glm5NextTextForCausalLM(nn.Module):
    """Text-only GLM-5.3-Flash: HF Glm5NextTextModel + lm_head (Glm5NextForConditionalGeneration minus vision)."""

    def __init__(self, text_model, lm_head, eos_token_ids):
        super().__init__()
        self.model = text_model
        self.lm_head = lm_head
        self.eos_token_ids = set(eos_token_ids)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, **kw):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache and past_key_values is not None,
            **kw,
        )
        h = out.last_hidden_state
        logits = self.lm_head(h[:, -logits_to_keep:] if logits_to_keep else h)
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=16, do_sample=False, **_):
        """Greedy decoding with GlmCache (batch 1, no padding)."""
        assert not do_sample and input_ids.shape[0] == 1
        if attention_mask is not None:
            assert bool(attention_mask.all()), "padding not supported"
        cache = GlmCache(self.model.config)
        ids, step = input_ids, input_ids
        for _ in range(max_new_tokens):
            logits = self.forward(step, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
            cache.seen += step.shape[1]
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            step = nxt
            if int(nxt) in self.eos_token_ids:
                break
        return ids


def text_config(model_path: str, num_layers: int | None):
    from models.demos.glm53_flash_d_p.reference.hf.configuration_glm5_next import Glm5NextTextConfig

    with open(os.path.join(model_path, "config.json")) as f:
        raw = json.load(f)["text_config"]
    cfg = Glm5NextTextConfig(**raw)
    cfg.layer_types = ["indexed_attention" if t == "deepseek_sparse_attention" else t for t in cfg.layer_types]
    if num_layers:
        cfg.num_hidden_layers = num_layers
        for name in ("layer_types", "mlp_layer_types", "indexer_types"):
            setattr(cfg, name, list(getattr(cfg, name))[:num_layers])
    cfg._attn_implementation = "eager"
    cfg.use_cache = False
    return cfg, raw


def checkpoint_name(name: str) -> str | tuple[str, ...]:
    """Module state-dict name (below Glm5NextTextModel) -> checkpoint tensor name(s)."""
    parts = name.split(".")
    if parts[0] == "layers":
        i, rest = parts[1], ".".join(parts[2:])
        p = f"{PREFIX}layers.{i}."
        for hc in ("attn", "ffn"):
            if rest.startswith(f"{hc}_hc."):
                return p + f"hc_{hc}_{rest.split('.')[-1]}"
        if rest == "self_attn.conv1d.weight":
            return tuple(p + f"self_attn.{n}_conv1d.weight" for n in "qkv")
        if rest.startswith("self_attn.forget_gate."):
            return p + "self_attn." + rest[len("self_attn.forget_gate.") :]
        return p + rest
    return PREFIX + name


def build_hf_model(model_path: str, num_layers: int | None = None, dtype=torch.bfloat16) -> Glm5NextTextForCausalLM:
    from models.demos.glm53_flash_d_p.reference.hf import modeling_glm5_next as mod

    cfg, raw = text_config(model_path, num_layers)
    with torch.device("meta"):
        tm = mod.Glm5NextTextModel(cfg)
    loader = WeightLoader(model_path)
    for i, layer in enumerate(tm.layers):
        if isinstance(layer.mlp, mod.Glm5NextTextMoE):
            layer.mlp.experts = PackedExperts(loader, i, cfg.n_routed_experts, cfg.swiglu_limit, dtype)

    sd = {}
    for name in tm.state_dict().keys():
        src = checkpoint_name(name)
        dt = torch.float32 if name.endswith(KEEP_FP32) else dtype
        if isinstance(src, tuple):
            sd[name] = torch.cat([loader.weight(s, torch.float32) for s in src]).to(dt)
        else:
            sd[name] = loader.weight(src, dt)
    missing, unexpected = tm.load_state_dict(sd, strict=True, assign=True)
    assert not missing and not unexpected, (missing, unexpected)
    for n, t in list(tm.named_parameters()) + list(tm.named_buffers()):
        assert t.device.type != "meta", f"{n} still on meta"

    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False, dtype=dtype)
    lm_head.weight = nn.Parameter(loader.weight("lm_head.weight", dtype), requires_grad=False)
    eos = raw.get("eos_token_id") or []
    model = Glm5NextTextForCausalLM(tm, lm_head, eos if isinstance(eos, list) else [eos])
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()
