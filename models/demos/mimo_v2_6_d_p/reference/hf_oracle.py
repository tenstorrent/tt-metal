# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The HF oracle for MiMo-V2.6-Flash-RL: the checkpoint's own modeling_mimo_v2.py with its weights dequantized.

The stock loader cannot run this checkpoint on CPU: transformers' fp8 path does not know the mxfp4 experts or the
per-TP-rank qkv layout, the index lists model_mtp.safetensors (not downloaded; MTP is out of scope), and the full
model is ~618 GB in bf16. So:

* the text-only MiMoV2ForCausalLM (vision_config / audio_config emptied, so neither encoder is built) is built on the
  meta device from the checkpoint's own class (trust_remote_code);
* every routed expert is replaced by ``PackedExpertMLP``, which keeps the mxfp4 weights + e8m0 scales and
  dequantizes them in its forward (same math as MiMoV2MLP.forward: down(silu(gate(x)) * up(x)));
* every other weight is loaded dequantized (qkv via weights.qkv_weight, the dense MLP via fp8_weight, the rest as
  stored) with ``load_state_dict(assign=True)``; the rotary modules are rebuilt off the meta device;
* compatibility shim: the modeling code (written for transformers 5.3) calls create_causal_mask /
  create_sliding_window_causal_mask with ``input_embeds=`` and ``cache_position=``; this transformers (5.12) takes
  ``inputs_embeds`` and no cache_position. The two names are wrapped in the modeling module's namespace to rename
  the one and drop the other; the mask semantics are the library's.
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.demos.mimo_v2_6_d_p.reference.weights import PackedExpert, WeightLoader, fp8_weight, qkv_weight


class PackedExpertMLP(nn.Module):
    """A routed expert kept in mxfp4; its forward matches MiMoV2MLP.forward with the dequantized weights."""

    def __init__(self, packed: PackedExpert, act_fn):
        super().__init__()
        self.packed = packed
        self.act_fn = act_fn

    def forward(self, hidden_states):
        gate, up, down = self.packed.weights(hidden_states.dtype)
        return F.linear(self.act_fn(F.linear(hidden_states, gate)) * F.linear(hidden_states, up), down)


def _mask_shim(fn):
    def wrapped(*args, input_embeds=None, cache_position=None, **kwargs):
        if input_embeds is not None and "inputs_embeds" not in kwargs:
            kwargs["inputs_embeds"] = input_embeds
        return fn(*args, **kwargs)

    wrapped.__wrapped__ = fn
    return wrapped


def _patch_masks(mod) -> None:
    import inspect

    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        fn = getattr(mod, name)
        fn = getattr(fn, "__wrapped__", fn) if getattr(fn, "_mimo_shim", False) else fn
        if "input_embeds" in inspect.signature(fn).parameters:
            continue  # a transformers that still takes the old names
        w = _mask_shim(fn)
        w._mimo_shim = True
        setattr(mod, name, w)


def build_hf_model(model_path: str, num_layers: int | None = None, dtype=torch.float32):
    from transformers import AutoConfig, GenerationConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if num_layers:
        cfg.num_hidden_layers = num_layers
    cfg.vision_config = {}
    cfg.audio_config = {}
    if hasattr(cfg, "quantization_config"):
        del cfg.quantization_config
    cfg._attn_implementation = "eager"
    cfg.dtype = dtype

    cls = get_class_from_dynamic_module("modeling_mimo_v2.MiMoV2ForCausalLM", model_path)
    mod = sys.modules[cls.__module__]
    _patch_masks(mod)

    with torch.device("meta"):
        model = cls(cfg)

    loader = WeightLoader(model_path)
    # routed experts: packed
    for i, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if isinstance(mlp, mod.MiMoV2MoE):
            act = mlp.experts[0].act_fn
            mlp.experts = nn.ModuleList(
                PackedExpertMLP(PackedExpert(loader, f"model.layers.{i}.mlp.experts.{e}."), act)
                for e in range(len(mlp.experts))
            )

    # everything else: dequantized
    sd = {}
    for name in model.state_dict().keys():
        if ".mlp.experts." in name:
            continue
        parts = name.split(".")
        if name.endswith("self_attn.qkv_proj.weight"):
            attn = model.model.layers[int(parts[2])].self_attn
            prefix = name[: -len("qkv_proj.weight")]
            sd[name] = qkv_weight(loader, prefix, (attn.q_size, attn.k_size, attn.v_size), dtype)
        elif loader.has(name + "_scale_inv"):
            sd[name] = fp8_weight(loader, name, dtype)
        else:
            sd[name] = loader.get(name).to(dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [m for m in missing if ".mlp.experts." not in m]
    assert not missing and not unexpected, (missing, unexpected)

    model.model.rotary_emb = mod.MiMoV2RotaryEmbedding(config=cfg, is_swa=False)
    model.model.swa_rotary_emb = mod.MiMoV2RotaryEmbedding(config=cfg, is_swa=True)
    for n, t in list(model.named_parameters()) + list(model.named_buffers()):
        assert t.device.type != "meta", f"{n} still on meta"
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    try:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
    except OSError:
        pass
    return model.eval()
