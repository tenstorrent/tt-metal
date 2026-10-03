# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Torch reference for one DeepSeek-V4.1-Flash decoder layer, using real checkpoint weights.

Runs the checkpoint's own ``inference/model.py`` unchanged: ``kernel`` (tilelang, CUDA only) is
replaced by ``ref_kernels`` and the layer's tensors are read straight from the safetensors shard
that holds them. Nothing here is specific to a device.

Usage::

    block = build_layer(2)
    outs = run_prefill_then_decode(block, tokens, n_decode=4)
"""

import importlib.util
import json
import os
import sys

import torch
from safetensors import safe_open

from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels

CKPT_DIR = os.environ.get("DSV41_CKPT", "/mnt/tt-data/ssinghal/deepseek-v41-flash")

_model_module = None
# REF_DTYPE=fp32: the whole reference (weights, activations, 4-stream residual) in float32 instead of the checkpoint's bf16 (noise-floor study)
REF_DTYPE = torch.float32 if os.environ.get("REF_DTYPE", "bf16") == "fp32" else torch.bfloat16


def load_model_module():
    """Import the checkpoint's model.py with `kernel` swapped for the torch stand-ins."""
    global _model_module
    if _model_module is not None:
        return _model_module
    inf = os.path.join(CKPT_DIR, "inference")
    sys.modules["kernel"] = ref_kernels
    if inf not in sys.path:
        sys.path.insert(0, inf)
    spec = importlib.util.spec_from_file_location("dsv41_model", os.path.join(inf, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dsv41_model"] = mod
    spec.loader.exec_module(mod)
    mod.world_size, mod.rank = 1, 0
    mod.default_dtype = torch.float8_e4m3fn
    _model_module = mod
    return mod


def model_args(max_batch_size: int = 4, max_seq_len: int = 2048):
    mod = load_model_module()
    cfg = json.load(open(os.path.join(CKPT_DIR, "inference", "config.json")))
    fields = set(mod.ModelArgs.__dataclass_fields__)
    cfg = {k: v for k, v in cfg.items() if k in fields}
    cfg.update(max_batch_size=max_batch_size, max_seq_len=max_seq_len)
    return mod.ModelArgs(**cfg)


def _shard_for(index: dict, key: str) -> str:
    return os.path.join(CKPT_DIR, index[key])


def build_layer(layer_id: int, max_batch_size: int = 4, max_seq_len: int = 2048):
    """Construct Block(layer_id) and fill it from the checkpoint. Returns the module in eval mode."""
    mod = load_model_module()
    torch.set_default_dtype(REF_DTYPE)
    args = model_args(max_batch_size, max_seq_len)
    # Block.__init__ reads module-level globals that Transformer would normally set.
    block = mod.Block(layer_id, args, engram_layout=None)
    index = json.load(open(os.path.join(CKPT_DIR, "model.safetensors.index.json")))["weight_map"]

    prefix = f"layers.{layer_id}."
    handles = {}
    missing, loaded = [], 0
    with torch.no_grad():
        for name, p in block.named_parameters():
            key = prefix + name
            if key not in index:
                missing.append(key)
                continue
            path = _shard_for(index, key)
            if path not in handles:
                handles[path] = safe_open(path, "pt")
            t = handles[path].get_tensor(key)
            if p.dtype == torch.float4_e2m1fn_x2:
                t = t.view(torch.float4_e2m1fn_x2)  # packed int8 on disk
            elif t.dtype == torch.float8_e4m3fn and p.dtype != torch.float8_e4m3fn:
                # e.g. wo_a: stored FP8 + scale, but the module holds bf16 (convert.py dequantises it)
                skey = key.replace(".weight", ".scale")
                assert skey in index, f"{key} is fp8 on disk but has no scale tensor"
                spath = _shard_for(index, skey)
                if spath not in handles:
                    handles[spath] = safe_open(spath, "pt")
                s = handles[spath].get_tensor(skey)
                bo, bi = t.size(0) // s.size(0), t.size(1) // s.size(1)
                assert bo == bi, f"{key}: non-square scale blocks {bo}x{bi}"
                t = ref_kernels.dequant_fp8_weight(t, s, bo).to(p.dtype)
            elif t.dtype != p.dtype:
                t = t.to(p.dtype)
            assert t.shape == p.shape, f"{key}: checkpoint {tuple(t.shape)} vs module {tuple(p.shape)}"
            p.data = t
            loaded += 1
    # bias_vl only matters for image tokens; absent from some layers' shards is fine
    unexpected = [m for m in missing if not m.endswith("bias_vl")]
    assert not unexpected, f"{len(unexpected)} tensors missing from checkpoint, e.g. {unexpected[:5]}"
    if REF_DTYPE != torch.bfloat16:  # params the module declares bf16 explicitly (wo_a, ...) follow the reference dtype
        for p_ in block.parameters():
            if p_.dtype == torch.bfloat16:
                p_.data = p_.data.to(REF_DTYPE)
    block.eval()
    block.loaded_tensors = loaded
    return block


def embed_tokens(token_ids: torch.Tensor) -> torch.Tensor:
    """Real input activations: embed.weight rows, expanded to hc_mult residual streams.

    Needs shard 2 (embed.weight). Returns ([b, s, hc, dim] bf16, identity pre_mix [b, s, hc]).
    """
    mod = load_model_module()
    index = json.load(open(os.path.join(CKPT_DIR, "model.safetensors.index.json")))["weight_map"]
    with safe_open(_shard_for(index, "embed.weight"), "pt") as f:
        w = f.get_tensor("embed.weight")
    h = w[token_ids].to(REF_DTYPE)
    h = h.unsqueeze(2).repeat(1, 1, 4, 1)
    return h, mod.make_identity_pre_mix(h, 4)


@torch.inference_mode()
def run_prefill_then_decode(block, h, pre_mix, decode_inputs):
    """Prefill `h` ([b, s, hc, dim]) at start_pos 0, then one decode step per entry of
    `decode_inputs` (list of ([b, 1, hc, dim], pre_mix[b,1,hc])). Returns the list of layer outputs
    (x, next_pre_mix), prefill first."""
    outs = []
    x, nxt = block(h, 0, pre_mix, None)
    outs.append((x, nxt))
    pos = h.size(1)
    for hd, pm in decode_inputs:
        x, nxt = block(hd, pos, pm, None)
        outs.append((x, nxt))
        pos += 1
    return outs


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def dequantized_attention_weights(block) -> dict:
    """Dequantised (bf16, [out, in]) attention tensors of a built reference block, for the TT module."""
    a = block.attn

    def dq(lin):
        w = lin.weight.data
        if w.dtype == torch.float8_e4m3fn:
            return ref_kernels.dequant_fp8_weight(w, lin.scale.data, 32).to(torch.bfloat16)
        return w.to(torch.bfloat16)

    return {
        "wq_a": dq(a.wq_a),
        "q_norm": a.q_norm.weight.data.float(),
        "wq_b": dq(a.wq_b),
        "wkv": dq(a.wkv),
        "kv_norm": a.kv_norm.weight.data.float(),
        "wo_a": dq(a.wo_a),
        "wo_b": dq(a.wo_b),
        "attn_sink": a.attn_sink.data.float(),
    }
