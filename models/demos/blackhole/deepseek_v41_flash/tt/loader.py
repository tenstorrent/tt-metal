# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Direct checkpoint -> host-tensor loader for one DeepSeek-V4.1-Flash layer (no CPU reference Block needed).

Returns what the TT modules take: dequantised bf16 attention tensors, compressor tensors for kv-source layers,
norm and mHC parameters, the RoPE frequency table of the layer's attention type, and the MoE tensors.
"""

import torch

from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards, load_moe_layer


def _fp8(sh, prefix):
    return ref_kernels.dequant_fp8_weight(sh.get(prefix + ".weight"), sh.get(prefix + ".scale"), 32).to(torch.bfloat16)


def layer_meta(layer_id: int):
    """compress_ratio, kv-source / index-source flags and the kv source this layer reads, from config.json."""
    args = R.model_args()
    ratio = args.compress_ratios[layer_id]
    sources = list(args.kv_source_layers)
    is_source = layer_id in sources
    src = max([s for s in sources if s <= layer_id], default=None) if ratio else None
    return {"ratio": ratio, "is_kv_source": is_source, "kv_source": src, "args": args}


def rope_freqs(layer_id: int, max_seq_len: int = 256):
    """complex [max_seq_len, 32]: plain RoPE for window-only layers, YaRN with compress_rope_theta otherwise."""
    mod = R.load_model_module()
    args = R.model_args()
    if args.compress_ratios[layer_id]:
        orig, theta = args.original_seq_len, args.compress_rope_theta
    else:
        orig, theta = 0, args.rope_theta
    return mod.precompute_freqs_cis(
        args.rope_head_dim, max_seq_len, orig, theta, args.rope_factor, args.beta_fast, args.beta_slow
    )


def load_layer(layer_id: int, with_moe: bool = True, max_seq_len: int = 256):
    sh = _Shards()
    p = f"layers.{layer_id}."
    meta = layer_meta(layer_id)
    attn = {
        "wq_a": _fp8(sh, p + "attn.wq_a"),
        "q_norm": sh.get(p + "attn.q_norm.weight").float(),
        "wq_b": _fp8(sh, p + "attn.wq_b"),
        "wkv": _fp8(sh, p + "attn.wkv"),
        "kv_norm": sh.get(p + "attn.kv_norm.weight").float(),
        "wo_a": _fp8(sh, p + "attn.wo_a"),
        "wo_b": _fp8(sh, p + "attn.wo_b"),
        "attn_sink": sh.get(p + "attn.attn_sink").float(),
    }
    comp = None
    if meta["is_kv_source"]:
        comp = {
            "wkv": sh.get(p + "attn.compressor.wkv.weight").float(),
            "norm": sh.get(p + "attn.compressor.norm.weight").float(),
        }
        if meta["ratio"] > 1:
            comp["wgate"] = sh.get(p + "attn.compressor.wgate.weight").float()
    out = {
        "meta": meta,
        "attn": attn,
        "compressor": comp,
        "freqs_cis": rope_freqs(layer_id, max_seq_len),
        "norms": {
            "attn_norm": sh.get(p + "attn_norm.weight").float(),
            "ffn_norm": sh.get(p + "ffn_norm.weight").float(),
        },
        "mhc": {
            n: (
                sh.get(p + f"hc_{n}_fn").float(),
                sh.get(p + f"hc_{n}_base").float(),
                sh.get(p + f"hc_{n}_scale").float(),
            )
            for n in ("attn", "ffn")
        },
    }
    if with_moe:
        out["moe"] = load_moe_layer(layer_id)
    return out
