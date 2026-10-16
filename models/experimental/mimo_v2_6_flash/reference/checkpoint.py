# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Load and dequantize the MiMo-V2.6-Flash checkpoint into canonical bf16 tensors.

The checkpoint is not in the layout the HF modeling code splits:

* ``qkv_proj`` is FP8 e4m3 with 128x128 block scales, and it is pre-sharded for
  ``ckpt_tp = num_key_value_heads`` (4) ranks: ``[Q_0|K_0|V_0|Q_1|K_1|V_1|...]``, with the block
  scales tiled per chunk (a global-attention chunk is 3392 rows, so 27 scale rows each, 108 in all).
  vLLM de-interleaves it the same way (``_shard_fp8_qkv_proj`` in vllm/model_executor/models/mimo_v2.py).
  We return canonical ``q_proj [64*192, 4096]``, ``k_proj [kv*192, 4096]`` and ``v_proj [kv*128, 4096]``.
* The dense MLP (layer 0) is FP8 with block scales; ``o_proj``, norms, router, embeddings and LM head are BF16.
* Routed experts are OCP MXFP4: U8 bytes holding two E2M1 values (low nibble first) plus one e8m0 scale per
  32 input elements (``store_dtype: mxfp4``; vLLM loads them with ``Mxfp4MoEMethod``, as for gpt-oss).
"""

import json
import os
from functools import lru_cache

import torch
from safetensors import safe_open

FP8_BLOCK = 128
MXFP4_BLOCK = 32

# E2M1 code -> value, codes 8..15 are the negatives of 0..7.
_FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)


def _cdiv(a, b):
    return (a + b - 1) // b


def dequant_fp8_block(weight, scale_inv, row_scale_index=None):
    """weight [R, C] fp8, scale_inv [ceil(R/128) or per-chunk rows, ceil(C/128)] fp32 -> fp32 [R, C]."""
    rows, cols = weight.shape
    if row_scale_index is None:
        row_scale_index = torch.arange(rows) // FP8_BLOCK
    col_scale_index = torch.arange(cols) // FP8_BLOCK
    scale = scale_inv.float()[row_scale_index][:, col_scale_index]
    return weight.float() * scale


def dequant_mxfp4(blocks, scales):
    """blocks [R, C/2] U8, scales [R, C/32] U8 (e8m0) -> fp32 [R, C]."""
    rows, half_cols = blocks.shape
    lo = _FP4_VALUES[(blocks & 0x0F).long()]
    hi = _FP4_VALUES[(blocks >> 4).long()]
    values = torch.stack((lo, hi), dim=-1).reshape(rows, half_cols * 2)
    exponent = scales.to(torch.int32) - 127
    values = values.reshape(rows, -1, MXFP4_BLOCK)
    return torch.ldexp(values, exponent.unsqueeze(-1)).reshape(rows, half_cols * 2)


def split_fused_qkv(weight, scale_inv, num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp):
    """De-interleave the per-chunk fused fp8 QKV into canonical fp32 (q, k, v) row blocks."""
    rows = weight.shape[0]
    rows_per_chunk = rows // ckpt_tp
    q_per_chunk = (num_heads // ckpt_tp) * head_dim
    k_per_chunk = (num_kv_heads // ckpt_tp) * head_dim
    v_per_chunk = (num_kv_heads // ckpt_tp) * v_head_dim
    assert q_per_chunk + k_per_chunk + v_per_chunk == rows_per_chunk, (rows, ckpt_tp)

    chunk_scale_rows = _cdiv(rows_per_chunk, FP8_BLOCK)
    r = torch.arange(rows)
    if scale_inv.shape[0] == ckpt_tp * chunk_scale_rows:
        row_scale_index = (r // rows_per_chunk) * chunk_scale_rows + (r % rows_per_chunk) // FP8_BLOCK
    else:
        assert scale_inv.shape[0] == _cdiv(rows, FP8_BLOCK), scale_inv.shape
        row_scale_index = r // FP8_BLOCK
    full = dequant_fp8_block(weight, scale_inv, row_scale_index)

    chunks = full.reshape(ckpt_tp, rows_per_chunk, -1)
    q = chunks[:, :q_per_chunk].reshape(num_heads * head_dim, -1)
    k = chunks[:, q_per_chunk : q_per_chunk + k_per_chunk].reshape(num_kv_heads * head_dim, -1)
    v = chunks[:, q_per_chunk + k_per_chunk :].reshape(num_kv_heads * v_head_dim, -1)
    return q, k, v


class MiMoCheckpoint:
    """Lazy per-tensor reader over the sharded safetensors checkpoint."""

    def __init__(self, model_path, config):
        self.model_path = model_path
        self.config = config
        with open(os.path.join(model_path, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self._handles = {}

    def _handle(self, filename):
        if filename not in self._handles:
            self._handles[filename] = safe_open(os.path.join(self.model_path, filename), framework="pt")
        return self._handles[filename]

    def has(self, name):
        return name in self.weight_map

    def get(self, name):
        return self._handle(self.weight_map[name]).get_tensor(name)

    def _fp8_or_bf16(self, prefix):
        weight = self.get(f"{prefix}.weight")
        if weight.dtype == torch.float8_e4m3fn:
            return dequant_fp8_block(weight, self.get(f"{prefix}.weight_scale_inv"))
        return weight.float()

    # ---- per-module loaders, all returning fp32 torch tensors in HF [out, in] layout ----

    def embedding(self):
        return self.get("model.embed_tokens.weight")

    def lm_head(self):
        return self.get("lm_head.weight")

    def final_norm(self):
        return self.get("model.norm.weight").float()

    def attention(self, layer_idx):
        cfg = self.config
        dims = cfg.attention_dims(layer_idx)
        prefix = f"model.layers.{layer_idx}.self_attn"
        w = self.get(f"{prefix}.qkv_proj.weight")
        s = self.get(f"{prefix}.qkv_proj.weight_scale_inv")
        q, k, v = split_fused_qkv(
            w,
            s,
            dims.num_heads,
            dims.num_kv_heads,
            dims.head_dim,
            dims.v_head_dim,
            ckpt_tp=cfg.num_key_value_heads,
        )
        out = {
            "q_proj": q,
            "k_proj": k,
            "v_proj": v,
            "o_proj": self.get(f"{prefix}.o_proj.weight").float(),
        }
        sink_name = f"{prefix}.attention_sink_bias"
        out["attention_sink_bias"] = self.get(sink_name).float() if self.has(sink_name) else None
        return out

    def norms(self, layer_idx):
        prefix = f"model.layers.{layer_idx}"
        return {
            "input_layernorm": self.get(f"{prefix}.input_layernorm.weight").float(),
            "post_attention_layernorm": self.get(f"{prefix}.post_attention_layernorm.weight").float(),
        }

    def dense_mlp(self, layer_idx):
        prefix = f"model.layers.{layer_idx}.mlp"
        return {name: self._fp8_or_bf16(f"{prefix}.{name}") for name in ("gate_proj", "up_proj", "down_proj")}

    def router(self, layer_idx):
        prefix = f"model.layers.{layer_idx}.mlp.gate"
        return {
            "weight": self.get(f"{prefix}.weight").float(),
            "e_score_correction_bias": self.get(f"{prefix}.e_score_correction_bias").float(),
        }

    def expert(self, layer_idx, expert_idx, dtype=torch.bfloat16):
        prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
        out = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            weight = self.get(f"{prefix}.{name}.weight")
            if weight.dtype == torch.uint8:
                out[name] = dequant_mxfp4(weight, self.get(f"{prefix}.{name}.weight_scale")).to(dtype)
            else:
                out[name] = self._fp8_or_bf16(f"{prefix}.{name}").to(dtype)
        return out

    def experts(self, layer_idx, dtype=torch.bfloat16):
        """All routed experts stacked: gate/up [E, I, H], down [E, H, I] in ``dtype``."""
        num_experts = self.config.n_routed_experts
        per_expert = [self.expert(layer_idx, e, dtype) for e in range(num_experts)]
        return {name: torch.stack([p[name] for p in per_expert]) for name in ("gate_proj", "up_proj", "down_proj")}


@lru_cache(maxsize=None)
def default_model_path():
    return os.environ.get("MIMO_MODEL_PATH", "/home/zni/models/MiMo-V2.6-Flash-MOPD")
