# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lazy tensor access to a Hy4-preview checkpoint, BF16 or the official MXFP8 release.

Tensor names are the checkpoint's own (``hc_attn_layer.hc_pre.hc_fn``, ``self_attn.linear_gate``,
``self_attn.learnable_sink_param``, ...), not the HF module names: HF ``hy_v4`` renames them on load
(``transformers.conversion_mapping``), the reference here reads them as stored.

MXFP8 (``quant_method: modelopt``, ``quant_algo: MXFP8``): a quantized ``X`` is float8_e4m3fn with a
``X_scale`` (experts) or ``X.weight_scale`` (linears) uint8 E8M0 exponent per 32 consecutive input
elements. Dequantization is ``fp8 * 2**(e - 127)``, exact in bf16 (3 mantissa bits, fp32 exponent range),
so the reference consumes the dequantized MXFP8 weights in the same dtype as the BF16 ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from models.demos.deepseek_v3_d_p.reference.hy4.config import Hy4RefConfig

MX_BLOCK = 32


def dequant_mxfp8(q: torch.Tensor, scale: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
    """``q`` float8 ``[..., K]`` with uint8 E8M0 ``scale`` ``[..., K / block]`` -> ``out_dtype``."""
    if scale.dtype != torch.uint8:
        raise ValueError(f"MXFP8 scale must be uint8 E8M0, got {scale.dtype}")
    k, nb = q.shape[-1], scale.shape[-1]
    if k % nb or q.shape[:-1] != scale.shape[:-1]:
        raise ValueError(f"scale {tuple(scale.shape)} does not tile weight {tuple(q.shape)}")
    block = k // nb
    exp = torch.exp2(scale.to(torch.float32) - 127.0)
    deq = q.to(torch.float32).reshape(*q.shape[:-1], nb, block) * exp.unsqueeze(-1)
    return deq.reshape(q.shape).to(out_dtype)


class Hy4Checkpoint:
    """Reads tensors on demand from ``model.safetensors.index.json`` (or a single ``model.safetensors``)."""

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        raw = json.loads((self.model_dir / "config.json").read_text())
        self.config = Hy4RefConfig.from_dict(raw)
        q = raw.get("quantization_config") or {}
        self.quant_algo = (q.get("quantization") or {}).get("quant_algo") or q.get("quant_algo")
        index = self.model_dir / "model.safetensors.index.json"
        if index.exists():
            self.weight_map = json.loads(index.read_text())["weight_map"]
        else:
            single = self.model_dir / "model.safetensors"
            with safe_open(str(single), framework="pt") as f:
                self.weight_map = {k: single.name for k in f.keys()}

    @property
    def precision(self) -> str:
        return "mxfp8" if (self.quant_algo or "").upper() == "MXFP8" else "bf16"

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def _raw(self, name: str) -> torch.Tensor:
        try:
            shard = self.weight_map[name]
        except KeyError:
            raise KeyError(f"{name} not in checkpoint {self.model_dir}") from None
        with safe_open(str(self.model_dir / shard), framework="pt") as f:
            return f.get_tensor(name)

    def _scale_name(self, name: str) -> str | None:
        for cand in (name + "_scale", name[: -len(".weight")] + ".weight_scale" if name.endswith(".weight") else None):
            if cand and cand in self.weight_map:
                return cand
        return None

    def get(self, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        """The tensor, dequantized if it is MXFP8; cast to ``dtype`` when given (native dtype otherwise)."""
        t = self._raw(name)
        if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            scale = self._scale_name(name)
            if scale is None:
                raise KeyError(f"{name} is {t.dtype} but has no MXFP8 scale tensor")
            t = dequant_mxfp8(t, self._raw(scale))
        return t if dtype is None else t.to(dtype)


# ---- random checkpoint for smoke tests -----------------------------------------------------------
def random_checkpoint_tensors(cfg: Hy4RefConfig, seed: int = 0, dtype=torch.float32) -> dict:
    """Every tensor a Hy4 checkpoint holds (checkpoint names), random but well-conditioned: linears at
    ~1/sqrt(fan_in), norms near 1, iHC mixes and sinks spread enough to matter. MTP is omitted (the
    prefill reference does not run it; HF ignores it on load)."""
    g = torch.Generator().manual_seed(seed)

    def lin(out_f, in_f):
        return (torch.randn(out_f, in_f, generator=g) / in_f**0.5).to(dtype)

    def norm(n):
        return (1.0 + 0.1 * torch.randn(n, generator=g)).to(dtype)

    D, H, hc = cfg.hidden_size, cfg.num_attention_heads, cfg.hc_mult
    t = {
        "model.embed_tokens.weight": torch.randn(cfg.vocab_size, D, generator=g).to(dtype),
        "lm_head.weight": lin(cfg.vocab_size, D),
        "model.norm.weight": norm(D),
        "model.hc_head.hc_head_fn": 0.5 * torch.randn(hc, hc * D, generator=g) / (hc * D) ** 0.5,
        "model.hc_head.hc_head_base": 0.3 * torch.randn(hc, generator=g),
        "model.hc_head.hc_head_scale": torch.tensor([0.7]),
    }
    for i in range(cfg.num_hidden_layers):
        p = f"model.layers.{i}."
        for hc_name in ("hc_attn_layer", "hc_mlp_layer"):
            t[p + f"{hc_name}.hc_pre.hc_fn"] = torch.randn(2 * hc, hc * D, generator=g) / (hc * D) ** 0.5
            t[p + f"{hc_name}.hc_pre.hc_base"] = 0.5 * torch.randn(2 * hc, generator=g)
            t[p + f"{hc_name}.hc_pre.hc_scale"] = 0.5 + torch.rand(2, generator=g)
        t[p + "input_layernorm.weight"] = norm(D)
        t[p + "post_attention_layernorm.weight"] = norm(D)
        a = p + "self_attn."
        t[a + "q_a_proj.weight"] = lin(cfg.q_lora_rank, D)
        t[a + "q_a_layernorm.weight"] = norm(cfg.q_lora_rank)
        t[a + "q_b_proj.weight"] = lin(H * cfg.qk_head_dim, cfg.q_lora_rank)
        t[a + "kv_a_proj_with_mqa.weight"] = lin(cfg.kvpe_dim, D)
        t[a + "kv_a_layernorm.weight"] = norm(cfg.kv_lora_rank)
        t[a + "kv_b_proj.weight"] = lin(H * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank)
        t[a + "o_proj.weight"] = lin(D, H * cfg.v_head_dim)
        t[a + "linear_gate.weight"] = lin(H * cfg.v_head_dim, D)
        t[a + "learnable_sink_param"] = torch.randn(H, generator=g)
        if cfg.is_full_indexer(i):
            x = a + "indexer."
            t[x + "wq_b.weight"] = lin(cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank)
            t[x + "wk.weight"] = lin(cfg.index_head_dim, D)
            t[x + "k_norm.weight"] = norm(cfg.index_head_dim)
            t[x + "k_norm.bias"] = 0.1 * torch.randn(cfg.index_head_dim, generator=g)
            t[x + "weights_proj.weight"] = lin(cfg.index_n_heads, D)
        m = p + "mlp."
        if cfg.is_dense(i):
            t[m + "gate_proj.weight"] = lin(cfg.intermediate_size, D)
            t[m + "up_proj.weight"] = lin(cfg.intermediate_size, D)
            t[m + "down_proj.weight"] = lin(D, cfg.intermediate_size)
        else:
            E, I = cfg.n_routed_experts, cfg.moe_intermediate_size
            t[m + "gate.weight"] = lin(E, D)
            t[m + "gate.e_score_correction_bias"] = 0.1 * torch.randn(E, generator=g)
            t[m + "experts.gate_up_proj"] = (torch.randn(E, 2 * I, D, generator=g) / D**0.5).to(dtype)
            t[m + "experts.down_proj"] = (torch.randn(E, D, I, generator=g) / I**0.5).to(dtype)
            S = I * cfg.n_shared_experts
            t[m + "shared_experts.gate_proj.weight"] = lin(S, D)
            t[m + "shared_experts.up_proj.weight"] = lin(S, D)
            t[m + "shared_experts.down_proj.weight"] = lin(D, S)
    return {k: v.contiguous() for k, v in t.items()}


def save_random_checkpoint(out_dir, cfg: Hy4RefConfig, seed: int = 0, dtype=torch.float32) -> Path:
    """Write a loadable random checkpoint (``config.json`` + ``model.safetensors``, checkpoint names)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(random_checkpoint_tensors(cfg, seed, dtype), str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(cfg.to_hf_dict(), indent=1))
    return out
