# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace config -> canonical architecture signature.

STRICT SCOPE: this module knows nothing about Tenstorrent hardware. No mesh
shapes, no TP/SP/EP, no bfp8/bfp4, no kernels. It answers exactly one
question: "what decoder architecture is this, mechanically?"

Two outputs per block, deliberately kept apart:
  mech  - categorical mechanism signature. Equality here means "same dataflow".
  shape - numeric dimensions. Differences here are a retune, not a rewrite.

Anything we cannot determine from the config is UNKNOWN, never guessed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

UNKNOWN = "unknown"


# ---------------------------------------------------------------- field aliases
# Confirmed by dumping every config.json bundled in tt-metal; see
# MODEL_TYPE_TRAITS below for mechanisms that no config field encodes.
def _get(cfg: dict, *names: str, default: Any = None) -> Any:
    for n in names:
        if n in cfg and cfg[n] is not None:
            return cfg[n]
    return default


N_LAYERS = ("num_hidden_layers", "n_layers", "num_layers")
HIDDEN = ("hidden_size", "dim", "d_model", "n_embd")
N_HEADS = ("num_attention_heads", "n_heads", "num_heads")
N_KV = ("num_key_value_heads", "n_kv_heads", "num_kv_heads")
FFN = ("intermediate_size", "ffn_dim", "n_inner")
EPS = ("rms_norm_eps", "norm_eps", "layer_norm_eps", "block_norm_eps")
# MoE: five spellings of the same three numbers across deepseek/gpt-oss/
# minimax/kimi/gemma4.
N_EXPERTS = ("num_local_experts", "n_routed_experts", "num_experts", "moe_num_experts")
TOP_K = ("num_experts_per_tok", "experts_per_token", "num_experts_per_token", "top_k_experts", "moe_top_k")
MOE_FFN = ("moe_intermediate_size", "expert_intermediate_size", "routed_expert_hidden_size")
N_SHARED = ("n_shared_experts", "num_shared_experts")
SHARED_FFN = ("shared_intermediate_size", "shared_expert_intermediate_size")


# --------------------------------------------------------- model_type traits
# Mechanisms that exist in the modeling code but have NO config field. Each
# entry is a claim about source we have read; keep the justification inline so
# the table stays auditable rather than folklore.
MODEL_TYPE_TRAITS: dict[str, dict[str, Any]] = {
    # Qwen3 applies per-head q_norm/k_norm unconditionally (no flag in config).
    "qwen3": {"qk_norm": "per_head"},
    "qwen3_vl": {"qk_norm": "per_head"},
    "qwen3_5": {"qk_norm": "per_head"},
    # GptOssAttention carries learned per-head sinks appended to the softmax,
    # and its MLP is the clamped swigluoai (config exposes only swiglu_limit).
    "gpt_oss": {"sinks": True, "glu": "swiglu_clamped", "qk_norm": "none", "norm_style": "standard"},
    # Gemma RMSNorm scales by (1 + w) and Gemma3+ adds per-head QK norm.
    "gemma3": {"norm_style": "gemma_1plus", "qk_norm": "per_head"},
    "gemma4": {"norm_style": "gemma_1plus", "qk_norm": "per_head"},
    # MLA carries q_a/kv_a LoRA norms, but no per-head QK norm.
    "deepseek_v3": {"qk_norm": "none", "norm_style": "standard", "sinks": False},
    "kimi_k3": {"qk_norm": "none", "norm_style": "standard", "sinks": False},
    # Plain llama/mistral/qwen2/phi3 families: no QK norm, no sinks, standard norm.
    "llama": {"qk_norm": "none", "sinks": False, "norm_style": "standard", "qkv_bias": False},
    "mistral": {"qk_norm": "none", "sinks": False, "norm_style": "standard", "qkv_bias": False},
    "ministral3": {"qk_norm": "none", "sinks": False, "norm_style": "standard", "qkv_bias": False},
    "mistral3": {"qk_norm": "none", "sinks": False, "norm_style": "standard", "qkv_bias": False},
    "qwen2": {"qk_norm": "none", "sinks": False, "norm_style": "standard", "qkv_bias": False},
    "qwen2_5_vl": {"qk_norm": "none", "sinks": False, "norm_style": "standard", "qkv_bias": False},
    "phi3": {"qk_norm": "none", "sinks": False, "norm_style": "standard"},
    "mllama": {"qk_norm": "none", "sinks": False, "norm_style": "standard", "qkv_bias": False},
    # M3 states qk_norm/gemma_norm itself; the absent projection biases and
    # absent sinks are only visible in the modeling source.
    "minimax_m3": {"qkv_bias": False, "sinks": False},
}

# Which mechanism fields change the dataflow (a rewrite) vs. only host-side
# parameters (a retune). Drives the identical/compatible/different verdict.
SEVERITY: dict[str, dict[str, str]] = {
    "attention": {
        "kind": "dataflow",
        "qk_norm": "dataflow",
        "rope_coverage": "dataflow",
        "sparsity": "dataflow",
        "sinks": "dataflow",
        "out_gate": "dataflow",
        "qkv_bias": "dataflow",
        "kv_shared_layers": "dataflow",
        "rope_scaling": "host",  # only changes the cos/sin table generation
    },
    "mlp": {
        "glu": "dataflow",
        "moe": "dataflow",
        "router_score": "dataflow",
        "router_bias": "dataflow",
        "shared_expert": "dataflow",
        "hybrid_schedule": "dataflow",
    },
    "norm": {"style": "dataflow", "eps": "host"},
    "embed": {"tied": "dataflow"},
    "global": {"quant": "ingest"},
}


@dataclass
class Signature:
    name: str
    source: str
    model_type: str
    architectures: list[str]
    has_vision: bool
    mech: dict[str, dict[str, Any]]
    shape: dict[str, dict[str, Any]]
    params: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "model_type": self.model_type,
            "architectures": self.architectures,
            "has_vision": self.has_vision,
            "mech": self.mech,
            "shape": self.shape,
            "params": self.params,
            "notes": self.notes,
        }


def unwrap(cfg: dict) -> tuple[dict, bool]:
    """Return (text-backbone config, has_vision). Handles nested text_config."""
    has_vision = "vision_config" in cfg
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        text = dict(cfg["text_config"])
        # quantization/architectures usually live at the top level
        for k in ("quantization_config", "architectures", "tie_word_embeddings"):
            text.setdefault(k, cfg.get(k))
        return text, has_vision
    return cfg, has_vision


def _rope(t: dict, head_dim: int) -> tuple[str, str, float]:
    """-> (coverage, scaling_type, theta)."""
    params = _get(t, "rope_parameters", "rope_scaling", default=None)
    theta = _get(t, "rope_theta", default=None)
    scaling = "none"
    if isinstance(params, dict):
        theta = params.get("rope_theta", theta)
        scaling = (params.get("rope_type") or params.get("type") or "default").lower()
        if scaling in ("default", "none"):
            scaling = "none"
    # coverage
    prf = _get(t, "partial_rotary_factor")
    rotary_dim = _get(t, "rotary_dim")
    if "qk_rope_head_dim" in t and "qk_nope_head_dim" in t:
        coverage = "partial"  # MLA splits rope/nope by construction
    elif prf is not None and float(prf) < 1.0:
        coverage = "partial"
    elif rotary_dim is not None and head_dim and int(rotary_dim) < int(head_dim):
        coverage = "partial"
    elif theta is None:
        coverage = "none"
    else:
        coverage = "full"
    return coverage, scaling, float(theta) if theta else 0.0


def _sparsity(t: dict) -> str:
    sac = _get(t, "sparse_attention_config", default={}) or {}
    if isinstance(sac, dict) and sac.get("use_sparse_attention"):
        return "block_sparse"
    layer_types = _get(t, "layer_types", default=None)
    if isinstance(layer_types, list) and layer_types:
        kinds = set(layer_types)
        if any("linear" in k for k in kinds):
            return "hybrid_linear"
        if any("sliding" in k for k in kinds):
            return "hybrid_sliding" if any("full" in k for k in kinds) else "sliding"
    if _get(t, "linear_attn_config") is not None:
        return "hybrid_linear"
    sw = _get(t, "sliding_window")
    if sw and _get(t, "use_sliding_window", default=True):
        return "sliding"
    return "dense"


def _glu(t: dict, traits: dict) -> str:
    if "swiglu_limit" in t or "swiglu_alpha" in t:
        return "swiglu_clamped"
    if traits.get("glu"):
        return traits["glu"]
    act = str(_get(t, "hidden_act", "hidden_activation", default="") or "").lower()
    if not act:
        return UNKNOWN
    if act in ("swigluoai", "swiglu_oai"):
        return "swiglu_clamped"
    if "silu" in act or "swish" in act or "swiglu" in act:
        return "swiglu_silu"
    if "gelu" in act or "geglu" in act:
        return "geglu"
    return act


def build(cfg: dict, name: str, source: str = "") -> Signature:
    t, has_vision = unwrap(cfg)
    inner = str(_get(t, "model_type", default=UNKNOWN))
    outer = str(_get(cfg, "model_type", default=inner))
    mt = inner if inner != UNKNOWN else outer
    # A nested text_config names itself (mllama -> "mllama_text_model"), so fall
    # back to the wrapper's model_type before giving up on the traits table.
    traits = MODEL_TYPE_TRAITS.get(inner) or MODEL_TYPE_TRAITS.get(outer) or {}
    notes: list[str] = []
    if not traits:
        notes.append(f"model_type '{mt}' has no traits entry: qk_norm/sinks/norm_style may be under-reported")

    hidden = int(_get(t, *HIDDEN, default=0) or 0)
    n_q = int(_get(t, *N_HEADS, default=0) or 0)
    n_kv = int(_get(t, *N_KV, default=n_q) or n_q)
    head_dim = int(_get(t, "head_dim", default=(hidden // n_q if n_q else 0)) or 0)
    layers = int(_get(t, *N_LAYERS, default=0) or 0)
    ffn = int(_get(t, *FFN, default=0) or 0)

    # ---- attention kind
    is_mla = "kv_lora_rank" in t and t.get("kv_lora_rank")
    if is_mla:
        kind = "MLA"
    elif n_kv == n_q:
        kind = "MHA"
    elif n_kv == 1:
        kind = "MQA"
    else:
        kind = "GQA"

    coverage, scaling, theta = _rope(t, head_dim)

    qk_norm = UNKNOWN
    if _get(t, "use_qk_norm") is not None:
        qk_norm = str(_get(t, "qk_norm_type", default="on")) if t.get("use_qk_norm") else "none"
    elif _get(t, "qk_layernorm") is not None:
        qk_norm = "on" if t["qk_layernorm"] else "none"
    elif "qk_norm" in traits:
        qk_norm = traits["qk_norm"]

    sinks = traits.get("sinks", UNKNOWN if mt not in MODEL_TYPE_TRAITS else False)
    out_gate = bool(
        _get(t, "attention_output_gate", "attn_output_gate", "mla_use_output_gate", default=False)
        or _get(t, "output_gate_type", default=None)
    )

    # ---- MoE
    n_experts = int(_get(t, *N_EXPERTS, default=0) or 0)
    top_k = int(_get(t, *TOP_K, default=0) or 0)
    moe_ffn = int(_get(t, *MOE_FFN, default=(ffn if n_experts else 0)) or 0)
    n_shared = int(_get(t, *N_SHARED, default=0) or 0)
    shared_ffn = int(_get(t, *SHARED_FFN, default=(moe_ffn if n_shared else 0)) or 0)
    dense_ffn = int(_get(t, "dense_intermediate_size", default=ffn) or ffn)
    is_moe = n_experts > 0

    score = str(_get(t, "scoring_func", "moe_router_activation_func", default="") or "").lower()
    if is_moe and not score:
        score = "softmax"  # HF default when unspecified
    router_bias = bool(
        _get(t, "use_routing_bias", default=False) or str(_get(t, "topk_method", default="")) == "noaux_tc"
    )

    # hybrid dense/MoE layer schedule
    mlf = _get(t, "moe_layer_freq")
    fkd = int(_get(t, "first_k_dense_replace", default=0) or 0)
    n_dense_layers = 0
    if isinstance(mlf, list):
        n_dense_layers = sum(1 for v in mlf if not v)
    elif fkd:
        n_dense_layers = fkd
    hybrid = bool(is_moe and n_dense_layers)

    norm_style = traits.get("norm_style", UNKNOWN)
    if _get(t, "use_gemma_norm") is not None:
        norm_style = "gemma_1plus" if t["use_gemma_norm"] else "standard"

    qb = _get(t, "attention_bias", "qkv_bias", default=None)
    if qb is None and "qkv_bias" in traits:
        qb = traits["qkv_bias"]
    quant = ((_get(t, "quantization_config", default=None) or {}) or {}).get("quant_method") or "none"

    mech = {
        "attention": {
            "kind": kind,
            "qk_norm": qk_norm,
            "rope_coverage": coverage,
            "rope_scaling": scaling,
            "sparsity": _sparsity(t),
            "sinks": sinks,
            "out_gate": out_gate,
            "qkv_bias": bool(qb) if qb is not None else UNKNOWN,
            "kv_shared_layers": bool(_get(t, "num_kv_shared_layers", default=0)),
        },
        "mlp": {
            "glu": _glu(t, traits),
            "moe": is_moe,
            "router_score": score or "n/a",
            "router_bias": router_bias if is_moe else "n/a",
            "shared_expert": (n_shared > 0) if is_moe else "n/a",
            "hybrid_schedule": hybrid,
        },
        "norm": {"style": norm_style, "eps": float(_get(t, *EPS, default=0.0) or 0.0)},
        "embed": {"tied": bool(_get(t, "tie_word_embeddings", default=False))},
        "global": {"quant": quant},
    }
    shape = {
        "attention": {
            "hidden": hidden,
            "n_q": n_q,
            "n_kv": n_kv,
            "head_dim": head_dim,
            "gqa_ratio": (n_q // n_kv) if n_kv else 0,
            "theta": theta,
            "max_pos": int(_get(t, "max_position_embeddings", default=0) or 0),
        },
        "mlp": {
            "ffn": ffn,
            "dense_ffn": dense_ffn,
            "moe_ffn": moe_ffn if is_moe else 0,
            "n_experts": n_experts,
            "top_k": top_k,
            "shared_ffn": shared_ffn,
        },
        "global": {
            "layers": layers,
            "dense_layers": n_dense_layers,
            "vocab": int(_get(t, "vocab_size", default=0) or 0),
        },
    }

    sig = Signature(
        name=name,
        source=source,
        model_type=mt,
        architectures=list(_get(t, "architectures", default=[]) or []),
        has_vision=has_vision,
        mech=mech,
        shape=shape,
        notes=notes,
    )
    sig.params = estimate_params(sig, t)
    return sig


def estimate_params(sig: Signature, t: dict) -> dict[str, float]:
    """Text-backbone parameter estimate (total and per-token active) from the
    config alone. Vision towers and MTP heads are excluded."""
    a, m, g = sig.shape["attention"], sig.shape["mlp"], sig.shape["global"]
    hidden, n_q, n_kv, hd = a["hidden"], a["n_q"], a["n_kv"], a["head_dim"]
    layers, vocab = g["layers"], g["vocab"]
    if not (hidden and layers):
        return {}

    if sig.mech["attention"]["kind"] == "MLA":
        q_lora = int(_get(t, "q_lora_rank", default=0) or 0)
        kv_lora = int(_get(t, "kv_lora_rank", default=0) or 0)
        nope, rope = int(_get(t, "qk_nope_head_dim", default=0) or 0), int(_get(t, "qk_rope_head_dim", default=0) or 0)
        v_hd = int(_get(t, "v_head_dim", default=hd) or hd)
        attn = (
            ((hidden * q_lora + q_lora * n_q * (nope + rope)) if q_lora else hidden * n_q * (nope + rope))
            + hidden * (kv_lora + rope)
            + kv_lora * n_q * (nope + v_hd)
            + n_q * v_hd * hidden
        )
    else:
        attn = 2 * hidden * n_q * hd + 2 * hidden * n_kv * hd

    dense_mlp = 3 * hidden * m["dense_ffn"]
    n_dense = g["dense_layers"] if sig.mech["mlp"]["hybrid_schedule"] else (0 if sig.mech["mlp"]["moe"] else layers)
    n_moe = layers - n_dense if sig.mech["mlp"]["moe"] else 0

    moe_total = moe_active = 0.0
    if n_moe:
        per_expert = 3 * hidden * m["moe_ffn"]
        shared = 3 * hidden * m["shared_ffn"] if m["shared_ffn"] else 0
        moe_total = m["n_experts"] * per_expert + shared + hidden * m["n_experts"]
        moe_active = m["top_k"] * per_expert + shared + hidden * m["n_experts"]

    embed = vocab * hidden * (1 if sig.mech["embed"]["tied"] else 2)
    if int(_get(t, "num_mtp_modules", "num_nextn_predict_layers", default=0) or 0):
        sig.notes.append("multi-token-prediction modules excluded from the parameter estimate")
    total = layers * attn + n_dense * dense_mlp + n_moe * moe_total + embed
    active = layers * attn + n_dense * dense_mlp + n_moe * moe_active + embed
    return {
        "total_B": round(total / 1e9, 1),
        "active_B": round(active / 1e9, 1),
        # Embedding conventions differ between vendors; this is the stable one.
        "active_no_embed_B": round((active - embed) / 1e9, 1),
        # Vendors quote one or the other of these two; a published "activated
        # parameter" count should land in [active_no_embed_B, active_embed_once_B].
        "active_embed_once_B": round((active - embed + vocab * hidden) / 1e9, 1),
        "gflop_per_token": round(2 * (active - embed) / 1e9, 1),
        "attn_per_layer_M": round(attn / 1e6, 1),
        "mlp_per_layer_M": round((moe_active if n_moe else dense_mlp) / 1e6, 1),
        "kv_bytes_per_token_bf16": layers * 2 * n_kv * hd * 2,
    }


def from_path(path: str, name: str | None = None) -> Signature:
    with open(path) as f:
        cfg = json.load(f)
    return build(cfg, name or path.split("/")[-2], source=path)
