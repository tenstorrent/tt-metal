# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Export a TTML Llama model to HuggingFace safetensors format.

Inverse of safetensors_loader.py.  Hardware-free: operates on plain numpy arrays.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np


def _permute_proj_rows(w: np.ndarray, n_heads: int) -> np.ndarray:
    """Inverse of _unpermute_proj_rows: interleaved TTML layout -> HF grouped layout."""
    R, C = w.shape
    D = R // n_heads
    return w.reshape(n_heads, D // 2, 2, C).transpose(0, 2, 1, 3).reshape(R, C)


def _from_4d_to_2d(arr: np.ndarray, is_norm: bool = False) -> np.ndarray:
    """Squeeze [1,1,R,C] -> [R,C] or [1,1,1,N] -> [N] (norms)."""
    if arr.ndim == 4:
        arr = arr.squeeze()
        if is_norm and arr.ndim == 0:
            arr = arr.reshape(1)
    return arr


def _normalize_config(config) -> dict:
    """Return a canonical flat dict from either CppLlamaConfig or Python LlamaConfig."""
    if hasattr(config, "num_heads"):
        # CppLlamaConfig
        cfg: dict = {
            "num_attention_heads": config.num_heads,
            "num_key_value_heads": config.num_groups,
            "hidden_size": config.embedding_dim,
            "intermediate_size": config.intermediate_dim if hasattr(config, "intermediate_dim") else None,
            "num_hidden_layers": config.num_blocks,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_sequence_length,
            "rope_theta": float(config.theta),
            "weight_tying": False,
        }
        sf = getattr(config, "scaling_factor", 0.0)
        orig = getattr(config, "original_context_length", 0)
        if sf and orig:
            cfg["rope_scaling"] = {
                "factor": float(sf),
                "high_freq_factor": float(getattr(config, "high_freq_factor", 4.0)),
                "low_freq_factor": float(getattr(config, "low_freq_factor", 1.0)),
                "original_max_position_embeddings": int(orig),
                "rope_type": "llama3",
            }
        else:
            cfg["rope_scaling"] = None
    else:
        # Python LlamaConfig
        from ttml.models import WeightTyingType

        cfg = {
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_position_embeddings,
            "rope_theta": float(config.rope_theta),
            "weight_tying": config.weight_tying == WeightTyingType.Enabled,
        }
        rs = config.rope_scaling
        if rs is not None and getattr(rs, "scaling_factor", 0.0) and getattr(rs, "original_context_length", 0):
            cfg["rope_scaling"] = {
                "factor": float(rs.scaling_factor),
                "high_freq_factor": float(rs.high_freq_factor),
                "low_freq_factor": float(rs.low_freq_factor),
                "original_max_position_embeddings": int(rs.original_context_length),
                "rope_type": "llama3",
            }
        else:
            cfg["rope_scaling"] = None

    return cfg


def _detect_prefix_and_block_fmt(keys: list[str]) -> tuple[str, str]:
    """Return (prefix, block_fmt) where block_fmt is 'llama_block_{}' or 'blocks/{}'."""
    for key in keys:
        if key.startswith("pipeline_parallel_llama/"):
            return "pipeline_parallel_llama", "llama_block_{}"
        if key.startswith("llama/"):
            return "llama", "llama_block_{}"
        if key.startswith("Llama/"):
            return "Llama", "blocks/{}"
    return "", "blocks/{}"


def _build_hf_state_dict(params: dict[str, np.ndarray], cfg: dict) -> dict[str, np.ndarray]:
    """Map TTML parameter names to HF tensor names with required transforms."""
    num_heads = cfg["num_attention_heads"]
    num_kv_heads = cfg["num_key_value_heads"]
    hidden_size = cfg["hidden_size"]
    head_dim = hidden_size // num_heads
    kv_split = num_kv_heads * head_dim
    vocab_size = cfg["vocab_size"]
    weight_tying = cfg["weight_tying"]
    num_layers = cfg["num_hidden_layers"]

    prefix, block_fmt = _detect_prefix_and_block_fmt(list(params.keys()))
    sep = "/" if prefix else ""

    def key(suffix: str) -> str:
        return f"{prefix}{sep}{suffix}"

    hf: dict[str, np.ndarray] = {}

    # Embedding
    emb_key = key("tok_emb/weight")
    if emb_key in params:
        w = params[emb_key]
        hf["model.embed_tokens.weight"] = w[:vocab_size]

    # Final norm
    norm_key = key("ln_fc/gamma")
    if norm_key in params:
        hf["model.norm.weight"] = params[norm_key]

    # LM head
    fc_key = key("fc/weight")
    if fc_key in params and not weight_tying:
        hf["lm_head.weight"] = params[fc_key][:vocab_size]

    # Per-layer weights
    for i in range(num_layers):
        blk = block_fmt.format(i)
        pfx = key(blk)

        attn_norm = params.get(f"{pfx}/attention_norm/gamma")
        if attn_norm is not None:
            hf[f"model.layers.{i}.input_layernorm.weight"] = attn_norm

        mlp_norm = params.get(f"{pfx}/mlp_norm/gamma")
        if mlp_norm is not None:
            hf[f"model.layers.{i}.post_attention_layernorm.weight"] = mlp_norm

        q = params.get(f"{pfx}/attention/q_linear/weight")
        if q is not None:
            hf[f"model.layers.{i}.self_attn.q_proj.weight"] = _permute_proj_rows(q, num_heads)

        kv = params.get(f"{pfx}/attention/kv_linear/weight")
        if kv is not None:
            k_rows = kv[:kv_split]
            v_rows = kv[kv_split:]
            hf[f"model.layers.{i}.self_attn.k_proj.weight"] = _permute_proj_rows(k_rows, num_kv_heads)
            hf[f"model.layers.{i}.self_attn.v_proj.weight"] = v_rows

        out = params.get(f"{pfx}/attention/out_linear/weight")
        if out is not None:
            hf[f"model.layers.{i}.self_attn.o_proj.weight"] = out

        w1 = params.get(f"{pfx}/mlp/w1/weight")
        if w1 is not None:
            hf[f"model.layers.{i}.mlp.gate_proj.weight"] = w1

        w3 = params.get(f"{pfx}/mlp/w3/weight")
        if w3 is not None:
            hf[f"model.layers.{i}.mlp.up_proj.weight"] = w3

        w2 = params.get(f"{pfx}/mlp/w2/weight")
        if w2 is not None:
            hf[f"model.layers.{i}.mlp.down_proj.weight"] = w2

    return hf


def _build_hf_config_dict(cfg: dict) -> dict:
    """Build a HF-compatible config.json dict."""
    intermediate_size = cfg.get("intermediate_size")
    if intermediate_size is None:
        hidden_size = cfg["hidden_size"]
        raw = int(8 / 3 * hidden_size)
        intermediate_size = math.ceil(raw / 256) * 256

    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": cfg["hidden_size"],
        "intermediate_size": intermediate_size,
        "num_hidden_layers": cfg["num_hidden_layers"],
        "num_attention_heads": cfg["num_attention_heads"],
        "num_key_value_heads": cfg["num_key_value_heads"],
        "vocab_size": cfg["vocab_size"],
        "max_position_embeddings": cfg["max_position_embeddings"],
        "rope_theta": cfg["rope_theta"],
        "rope_scaling": cfg.get("rope_scaling"),
        "tie_word_embeddings": cfg.get("weight_tying", False),
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0",
    }


def save_to_hf_format(
    merged_params: dict[str, np.ndarray],
    output_dir: str | os.PathLike,
    config,
    tokenizer_source: str | os.PathLike | None = None,
) -> None:
    """Export TTML Llama weights to a HuggingFace-loadable directory.

    Args:
        merged_params: All ranks merged, numpy float32 arrays (may be 4D).
        output_dir: Destination directory (created if absent).
        config: CppLlamaConfig or Python LlamaConfig instance.
        tokenizer_source: Optional path to copy tokenizer files from.
    """
    from safetensors.numpy import save_file

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = _normalize_config(config)

    # Squeeze 4D tensors to 2D/1D before transforms
    squeezed: dict[str, np.ndarray] = {}
    for name, arr in merged_params.items():
        is_norm = name.endswith("/gamma")
        squeezed[name] = _from_4d_to_2d(arr.astype(np.float32), is_norm=is_norm)

    state_dict = _build_hf_state_dict(squeezed, cfg)

    save_file(state_dict, str(out / "model.safetensors"))

    config_dict = _build_hf_config_dict(cfg)
    with open(out / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    gen_config = {"do_sample": True, "temperature": 0.6, "top_p": 0.9}
    with open(out / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    if tokenizer_source is not None:
        src = Path(tokenizer_source)
        for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            src_file = src / fname
            if src_file.exists():
                shutil.copy2(src_file, out / fname)
