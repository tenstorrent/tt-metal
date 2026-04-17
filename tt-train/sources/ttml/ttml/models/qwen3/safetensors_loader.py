# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Safetensors I/O for Python Qwen3.

Two entry points:
  - :func:`load_from_hf` loads HuggingFace ``Qwen3ForCausalLM`` safetensors into
    a :class:`Qwen3` instance, applying RoPE-pair unpermutation to Q/K
    weights/biases, QK-norm unpermutation, and vocab tile-padding.
  - :func:`export_hf_model` re-permutes and crops back to HF layout and writes a
    safetensors directory that ``AutoModelForCausalLM.from_pretrained`` can load.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict

import ml_dtypes
import numpy as np

import ttnn
import ttml

from .. import WeightTyingType


# ---------------------------------------------------------------------------
# Weight-permutation helpers (module-private)
# ---------------------------------------------------------------------------


def _unpermute_proj_rows(w: np.ndarray, n_heads: int) -> np.ndarray:
    """HF ``[first_half, second_half]`` → ttml interleaved ``[r0, i0, r1, i1, ...]``."""
    if w.ndim == 1:
        total = w.shape[0]
        D = total // n_heads
        half = D // 2
        wv = w.reshape(n_heads, D)
        first_half = wv[:, :half]
        second_half = wv[:, half:]
        return np.stack([first_half, second_half], axis=2).reshape(total)
    if w.ndim == 2:
        rows, cols = w.shape
        D = rows // n_heads
        half = D // 2
        wv = w.reshape(n_heads, D, cols)
        first_half = wv[:, :half, :]
        second_half = wv[:, half:, :]
        return np.stack([first_half, second_half], axis=2).reshape(rows, cols)
    raise ValueError(f"Expected 1D or 2D tensor, got {w.ndim}D")


def _unpermute_norm_weights(w: np.ndarray) -> np.ndarray:
    """QK-Norm: HF ``[x1, x2, ..., y1, y2, ...]`` → ttml ``[x1, y1, x2, y2, ...]``."""
    head_dim = w.shape[0]
    if head_dim % 2 != 0:
        raise ValueError(f"QK-Norm weight dim must be even; got {head_dim}")
    half = head_dim // 2
    return w.reshape(2, half).T.reshape(-1)


def _repermute_proj_rows(w: np.ndarray, n_heads: int) -> np.ndarray:
    if w.ndim == 1:
        total = w.shape[0]
        D = total // n_heads
        wv = w.reshape(n_heads, D)
        reals = wv[:, 0::2]
        imags = wv[:, 1::2]
        return np.concatenate([reals, imags], axis=1).reshape(total)
    if w.ndim == 2:
        rows, cols = w.shape
        D = rows // n_heads
        wv = w.reshape(n_heads, D, cols)
        reals = wv[:, 0::2, :]
        imags = wv[:, 1::2, :]
        return np.concatenate([reals, imags], axis=1).reshape(rows, cols)
    raise ValueError(f"Expected 1D or 2D tensor, got {w.ndim}D")


def _repermute_norm_weights(w: np.ndarray) -> np.ndarray:
    head_dim = w.shape[0]
    half = head_dim // 2
    return w.reshape(half, 2).T.reshape(-1)


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def _pad_to_tile(arr: np.ndarray, tgt_rows: int, tgt_cols: int) -> np.ndarray:
    src_rows, src_cols = arr.shape
    if src_rows == tgt_rows and src_cols == tgt_cols:
        return arr
    out = np.zeros((tgt_rows, tgt_cols), dtype=arr.dtype)
    cr = min(src_rows, tgt_rows)
    cc = min(src_cols, tgt_cols)
    out[:cr, :cc] = arr[:cr, :cc]
    return out


def _crop_to_hf_2d(arr: np.ndarray, rows: int, cols: int) -> np.ndarray:
    return arr[:rows, :cols]


def _crop_to_hf_1d(arr: np.ndarray, dim: int) -> np.ndarray:
    return arr[:dim]


def _to_bf16_4d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr.reshape(1, 1, 1, -1)
    elif arr.ndim == 2:
        arr = arr.reshape(1, 1, arr.shape[0], arr.shape[1])
    return arr.astype(ml_dtypes.bfloat16)


def _assign(param, arr_4d: np.ndarray) -> None:
    param.assign(ttml.autograd.Tensor.from_numpy(arr_4d, layout=ttnn.Layout.TILE))


# ---------------------------------------------------------------------------
# HF ↔ ttml name mapping
# ---------------------------------------------------------------------------


def _hf_to_ttml_name(hf_name: str, *, root: str = "Qwen3") -> str:
    """``model.layers.0.self_attn.q_proj.weight`` → ``Qwen3/model/layers/0/self_attn/q_proj/weight``.

    ``lm_head.weight`` has no ``model.`` prefix in HF, so it stays at the top
    level; adding the root prefix gives ``Qwen3/lm_head/weight``.
    """
    return f"{root}/{hf_name.replace('.', '/')}"


def _ttml_to_hf_name(ttml_name: str, *, root: str = "Qwen3") -> str:
    prefix = f"{root}/"
    assert ttml_name.startswith(prefix), f"expected ttml name to start with {prefix!r}; got {ttml_name!r}"
    return ttml_name[len(prefix) :].replace("/", ".")


def _proj_transform_heads(hf_name: str, num_attention_heads: int, num_key_value_heads: int) -> int | None:
    """Return the head count to use when un/repermuting, or None if not a Q/K projection."""
    if ".self_attn.q_proj." in hf_name:
        return num_attention_heads
    if ".self_attn.k_proj." in hf_name:
        return num_key_value_heads
    return None


def _is_norm_proj(hf_name: str) -> bool:
    return hf_name.endswith(".self_attn.q_norm.weight") or hf_name.endswith(".self_attn.k_norm.weight")


# ---------------------------------------------------------------------------
# Public API: load HF → ttml
# ---------------------------------------------------------------------------


def load_from_hf(model, safetensors_path, config) -> None:
    """Load HuggingFace ``Qwen3ForCausalLM`` safetensors into a Python Qwen3 model.

    Args:
        model: A :class:`~ttml.models.qwen3.Qwen3` instance.
        safetensors_path: Directory containing one or more ``.safetensors`` shards.
        config: The :class:`~ttml.models.qwen3.Qwen3Config` used to build the model.
    """
    from safetensors.numpy import load_file

    directory = Path(safetensors_path)
    shards = sorted(directory.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in {directory}")

    hf_tensors: Dict[str, np.ndarray] = {}
    for shard in shards:
        hf_tensors.update(load_file(str(shard)))

    parameters = model.parameters()
    tied = config.weight_tying == WeightTyingType.Enabled
    loaded: set[str] = set()

    for hf_name, hf_arr in hf_tensors.items():
        arr = hf_arr.astype(np.float32)

        if tied and hf_name == "lm_head.weight":
            # Tied: HF may or may not emit this key; either way the tied embedding covers it.
            continue

        heads = _proj_transform_heads(hf_name, config.num_attention_heads, config.num_key_value_heads)
        if heads is not None:
            arr = _unpermute_proj_rows(arr, heads)
        elif _is_norm_proj(hf_name):
            arr = _unpermute_norm_weights(arr)

        ttml_name = _hf_to_ttml_name(hf_name)
        if ttml_name not in parameters:
            continue

        param = parameters[ttml_name]
        shape = param.shape()

        if arr.ndim == 2:
            tgt_rows, tgt_cols = shape[-2], shape[-1]
            r, c = arr.shape
            if r == tgt_rows and c == tgt_cols:
                pass
            elif c == tgt_rows and r == tgt_cols:
                arr = arr.T
            elif r <= tgt_rows and c <= tgt_cols:
                arr = _pad_to_tile(arr, tgt_rows, tgt_cols)
            else:
                raise RuntimeError(f"shape mismatch for {hf_name}: HF ({r}x{c}) vs ttml ({tgt_rows}x{tgt_cols})")
        elif arr.ndim == 1:
            tgt_dim = shape[-1]
            if arr.shape[0] > tgt_dim:
                raise RuntimeError(f"shape mismatch for {hf_name}: HF ({arr.shape[0]},) vs ttml ({tgt_dim},)")
            if arr.shape[0] != tgt_dim:
                padded = np.zeros((tgt_dim,), dtype=arr.dtype)
                padded[: arr.shape[0]] = arr
                arr = padded

        _assign(param, _to_bf16_4d(arr))
        loaded.add(ttml_name)

    unused = sorted(p for p in parameters if p not in loaded)
    if tied:
        # The tied lm_head weight is loaded via embed_tokens; don't flag it as missing.
        unused = [p for p in unused if not p.endswith("/lm_head/weight")]
    if unused:
        print(f"Warning: {len(unused)} model parameters were NOT loaded from safetensors:")
        for name in unused:
            print(f"  - {name}")


# ---------------------------------------------------------------------------
# Public API: export ttml → HF
# ---------------------------------------------------------------------------


def _build_hf_shapes(config) -> Dict[str, tuple]:
    shapes: Dict[str, tuple] = {}
    q_dim = config.num_attention_heads * config.head_dim
    kv_dim = config.num_key_value_heads * config.head_dim

    shapes["model.embed_tokens.weight"] = (config.vocab_size, config.hidden_size)
    if config.weight_tying == WeightTyingType.Disabled:
        shapes["lm_head.weight"] = (config.vocab_size, config.hidden_size)

    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        shapes[f"{p}.self_attn.q_proj.weight"] = (q_dim, config.hidden_size)
        shapes[f"{p}.self_attn.k_proj.weight"] = (kv_dim, config.hidden_size)
        shapes[f"{p}.self_attn.v_proj.weight"] = (kv_dim, config.hidden_size)
        shapes[f"{p}.self_attn.o_proj.weight"] = (config.hidden_size, q_dim)
        if config.attention_bias:
            shapes[f"{p}.self_attn.q_proj.bias"] = (q_dim,)
            shapes[f"{p}.self_attn.k_proj.bias"] = (kv_dim,)
            shapes[f"{p}.self_attn.v_proj.bias"] = (kv_dim,)
            shapes[f"{p}.self_attn.o_proj.bias"] = (config.hidden_size,)
        shapes[f"{p}.self_attn.q_norm.weight"] = (config.head_dim,)
        shapes[f"{p}.self_attn.k_norm.weight"] = (config.head_dim,)
        shapes[f"{p}.input_layernorm.weight"] = (config.hidden_size,)
        shapes[f"{p}.post_attention_layernorm.weight"] = (config.hidden_size,)
        shapes[f"{p}.mlp.gate_proj.weight"] = (config.intermediate_size, config.hidden_size)
        shapes[f"{p}.mlp.up_proj.weight"] = (config.intermediate_size, config.hidden_size)
        shapes[f"{p}.mlp.down_proj.weight"] = (config.hidden_size, config.intermediate_size)

    shapes["model.norm.weight"] = (config.hidden_size,)
    return shapes


_HF_CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


def export_hf_model(model, config, out_dir, hf_source_dir=None) -> str:
    """Export the model in HF-compatible format.

    Applies inverse transforms (re-permute Q/K rows, re-permute QK-Norm weights,
    crop vocab/intermediate back to their HF shape) and writes
    ``model.safetensors`` to ``out_dir``. When ``hf_source_dir`` is provided,
    tokenizer + config files are copied alongside so the directory loads with
    ``AutoModelForCausalLM.from_pretrained``.
    """
    from safetensors.numpy import save_file

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    hf_shapes = _build_hf_shapes(config)
    parameters = model.parameters()

    hf_state: Dict[str, np.ndarray] = {}
    for ttml_name, param in parameters.items():
        hf_name = _ttml_to_hf_name(ttml_name)
        if hf_name not in hf_shapes:
            # When tied, lm_head.weight is omitted from hf_shapes because HF only
            # emits model.embed_tokens.weight. Any other name here is unexpected.
            continue
        arr = param.to_numpy(ttnn.DataType.FLOAT32).squeeze()

        heads = _proj_transform_heads(hf_name, config.num_attention_heads, config.num_key_value_heads)
        if heads is not None:
            arr = _repermute_proj_rows(arr, heads)
        elif _is_norm_proj(hf_name):
            arr = _repermute_norm_weights(arr)

        hf_shape = hf_shapes[hf_name]
        if arr.ndim == 2:
            arr = _crop_to_hf_2d(arr, hf_shape[0], hf_shape[1])
        elif arr.ndim == 1:
            arr = _crop_to_hf_1d(arr, hf_shape[0])

        hf_state[hf_name] = arr.astype(ml_dtypes.bfloat16)

    save_file(hf_state, str(out_path / "model.safetensors"))

    # Minimal single-shard index so HF treats this as a valid model dir.
    index = {"metadata": {"total_size": 0}, "weight_map": {}}
    with open(out_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)

    if hf_source_dir is not None and os.path.isdir(hf_source_dir):
        for fname in _HF_CONFIG_FILES:
            src = os.path.join(hf_source_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, out_path / fname)

    return str(out_path)
