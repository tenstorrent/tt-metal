# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load HuggingFace Qwen3 safetensors weights into a ttml Qwen3 model.

Handles:
- Q/K weight/bias unpermutation for TTML's interleaved RoPE
- QK-Norm weight unpermutation
- Embedding/LM-head padding for tile alignment
- Shape conversion from HF 2D to TTML 4D
- Weight tying (shared embedding / LM head)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import ml_dtypes

import ttnn
import ttml


# =====================================================================
# Weight permutation helpers (numpy)
# =====================================================================


def _unpermute_proj_rows(w: np.ndarray, n_heads: int) -> np.ndarray:
    """Reorder Q/K projection rows from HF grouped layout to interleaved pairs.

    HF stores rows as [first_half, second_half] per head.
    TTML's RoPE expects interleaved: [0, half, 1, half+1, ...].

    Works for both 2D weights and 1D biases.
    """
    if w.ndim == 1:
        total = w.shape[0]
        D = total // n_heads
        half = D // 2
        out = np.empty_like(w)
        for h in range(n_heads):
            base = h * D
            for i in range(half):
                out[base + 2 * i] = w[base + i]
                out[base + 2 * i + 1] = w[base + half + i]
        return out

    rows, cols = w.shape
    assert rows % n_heads == 0, f"rows {rows} not divisible by n_heads {n_heads}"
    D = rows // n_heads
    assert D % 2 == 0, f"rows per head {D} must be even"

    out = np.empty_like(w)
    half = D // 2
    for h in range(n_heads):
        base = h * D
        for i in range(half):
            out[base + 2 * i] = w[base + i]
            out[base + 2 * i + 1] = w[base + half + i]
    return out


def _unpermute_norm_weights(w: np.ndarray) -> np.ndarray:
    """Reorder QK-Norm: HF [x1,x2,...,y1,y2,...] -> TTML [x1,y1,x2,y2,...].

    Works for 1D weight vectors.
    """
    head_dim = w.shape[0]
    assert head_dim % 2 == 0
    half = head_dim // 2
    reshaped = w.reshape(2, half)
    return reshaped.T.ravel().copy()


def _pad_and_resize(arr: np.ndarray, tgt_rows: int, tgt_cols: int) -> np.ndarray:
    """Pad or crop a 2D array to (tgt_rows, tgt_cols)."""
    src_rows, src_cols = arr.shape
    if src_rows == tgt_rows and src_cols == tgt_cols:
        return arr

    out = np.zeros((tgt_rows, tgt_cols), dtype=arr.dtype)
    cr = min(src_rows, tgt_rows)
    cc = min(src_cols, tgt_cols)
    out[:cr, :cc] = arr[:cr, :cc]

    need_random = tgt_rows > src_rows or tgt_cols > src_cols
    if need_random:
        rng = np.random.default_rng()
        if tgt_rows > src_rows:
            out[cr:, :] = rng.normal(0.0, 0.02, (tgt_rows - cr, tgt_cols)).astype(arr.dtype)
        if tgt_cols > src_cols:
            out[:cr, cc:] = rng.normal(0.0, 0.02, (cr, tgt_cols - cc)).astype(arr.dtype)
    return out


def _to_bf16_4d(arr: np.ndarray) -> np.ndarray:
    """Convert to bfloat16 and reshape to 4D [1, 1, *, *]."""
    if arr.ndim == 1:
        arr = arr.reshape(1, 1, 1, -1)
    elif arr.ndim == 2:
        arr = arr.reshape(1, 1, arr.shape[0], arr.shape[1])
    return arr.astype(ml_dtypes.bfloat16)


def _assign_tensor(param, arr_4d: np.ndarray) -> None:
    restored = ttml.autograd.Tensor.from_numpy(arr_4d, layout=ttnn.Layout.TILE)
    param.assign(restored)


# =====================================================================
# Main loader
# =====================================================================


def load_from_safetensors(
    model: ttml.modules.AbstractModuleBase,
    safetensors_path: str | os.PathLike,
    config,
) -> None:
    """Load HuggingFace Qwen3 .safetensors weights into a ttml Qwen3 model.

    Args:
        model: A Qwen3ForCausalLM model instance.
        safetensors_path: Path to directory containing .safetensors file(s).
        config: The Qwen3Config used to build the model.
    """
    from safetensors.numpy import load_file

    safetensors_dir = Path(safetensors_path)
    st_files = sorted(safetensors_dir.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors files found in {safetensors_dir}")

    all_tensors: Dict[str, np.ndarray] = {}
    for f in st_files:
        print(f"Loading safetensors file: {f}")
        all_tensors.update(load_file(str(f)))

    parameters = model.parameters()
    used_params: set[str] = set()
    ignored_hf: set[str] = set()

    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    tie = model.tie_word_embeddings

    # Determine root prefix from actual parameter names
    any_key = next(iter(parameters))
    root_prefix = any_key.split("/")[0]

    def get_param(name: str):
        if name not in parameters:
            raise RuntimeError(f"Parameter {name} not found in model")
        used_params.add(name)
        return parameters[name]

    def _load_weight(param, hf_arr, unpermute_fn=None, unpermute_args=()):
        """Load a single weight: apply optional transform, pad/resize, assign."""
        hf_arr = hf_arr.astype(np.float32)
        if unpermute_fn is not None:
            hf_arr = unpermute_fn(hf_arr, *unpermute_args)

        tgt_shape = param.shape()
        if hf_arr.ndim == 2:
            tgt_rows, tgt_cols = tgt_shape[-2], tgt_shape[-1]
            r, c = hf_arr.shape
            if r == tgt_rows and c == tgt_cols:
                pass
            elif c == tgt_rows and r == tgt_cols:
                hf_arr = hf_arr.T
            else:
                hf_arr = _pad_and_resize(hf_arr, tgt_rows, tgt_cols)
        elif hf_arr.ndim == 1:
            tgt_dim = tgt_shape[-1]
            if hf_arr.shape[0] != tgt_dim:
                padded = np.zeros(tgt_dim, dtype=hf_arr.dtype)
                n = min(hf_arr.shape[0], tgt_dim)
                padded[:n] = hf_arr[:n]
                hf_arr = padded

        _assign_tensor(param, _to_bf16_4d(hf_arr))

    for hf_name, hf_arr in all_tensors.items():
        # ── Embedding ──
        if hf_name in ("model.embed_tokens.weight", "embed_tokens.weight"):
            if tie:
                # When tied, we skip embedding — lm_head.weight handles it
                continue
            param = get_param(f"{root_prefix}/model/embed_tokens/weight")
            _load_weight(param, hf_arr)
            continue

        # ── LM head ──
        if hf_name == "lm_head.weight":
            if tie:
                # Tied: load into embedding (which shares weight with lm_head usage)
                param = get_param(f"{root_prefix}/model/embed_tokens/weight")
            else:
                param = get_param(f"{root_prefix}/lm_head/weight")
            _load_weight(param, hf_arr)
            continue

        # ── Final RMSNorm ──
        if hf_name in ("model.norm.weight", "norm.weight"):
            param = get_param(f"{root_prefix}/model/norm/gamma")
            _load_weight(param, hf_arr)
            continue

        # ── Per-layer weights ──
        matched = False
        for i in range(config.num_hidden_layers):
            pfx = f"model.layers.{i}"
            pfx2 = f"layers.{i}"
            tp = f"{root_prefix}/model/layers/{i}"

            # input_layernorm
            if hf_name in (f"{pfx}.input_layernorm.weight", f"{pfx2}.input_layernorm.weight"):
                param = get_param(f"{tp}/input_layernorm/gamma")
                _load_weight(param, hf_arr)
                matched = True
                break

            # post_attention_layernorm
            if hf_name in (f"{pfx}.post_attention_layernorm.weight", f"{pfx2}.post_attention_layernorm.weight"):
                param = get_param(f"{tp}/post_attention_layernorm/gamma")
                _load_weight(param, hf_arr)
                matched = True
                break

            # q_proj weight
            if hf_name in (f"{pfx}.self_attn.q_proj.weight", f"{pfx2}.self_attn.q_proj.weight"):
                param = get_param(f"{tp}/self_attn/q_proj/weight")
                _load_weight(param, hf_arr, _unpermute_proj_rows, (num_heads,))
                matched = True
                break

            # q_proj bias
            if hf_name in (f"{pfx}.self_attn.q_proj.bias", f"{pfx2}.self_attn.q_proj.bias"):
                param = get_param(f"{tp}/self_attn/q_proj/bias")
                _load_weight(param, hf_arr, _unpermute_proj_rows, (num_heads,))
                matched = True
                break

            # k_proj weight
            if hf_name in (f"{pfx}.self_attn.k_proj.weight", f"{pfx2}.self_attn.k_proj.weight"):
                param = get_param(f"{tp}/self_attn/k_proj/weight")
                _load_weight(param, hf_arr, _unpermute_proj_rows, (num_kv_heads,))
                matched = True
                break

            # k_proj bias
            if hf_name in (f"{pfx}.self_attn.k_proj.bias", f"{pfx2}.self_attn.k_proj.bias"):
                param = get_param(f"{tp}/self_attn/k_proj/bias")
                _load_weight(param, hf_arr, _unpermute_proj_rows, (num_kv_heads,))
                matched = True
                break

            # v_proj weight
            if hf_name in (f"{pfx}.self_attn.v_proj.weight", f"{pfx2}.self_attn.v_proj.weight"):
                param = get_param(f"{tp}/self_attn/v_proj/weight")
                _load_weight(param, hf_arr)
                matched = True
                break

            # v_proj bias
            if hf_name in (f"{pfx}.self_attn.v_proj.bias", f"{pfx2}.self_attn.v_proj.bias"):
                param = get_param(f"{tp}/self_attn/v_proj/bias")
                _load_weight(param, hf_arr)
                matched = True
                break

            # o_proj weight
            if hf_name in (f"{pfx}.self_attn.o_proj.weight", f"{pfx2}.self_attn.o_proj.weight"):
                param = get_param(f"{tp}/self_attn/o_proj/weight")
                _load_weight(param, hf_arr)
                matched = True
                break

            # o_proj bias
            if hf_name in (f"{pfx}.self_attn.o_proj.bias", f"{pfx2}.self_attn.o_proj.bias"):
                param = get_param(f"{tp}/self_attn/o_proj/bias")
                _load_weight(param, hf_arr)
                matched = True
                break

            # q_norm weight
            if hf_name in (f"{pfx}.self_attn.q_norm.weight", f"{pfx2}.self_attn.q_norm.weight"):
                param = get_param(f"{tp}/self_attn/q_norm/gamma")
                _load_weight(param, hf_arr, _unpermute_norm_weights)
                matched = True
                break

            # k_norm weight
            if hf_name in (f"{pfx}.self_attn.k_norm.weight", f"{pfx2}.self_attn.k_norm.weight"):
                param = get_param(f"{tp}/self_attn/k_norm/gamma")
                _load_weight(param, hf_arr, _unpermute_norm_weights)
                matched = True
                break

            # gate_proj
            if hf_name in (f"{pfx}.mlp.gate_proj.weight", f"{pfx2}.mlp.gate_proj.weight"):
                param = get_param(f"{tp}/mlp/gate_proj/weight")
                _load_weight(param, hf_arr)
                matched = True
                break

            # up_proj
            if hf_name in (f"{pfx}.mlp.up_proj.weight", f"{pfx2}.mlp.up_proj.weight"):
                param = get_param(f"{tp}/mlp/up_proj/weight")
                _load_weight(param, hf_arr)
                matched = True
                break

            # down_proj
            if hf_name in (f"{pfx}.mlp.down_proj.weight", f"{pfx2}.mlp.down_proj.weight"):
                param = get_param(f"{tp}/mlp/down_proj/weight")
                _load_weight(param, hf_arr)
                matched = True
                break

        if not matched:
            ignored_hf.add(hf_name)

    # ── Report ──
    unused = [n for n in parameters if n not in used_params]
    if unused:
        print(f"Warning: {len(unused)} model parameters were NOT loaded from safetensors:")
        for n in unused:
            print(f"  - {n}")
    else:
        print(f"All {len(parameters)} parameters successfully loaded.")

    if ignored_hf:
        print(f"Note: {len(ignored_hf)} HF tensors were ignored (no mapping):")
        for n in sorted(ignored_hf):
            print(f"  - {n}")
