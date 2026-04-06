# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Load HuggingFace Llama safetensors weights into a Python Llama model.

Port of the C++ loader in tt-train/sources/ttml/models/llama.cpp.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import ml_dtypes

import ttnn
import ttml

from . import LlamaConfig


def _unpermute_proj_rows(w: np.ndarray, n_heads: int) -> np.ndarray:
    """Reorder Q/K projection rows from HF grouped layout to interleaved pairs.

    HF stores rows as [first_half, second_half] per head.
    TTML's RoPE expects interleaved: [0, half, 1, half+1, ...].
    """
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


def _pad_and_resize(arr: np.ndarray, tgt_rows: int, tgt_cols: int) -> np.ndarray:
    """Pad or crop a 2D array to (tgt_rows, tgt_cols).

    Extra rows/cols are filled with small random values (N(0, 0.02))
    to avoid dead neurons, matching C++ behavior.
    """
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


def load_from_safetensors(
    model: ttml.modules.AbstractModuleBase,
    safetensors_path: str | os.PathLike,
    config: LlamaConfig,
) -> None:
    """Load HuggingFace Llama .safetensors weights into a Python Llama model.

    Handles:
    - Q/K weight unpermutation for TTML's interleaved RoPE
    - K/V concatenation into kv_linear
    - Embedding padding when model vocab_size differs from HF
    - Shape conversion from HF 2D to TTML 4D

    Args:
        model: A Python Llama model instance.
        safetensors_path: Path to directory containing .safetensors file(s).
        config: The LlamaConfig used to build the model.
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

    k_staged: Dict[int, np.ndarray] = {}
    v_staged: Dict[int, np.ndarray] = {}

    def get_param(name: str):
        if name not in parameters:
            raise RuntimeError(f"Parameter {name} not found in model")
        used_params.add(name)
        return parameters[name]

    def try_combine_kv(layer_idx: int) -> None:
        if layer_idx not in k_staged or layer_idx not in v_staged:
            return

        param_name = f"Llama/blocks/{layer_idx}/attention/kv_linear/weight"
        param = get_param(param_name)
        tgt_shape = param.shape()
        tgt_rows, tgt_cols = tgt_shape[-2], tgt_shape[-1]

        k = k_staged.pop(layer_idx)
        v = v_staged.pop(layer_idx)

        combined = np.concatenate([k, v], axis=0)
        cr, cc = combined.shape
        if cr != tgt_rows or cc != tgt_cols:
            if cc == tgt_rows and cr == tgt_cols:
                combined = combined.T
            else:
                raise RuntimeError(
                    f"KV concat shape mismatch at layer {layer_idx}: "
                    f"combined=({cr}x{cc}), target=({tgt_rows}x{tgt_cols})"
                )

        _assign_tensor(param, _to_bf16_4d(combined))
        print(f"  Combined k_proj + v_proj -> kv_linear for layer {layer_idx}")

    weight_tying = config.weight_tying

    for hf_name, hf_arr in all_tensors.items():
        hf_arr = hf_arr.astype(np.float32)
        print(f"Loading tensor: {hf_name}, shape={hf_arr.shape}, dtype={hf_arr.dtype}")

        # ── Embedding ──
        if hf_name in (
            "model.embed_tokens.weight",
            "embed_tokens.weight",
            "transformer.wte.weight",
            "wte.weight",
            "model.wte.weight",
        ):
            from ttml.models import WeightTyingType

            emb_param_name = "Llama/fc/weight" if weight_tying == WeightTyingType.Enabled else "Llama/tok_emb/weight"
            param = get_param(emb_param_name)
            tgt = param.shape()
            resized = _pad_and_resize(hf_arr, tgt[-2], tgt[-1])
            _assign_tensor(param, _to_bf16_4d(resized))
            continue

        # ── LM head ──
        if hf_name == "lm_head.weight":
            from ttml.models import WeightTyingType

            if weight_tying == WeightTyingType.Disabled:
                param = get_param("Llama/fc/weight")
                tgt = param.shape()
                resized = _pad_and_resize(hf_arr, tgt[-2], tgt[-1])
                _assign_tensor(param, _to_bf16_4d(resized))
            continue

        # ── Final RMSNorm ──
        if hf_name in ("model.norm.weight", "norm.weight"):
            param = get_param("Llama/ln_fc/gamma")
            _assign_tensor(param, _to_bf16_4d(hf_arr))
            continue

        # ── Per-layer weights ──
        matched = False
        for i in range(config.num_hidden_layers):
            pfx = f"model.layers.{i}"
            pfx2 = f"layers.{i}"

            # input_layernorm
            if hf_name in (
                f"{pfx}.input_layernorm.weight",
                f"{pfx2}.input_layernorm.weight",
            ):
                param = get_param(f"Llama/blocks/{i}/attention_norm/gamma")
                _assign_tensor(param, _to_bf16_4d(hf_arr))
                matched = True
                break

            # post_attention_layernorm
            if hf_name in (
                f"{pfx}.post_attention_layernorm.weight",
                f"{pfx2}.post_attention_layernorm.weight",
            ):
                param = get_param(f"Llama/blocks/{i}/mlp_norm/gamma")
                _assign_tensor(param, _to_bf16_4d(hf_arr))
                matched = True
                break

            # q_proj
            if hf_name in (
                f"{pfx}.self_attn.q_proj.weight",
                f"{pfx2}.self_attn.q_proj.weight",
            ):
                w = _unpermute_proj_rows(hf_arr, num_heads)
                param = get_param(f"Llama/blocks/{i}/attention/q_linear/weight")
                tgt = param.shape()
                tr, tc = tgt[-2], tgt[-1]
                r, c = w.shape
                if r == tr and c == tc:
                    pass
                elif c == tr and r == tc:
                    w = np.ascontiguousarray(w.T)
                else:
                    raise RuntimeError(f"q_proj shape mismatch layer {i}: ({r}x{c}) vs ({tr}x{tc})")
                _assign_tensor(param, _to_bf16_4d(w))
                matched = True
                break

            # k_proj (stage for kv concat)
            if hf_name in (
                f"{pfx}.self_attn.k_proj.weight",
                f"{pfx2}.self_attn.k_proj.weight",
            ):
                k_staged[i] = _unpermute_proj_rows(hf_arr, num_kv_heads)
                try_combine_kv(i)
                matched = True
                break

            # v_proj (stage for kv concat, no unpermute)
            if hf_name in (
                f"{pfx}.self_attn.v_proj.weight",
                f"{pfx2}.self_attn.v_proj.weight",
            ):
                v_staged[i] = hf_arr
                try_combine_kv(i)
                matched = True
                break

            # o_proj
            if hf_name in (
                f"{pfx}.self_attn.o_proj.weight",
                f"{pfx2}.self_attn.o_proj.weight",
            ):
                param = get_param(f"Llama/blocks/{i}/attention/out_linear/weight")
                tgt = param.shape()
                tr, tc = tgt[-2], tgt[-1]
                r, c = hf_arr.shape
                w = hf_arr
                if r == tr and c == tc:
                    pass
                elif c == tr and r == tc:
                    w = np.ascontiguousarray(w.T)
                else:
                    raise RuntimeError(f"o_proj shape mismatch layer {i}: ({r}x{c}) vs ({tr}x{tc})")
                _assign_tensor(param, _to_bf16_4d(w))
                matched = True
                break

            # gate_proj -> w1
            if hf_name in (
                f"{pfx}.mlp.gate_proj.weight",
                f"{pfx2}.mlp.gate_proj.weight",
            ):
                param = get_param(f"Llama/blocks/{i}/mlp/w1/weight")
                tgt = param.shape()
                tr, tc = tgt[-2], tgt[-1]
                r, c = hf_arr.shape
                w = hf_arr
                if r == tr and c == tc:
                    pass
                elif c == tr and r == tc:
                    w = np.ascontiguousarray(w.T)
                else:
                    raise RuntimeError(f"gate_proj shape mismatch layer {i}: ({r}x{c}) vs ({tr}x{tc})")
                _assign_tensor(param, _to_bf16_4d(w))
                matched = True
                break

            # up_proj -> w3
            if hf_name in (
                f"{pfx}.mlp.up_proj.weight",
                f"{pfx2}.mlp.up_proj.weight",
            ):
                param = get_param(f"Llama/blocks/{i}/mlp/w3/weight")
                tgt = param.shape()
                tr, tc = tgt[-2], tgt[-1]
                r, c = hf_arr.shape
                w = hf_arr
                if r == tr and c == tc:
                    pass
                elif c == tr and r == tc:
                    w = np.ascontiguousarray(w.T)
                else:
                    raise RuntimeError(f"up_proj shape mismatch layer {i}: ({r}x{c}) vs ({tr}x{tc})")
                _assign_tensor(param, _to_bf16_4d(w))
                matched = True
                break

            # down_proj -> w2
            if hf_name in (
                f"{pfx}.mlp.down_proj.weight",
                f"{pfx2}.mlp.down_proj.weight",
            ):
                param = get_param(f"Llama/blocks/{i}/mlp/w2/weight")
                tgt = param.shape()
                tr, tc = tgt[-2], tgt[-1]
                r, c = hf_arr.shape
                w = hf_arr
                if r == tr and c == tc:
                    pass
                elif c == tr and r == tc:
                    w = np.ascontiguousarray(w.T)
                else:
                    raise RuntimeError(f"down_proj shape mismatch layer {i}: ({r}x{c}) vs ({tr}x{tc})")
                _assign_tensor(param, _to_bf16_4d(w))
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
