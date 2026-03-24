# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Load HuggingFace GPT-2 safetensors weights into a Python NanoGPT model.

Port of the C++ loader in tt-train/sources/ttml/models/gpt2.cpp.
GPT-2 uses Conv1D layers whose weights are stored transposed relative to
standard linear layers, so attn and mlp weights need a transpose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import ml_dtypes

import ttnn
import ttml


def _pad_rows(arr: np.ndarray, tgt_rows: int) -> np.ndarray:
    """Pad embedding rows to match the model's (tile-aligned) vocab size."""
    src_rows, cols = arr.shape
    if src_rows >= tgt_rows:
        return arr[:tgt_rows]
    out = np.zeros((tgt_rows, cols), dtype=arr.dtype)
    out[:src_rows] = arr
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


def load_gpt2_from_safetensors(
    model: ttml.modules.AbstractModuleBase,
    safetensors_path: str,
    config: "NanoGPTConfig",
) -> None:
    """Load HuggingFace GPT-2 .safetensors weights into a Python NanoGPT model.

    Handles:
    - Conv1D weight transposition (GPT-2 stores linear weights transposed)
    - Embedding padding when model vocab_size differs from HF
    - Weight tying (wte.weight shared between embedding and output projection)

    Args:
        model: A Python NanoGPT model instance.
        safetensors_path: Path to directory containing .safetensors file(s).
        config: The NanoGPTConfig used to build the model.
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

    def get_param(name: str):
        if name not in parameters:
            raise RuntimeError(f"Parameter {name} not found in model. Available: {list(parameters.keys())}")
        used_params.add(name)
        return parameters[name]

    weight_tying = config.weight_tying

    for hf_name, hf_arr in all_tensors.items():
        hf_arr = hf_arr.astype(np.float32)

        # wte.weight → embedding / output projection
        if hf_name == "wte.weight":
            from ttml.models import WeightTyingType

            param_name = "NanoGPT/fc/weight" if weight_tying == WeightTyingType.Enabled else "NanoGPT/tok_emb/weight"
            param = get_param(param_name)
            tgt_rows = param.shape()[-2]
            resized = _pad_rows(hf_arr, tgt_rows)
            _assign_tensor(param, _to_bf16_4d(resized))
            continue

        # wpe.weight → positional embedding
        if hf_name == "wpe.weight":
            param = get_param("NanoGPT/pos_emb/weight")
            _assign_tensor(param, _to_bf16_4d(hf_arr))
            continue

        # ln_f.weight / ln_f.bias → final layer norm
        if hf_name == "ln_f.weight":
            param = get_param("NanoGPT/ln_f_gamma")
            _assign_tensor(param, _to_bf16_4d(hf_arr))
            continue
        if hf_name == "ln_f.bias":
            param = get_param("NanoGPT/ln_f_beta")
            _assign_tensor(param, _to_bf16_4d(hf_arr))
            continue

        # Per-block weights
        matched = False
        for i in range(config.n_layer):
            pfx = f"h.{i}"

            if hf_name == f"{pfx}.ln_1.weight":
                _assign_tensor(get_param(f"NanoGPT/blocks/{i}/ln1/gamma"), _to_bf16_4d(hf_arr))
                matched = True
                break
            if hf_name == f"{pfx}.ln_1.bias":
                _assign_tensor(get_param(f"NanoGPT/blocks/{i}/ln1/beta"), _to_bf16_4d(hf_arr))
                matched = True
                break
            if hf_name == f"{pfx}.ln_2.weight":
                _assign_tensor(get_param(f"NanoGPT/blocks/{i}/ln2/gamma"), _to_bf16_4d(hf_arr))
                matched = True
                break
            if hf_name == f"{pfx}.ln_2.bias":
                _assign_tensor(get_param(f"NanoGPT/blocks/{i}/ln2/beta"), _to_bf16_4d(hf_arr))
                matched = True
                break

            # GPT-2 Conv1D weights are stored as (in_features, out_features),
            # standard linear expects (out_features, in_features) → transpose.
            if hf_name == f"{pfx}.attn.c_attn.weight":
                _assign_tensor(
                    get_param(f"NanoGPT/blocks/{i}/attention/qkv_linear/weight"),
                    _to_bf16_4d(hf_arr.T),
                )
                matched = True
                break
            if hf_name == f"{pfx}.attn.c_attn.bias":
                _assign_tensor(get_param(f"NanoGPT/blocks/{i}/attention/qkv_linear/bias"), _to_bf16_4d(hf_arr))
                matched = True
                break
            if hf_name == f"{pfx}.attn.c_proj.weight":
                _assign_tensor(
                    get_param(f"NanoGPT/blocks/{i}/attention/out_linear/weight"),
                    _to_bf16_4d(hf_arr.T),
                )
                matched = True
                break
            if hf_name == f"{pfx}.attn.c_proj.bias":
                _assign_tensor(get_param(f"NanoGPT/blocks/{i}/attention/out_linear/bias"), _to_bf16_4d(hf_arr))
                matched = True
                break
            if hf_name == f"{pfx}.mlp.c_fc.weight":
                _assign_tensor(
                    get_param(f"NanoGPT/blocks/{i}/mlp/fc1/weight"),
                    _to_bf16_4d(hf_arr.T),
                )
                matched = True
                break
            if hf_name == f"{pfx}.mlp.c_fc.bias":
                _assign_tensor(get_param(f"NanoGPT/blocks/{i}/mlp/fc1/bias"), _to_bf16_4d(hf_arr))
                matched = True
                break
            if hf_name == f"{pfx}.mlp.c_proj.weight":
                _assign_tensor(
                    get_param(f"NanoGPT/blocks/{i}/mlp/fc2/weight"),
                    _to_bf16_4d(hf_arr.T),
                )
                matched = True
                break
            if hf_name == f"{pfx}.mlp.c_proj.bias":
                _assign_tensor(get_param(f"NanoGPT/blocks/{i}/mlp/fc2/bias"), _to_bf16_4d(hf_arr))
                matched = True
                break

        if not matched:
            ignored_hf.add(hf_name)

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
