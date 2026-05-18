# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared Hugging Face download / dequantization helpers for the Devstral-2-123B tests.

The Devstral-2-123B checkpoint on the Hub is stored in FineGrained FP8 with per-block weight
scales. The TT path expects ``bf16`` host tensors. Loading the full checkpoint is out of scope
for unit tests, so each test downloads only the safetensor shards it needs via the index file
and dequantizes FP8 weights with their ``weight_scale_inv`` tensors when present.

If Hub access fails (gated repo, offline CI, etc.) the helpers raise; tests are expected to
``pytest.skip`` on that signal.
"""

from __future__ import annotations

import json
import os

import torch
from transformers import AutoConfig
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

DEVSTRAL2_LARGE_REPO_ID = "mistralai/Devstral-2-123B-Instruct-2512"

_FP8_DTYPES = tuple(
    dt for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz") if (dt := getattr(torch, name, None)) is not None
)


def load_text_config() -> Ministral3Config:
    """Download (cached) the HF config and return the inner ``Ministral3Config``.

    The Devstral-2 repo may publish a multimodal wrapper config; the underlying text stack is
    ``Ministral3Config`` either way. We always return the text config so the rest of the test
    can treat it uniformly.
    """
    hf_cfg = AutoConfig.from_pretrained(
        DEVSTRAL2_LARGE_REPO_ID,
        trust_remote_code=True,
        local_files_only=os.getenv("CI") == "true",
    )
    text = getattr(hf_cfg, "text_config", None) or hf_cfg
    if not isinstance(text, Ministral3Config):
        raise TypeError(f"Expected Ministral3Config, got {type(text)!r}")
    return text


def dequantize_fp8_weight(weight: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Dequant matching HF ``Fp8Dequantize._dequantize_one`` (scalar or per-block ``weight_scale_inv``)."""
    if scale_inv.numel() == 1:
        out_dtype = (
            scale_inv.dtype if scale_inv.dtype.is_floating_point and scale_inv.element_size() >= 2 else torch.bfloat16
        )
        return (weight.to(torch.float32) * scale_inv.to(torch.float32)).to(out_dtype)

    quantized_fp32 = weight.to(torch.float32)
    rows, cols = quantized_fp32.shape[-2:]
    scale_rows, scale_cols = scale_inv.shape[-2:]
    if rows % scale_rows or cols % scale_cols:
        raise ValueError(f"Weight shape ({rows}, {cols}) not divisible by scale grid ({scale_rows}, {scale_cols}).")
    block_m = rows // scale_rows
    block_n = cols // scale_cols
    out_dtype = (
        scale_inv.dtype if scale_inv.dtype.is_floating_point and scale_inv.element_size() >= 2 else torch.bfloat16
    )
    original_shape = quantized_fp32.shape
    q = quantized_fp32.reshape(-1, scale_rows, block_m, scale_cols, block_n)
    s = scale_inv.to(torch.float32).reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)
    return (q * s).to(out_dtype).reshape(original_shape)


def to_bf16_host_if_fp8(t: torch.Tensor, scale_inv: torch.Tensor | None = None) -> torch.Tensor:
    """FP8 → bf16 on host. Uses block scales when ``scale_inv`` is provided."""
    if scale_inv is not None and _FP8_DTYPES and t.dtype in _FP8_DTYPES:
        return dequantize_fp8_weight(t, scale_inv).to(torch.bfloat16)
    if _FP8_DTYPES and t.dtype in _FP8_DTYPES:
        return t.to(torch.bfloat16)
    return t


def load_hf_tensors_for_keys(keys: list[str]) -> dict[str, torch.Tensor]:
    """Download shards for ``keys`` and return host tensors (weights dequantized to bf16 when FP8)."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import safe_open as safetensors_safe_open

    index_path = hf_hub_download(
        repo_id=DEVSTRAL2_LARGE_REPO_ID,
        filename="model.safetensors.index.json",
        local_files_only=os.getenv("CI") == "true",
    )
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    fetch_keys: set[str] = set(keys)
    for key in keys:
        if key.endswith(".weight"):
            scale_key = key[: -len(".weight")] + ".weight_scale_inv"
            if scale_key in weight_map:
                fetch_keys.add(scale_key)

    raw: dict[str, torch.Tensor] = {}
    for key in fetch_keys:
        if key not in weight_map:
            if key in keys:
                raise KeyError(f"Key {key!r} not in weight_map for {DEVSTRAL2_LARGE_REPO_ID}")
            continue
        shard_path = hf_hub_download(
            repo_id=DEVSTRAL2_LARGE_REPO_ID,
            filename=weight_map[key],
            local_files_only=os.getenv("CI") == "true",
        )
        with safetensors_safe_open(shard_path, framework="pt", device="cpu") as sf:
            if key not in sf.keys():
                raise KeyError(f"Key {key!r} missing from shard {weight_map[key]}")
            raw[key] = sf.get_tensor(key).clone()

    out: dict[str, torch.Tensor] = {}
    for key in keys:
        t = raw[key]
        if key.endswith(".weight"):
            scale_key = key[: -len(".weight")] + ".weight_scale_inv"
            scale = raw.get(scale_key)
            out[key] = to_bf16_host_if_fp8(t, scale).to(torch.bfloat16)
        else:
            out[key] = t
    return out
