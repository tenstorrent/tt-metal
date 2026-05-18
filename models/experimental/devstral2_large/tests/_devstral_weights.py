# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared Hugging Face download / dequantization helpers for the Devstral-2-123B tests.

The Devstral-2-123B checkpoint on the Hub is stored in FineGrained FP8 with per-block weight
scales. The TT path expects ``bf16`` host tensors. Loading the full checkpoint is out of scope
for unit tests, so each test downloads only the safetensor shards it needs via the index file
and casts FP8 dtypes directly to ``bf16`` — both the HF reference module *and* the TT module
see the same crude-dequantized weights, so the PCC comparison still measures TT-vs-HF parity.

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


def load_hf_tensors_for_keys(keys: list[str]) -> dict[str, torch.Tensor]:
    """Download only the safetensor shards that contain ``keys`` and return them on host."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import safe_open as safetensors_safe_open

    index_path = hf_hub_download(
        repo_id=DEVSTRAL2_LARGE_REPO_ID,
        filename="model.safetensors.index.json",
        local_files_only=os.getenv("CI") == "true",
    )
    with open(index_path, encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    out: dict[str, torch.Tensor] = {}
    for key in keys:
        if key not in weight_map:
            raise KeyError(f"Key {key!r} not in weight_map for {DEVSTRAL2_LARGE_REPO_ID}")
        shard_path = hf_hub_download(
            repo_id=DEVSTRAL2_LARGE_REPO_ID,
            filename=weight_map[key],
            local_files_only=os.getenv("CI") == "true",
        )
        with safetensors_safe_open(shard_path, framework="pt", device="cpu") as sf:
            if key not in sf.keys():
                raise KeyError(f"Key {key!r} missing from shard {weight_map[key]}")
            out[key] = sf.get_tensor(key).clone()
    return out


def to_bf16_host_if_fp8(t: torch.Tensor) -> torch.Tensor:
    """Crude FP8 → bf16 cast (no scale application).

    HF stores Devstral-2-123B linear weights in ``float8_e4m3fn`` with companion per-block scale
    tensors. Real inference dequantizes via those scales; for parity testing we cast directly
    (both HF ref and TT module see the same numerically-degraded weights, so PCC still
    isolates TT-vs-HF disagreement from dequantization fidelity).
    """
    fp8_dtypes = tuple(
        dt
        for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz")
        if (dt := getattr(torch, name, None)) is not None
    )
    if fp8_dtypes and t.dtype in fp8_dtypes:
        return t.to(torch.bfloat16)
    return t
