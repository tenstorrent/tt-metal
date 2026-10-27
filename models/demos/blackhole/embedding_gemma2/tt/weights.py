# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Setup-time weight helpers: checkpoint loading and the ttnn weight cache.

Nothing in this file runs inside a forward pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import ttnn

TEXT_PREFIX = "language_model."


def resolve_snapshot(model_id_or_path: str) -> Path:
    """Local directory holding config.json + model.safetensors for `model_id_or_path`."""
    p = Path(model_id_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id_or_path, allow_patterns=["*.json", "*.safetensors", "tokenizer*"]))


def load_text_config_dict(snapshot: Path) -> dict:
    with open(snapshot / "config.json") as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def load_text_state_dict(snapshot: Path) -> dict[str, torch.Tensor]:
    """Text-tower tensors with the `language_model.` prefix stripped (bf16, as stored)."""
    from safetensors import safe_open

    out = {}
    for f in sorted(snapshot.glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            for k in sf.keys():
                if k.startswith(TEXT_PREFIX):
                    out[k[len(TEXT_PREFIX) :]] = sf.get_tensor(k)
    if not out:
        raise FileNotFoundError(f"no `{TEXT_PREFIX}*` tensors under {snapshot}")
    return out


def substate(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    prefix = prefix if prefix.endswith(".") else prefix + "."
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def cache_name(cache_dir: Path | None, name: str) -> str | None:
    if cache_dir is None:
        return None
    return str(Path(cache_dir) / name)


def as_weight(
    tensor: torch.Tensor,
    device,
    *,
    cache_dir: Path | None,
    name: str,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Upload one weight, reading/writing `<cache_dir>/<name>` when a cache dir is given."""
    return ttnn.as_tensor(
        tensor.contiguous(),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
        cache_file_name=cache_name(cache_dir, name),
    )


def linear_weight(w: torch.Tensor) -> torch.Tensor:
    """HF nn.Linear weight [out, in] -> ttnn.linear operand [1, 1, in, out]."""
    return w.transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0)


def norm_weight(w: torch.Tensor) -> torch.Tensor:
    """RMSNorm weight [dim] -> [1, 1, dim // 32, 32] ROW_MAJOR operand for ttnn.rms_norm."""
    return w.reshape(1, 1, -1, ttnn.TILE_SIZE)
