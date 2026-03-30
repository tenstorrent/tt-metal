# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Load individual tensors from a Hugging Face **local snapshot** (no Hub calls).

Expects a directory containing ``config.json`` and either ``model.safetensors`` or
``model.safetensors.index.json`` + shard files (standard HF layout).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

try:
    from safetensors import safe_open
except ImportError as exc:
    safe_open = None  # type: ignore[misc, assignment]
    _SAFEIMPORT_ERR = exc
else:
    _SAFEIMPORT_ERR = None


def read_snapshot_config(snapshot_dir: Path) -> dict[str, Any]:
    p = Path(snapshot_dir) / "config.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing config.json under {snapshot_dir}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_tensor_from_snapshot(snapshot_dir: Path, tensor_name: str) -> torch.Tensor:
    """Mmap-read a single tensor from sharded or single-file safetensors in ``snapshot_dir``."""
    if safe_open is None:
        raise ImportError("safetensors is required for local snapshot weight load") from _SAFEIMPORT_ERR
    root = Path(snapshot_dir)
    index_path = root / "model.safetensors.index.json"
    single_path = root / "model.safetensors"
    if index_path.is_file():
        with open(index_path, encoding="utf-8") as f:
            idx = json.load(f)
        weight_map: dict[str, str] = idx["weight_map"]
        if tensor_name not in weight_map:
            raise KeyError(f"Tensor {tensor_name!r} not in {index_path} (check num_hidden_layers / naming)")
        shard = root / weight_map[tensor_name]
        if not shard.is_file():
            raise FileNotFoundError(f"Shard missing: {shard}")
        with safe_open(str(shard), framework="pt", device="cpu") as sf:
            if tensor_name not in sf.keys():
                raise KeyError(f"{tensor_name} not in shard {shard}")
            return sf.get_tensor(tensor_name).float().clone()
    if single_path.is_file():
        with safe_open(str(single_path), framework="pt", device="cpu") as sf:
            if tensor_name not in sf.keys():
                raise KeyError(f"{tensor_name} not in {single_path}")
            return sf.get_tensor(tensor_name).float().clone()
    raise FileNotFoundError(
        f"No model.safetensors.index.json or model.safetensors in {root} "
        "(local DeepSeek snapshot must use safetensors format)."
    )


def verify_record_dims_vs_snapshot(
    *,
    record_hidden_size: int,
    record_max_token_id: int,
    snapshot_dir: Path,
) -> None:
    """Raise ``ValueError`` if the record does not match ``config.json`` in the snapshot."""
    cfg = read_snapshot_config(snapshot_dir)
    h = int(cfg.get("hidden_size", 0))
    v = int(cfg.get("vocab_size", 0))
    if h <= 0 or v <= 0:
        raise ValueError(f"Invalid hidden_size/vocab_size in {snapshot_dir}/config.json")
    if record_hidden_size != h:
        raise ValueError(
            f"Record hidden_size {record_hidden_size} != snapshot config hidden_size {h}"
        )
    if record_max_token_id >= v:
        raise ValueError(
            f"Record max token id {record_max_token_id} >= snapshot vocab_size {v}"
        )


# Keys for a **small** safetensors file used with NextN-only MTP draft (not a full checkpoint).
_EMBED_KEYS_ORDER = ("embed_tokens.weight", "model.embed_tokens.weight")
_HEAD_KEYS_ORDER = (
    "shared_head.head.weight",
    "model.layers.0.shared_head.head.weight",
    "model.layers.61.shared_head.head.weight",  # R1-class MTP index when num_hidden_layers==61
)


def load_nextn_mtp_auxiliary_safetensors(aux_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ``(embed_tokens, shared_head_head)`` from a single small ``.safetensors`` file.

    Use when ``nextn_layer_parameters.safetensors`` does not already include embed + lm head:
    supply a small file with the tensors listed in ``_EMBED_KEYS_ORDER`` / ``_HEAD_KEYS_ORDER``.
    """
    if safe_open is None:
        raise ImportError("safetensors is required") from _SAFEIMPORT_ERR
    p = Path(aux_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Auxiliary safetensors not found: {p}")
    with safe_open(str(p), framework="pt", device="cpu") as sf:
        keys = list(sf.keys())
        embed = None
        for k in _EMBED_KEYS_ORDER:
            if k in keys:
                embed = sf.get_tensor(k).float().clone()
                break
        head = None
        for k in _HEAD_KEYS_ORDER:
            if k in keys:
                head = sf.get_tensor(k).float().clone()
                break
        if embed is None:
            raise KeyError(
                f"No embed tensor in {p}; expected one of {_EMBED_KEYS_ORDER}. Keys present: {keys}"
            )
        if head is None:
            raise KeyError(
                f"No shared head tensor in {p}; expected one of {_HEAD_KEYS_ORDER}. Keys present: {keys}"
            )
    return embed, head


def verify_record_dims_vs_config_dir(
    *,
    record_hidden_size: int,
    record_max_token_id: int,
    config_dir: Path,
) -> None:
    """Same as ``verify_record_dims_vs_snapshot`` but name clarifies any ``config.json`` dir (e.g. NextN repo)."""
    verify_record_dims_vs_snapshot(
        record_hidden_size=record_hidden_size,
        record_max_token_id=record_max_token_id,
        snapshot_dir=Path(config_dir),
    )
