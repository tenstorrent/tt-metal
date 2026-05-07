# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read decoder-layer tensors from a Mistral-3 style Hugging Face ``safetensors`` snapshot.

Checkpoint keys follow::

    language_model.model.layers.<L>.<submodule>.<param>

Some snapshots use ``model.layers.<L>.`` instead of ``language_model.model.layers.<L>.``.

Used by :mod:`models.demos.mistral_small_4_119B.tt.mla.mla1d` to slice ``self_attn.*``
and by other layer loaders (MoE, norms, etc.).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


def _maybe_cast_loaded_weight(tensor: torch.Tensor) -> torch.Tensor:
    for dt_name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        dt = getattr(torch, dt_name, None)
        if dt is not None and tensor.dtype == dt:
            return tensor.to(torch.float32).to(torch.bfloat16)

    if tensor.dtype == torch.uint8:
        raise RuntimeError(
            "Checkpoint tensor is UINT8 (blocked FP8); use transformers ``from_pretrained`` "
            "with FP8 dequantization or bf16 shards."
        )

    if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return tensor

    return tensor.to(torch.bfloat16)


def _load_index(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}. Expected a Hugging Face sharded layout next to shard files.")
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def _pick_layer_prefix(weight_map: dict[str, str], layer_idx: int) -> str:
    candidates = (
        f"language_model.model.layers.{layer_idx}.",
        f"model.layers.{layer_idx}.",
    )
    for prefix in candidates:
        probe = prefix + "self_attn."
        if any(k.startswith(probe) for k in weight_map):
            return prefix
    raise KeyError(
        f"No decoder layer prefix found for layer {layer_idx} "
        f"(tried language_model.model.layers.* and model.layers.*)."
    )


def _shard_groups(full_keys: list[str], weight_map: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for fk in full_keys:
        if fk not in weight_map:
            raise KeyError(f"Tensor missing from checkpoint index: {fk}")
        out[weight_map[fk]].append(fk)
    return dict(out)


def _read_shard_tensors(shard_path: Path, keys: list[str]) -> dict[str, torch.Tensor]:
    if not shard_path.is_file():
        raise FileNotFoundError(f"Shard file missing: {shard_path}")
    out: dict[str, torch.Tensor] = {}
    with safe_open(shard_path, framework="pt", device="cpu") as sf:
        keys_avail = set(sf.keys())
        for k in keys:
            if k not in keys_avail:
                raise KeyError(f"Key {k} not present in {shard_path}")
            out[k] = _maybe_cast_loaded_weight(sf.get_tensor(k))
    return out


def read_decoder_layer_tensors_from_sharded_checkpoint(
    model_dir: str | Path,
    layer_idx: int,
) -> tuple[dict[str, torch.Tensor], str]:
    """Load **all** tensors for decoder ``layer_idx`` with keys relative to the layer root.

    Example relative keys::

        self_attn.q_a_proj.weight
        mlp.gate.weight

    Returns:
        ``(state_dict_relative_to_layer, checkpoint_prefix_string)`` where the prefix is the
        stripped checkpoint prefix (e.g. ``language_model.model.layers.0.``).
    """
    model_dir = Path(model_dir).resolve()
    weight_map = _load_index(model_dir)
    prefix = _pick_layer_prefix(weight_map, layer_idx)

    full_keys = [k for k in weight_map if k.startswith(prefix)]
    if not full_keys:
        raise ValueError(f"No tensors indexed for layer prefix {prefix!r}")

    shard_groups = _shard_groups(full_keys, weight_map)
    relative: dict[str, torch.Tensor] = {}
    plen = len(prefix)
    for shard_name, keys in shard_groups.items():
        tensors = _read_shard_tensors(model_dir / shard_name, keys)
        for fk, tensor in tensors.items():
            relative[fk[plen:]] = tensor

    return relative, prefix


__all__ = ["read_decoder_layer_tensors_from_sharded_checkpoint"]
