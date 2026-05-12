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


def read_decoder_layer_weight(model_dir: str | Path, layer_idx: int, relative_key: str) -> torch.Tensor:
    """Load one tensor from a decoder layer in a sharded ``safetensors`` tree.

    ``relative_key`` is the key returned by :func:`read_decoder_layer_tensors_from_sharded_checkpoint`,
    e.g. ``\"input_layernorm.weight\"`` or ``\"self_attn.q_a_layernorm.weight\"``.
    """
    tensors, _ = read_decoder_layer_tensors_from_sharded_checkpoint(model_dir, layer_idx)
    if relative_key not in tensors:
        sample = sorted(tensors.keys())[:20]
        raise KeyError(f"Missing {relative_key!r} in layer {layer_idx}; sample keys: {sample}")
    return tensors[relative_key]


def _pick_lm_head_checkpoint_keys(weight_map: dict[str, str]) -> tuple[str, str]:
    """Resolve full checkpoint keys for final RMSNorm and vocab projection (or tied embeddings)."""
    norm_key = None
    for cand in (
        "language_model.model.norm.weight",
        "model.language_model.model.norm.weight",
    ):
        if cand in weight_map:
            norm_key = cand
            break
    if norm_key is None:
        matches = [k for k in weight_map if k.endswith(".model.norm.weight") and ".layers." not in k]
        if not matches:
            raise KeyError("Could not find final RMSNorm (model.norm) weight in checkpoint index")
        norm_key = sorted(matches, key=lambda s: (len(s), s))[0]

    lm_key = None
    for cand in (
        "language_model.lm_head.weight",
        "model.lm_head.weight",
        "lm_head.weight",
    ):
        if cand in weight_map:
            lm_key = cand
            break
    if lm_key is None:
        for cand in (
            "language_model.model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
        ):
            if cand in weight_map:
                lm_key = cand
                break
    if lm_key is None:
        raise KeyError(
            "Could not find lm_head.weight or tied embed_tokens.weight in checkpoint index "
            "(extend _pick_lm_head_checkpoint_keys if your snapshot uses different names)."
        )

    return norm_key, lm_key


def read_lm_head_checkpoint_tensors(model_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load final norm + LM projection tensors from a Hugging Face sharded ``safetensors`` tree.

    Returns keys expected by :meth:`Mistral4LMHead.convert_weights`:

    - ``norm.weight`` — shape ``[hidden_size]``
    - ``lm_head.weight`` — shape ``[vocab_size, hidden_size]`` (may be read from tied ``embed_tokens``)

    Only the shards that contain these tensors are opened (no full-model load).
    """
    model_dir = Path(model_dir).resolve()
    weight_map = _load_index(model_dir)
    norm_key, lm_key = _pick_lm_head_checkpoint_keys(weight_map)
    shard_groups = _shard_groups([norm_key, lm_key], weight_map)
    raw: dict[str, torch.Tensor] = {}
    for shard_name, keys in shard_groups.items():
        raw.update(_read_shard_tensors(model_dir / shard_name, keys))

    return {
        "norm.weight": raw[norm_key],
        "lm_head.weight": raw[lm_key],
    }


def _pick_embed_tokens_checkpoint_key(weight_map: dict[str, str]) -> str:
    for cand in (
        "language_model.model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
    ):
        if cand in weight_map:
            return cand
    matches = [
        k
        for k in weight_map
        if k.endswith("embed_tokens.weight") and "vision" not in k.lower() and "image" not in k.lower()
    ]
    if not matches:
        raise KeyError(
            "Could not find embed_tokens.weight in checkpoint index "
            "(extend _pick_embed_tokens_checkpoint_key for your snapshot layout)."
        )
    return sorted(matches, key=lambda s: (len(s), s))[0]


def read_embed_tokens_checkpoint_tensor(model_dir: str | Path) -> torch.Tensor:
    """Load ``embed_tokens.weight`` ``[vocab_size, hidden_size]`` from sharded ``safetensors`` (CPU bf16).

    Opens only the shard(s) that contain this tensor.
    """
    model_dir = Path(model_dir).resolve()
    weight_map = _load_index(model_dir)
    key = _pick_embed_tokens_checkpoint_key(weight_map)
    shard_groups = _shard_groups([key], weight_map)
    raw: dict[str, torch.Tensor] = {}
    for shard_name, keys in shard_groups.items():
        raw.update(_read_shard_tensors(model_dir / shard_name, keys))
    return raw[key]


__all__ = [
    "read_decoder_layer_tensors_from_sharded_checkpoint",
    "read_decoder_layer_weight",
    "read_embed_tokens_checkpoint_tensor",
    "read_lm_head_checkpoint_tensors",
]
