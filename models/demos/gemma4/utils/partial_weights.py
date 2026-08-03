# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Read a subset of tensors out of a HuggingFace checkpoint.

Gemma4's device weights are cached under ``TT_CACHE_PATH``, and ``ttnn.as_tensor``
ignores its host-tensor argument entirely whenever the cache file already exists
(see ``ttnn/ttnn/operations/core.py``). A model can therefore be rebuilt from the
on-disk cache with an almost-empty state dict — which for the 31B variant skips a
~59 GB checkpoint read and ~62 GB of host RAM.

Two pieces still have to come from the checkpoint:

* ``layer_scalar`` — ``Gemma4DecoderLayer`` reads it as a *Python float*, so it is
  never written to the tensor cache. It silently defaults to 1.0 when absent,
  while the real 31B values are 0.089, 0.065, 0.992, ... — i.e. wrong numerics
  with no other failure signal.
* ``embed_tokens.weight`` — ``Gemma4Model`` only *constructs* its embedding and
  lm_head tensors when that key is present in the state dict, even though both
  cache files exist.

``load_cache_completion_state`` returns exactly those two. ``load_layer_state``
additionally pulls one decoder layer's full weights, which is enough to build a
host-side HuggingFace reference layer without loading the rest of the model.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from loguru import logger

_INDEX_FILE = "model.safetensors.index.json"
_SINGLE_FILE = "model.safetensors"

# Text-decoder prefixes. Gemma4 checkpoints nest the text model under
# ``model.language_model`` (the multimodal variants, including 31B) or directly
# under ``model``. The vision tower has its own ``model.vision_tower.encoder.layers.*``
# namespace, which neither prefix matches.
_LAYER_PREFIXES = ("model.language_model.layers.", "model.layers.")
_EMBED_KEYS = ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")


def resolve_checkpoint_dir(model_path) -> Path:
    """Return the local directory holding the checkpoint's safetensors shards.

    Accepts either a local checkpoint directory or a HuggingFace repo id; for a
    repo id the already-downloaded snapshot is located through the HF cache with
    ``local_files_only=True``, so this never reaches the network.
    """
    path = Path(model_path)
    if path.is_dir():
        return path

    from huggingface_hub import hf_hub_download

    for filename in (_INDEX_FILE, _SINGLE_FILE):
        try:
            resolved = hf_hub_download(str(model_path), filename, local_files_only=True)
        except Exception:  # not in the cache under this name — try the next one
            continue
        return Path(resolved).parent

    raise FileNotFoundError(
        f"No local checkpoint for {model_path!r}: it is neither a directory nor a HuggingFace "
        f"snapshot containing {_INDEX_FILE} or {_SINGLE_FILE}. Point HF_HOME at the populated "
        f"cache (and set HF_HUB_OFFLINE=1), or pass a checkpoint directory."
    )


def _weight_map(checkpoint_dir: Path) -> dict[str, str]:
    """Map every tensor key to the name of the shard file holding it.

    Uses the safetensors index when present (one small JSON read); otherwise
    falls back to reading just the shard *headers*, which safe_open does without
    materializing any tensor data.
    """
    index_path = checkpoint_dir / _INDEX_FILE
    if index_path.is_file():
        with open(index_path) as f:
            return json.load(f)["weight_map"]

    shards = sorted(checkpoint_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No *.safetensors shards under {checkpoint_dir}")

    from safetensors import safe_open

    weight_map: dict[str, str] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                weight_map[key] = shard.name
    return weight_map


def load_state_dict_subset(model_path, key_filter) -> dict[str, torch.Tensor]:
    """Load only the tensors whose key satisfies ``key_filter``.

    Each shard is opened once and only the selected tensors are materialized, so
    the cost scales with the keys requested rather than with the checkpoint size.
    float32 tensors are cast to bfloat16, matching
    ``Gemma4ModelArgs.load_state_dict``.
    """
    from safetensors import safe_open

    checkpoint_dir = resolve_checkpoint_dir(model_path)
    weight_map = _weight_map(checkpoint_dir)

    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        if key_filter(key):
            by_shard[shard].append(key)

    state_dict: dict[str, torch.Tensor] = {}
    for shard, keys in sorted(by_shard.items()):
        with safe_open(str(checkpoint_dir / shard), framework="pt") as f:
            for key in sorted(keys):
                tensor = f.get_tensor(key)
                state_dict[key] = tensor.to(torch.bfloat16) if tensor.dtype == torch.float32 else tensor
    return state_dict


def _is_text_layer_scalar(key: str) -> bool:
    return key.endswith(".layer_scalar") and key.startswith(_LAYER_PREFIXES)


def load_cache_completion_state(model_path) -> dict[str, torch.Tensor]:
    """The state-dict entries a cache-only ``Gemma4Model`` build still needs.

    Returns every text-decoder ``layer_scalar`` plus the token embedding weight.
    Everything else (attention, MLP, all per-layer norms, the final norm) resolves
    from the tensor cache with an empty sub-state.
    """
    state = load_state_dict_subset(
        model_path,
        lambda k: _is_text_layer_scalar(k) or k in _EMBED_KEYS,
    )

    scalars = [k for k in state if k.endswith(".layer_scalar")]
    embed = [k for k in state if k in _EMBED_KEYS]
    if not scalars:
        raise ValueError(
            f"No text-decoder layer_scalar tensors found in {model_path}. Without them every "
            f"layer silently falls back to layer_scalar=1.0, which is wrong for Gemma4."
        )
    if not embed:
        raise ValueError(
            f"No token embedding weight found in {model_path} (looked for {_EMBED_KEYS}). "
            f"Gemma4Model skips building embedding + lm_head when it is absent."
        )

    embed_gib = state[embed[0]].numel() * state[embed[0]].element_size() / 2**30
    logger.info(
        f"Cache-completion state: {len(scalars)} layer_scalar values + "
        f"{embed[0]} ({embed_gib:.1f} GiB); everything else loads from the tensor cache"
    )
    return state


def load_layer_state(model_path, layer_idx) -> dict[str, torch.Tensor]:
    """One decoder layer's full weights, keyed by HF module name.

    Keys come back relative to the layer (``self_attn.q_proj.weight``,
    ``mlp.gate_proj.weight``, ``layer_scalar``, ...), which is exactly what
    ``Gemma4TextDecoderLayer.load_state_dict`` expects.
    """
    prefixes = tuple(f"{p}{layer_idx}." for p in _LAYER_PREFIXES)
    state = load_state_dict_subset(model_path, lambda k: k.startswith(prefixes))

    stripped: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                stripped[key[len(prefix) :]] = tensor
                break

    if not stripped:
        raise ValueError(f"No weights for decoder layer {layer_idx} in {model_path}")
    return stripped
