# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Indexed safetensor loading of one Qwen Gated DeltaNet layer.

A checkpoint directory holds ``config.json`` and ``model.safetensors.index.json`` plus the shard files the requested
layer lives in. The index may be the hub's full index or a partial one listing only the fetched layer
(``tests/gdn/prepare.py``). The layer's keys are ``<root>layers.<i>.linear_attn.<name>``, with the root read off the
config's shape: ``model.language_model.`` when it nests a ``text_config`` (27B, 35B-A3B, Flash-Next) and ``model.``
when it is flat (2.4T).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig
from models.demos.deepseek_v3_d_p.reference.gdn.qwen_models import QWEN_GDN_MODELS
from models.demos.deepseek_v3_d_p.reference.gdn.weights import GDN_WEIGHT_NAMES, validate_gdn_weights
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import kda_state_dict_sha256

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


def gdn_checkpoint_dir(model: str, layer_idx: int) -> Path:
    """Local partial checkpoint of one layer: ``<weights root>/<org>--<name>/<revision>/gdn-layer<i>``.

    The weights root is ``GDN_WEIGHTS_ROOT`` when set, else ``<checkout>/.weights`` (the GDN baseline harness's
    root; this sub-directory keeps the two tools' files apart).
    """
    source = QWEN_GDN_MODELS[model]
    root = Path(os.environ.get("GDN_WEIGHTS_ROOT", _REPOSITORY_ROOT / ".weights"))
    return root / source.local_name / source.revision / f"gdn-layer{layer_idx}"


# Content identity of a layer-local weight dict: sorted names, dtypes, shapes and bytes (KDA's digest is generic).
gdn_state_dict_sha256 = kda_state_dict_sha256


def gdn_model_root(model_config: dict[str, Any]) -> str:
    """Key root of the text tower: ``model.language_model.`` with ``text_config``, ``model.`` without."""
    return "model.language_model." if "text_config" in model_config else "model."


def gdn_layer_prefix(layer_idx: int, model_root: str) -> str:
    if layer_idx < 0:
        raise ValueError(f"layer_idx must be nonnegative, got {layer_idx}")
    return f"{model_root}layers.{layer_idx}.linear_attn."


def gdn_layer_keys(weight_map: dict[str, str], layer_idx: int, model_root: str) -> dict[str, str]:
    """Map every canonical GDN weight name of one layer to its checkpoint key; reject missing and unknown keys.

    Every ``linear_attn.*`` key of the layer must be canonical, so another fused layout (Qwen3-Next ``in_proj_qkvz``,
    ``in_proj_ba``) or a block-quantized scale is an error rather than silently ignored.
    """
    prefix = gdn_layer_prefix(layer_idx, model_root)
    present = {key[len(prefix) :] for key in weight_map if key.startswith(prefix)}
    unknown = sorted(present - set(GDN_WEIGHT_NAMES))
    if unknown:
        raise ValueError(f"layer {layer_idx} has linear_attn weights GDN does not model: {unknown}")
    missing = [name for name in GDN_WEIGHT_NAMES if name not in present]
    if missing:
        raise ValueError(f"layer {layer_idx} checkpoint index is missing GDN weights: {missing}")
    return {name: prefix + name for name in GDN_WEIGHT_NAMES}


def load_gdn_layer_state_dict(checkpoint_dir: Path, layer_idx: int, config: GDNConfig) -> dict[str, torch.Tensor]:
    """Load one GDN layer's canonical layer-local weights (checkpoint dtype) and validate them against ``config``."""
    checkpoint_dir = Path(checkpoint_dir)
    model_config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    if GDNConfig.from_model_config(model_config) != config:
        raise ValueError(f"{checkpoint_dir}/config.json describes another GDN layer than {config}")
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing safetensor index: {index_path}")
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    keys = gdn_layer_keys(weight_map, layer_idx, gdn_model_root(model_config))
    shards = sorted({weight_map[key] for key in keys.values()})
    missing_shards = [name for name in shards if not (checkpoint_dir / name).is_file()]
    if missing_shards:
        raise FileNotFoundError(f"missing checkpoint shard(s) of layer {layer_idx}: {missing_shards}")
    state_dict = {}
    for shard in shards:
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as handle:
            for name, key in keys.items():
                if weight_map[key] == shard:
                    state_dict[name] = handle.get_tensor(key)
    validate_gdn_weights(state_dict, config)
    return state_dict
