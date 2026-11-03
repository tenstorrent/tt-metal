# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read-only version of the functional stage's pinned layer oracle loader."""

import importlib
import json

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .run_decoder import MODEL, REVISION, ROOT, SNAPSHOT, pcc

__all__ = ["REVISION", "ROOT", "pcc", "reference"]


def reference(layer, synthetic=False):
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True)
    config._attn_implementation = "sdpa"
    cls = get_class_from_dynamic_module(
        "modeling_k2_horizon.K2HorizonDecoderLayer", MODEL, revision=REVISION, local_files_only=True
    )
    with torch.device("meta"):
        model = cls(config, layer)
    sd = {}
    if synthetic:
        stats_path = ROOT / f"doc/functional_decoder/weight_stats_layer_{layer}.json"
        for stat in json.loads(stats_path.read_text()):
            sd[stat["name"]] = (torch.randn(stat["shape"]) * stat["std"] + stat["mean"]).bfloat16()
    else:
        index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
        prefix = f"model.layers.{layer}."
        for shard in sorted({f for k, f in index.items() if k.startswith(prefix)}):
            with safe_open(SNAPSHOT / shard, framework="pt") as f:
                for k in f.keys():
                    if k.startswith(prefix):
                        sd[k.removeprefix(prefix)] = f.get_tensor(k)
    model.load_state_dict(sd, assign=True)
    model.eval()
    rope = importlib.import_module(cls.__module__).K2HorizonRotaryEmbedding(config)
    return config, model, rope, sd
