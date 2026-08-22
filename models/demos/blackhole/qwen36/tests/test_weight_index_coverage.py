# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only coverage check: the checkpoint on HF_MODEL loads and remaps completely.

Catches checkpoint-layout drift (shard naming, key prefixes, split/fused GDN
projections) without burning a device job. Two assertions:

1. The shard glob used by the unit-test loader finds every file the safetensors
   index references (the 3.8 checkpoint uses standard HF ``model-XXXXX-of-XXXXX``
   naming, which an older ``model.safetensors-*`` pattern missed entirely).
2. ``remap_qwen36_state_dict`` over shape-only zero tensors for one GDN layer, one
   full-attention layer, and the globals yields every internal key the component
   tests consume, with no ``model.``/``model.language_model.`` prefix leftovers.

Pure CPU: tensor data is never read from disk (safetensors headers only).
"""

import glob
import json
import os
from pathlib import Path

import pytest
import torch

CKPT = os.environ.get("HF_MODEL", "")


def _index_path():
    return Path(CKPT) / "model.safetensors.index.json"


pytestmark = pytest.mark.skipif(
    not CKPT or not CKPT.startswith("/") or not _index_path().is_file(),
    reason="needs HF_MODEL pointing at a local sharded checkpoint dir",
)


def test_shard_glob_covers_index():
    with open(_index_path()) as f:
        weight_map = json.load(f)["weight_map"]
    indexed_files = set(weight_map.values())
    globbed = {os.path.basename(p) for p in glob.glob(f"{CKPT}/*.safetensors")}
    missing = indexed_files - globbed
    assert not missing, f"shard glob misses {len(missing)} files from the index, e.g. {sorted(missing)[:3]}"


def test_remap_covers_gdn_and_attention_layers():
    from safetensors import safe_open

    from models.demos.blackhole.qwen36.tt.weight_mapping import remap_qwen36_state_dict

    with open(Path(CKPT) / "config.json") as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    layer_types = tc["layer_types"]
    gdn_layer = layer_types.index("linear_attention")
    attn_layer = layer_types.index("full_attention")

    with open(_index_path()) as f:
        weight_map = json.load(f)["weight_map"]

    def _wanted(key):
        if ".visual" in key or key.startswith("mtp"):
            return False
        stripped = key
        for prefix in ("model.language_model.", "model."):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                break
        if stripped.startswith("layers."):
            idx = int(stripped.split(".")[1])
            return idx in (gdn_layer, attn_layer)
        return True  # globals: embeddings, final norm, lm_head

    picked = {k: fn for k, fn in weight_map.items() if _wanted(k)}
    assert picked, "no keys selected from the index"

    # Shape-only load: zero tensors with the true shapes (headers only, no data I/O).
    raw = {}
    by_file = {}
    for key, fn in picked.items():
        by_file.setdefault(fn, []).append(key)
    for fn, keys in by_file.items():
        with safe_open(str(Path(CKPT) / fn), framework="pt", device="cpu") as sf:
            for key in keys:
                raw[key] = torch.zeros(sf.get_slice(key).get_shape(), dtype=torch.bfloat16)

    sd = remap_qwen36_state_dict(raw)

    leftovers = [k for k in sd if k.startswith("model.") or "language_model" in k]
    assert not leftovers, f"unstripped prefixes: {leftovers[:5]}"

    gdn = f"layers.{gdn_layer}.linear_attn"
    attn = f"layers.{attn_layer}.self_attn"
    required = [
        f"{gdn}.qkv_proj.weight",
        f"{gdn}.q_conv.weight",
        f"{gdn}.k_conv.weight",
        f"{gdn}.v_conv.weight",
        f"{gdn}.in_proj_z.weight",
        f"{gdn}.in_proj_a.weight",
        f"{gdn}.in_proj_b.weight",
        f"{gdn}.out_proj.weight",
        f"{gdn}.A_log",
        f"{gdn}.dt_bias",
        f"{gdn}.norm.weight",
        f"layers.{gdn_layer}.mlp.gate_proj.weight",
        f"layers.{gdn_layer}.mlp.up_proj.weight",
        f"layers.{gdn_layer}.mlp.down_proj.weight",
        f"layers.{gdn_layer}.input_layernorm.weight",
        f"layers.{gdn_layer}.post_attention_layernorm.weight",
        f"{attn}.q_proj.weight",
        f"{attn}.k_proj.weight",
        f"{attn}.v_proj.weight",
        f"{attn}.o_proj.weight",
        "tok_embeddings.weight",
        "norm.weight",
    ]
    missing = [k for k in required if k not in sd]
    assert not missing, f"remap missing internal keys: {missing}"
