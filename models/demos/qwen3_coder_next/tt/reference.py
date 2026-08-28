# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SOURCE A access: the HuggingFace reference for `Qwen/Qwen3-Coder-Next`.

Everything in this module is SETUP / GOLDEN only -- none of it is on the TT forward path.

`Qwen/Qwen3-Coder-Next` is a 48-layer, 512-expert hybrid MoE (~80B params, 159 GB bf16).  A
depth-capped reference is therefore the unit of work for bring-up: `load_reference(layers=L)`
materialises the FIRST L decoder layers with their REAL checkpoint weights and the real
embeddings / final norm / lm_head, by handing `from_pretrained` a snapshot view whose
safetensors index has been filtered to just the shards those layers live in.  Nothing is
randomly initialised and nothing is quantised -- it is the real model, truncated in depth.

The cap matters because `config.layer_types` alternates `linear_attention` x3 then
`full_attention` (`full_attention_interval=4`), so L must be a multiple of 4 for the capped
stack to contain BOTH token-mixer kinds -- and therefore to exercise both the `gated_delta_net`
and the `attention` graduated stubs.  4 is the smallest such stack, and was the default
while only a depth-capped build fit.  `DEFAULT_LAYERS` is now the FULL 48: the whole model
is resident on 8 chips at TP=8 x DP=4 (e2e PCC 0.9869, worst-step 0.9734), so a capped
build is no longer the honest default.  Set TT_QWEN3_LAYERS=4 for the fast capped stack.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch

MODEL_ID = "Qwen/Qwen3-Coder-Next"

# The FULL model.  48 layers at TP=8 x DP=4 is the topology bring-up proved and the one every
# published number describes; defaulting to a 4-layer cap made every unqualified run silently
# measure a different model.  4 remains the smallest depth whose layer_types cover BOTH token
# mixers (3 x linear_attention + 1 x full_attention) -- set TT_QWEN3_LAYERS=4 to get it back.
# Everything else in the stack (embeddings, both norms, the MoE block, the lm_head) is present
# at any depth.
DEFAULT_LAYERS = 48

_CACHE_ROOT = Path(os.environ.get("TT_QWEN3_TRIM_CACHE", Path.home() / ".cache" / "tt_qwen3_coder_next"))


def hf_snapshot() -> Path:
    """Local path of the pre-fetched `Qwen/Qwen3-Coder-Next` snapshot."""
    override = os.environ.get("TT_QWEN3_SNAPSHOT")
    if override:
        return Path(override)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.txt", "*.jinja", "*.safetensors"]))


def trimmed_snapshot(layers: int) -> Path:
    """A snapshot VIEW containing only the shards needed for the first `layers` decoder layers.

    The safetensors files are symlinked (no copy), the index is filtered to the retained keys and
    `config.json` gets `num_hidden_layers = layers`, so `from_pretrained` reads ~L/48 of the 159 GB
    checkpoint instead of all of it.
    """
    src = hf_snapshot()
    dst = _CACHE_ROOT / f"L{layers}"
    stamp = dst / ".complete"
    if stamp.exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)

    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    keep: dict[str, str] = {}
    for key, shard in weight_map.items():
        m = re.match(r"model\.layers\.(\d+)\.", key)
        if m is None or int(m.group(1)) < layers:
            keep[key] = shard

    for shard in sorted(set(keep.values())):
        link = dst / shard
        if not link.exists():
            link.symlink_to(src / shard)

    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": index.get("metadata", {}), "weight_map": keep})
    )

    config = json.loads((src / "config.json").read_text())
    config["num_hidden_layers"] = int(layers)
    config.pop("layer_types", None)
    (dst / "config.json").write_text(json.dumps(config, indent=2))

    for name in (
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    ):
        if (src / name).exists():
            (dst / name).write_text((src / name).read_text())

    stamp.write_text("ok\n")
    return dst


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(hf_snapshot()))


def load_reference(layers: int | None = DEFAULT_LAYERS, dtype=torch.bfloat16):
    """Return `(hf_model, tokenizer)` for the depth-capped real checkpoint.

    `layers=None` loads all 48 layers (the full 159 GB model).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = hf_snapshot() if layers is None else trimmed_snapshot(int(layers))
    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    return model, tokenizer


def encode_prompt(tokenizer, prompt: str, *, chat: bool = True) -> torch.Tensor:
    """SOURCE A input construction: the real HF chat template / tokenizer."""
    if chat and getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt")["input_ids"]
