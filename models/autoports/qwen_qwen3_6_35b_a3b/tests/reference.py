# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only canonical checkpoint and HF layer boundary."""

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding

MODEL = "Qwen/Qwen3.6-35B-A3B"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
SNAPSHOT = (
    Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
    / REVISION
)


def config():
    source = (
        SNAPSHOT if (SNAPSHOT / "config.json").exists() else Path(__file__).parents[1] / "doc/functional_decoder/hf"
    )
    c = AutoConfig.from_pretrained(source, local_files_only=True).text_config
    c._attn_implementation = "sdpa"
    return c


def weights(layer_idx):
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}."
    selected = {k: v for k, v in index.items() if k.startswith(prefix)}
    assert selected, prefix
    result = {}
    for shard in sorted(set(selected.values())):
        with safe_open(SNAPSHOT / shard, framework="pt", device="cpu") as f:
            for k, v in selected.items():
                if v == shard:
                    result[k[len(prefix) :]] = f.get_tensor(k)
    return result


def layer(state, layer_idx):
    with torch.device("meta"):
        ref = Qwen3_5MoeDecoderLayer(config(), layer_idx)
    ref.load_state_dict(state, assign=True, strict=True)
    return ref.eval()


def hf_forward(hf, x, cache, start):
    pos = torch.arange(start, start + x.shape[1]).unsqueeze(0)
    rope = Qwen3_5MoeTextRotaryEmbedding(config())(x, pos)
    mask = None
    if hf.layer_type == "full_attention":
        allowed = torch.arange(start + x.shape[1])[None, :] <= pos[0, :, None]
        mask = torch.zeros_like(allowed, dtype=x.dtype).masked_fill(~allowed, float("-inf"))[None, None]
    with torch.no_grad():
        return hf(x, position_embeddings=rope, attention_mask=mask, past_key_values=cache, position_ids=pos)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    if torch.equal(a, b):
        return 1.0
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm())).item()
