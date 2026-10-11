# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Selective vision checkpoint loading, never a second full language model."""

import json
from pathlib import Path

import torch
from safetensors import safe_open


def vision_state_dict(snapshot):
    snapshot = Path(snapshot)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.visual."
    selected = {name: shard for name, shard in index.items() if name.startswith(prefix)}
    if not selected:
        raise ValueError("Checkpoint contains no model.visual weights")
    state = {}
    for shard in sorted(set(selected.values())):
        path = (snapshot / shard).resolve()
        if not path.is_relative_to(snapshot.resolve()):
            raise ValueError("Vision shard path escapes the checkpoint")
        with safe_open(path, framework="pt", device="cpu") as source:
            for name, filename in selected.items():
                if filename == shard:
                    state[name.removeprefix(prefix)] = source.get_tensor(name)
    return state


def load_reference_vision(snapshot, vision_config):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel, Qwen3_5VisionRotaryEmbedding

    # Parameters are assigned directly from safetensors; allocating random FP32
    # copies first doubles startup memory. The one nonpersistent rotary buffer
    # is initialized on CPU after assignment because it is absent from weights.
    with torch.device("meta"):
        reference = Qwen3_5VisionModel(vision_config)
    reference.load_state_dict(vision_state_dict(snapshot), strict=True, assign=True)
    reference.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(vision_config.hidden_size // vision_config.num_heads // 2)
    if any(tensor.is_meta for tensor in (*reference.parameters(), *reference.buffers())):
        raise ValueError("Vision-only reference still contains uninitialized meta tensors")
    reference.requires_grad_(False)
    return reference.eval()
