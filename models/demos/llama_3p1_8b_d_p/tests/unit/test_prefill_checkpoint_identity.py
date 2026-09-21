# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only real-checkpoint identity gate before allocating the full model."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from models.demos.llama_3p1_8b_d_p.tt.weights import CheckpointWeights

CHECKPOINT = Path(os.environ.get("LLAMA31_8B_CHECKPOINT", "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"))


# Every real layer and every weight must equal the independently named HF tensor byte for byte.
# This catches swapped layer mappings even when observer indices still report the correct order.
# Only one layer plus one comparison tensor is resident, bounding host memory during this gate.
@pytest.mark.parametrize("layer_idx", range(32))
def test_prefill_raw_checkpoint_layer_identity(layer_idx):
    loaded = CheckpointWeights(CHECKPOINT).layer(layer_idx)
    index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())["weight_map"]
    expected_names = (
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    assert set(loaded) == set(expected_names)
    for name in expected_names:
        full_name = f"model.layers.{layer_idx}.{name}"
        with safe_open(CHECKPOINT / index[full_name], framework="pt", device="cpu") as shard:
            raw = shard.get_tensor(full_name)
        assert loaded[name].dtype == raw.dtype
        assert torch.equal(loaded[name], raw), (layer_idx, name)
