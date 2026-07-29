# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Official-weight HF-vs-TTNN decode check for representative full-attention layer 3."""

import os
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.optimized_full_attention_synthetic_pcc import _hf_layer
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    MODEL_ID,
    MODEL_REVISION,
    OptimizedDecoder,
    _to_device,
)
from models.common.utility_functions import comp_pcc

LAYER = 3
SNAPSHOT = Path("/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots") / MODEL_REVISION
SHARDS = tuple(f"model-{index:05d}-of-00015.safetensors" for index in (4, 6, 7, 8))


def _real_state():
    prefix = f"model.language_model.layers.{LAYER}."
    state = {}
    for shard_name in SHARDS:
        shard = SNAPSHOT / shard_name
        if not shard.is_file():
            raise FileNotFoundError(f"Required official shard is missing: {shard}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            state.update({key: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)})
    return state


@torch.no_grad()
def run():
    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260729)
    config = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION).text_config
    config._attn_implementation = "eager"
    state = _real_state()
    hf_layer = _hf_layer(config, state)
    hidden = (torch.randn(1, 1, config.hidden_size) * 0.2).bfloat16()
    positions_cpu = torch.zeros((1, 1), dtype=torch.long)
    position_ids = positions_cpu.unsqueeze(0).expand(3, -1, -1)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    reference = hf_layer(
        hidden,
        position_embeddings=rotary(hidden, position_ids),
        position_ids=positions_cpu,
        attention_mask=None,
        past_key_values=DynamicCache(config=config),
    )

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=LAYER,
            mesh_device=mesh,
            batch=1,
            max_context=64,
            page_size=64,
            candidate=os.environ.get("QWEN_OPT_CANDIDATE", "default"),
        )
        output = decoder.decode_forward(
            hidden_states=_to_device(hidden.unsqueeze(0), mesh_device=mesh),
            page_table=_to_device(
                torch.zeros((1, 1), dtype=torch.int32),
                mesh_device=mesh,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.int32,
            ),
            current_positions=_to_device(
                torch.zeros(1, dtype=torch.uint32),
                mesh_device=mesh,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
            ),
        )
        ttnn.synchronize_device(mesh)
        actual = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).squeeze(0)
        passed, message = comp_pcc(reference.float(), actual.float(), 0.995)
        print("FULL_ATTENTION_REAL_WEIGHT_DECODE_PCC", message)
        assert passed, message
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    run()
