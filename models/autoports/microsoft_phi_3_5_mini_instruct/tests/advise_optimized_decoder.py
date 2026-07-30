# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shard-advisor capture target for the dense Phi-3.5 optimized decoder."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import torch

import ttnn

MODEL_DIR = os.environ.get("SHARD_ADVISE_MODEL_DIR", "/home/mvasiljevic/tt-metal")
BATCH = int(os.environ.get("SHARD_ADVISE_BATCH", "32"))
MAX_CONTEXT = int(os.environ.get("SHARD_ADVISE_CONTEXT", "128"))
CONFIG_PATH = os.environ.get(
    "SHARD_ADVISE_CONFIG",
    "/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/"
    "snapshots/2fe192450127e6a83f7441aef6e3ca586c338b77/config.json",
)

if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizedDecoder  # noqa: E402

LAYER_IDX = 0
_DECODER = None
_KWARGS = None


def _synthetic_state(config):
    generator = torch.Generator().manual_seed(20260728)
    prefix = f"model.layers.{LAYER_IDX}."

    def sample(shape, mean, std):
        return (torch.randn(*shape, generator=generator) * std + mean).to(torch.bfloat16)

    return {
        prefix + "input_layernorm.weight": sample((config.hidden_size,), 0.00829245, 0.02295496),
        prefix + "post_attention_layernorm.weight": sample((config.hidden_size,), 0.03923744, 0.00945584),
        prefix
        + "self_attn.qkv_proj.weight": sample((3 * config.hidden_size, config.hidden_size), 0.00000262, 0.02379715),
        prefix + "self_attn.o_proj.weight": sample((config.hidden_size, config.hidden_size), -0.00000081, 0.01751270),
        prefix
        + "mlp.gate_up_proj.weight": sample(
            (2 * config.intermediate_size, config.hidden_size), -0.00001401, 0.03248470
        ),
        prefix + "mlp.down_proj.weight": sample((config.hidden_size, config.intermediate_size), 0.00000275, 0.03603584),
    }


def _from_torch(tensor, device, *, dtype, layout):
    return ttnn.from_torch(
        tensor,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def decode(hidden):
    return _DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    global _DECODER, _KWARGS
    with open(CONFIG_PATH) as config_file:
        config = SimpleNamespace(**json.load(config_file))
    _DECODER = OptimizedDecoder.from_state_dict(
        _synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        batch=BATCH,
        max_context=MAX_CONTEXT,
    )
    hidden = torch.randn(
        BATCH,
        1,
        config.hidden_size,
        generator=torch.Generator().manual_seed(20260730 + BATCH),
    ).to(torch.bfloat16)
    tt_hidden = _from_torch(
        hidden.transpose(0, 1).unsqueeze(0),
        device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    key_cache, value_cache = _DECODER.create_paged_kv_cache()
    blocks = MAX_CONTEXT // 32
    page_table = torch.arange(BATCH * blocks, dtype=torch.int32).reshape(BATCH, blocks).flip(-1)
    _KWARGS = {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "page_table": _from_torch(page_table, device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
        "current_positions": _from_torch(
            torch.zeros(BATCH, dtype=torch.int32),
            device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "use_long_rope": False,
    }
    return (tt_hidden,)
