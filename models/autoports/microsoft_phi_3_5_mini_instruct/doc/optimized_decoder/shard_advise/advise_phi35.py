# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shard-advisor capture target for the optimized Phi-3.5 dense decoder.

This is build-time tooling, not part of the runtime model.  It traces one
serving-batch decode invocation using the same synthetic-state and tensor
builders as ``tests/test_optimized_decoder.py``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn

TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    # Append: prepending the checkout would let tt-metal's source ``ttnn/``
    # directory shadow the advisor environment's installed ttnn package.
    sys.path.append(TT_METAL_ROOT)

from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizedDecoder  # noqa: E402

BATCH = int(os.environ.get("SHARD_ADVISE_BATCH", "32"))
MAX_SEQ_LEN = int(os.environ.get("SHARD_ADVISE_SEQ", "128"))
HF_CONFIG_PATH = Path(
    os.environ.get(
        "PHI35_CONFIG_PATH",
        "/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/snapshots/"
        "2fe192450127e6a83f7441aef6e3ca586c338b77/config.json",
    )
)
WEIGHT_SHAPES = {
    "self_attn.qkv_proj.weight": (9216, 3072),
    "self_attn.o_proj.weight": (3072, 3072),
    "mlp.gate_up_proj.weight": (16384, 3072),
    "mlp.down_proj.weight": (3072, 8192),
    "input_layernorm.weight": (3072,),
    "post_attention_layernorm.weight": (3072,),
}

_DECODER = None
_KWARGS = None


def _config():
    return SimpleNamespace(**json.loads(HF_CONFIG_PATH.read_text()))


def _synthetic_state_dict():
    generator = torch.Generator().manual_seed(20260729)
    return {
        name: torch.randn(shape, generator=generator, dtype=torch.float32) * 0.01
        for name, shape in WEIGHT_SHAPES.items()
    }


def _position_tensors(device):
    values = torch.zeros((BATCH,), dtype=torch.int32)
    current_pos = ttnn.from_torch(values, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    position_ids = ttnn.from_torch(
        values.to(torch.uint32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    return current_pos, position_ids


def _build(device):
    config = _config()
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state_dict(),
        hf_config=config,
        layer_idx=0,
        mesh_device=device,
        batch=BATCH,
        max_position_embeddings=MAX_SEQ_LEN,
    )
    kv_cache = OptimizedDecoder.allocate_paged_kv_cache(
        hf_config=config,
        mesh_device=device,
        max_batch_size=BATCH,
        max_seq_len=MAX_SEQ_LEN,
    )
    hidden = torch.randn(
        1,
        1,
        BATCH,
        config.hidden_size,
        generator=torch.Generator().manual_seed(20260730),
        dtype=torch.bfloat16,
    )
    tt_hidden = ttnn.from_torch(
        hidden,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    page_table = ttnn.from_torch(
        torch.arange(BATCH, dtype=torch.int32).reshape(BATCH, 1),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    current_pos, position_ids = _position_tensors(device)
    kwargs = {
        "current_pos": current_pos,
        "position_ids": position_ids,
        "page_table": page_table,
        "kv_cache": kv_cache,
        "rope_sequence_length": 1,
    }
    return decoder, kwargs, tt_hidden


def decode(hidden):
    return _DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, _KWARGS, hidden = _build(device)
    return (hidden,)
