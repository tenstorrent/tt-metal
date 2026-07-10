# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""`ttnn-advise capture` target for the qwen3_6_27b OptimizedDecoder (decode step).

Qwen3.6-27B is a HYBRID gated-delta model: 16 ``full_attention`` layers and 48
``linear_attention`` (gated-delta / Mamba-like recurrent) layers, with a DENSE
SwiGLU MLP (gate/up/down). It is NOT a Mixture-of-Experts model -- there is no
expert routing (no topk / gather / grouped matmul) anywhere in the decoder.

Mirrors tests/test_optimized_decoder_perf.py so the advisor traces the same decode
graph the profiler measures.

Exposes:
  * decode(hidden)        -- one decode step through the model's ttnn path
  * make_inputs(device)   -- builds the layer + decode inputs on device

Modes (env QWEN36_ADVISE_LAYER):
  * 3 (default) -> full_attention  (q/q_gate/k/v/o projections + dense MLP)
  * 0           -> linear_attention (gated-delta recurrent + dense MLP)

Run (PYTHONPATH must include the snapshot root AND tt-metal for models.common):
  source .agents/skills/shard-advise/scripts/bootstrap.sh
  export PYTHONPATH=<snapshot-root>:/localdev/mvasiljevic/tt-metal:$PYTHONPATH
  ttnn-advise capture .agents/skills/shard-advise/scripts/advise_qwen36_decoder.py:decode --out /tmp/qwen36-advice
"""
import os
import sys

import torch

import ttnn

SNAPSHOT_ROOT = os.environ.get(
    "QWEN36_ADVISE_SNAPSHOT_ROOT",
    "/localdev/mvasiljevic/agentic-research/experiment-7/code/runs/snapshots/qwen36-27b",
)
TTMETAL_ROOT = os.environ.get("QWEN36_ADVISE_TTMETAL_ROOT", "/localdev/mvasiljevic/tt-metal")
LAYER_IDX = int(os.environ.get("QWEN36_ADVISE_LAYER", "3"))  # 3 = full_attention, 0 = linear_attention
BLOCK_SIZE = int(os.environ.get("QWEN36_ADVISE_BLOCK", "32"))
SEQ_LEN = int(os.environ.get("QWEN36_ADVISE_SEQ", "4"))

# Append (do NOT prepend): the tt-metal root contains a `ttnn/` dir that would
# shadow the real ttnn package if placed ahead of it.
for _p in (SNAPSHOT_ROOT, TTMETAL_ROOT):
    if _p not in sys.path:
        sys.path.append(_p)


_DECODER = None
_KWARGS = None


def _build(device):
    from models.autoports.qwen3_6_27b.tests.test_functional_decoder import (
        _position_embeddings,
        _synthetic_layer_state_dict,
        _text_config,
        _to_tt,
        _to_tt_position_embeddings,
    )
    from models.autoports.qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder

    config = _text_config()
    layer_type = config.layer_types[LAYER_IDX]
    state = _synthetic_layer_state_dict(config, LAYER_IDX)
    decoder = OptimizedDecoder.from_state_dict(state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=device)

    decode_hidden = (torch.randn(1, 1, config.hidden_size, dtype=torch.bfloat16) * 0.1).to(torch.bfloat16)
    tt_decode_hidden = _to_tt(decode_hidden.unsqueeze(0), device)

    if layer_type == "full_attention":
        decode_cos, decode_sin = _position_embeddings(config, decode_hidden, SEQ_LEN)
        key_cache = ttnn.zeros(
            [2, config.num_key_value_heads, BLOCK_SIZE, config.head_dim],
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        value_cache = ttnn.zeros(
            [2, config.num_key_value_heads, BLOCK_SIZE, config.head_dim],
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        page_table = ttnn.Tensor(torch.tensor([[1]], dtype=torch.int32), ttnn.int32).to(device)
        current_pos = ttnn.Tensor(torch.tensor([SEQ_LEN], dtype=torch.int32), ttnn.int32).to(device)
        kwargs = dict(
            position_embeddings=_to_tt_position_embeddings(decode_cos, decode_sin, device, mode="decode"),
            kv_cache=(key_cache, value_cache),
            page_table=page_table,
            current_pos=current_pos,
        )
    else:
        linear_state = decoder.create_empty_linear_state(batch_size=1)
        kwargs = dict(linear_state=linear_state)

    return decoder, kwargs, tt_decode_hidden


def decode(hidden):
    return _DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, _KWARGS, hidden = _build(device)
    return (hidden,)
