# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""`ttnn-advise capture` target for the gemma_4_26b_a4b_it OptimizedDecoder (decode step).

gemma-4-26B-A4B-it is a GENUINE Mixture-of-Experts model: every layer has an MoE
block (enable_moe_block=True) with a 128-expert router (top_k=8) alongside a dense
SwiGLU MLP. This is the key test for whether the shard-advise tracer can model an
MoE expert path (router softmax/topk + per-expert unrolled matmuls).

Representative decode layer: FULL_LAYER_IDX=5 (full_attention). Mirrors
tests/test_optimized_decoder.py decode construction (paged kv cache, rope, current_pos).

Exposes:
  * decode(hidden)        -- one decode step through the model's ttnn path
  * make_inputs(device)   -- builds the layer + decode inputs on device

Run (PYTHONPATH must include the snapshot root AND tt-metal for models.common):
  source .agents/skills/shard-advise/scripts/bootstrap.sh
  export PYTHONPATH=<snapshot-root>:/localdev/mvasiljevic/tt-metal:$PYTHONPATH
  ttnn-advise capture .agents/skills/shard-advise/scripts/advise_gemma26_decoder.py:decode --out /tmp/gemma26-advice
"""
import os
import sys

import torch

import ttnn

SNAPSHOT_ROOT = os.environ.get(
    "GEMMA26_ADVISE_SNAPSHOT_ROOT",
    "/localdev/mvasiljevic/agentic-research/experiment-7/code/runs/snapshots/gemma4-26b-a4b-it",
)
TTMETAL_ROOT = os.environ.get("GEMMA26_ADVISE_TTMETAL_ROOT", "/localdev/mvasiljevic/tt-metal")
LAYER_IDX = int(os.environ.get("GEMMA26_ADVISE_LAYER", "5"))  # 5 = full_attention

# Append (do NOT prepend): the tt-metal root contains a `ttnn/` dir that would
# shadow the real ttnn package if placed ahead of it.
for _p in (SNAPSHOT_ROOT, TTMETAL_ROOT):
    if _p not in sys.path:
        sys.path.append(_p)


_DECODER = None
_KWARGS = None


def _build(device):
    from models.autoports.gemma_4_26b_a4b_it.tests.test_functional_decoder import (
        _hidden_states,
        _make_synthetic_state_dict,
        _paged_cache_pair,
        _position_embeddings,
        _real_text_config,
    )
    from models.autoports.gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder

    cfg = _real_text_config()
    layer_type = cfg.layer_types[LAYER_IDX]
    state = _make_synthetic_state_dict(cfg, LAYER_IDX)
    decoder = OptimizedDecoder.from_state_dict(state, hf_config=cfg, layer_idx=LAYER_IDX, mesh_device=device)

    # Build seq_len=2 hidden + rope tables, then take the decode (position 1) slice.
    hidden_states = _hidden_states(seq_len=2, cfg=cfg)
    position_embeddings, _ = _position_embeddings(cfg, layer_type, hidden_states)

    decode_hidden = hidden_states[:, 1:2, :].contiguous()
    decode_cos = position_embeddings[0][:, 1:2, :].contiguous()
    decode_sin = position_embeddings[1][:, 1:2, :].contiguous()

    tt_hidden = ttnn.from_torch(decode_hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_cos = ttnn.from_torch(decode_cos, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_sin = ttnn.from_torch(decode_sin, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    key_cache, value_cache, page_table = _paged_cache_pair(ttnn, decoder, device)
    current_pos = ttnn.from_torch(
        torch.ones((1,), dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    kwargs = dict(
        position_embeddings=(tt_cos, tt_sin),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_pos=current_pos,
    )
    return decoder, kwargs, tt_hidden


def decode(hidden):
    return _DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, _KWARGS, hidden = _build(device)
    return (hidden,)
