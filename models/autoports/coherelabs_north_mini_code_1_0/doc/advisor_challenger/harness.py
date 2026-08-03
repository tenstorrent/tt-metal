# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""North-Mini hooks for advisor-challenger's fixed timing protocol."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder import (
    REAL_REVISION,
    _decode_inputs,
    _page_table,
    _real_layer_one_state,
    _synthetic_state,
    _to_tt,
)
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_optimized_decoder import (
    _real_dense_layer_zero_state,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import MODEL_ID, OptimizedDecoder


_TEMPLATE = Path(__file__).resolve().parents[5] / ".agents/skills/advisor-challenger/scripts/harness_template.py"
_SPEC = importlib.util.spec_from_file_location("north_advisor_challenger_protocol", _TEMPLATE)
protocol = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(protocol)

LAYER_BY_KIND = {
    "dense_full_forced_rope": 0,
    "sliding_rope_moe": 1,
    "full_no_rope_moe": 4,
}


def build(device, policy: dict):
    kind = os.environ.get("CHALLENGER_LAYER_KIND", "sliding_rope_moe")
    layer_idx = LAYER_BY_KIND[kind]
    config = AutoConfig.from_pretrained(MODEL_ID, revision=REAL_REVISION, local_files_only=True)
    real_weights = os.environ.get("CHALLENGER_REAL_WEIGHTS", "1") == "1"
    if real_weights and layer_idx == 0:
        state = _real_dense_layer_zero_state()
    elif real_weights and layer_idx == 1:
        state = _real_layer_one_state()
    else:
        state = _synthetic_state(config, layer_idx, sparse_weights=True)
    candidate = os.environ.get("CHALLENGER_CANDIDATE", policy.get("candidate", "default"))
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=device,
        batch=protocol.BATCH,
        max_cache_len=32,
        candidate=candidate,
    )
    generator = torch.Generator().manual_seed(18000 + layer_idx)
    hidden = _to_tt(
        (torch.randn(1, protocol.BATCH, 1, config.hidden_size, generator=generator) * 0.02).to(torch.bfloat16),
        device,
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(
        _page_table(protocol.BATCH, 1), device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    current, cos, sin = _decode_inputs(decoder, config, device, [0] * protocol.BATCH)
    return {
        "decoder": decoder,
        "hidden": hidden,
        "kwargs": {
            "key_cache": key_cache,
            "value_cache": value_cache,
            "page_table": page_table,
            "current_positions": current,
            "position_cos": cos,
            "position_sin": sin,
        },
    }


def decode(state):
    return state["decoder"].decode_forward(state["hidden"], **state["kwargs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="incumbent")
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy", default=None)
    args = parser.parse_args()
    protocol.build = build
    protocol.decode = decode
    default_policy = f"models/autoports/{protocol.MODEL_DIR}/doc/advisor_challenger/incumbent.json"
    if args.label == "incumbent" and not args.policy:
        raise SystemExit("--policy is required for the incumbent")
    protocol.measure(args.label, args.out, args.policy or default_policy)
