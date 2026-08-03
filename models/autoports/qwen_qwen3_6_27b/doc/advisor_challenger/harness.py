# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B hooks for advisor-challenger's fixed timing harness."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import LAYER as FULL_LAYER
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import _state as full_state
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import LAYER as LINEAR_LAYER
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import _state as linear_state
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder
from models.autoports.qwen_qwen3_6_27b.doc.advisor_challenger import harness_template as protocol


def build(device, policy: dict):
    kind = protocol.os.environ["CHALLENGER_LAYER_KIND"]
    policy = dict(policy)
    policy["candidate"] = protocol.os.environ.get("CHALLENGER_CANDIDATE", policy["candidate"])
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    layer_idx, state_builder = (FULL_LAYER, full_state) if kind == "full_attention" else (LINEAR_LAYER, linear_state)
    decoder = OptimizedDecoder.from_state_dict(
        state_builder(config), hf_config=config, layer_idx=layer_idx, mesh_device=device,
        batch=protocol.BATCH, max_context=64, page_size=64, **policy,
    )
    torch.manual_seed(20260803)
    hidden = _to_device(
        (torch.randn(1, 1, protocol.BATCH, config.hidden_size) * 0.2).bfloat16(), mesh_device=device
    )
    page_table = _to_device(
        torch.arange(protocol.BATCH, dtype=torch.int32).reshape(protocol.BATCH, 1),
        mesh_device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32,
    )
    positions = _to_device(
        torch.zeros(protocol.BATCH, dtype=torch.uint32), mesh_device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32,
    )
    return decoder, hidden, page_table, positions


def decode(state):
    decoder, hidden, page_table, positions = state
    return decoder.decode_forward(hidden_states=hidden, page_table=page_table, current_positions=positions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="incumbent")
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy", default=None)
    args = parser.parse_args()
    protocol._module.build = build
    protocol._module.decode = decode
    default_policy = f"models/autoports/{protocol.MODEL_DIR}/doc/advisor_challenger/incumbent.json"
    if args.label == "incumbent" and not args.policy:
        raise SystemExit("--policy is required for the incumbent")
    protocol.measure(args.label, args.out, args.policy or default_policy)
