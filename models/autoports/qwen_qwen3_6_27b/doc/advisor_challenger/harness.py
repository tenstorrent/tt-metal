"""Qwen3.6 hooks for advisor-challenger's fixed timing harness."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import _state as full_state
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import _state as linear_state
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder

_template_path = Path(__file__).parents[5] / ".agents/skills/advisor-challenger/scripts/harness_template.py"
_spec = importlib.util.spec_from_file_location("advisor_challenger_harness_template", _template_path)
template = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(template)


def build(device, policy):
    kind = template.os.environ["CHALLENGER_LAYER_KIND"]
    layer_idx = {"linear_attention": 0, "full_attention": 3}[kind]
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    state = (linear_state if kind == "linear_attention" else full_state)(config)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=device,
        batch=template.BATCH,
        max_context=64,
        page_size=64,
        **policy,
    )
    torch.manual_seed(20260803)
    hidden = _to_device(
        (torch.randn(1, 1, template.BATCH, config.hidden_size) * 0.2).bfloat16(),
        mesh_device=device,
    )
    page_table = _to_device(
        torch.arange(template.BATCH, dtype=torch.int32).reshape(template.BATCH, 1),
        mesh_device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
    )
    positions = _to_device(
        torch.zeros(template.BATCH, dtype=torch.uint32),
        mesh_device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    return decoder, hidden, page_table, positions


def decode(state):
    decoder, hidden, page_table, positions = state
    return decoder.decode_forward(
        hidden_states=hidden,
        page_table=page_table,
        current_positions=positions,
    )


template.build = build
template.decode = decode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="incumbent")
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy")
    args = parser.parse_args()
    default_policy = f"models/autoports/{template.MODEL_DIR}/doc/advisor_challenger/incumbent.json"
    if args.label == "incumbent" and not args.policy:
        raise SystemExit("--policy is required for the incumbent")
    template.measure(args.label, args.out, args.policy or default_policy)
