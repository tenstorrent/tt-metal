# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact repeated-run determinism check for the optimized decoder."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import LAYER as FULL_LAYER
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import _state as full_state
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import LAYER as LINEAR_LAYER
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import _state as linear_state
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import POLICIES, OptimizedDecoder


@torch.no_grad()
def _one(config, state, layer, hidden, mode, candidate):
    batch, sequence, _ = hidden.shape
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer,
            mesh_device=mesh,
            batch=batch,
            max_context=64,
            page_size=64,
            candidate=candidate,
        )
        hidden_device_shape = (
            (1, 1, batch, config.hidden_size) if mode == "decode" else (1, batch, sequence, config.hidden_size)
        )
        hidden_tt = _to_device(hidden.reshape(hidden_device_shape), mesh_device=mesh)
        page_table = _to_device(
            torch.arange(batch, dtype=torch.int32).reshape(batch, 1),
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        )
        position_values = (
            torch.zeros(batch, dtype=torch.uint32)
            if mode == "decode"
            else torch.arange(sequence, dtype=torch.int64).to(torch.uint32).reshape(1, -1).expand(batch, -1)
        )
        positions = _to_device(
            position_values,
            mesh_device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )
        forward = decoder.decode_forward if mode == "decode" else decoder.prefill_forward
        output = forward(
            hidden_states=hidden_tt,
            page_table=page_table,
            current_positions=positions,
        )
        ttnn.synchronize_device(mesh)
        return ttnn.to_torch(ttnn.get_device_tensors(output)[0])
    finally:
        ttnn.close_mesh_device(mesh)


def run(kind, mode, batch, candidate="default"):
    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260730)
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    if kind == "full":
        layer, state, sequence = FULL_LAYER, full_state(config), 33
    else:
        layer, state, sequence = LINEAR_LAYER, linear_state(config), 5
    if mode == "decode":
        sequence = 1
    hidden = (torch.randn(batch, sequence, config.hidden_size) * 0.2).bfloat16()
    first = _one(config, state, layer, hidden, mode, candidate)
    second = _one(config, state, layer, hidden, mode, candidate)
    assert torch.equal(first, second), f"{kind} {mode} B{batch} is not bit-exact"
    print(
        "OPTIMIZED_DETERMINISM",
        f"kind={kind}",
        f"mode={mode}",
        f"batch={batch}",
        f"candidate={candidate}",
        "bit_exact=True",
    )
    return {
        "kind": kind,
        "mode": mode,
        "batch": batch,
        "candidate": candidate,
        "bit_exact": True,
        "path": "optimized",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("full", "linear"), required=True)
    parser.add_argument("--mode", choices=("decode", "prefill"), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 32), default=1)
    parser.add_argument("--candidate", choices=sorted(POLICIES), default="default")
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    result = run(args.kind, args.mode, args.batch, args.candidate)
    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
