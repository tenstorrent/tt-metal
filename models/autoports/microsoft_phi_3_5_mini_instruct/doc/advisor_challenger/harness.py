"""Phi-3.5 adapter for advisor-challenger's fixed timing protocol."""
from __future__ import annotations

import os
import runpy

import torch

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _config,
    _page_table,
    _positions,
    _real_state,
    _to_tt_decode,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizationPolicy, OptimizedDecoder


def _dtype(name):
    return {"bfloat4_b": ttnn.bfloat4_b, "bfloat8_b": ttnn.bfloat8_b, "bfloat16": ttnn.bfloat16}[name]


def _fidelity(name):
    return {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}[name]


def _policy(record):
    values = dict(record)
    for key in ("attention_weight_dtype", "gate_up_weight_dtype", "down_weight_dtype", "kv_cache_dtype"):
        values[key] = _dtype(values[key])
    for key in ("attention_math_fidelity", "gate_up_math_fidelity", "down_math_fidelity"):
        values[key] = _fidelity(values[key])
    return OptimizationPolicy(**values)


def build(device, policy):
    batch = int(os.environ["CHALLENGER_DECODE_BATCH"])
    config = _config()
    decoder = OptimizedDecoder.from_state_dict(
        _real_state(), hf_config=config, layer_idx=LAYER_IDX, mesh_device=device,
        batch=batch, max_context=128, optimization_policy=_policy(policy),
    )
    generator = torch.Generator().manual_seed(232)
    hidden = torch.randn(batch, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    return {
        "decoder": decoder,
        "hidden": _to_tt_decode(hidden, device),
        "page_table": _page_table(batch, 128, device, permute=True),
        "positions": _positions([33] * batch, device),
        "caches": decoder.create_paged_kv_cache(),
    }


def decode(state):
    key_cache, value_cache = state["caches"]
    return state["decoder"].decode_forward(
        state["hidden"], key_cache=key_cache, value_cache=value_cache,
        page_table=state["page_table"], current_positions=state["positions"], use_long_rope=False,
    )


if __name__ == "__main__":
    template = runpy.run_path(".agents/skills/advisor-challenger/scripts/harness_template.py", run_name="challenger_template")
    template["build"].__globals__["build"] = build
    template["decode"].__globals__["decode"] = decode
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="incumbent")
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy")
    args = parser.parse_args()
    default_policy = f"models/autoports/{os.environ['CHALLENGER_MODEL_DIR']}/doc/advisor_challenger/incumbent.json"
    template["measure"](args.label, args.out, args.policy or default_policy)
