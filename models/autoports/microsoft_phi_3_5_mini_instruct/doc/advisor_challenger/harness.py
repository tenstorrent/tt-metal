"""Phi-3.5 hooks for advisor-challenger's fixed timing harness."""
from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
TEMPLATE = ROOT / ".agents/skills/advisor-challenger/scripts/harness_template.py"
spec = importlib.util.spec_from_file_location("advisor_challenger_harness_template", TEMPLATE)
fixed = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fixed)


def build(device, policy: dict):
    from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
        LAYER_IDX,
        _config,
        _page_table,
        _positions,
        _synthetic_state,
        _to_tt_decode,
    )
    from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import (
        OptimizationPolicy,
        OptimizedDecoder,
    )

    if policy != {"policy_name": "final"}:
        raise ValueError(f"unsupported frozen policy: {policy}")
    config = _config()
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state(config), hf_config=config, layer_idx=LAYER_IDX,
        mesh_device=device, batch=fixed.BATCH, max_context=128,
        policy=replace(
            OptimizationPolicy(),
            advisor_rope_l1_chain=os.environ.get("PHI35_ADVISOR_ROPE_L1") == "1",
        ),
    )
    hidden = torch.randn(
        fixed.BATCH, 1, config.hidden_size,
        generator=torch.Generator().manual_seed(52), dtype=torch.bfloat16,
    )
    tt_hidden = _to_tt_decode(hidden, device)
    page_table = _page_table(fixed.BATCH, 128, device, permute=True)
    current_positions = _positions([127] * fixed.BATCH, device)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    return decoder, tt_hidden, key_cache, value_cache, page_table, current_positions


def decode(state):
    decoder, hidden, key_cache, value_cache, page_table, current_positions = state
    return decoder.decode_forward(
        hidden, key_cache=key_cache, value_cache=value_cache,
        page_table=page_table, current_positions=current_positions, use_long_rope=False,
    )


fixed.build = build
fixed.decode = decode

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="incumbent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy")
    args = ap.parse_args()
    default_policy = ROOT / "models/autoports" / fixed.MODEL_DIR / "doc/advisor_challenger/incumbent.json"
    if args.label == "incumbent" and not args.policy:
        raise SystemExit("--policy is required for the incumbent")
    record = fixed.measure(args.label, args.out, args.policy or str(default_policy))
    if args.label == "incumbent":
        record["layer_counts"] = {"dense": 32}
        record["total_layers"] = 32
        with open(args.out, "w") as fh:
            json.dump(record, fh, indent=2)
