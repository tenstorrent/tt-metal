# SPDX-License-Identifier: Apache-2.0
"""North-Mini hooks for the advisor-challenger timing template."""

import importlib.util
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

import torch

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tests import test_functional_decoder as tests
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import OptimizedDecoder

_template_path = Path(__file__).parents[5] / ".agents/skills/advisor-challenger/scripts/harness_template.py"
_spec = importlib.util.spec_from_file_location("advisor_challenger_harness_template", _template_path)
template = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(template)
_executed = {}


def _json_value(value):
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def build(device, policy):
    layer_idx = int(os.environ["CHALLENGER_LAYER_IDX"])
    config = tests._config()
    state_dict = tests._synthetic_state(config, layer_idx)
    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=device,
        batch=template.BATCH,
        max_cache_len=32,
        candidate=os.environ.get("CHALLENGER_CANDIDATE", policy["candidate"]),
    )
    _executed.update(
        candidate=os.environ.get("CHALLENGER_CANDIDATE", policy["candidate"]),
        policy={field.name: _json_value(getattr(decoder.policy, field.name)) for field in fields(decoder.policy)},
    )
    generator = torch.Generator().manual_seed(29000 + layer_idx)
    hidden = tests._to_tt(
        (torch.randn(1, template.BATCH, 1, config.hidden_size, generator=generator) * 0.02).to(torch.bfloat16),
        device,
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = tests._to_tt(
        tests._page_table(template.BATCH, 1), device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    current, cos, sin = tests._decode_inputs(decoder, config, device, [0] * template.BATCH)
    return (
        decoder,
        hidden,
        {
            "key_cache": key_cache,
            "value_cache": value_cache,
            "page_table": page_table,
            "current_positions": current,
            "position_cos": cos if decoder.use_rope else None,
            "position_sin": sin if decoder.use_rope else None,
        },
    )


def decode(state):
    decoder, hidden, kwargs = state
    return decoder.decode_forward(hidden, **kwargs)


template.build = build
template.decode = decode

if __name__ == "__main__":
    args = template.argparse.ArgumentParser()
    args.add_argument("--label", default="incumbent")
    args.add_argument("--out", required=True)
    args.add_argument("--policy", default=None)
    parsed = args.parse_args()
    default_policy = f"models/autoports/{template.MODEL_DIR}/doc/advisor_challenger/incumbent.json"
    if parsed.label == "incumbent" and not parsed.policy:
        raise SystemExit("--policy is required for incumbent")
    policy_path = parsed.policy or default_policy
    record = template.measure(parsed.label, parsed.out, policy_path)
    record.update(
        executed_candidate=_executed["candidate"],
        executed_policy=_executed["policy"],
        invocation={
            "argv": sys.argv,
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "CHALLENGER_MODEL_DIR",
                    "CHALLENGER_DECODE_BATCH",
                    "CHALLENGER_REQUESTED_DECODE_BATCH",
                    "CHALLENGER_LAYER_IDX",
                    "CHALLENGER_CANDIDATE",
                )
            },
            "policy_path": policy_path,
        },
    )
    Path(parsed.out).write_text(json.dumps(record, indent=2) + "\n")
