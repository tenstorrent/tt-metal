# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""North-Mini hooks for the advisor-challenger timing template."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoConfig

import ttnn

# Import the fixed protocol as a module so this file only supplies its two model hooks.
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[5]
_SPEC = spec_from_file_location(
    "advisor_challenger_harness_template",
    _ROOT / ".agents/skills/advisor-challenger/scripts/harness_template.py",
)
assert _SPEC and _SPEC.loader
_TEMPLATE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TEMPLATE)

from models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder import (
    REAL_REVISION,
    _decode_inputs,
    _page_table,
    _synthetic_state,
    _to_tt,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import MODEL_ID
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import OptimizedDecoder


def build(device, policy: dict):
    layer_idx = int(policy.get("layer_idx", 0))
    candidate = policy.get("candidate", "default")
    config = AutoConfig.from_pretrained(MODEL_ID, revision=REAL_REVISION, local_files_only=True)
    state_dict = _synthetic_state(config, layer_idx, sparse_weights=layer_idx != 0)
    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=device,
        batch=_TEMPLATE.BATCH,
        max_cache_len=32,
        candidate=candidate,
    )
    generator = torch.Generator().manual_seed(20260803 + layer_idx)
    hidden = _to_tt(
        (torch.randn(1, _TEMPLATE.BATCH, 1, config.hidden_size, generator=generator) * 0.02).to(torch.bfloat16),
        device,
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(
        _page_table(_TEMPLATE.BATCH, 1),
        device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    current, cos, sin = _decode_inputs(decoder, config, device, [0] * _TEMPLATE.BATCH)
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


_TEMPLATE.build = build
_TEMPLATE.decode = decode


def profile_once():
    """Emit one bounded profiler replay; this path records no timing result."""
    try:
        from tracy import signpost
    except ImportError as exc:
        raise SystemExit("profile_once requires tracy") from exc
    import json

    policy_path = _PROFILE_ARGS.policy
    policy = json.load(open(policy_path))["shipped_policy"]
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        state = build(device, policy)
        decode(state)
        ttnn.synchronize_device(device)
        trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        decode(state)
        ttnn.end_trace_capture(device, trace_id, cq_id=0)
        for _ in range(10):
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(device)
        signpost(header="PERF_DECODE")
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(device)
        signpost(header="PERF_DECODE_END")
        ttnn.release_trace(device, trace_id)
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="incumbent")
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy")
    parser.add_argument("--profile-once", action="store_true")
    args = parser.parse_args()
    _PROFILE_ARGS = args
    if args.profile_once:
        if not args.policy:
            raise SystemExit("--profile-once requires --policy")
        profile_once()
        raise SystemExit(0)
    if args.label == "incumbent" and not args.policy:
        raise SystemExit("--policy is required for the incumbent run")
    default_policy = (
        f"models/autoports/{_TEMPLATE.MODEL_DIR}/doc/advisor_challenger/incumbent.json"
    )
    _TEMPLATE.measure(args.label, args.out, args.policy or default_policy)
