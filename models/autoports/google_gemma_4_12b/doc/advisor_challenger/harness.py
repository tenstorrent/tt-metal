"""Gemma-4 hooks for the unmodified advisor-challenger harness protocol."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

from common import build_state, decode as decode_state


TEMPLATE = Path(__file__).resolve().parents[5] / ".agents/skills/advisor-challenger/scripts/harness_template.py"
spec = importlib.util.spec_from_file_location("advisor_challenger_harness_template", TEMPLATE)
template = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(template)


def build(device, policy):
    return build_state(device, policy, os.environ["CHALLENGER_LAYER_KIND"], template.BATCH)


def decode(state):
    if os.environ.get("CHALLENGER_PROFILE_EAGER") == "1" and not getattr(decode, "profiled", False):
        from tracy import signpost

        decode.profiled = True
        signpost(header="PERF_DECODE")
        result = decode_state(state)
        import ttnn

        ttnn.synchronize_device(state["decoder"].mesh_device)
        signpost(header="PERF_DECODE_END")
        return result
    return decode_state(state)


template.build = build
template.decode = decode


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="incumbent")
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy")
    args = parser.parse_args()
    default = Path(__file__).with_name("incumbent.json")
    if args.label == "incumbent" and not args.policy:
        raise SystemExit("--policy is required for the incumbent")
    template.measure(args.label, args.out, args.policy or str(default))
