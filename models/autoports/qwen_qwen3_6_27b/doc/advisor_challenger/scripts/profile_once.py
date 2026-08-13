"""One eager decode replay solely for op-level accounting; latency is never read from this run."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import ttnn
from tracy import signpost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    args = ap.parse_args()
    harness_path = Path(__file__).with_name("harness.py")
    spec = importlib.util.spec_from_file_location("challenger_harness", harness_path)
    harness = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(harness)
    policy = json.load(open(args.policy))["shipped_policy"]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        state = harness.build(mesh, policy)
        harness.decode(state)
        ttnn.synchronize_device(mesh)
        signpost(header="PERF_DECODE")
        harness.decode(state)
        ttnn.synchronize_device(mesh)
        signpost(header="PERF_DECODE_END")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
