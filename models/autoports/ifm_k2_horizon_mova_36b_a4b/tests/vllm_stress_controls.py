# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Recheck inherited stress behavior against exact Stage9 and standalone controls."""

import argparse
import json
from pathlib import Path

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1] / "readiness_vllm"
control = json.loads((root / "gravity_server_controls.json").read_text())
standalone = json.loads((root / "gravity_standalone_control.json").read_text())
runs = []
for old, reference in zip(control["runs"], standalone["runs"], strict=True):
    payload = old["request"]
    assert payload["prompt"] == reference["prompt_token_ids"]
    response = requests.post("http://localhost:8000/v1/completions", json=payload, timeout=240)
    response.raise_for_status()
    body = response.json()
    actual = body["choices"][0]["token_ids"]
    prior = old["response"]["choices"][0]["token_ids"]
    runs.append(
        dict(
            request=payload,
            response=body,
            matches_stage9=actual == prior,
            matches_standalone=actual == reference["token_ids"],
        )
    )
result = dict(
    status="pass" if all(r["matches_stage9"] and r["matches_standalone"] for r in runs) else "fail",
    control_paths=[str(root / "gravity_server_controls.json"), str(root / "gravity_standalone_control.json")],
    scope="Serving equivalence on the inherited repetitive filler prompt; not a claim of good output or checkpoint root cause",
    runs=runs,
)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(result["status"])
assert result["status"] == "pass"
