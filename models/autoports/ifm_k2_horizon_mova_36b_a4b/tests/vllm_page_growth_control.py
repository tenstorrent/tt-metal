# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Actual scheduler page growth: resident device tokens versus authoritative host control."""

import argparse
import json
from pathlib import Path

import requests

root = Path(__file__).resolve().parents[1] / "readiness_vllm"
parser = argparse.ArgumentParser()
parser.add_argument("--mixed", type=Path, default=root / "mixed_shapes.json")
parser.add_argument("--output", type=Path, default=root / "page_growth_control.json")
args = parser.parse_args()
mixed = json.loads(args.mixed.read_text())
rows = mixed["results"][:4]
runs = []
for host in (False, True):
    payload = dict(
        model="IFM/K2-Horizon-MoVA-36B-A4B",
        prompt=[r["prompt_token_ids"] for r in rows],
        max_tokens=160,
        ignore_eos=True,
        temperature=0,
        return_token_ids=True,
    )
    if host:
        payload["logprobs"] = 1
    response = requests.post("http://localhost:8000/v1/completions", json=payload, timeout=240)
    response.raise_for_status()
    body = response.json()
    tokens = [c["token_ids"] for c in sorted(body["choices"], key=lambda c: c["index"])]
    assert len(tokens) == 4 and all(len(t) == 160 for t in tokens)
    runs.append(dict(host=host, request=payload, response=body, tokens=tokens))
result = dict(
    status="pass" if runs[0]["tokens"] == runs[1]["tokens"] else "fail",
    prompt_lengths=[r["length"] for r in rows],
    output_tokens=160,
    consumed_page_boundaries=[[p for p in range(r["length"], r["length"] + 159) if p % 64 == 0] for r in rows],
    control="logprobs=1 forces host on pinned four-device plugin; host decode requires reload_inputs=True, no resident overlap",
    page_owner="vLLM scheduler; client supplies no page table or reload flags",
    runs=runs,
)
args.output.write_text(json.dumps(result, indent=2))
print(result["status"])
assert result["status"] == "pass", "Resident scheduler growth differs from authoritative host control"
