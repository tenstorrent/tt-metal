# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serving log-softmax determinism via explicit host compatibility, outside performance."""

import argparse
import json
from pathlib import Path

import requests

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=root / "readiness_vllm/logit_determinism.json")
args = parser.parse_args()
metadata = json.loads((root / "readiness_vllm/qualitative_prompt_format.json").read_text())
prompts = [p["tokens"] for p in metadata["prompts"][:3]]
results = []
for order in ([0], [0], [0, 1, 2], [2, 1, 0], [1, 0, 2]):
    r = requests.post(
        "http://localhost:8000/v1/completions",
        json={
            "model": "IFM/K2-Horizon-MoVA-36B-A4B",
            "prompt": [prompts[i] for i in order],
            "temperature": 0,
            "max_tokens": 4,
            "ignore_eos": True,
            "logprobs": 20,
            "return_token_ids": True,
        },
        timeout=240,
    )
    r.raise_for_status()
    results.append({"order": order, "response": r.json()})
reference = {}
differences = []
for run in results:
    for c in run["response"]["choices"]:
        prompt = run["order"][c["index"]]
        item = (c["text"], c["logprobs"])
        if prompt in reference and reference[prompt] != item:
            differences.append({"prompt": prompt, "order": run["order"]})
        else:
            reference[prompt] = item
out = {
    "status": "pass" if not differences else "fail",
    "scope": "reported top20 normalized logits for four autoregressive steps; host compatibility only",
    "differences": differences,
    "runs": results,
    "standalone_baseline": "doc/datatype_sweep/selected_contract_b32_p256.json (repeat_logits_bitwise and batch_slot_logits_bitwise)",
}
args.output.write_text(json.dumps(out, indent=2))
print(out["status"], differences)
if differences:
    raise SystemExit(1)
