# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Matched CPU QSA GR accumulation counterfactuals; no replacement acceptance gate."""
import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from models.autoports.qwen_qwen3_8_flash_next.tests import probe_qsa_decoder as probe


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--device-output-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tensor-output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    _, model, rope, _, provenance = probe.load_reference(a.snapshot)
    device = torch.load(a.device_output_cache, map_location="cpu", weights_only=False)["rows"]
    by_length = {r["length"]: r for r in device if r["metric_index"] == 0}
    policies = {
        "literal": [],
        "attn_down_fp64": ["attn_hyper_connection.input_mix_weight_down"],
        "mlp_down_fp64": ["mlp_hyper_connection.input_mix_weight_down"],
        "both_down_fp64": ["attn_hyper_connection.input_mix_weight_down", "mlp_hyper_connection.input_mix_weight_down"],
        "all_gr_fp64": [
            f"{block}_hyper_connection.input_mix_weight_{direction}"
            for block in ("attn", "mlp")
            for direction in ("down", "up")
        ],
    }
    report = dict(scope=__doc__, accepted=False, device_executed=False, reference=provenance, rows=[])
    tensors = []
    for policy, paths in policies.items():
        with ExitStack() as stack:
            for path in paths:
                module = model.get_submodule(path)

                def projection(x, module=module):
                    return F.linear(x.double(), module.weight.double()).to(x.dtype)

                stack.enter_context(patch.object(module, "forward", projection))
            for length, original in by_length.items():
                fixture = probe.make_fixture(model, rope, length=length)
                assert torch.equal(fixture["prompt"], original["residual"])
                expected, actual = original["expected"], fixture["expected_prefill"]
                if policy == "literal":
                    assert torch.equal(expected, actual), "Literal independent HF rerun must match exported oracle"
                literal = probe.metrics(expected, actual, fixture["prompt"])
                matched = probe.metrics(actual, original["actual"], fixture["prompt"])
                row = dict(policy=policy, length=length, literal_vs_control=literal, device_vs_control=matched)
                report["rows"].append(row)
                tensors.append(dict(policy=policy, length=length, output=actual, prompt=fixture["prompt"]))
                print(
                    json.dumps(
                        dict(
                            policy=policy,
                            length=length,
                            literal_min=min(r["pcc"] for r in literal["per_row"]),
                            device_min=min(r["pcc"] for r in matched["per_row"]),
                        )
                    ),
                    flush=True,
                )
    assert "ttnn" not in sys.modules
    report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["device_cache_sha256"] = hashlib.sha256(a.device_output_cache.read_bytes()).hexdigest()
    torch.save(dict(rows=tensors), a.tensor_output)
    a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
