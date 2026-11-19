# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Independent CPU HF control for saved long-QSA timing inputs and outputs.

The raw timing probe retains its extra .995 per-row predicate. This control
reports that predicate separately from the user's whole-output acceptance bar.
No tensors from this control are supplied to device execution.
"""

import argparse
import copy
import json
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import ReferenceCache
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_decoder import load_reference, metric_passes, metrics


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--device-outputs", type=Path, required=True)
    parser.add_argument("--device-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    saved = torch.load(args.device_outputs, map_location="cpu", weights_only=False)
    report = json.loads(args.device_report.read_text())
    assert report["layer"] == 3 and report["factory"] == "optimized"
    config, reference, rope, weights, provenance = load_reference(args.snapshot)
    del config, weights
    prompt = saved["prompt"].reshape(1, -1, 10240)
    tokens = [value.reshape(1, 1, 10240) for value in saved["tokens"]]
    length = prompt.shape[1]
    assert length == report["seq_len"]
    cos, sin = rope(torch.cat([prompt, tokens[0]], dim=1), torch.arange(length + 1).reshape(1, -1))
    causal = torch.zeros(1, 1, length, length, dtype=torch.bfloat16)
    causal.masked_fill_(torch.ones(length, length, dtype=torch.bool).triu(1), torch.finfo(torch.bfloat16).min)
    cache = ReferenceCache()
    expected = reference(
        prompt, position_embeddings=(cos[:, :length], sin[:, :length]), attention_mask=causal, past_key_values=cache
    )
    decodes = [
        reference(
            token,
            position_embeddings=(cos, sin),
            attention_mask=torch.zeros(1, 1, 1, length + 1, dtype=torch.bfloat16),
            past_key_values=copy.deepcopy(cache),
        )
        for token in tokens
    ]

    def accepts(metric):
        return (
            metric["finite"]
            and metric["pcc"] >= 0.995
            and metric["contribution_pcc"] >= 0.995
            and all(row["pcc"] >= 0.99 and row["contribution_pcc"] >= 0.99 for row in metric["per_row"])
        )

    result = dict(
        source_device_outputs=str(args.device_outputs),
        source_device_report=str(args.device_report),
        provenance=provenance,
        scope="Full real-weight HF layer3, exact saved timing inputs, independent reset for each decode input",
        contract="Whole-output >=.995; per-row diagnostic floor .99; exact replay/state mandatory",
        variants={},
    )
    for name, outputs in saved["variants"].items():
        prefill = metrics(expected, outputs["prefill"], prompt)
        decode = [metrics(e, actual, token) for e, actual, token in zip(decodes, outputs["decode"], tokens)]
        timing_row = next(row for row in report["rows"] if row["variant"] == name)
        saved_comparison = metrics(
            saved["variants"]["baseline"]["prefill"].reshape(prompt.shape), outputs["prefill"], prompt
        )
        assert abs(saved_comparison["pcc"] - timing_row["prefill_pcc"]["whole_output_global_pcc"]) < 1e-6
        assert (
            abs(
                min(row["pcc"] for row in saved_comparison["per_row"])
                - timing_row["prefill_pcc"]["whole_output_min_row_pcc"]
            )
            < 1e-6
        )
        replay = all(row["exact_output"] and row["exact_state"] for row in timing_row["replays"])
        result["variants"][name] = dict(
            prefill=prefill,
            decode=decode,
            legacy_strict_hf_passed=all(metric_passes(value) for value in [prefill, *decode]),
            exact_replay_state=replay,
            passed=replay and all(accepts(value) for value in [prefill, *decode]),
        )
    result["passed"] = result["variants"]["default"]["passed"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {"passed": row["passed"], "prefill_pcc": row["prefill"]["pcc"]}
                for name, row in result["variants"].items()
            }
        )
    )
    assert result["passed"]


if __name__ == "__main__":
    main()
