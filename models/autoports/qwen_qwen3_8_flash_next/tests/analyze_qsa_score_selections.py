# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only HF attention on the native/SFPU score diagnostic's selected IDs."""

import argparse
import json
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_context_window import metrics


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/ttuser/qwen38-flash"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    fixture = torch.load(args.root / "qsa_context_window_222272.pt", weights_only=True, mmap=True)
    capture = torch.load(args.root / "qsa_sfpu_scores.pt", weights_only=True, mmap=True)
    _, model, _, _, provenance = load_reference(args.snapshot)
    assert provenance == fixture["provenance"]
    from transformers.models.qwen4_exp.modeling_qwen4_exp import repeat_kv

    keys = torch.cat((fixture["prefix_state"]["keys"], fixture["projections"]["key"]), 2)
    values = torch.cat((fixture["prefix_state"]["values"], fixture["projections"]["value"]), 2)
    report = {
        "scope": "CPU HF attention with score-policy selected IDs; isolates selection impact, not device acceptance",
        "policies": {},
    }
    for policy in ("native", "sfpu"):
        outputs, changes = [], []
        for row in range(32):
            position = fixture["prefix"] + row
            complete = (position + 1) // 4
            blocks = capture[policy]["score"][0, 0, row, :complete].topk(512).indices
            selected = (blocks[:, None] * 4 + torch.arange(4)[None]).flatten()
            selected = torch.cat((selected, torch.arange(complete * 4, position + 1)))
            changes.append(len(set(selected.tolist()) ^ set(fixture["selected_ids"][row].tolist())))
            query = fixture["projections"]["query"][:, :, row : row + 1]
            k, v = repeat_kv(keys[:, :, selected], 12), repeat_kv(values[:, :, selected], 12)
            probabilities = torch.softmax(
                (query @ k.transpose(-1, -2)) * (256**-0.5), -1, dtype=torch.float32
            ).bfloat16()
            result = probabilities @ v
            result *= fixture["projections"]["gate"][:, :, row : row + 1].sigmoid()
            outputs.append(model.o_proj(result.reshape(1, 1, 6144)))
        actual = torch.cat(outputs, 1)
        measured = metrics(fixture["saved_expected"], actual)
        report["policies"][policy] = {
            "output_accuracy": {key: value for key, value in measured.items() if not isinstance(value, list)},
            "per_row_pcc": measured["row_pcc_fp64"],
            "changed_token_set_sizes": changes,
            "passed_window_row_gate": min(measured["row_pcc_fp64"]) >= 0.995,
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
