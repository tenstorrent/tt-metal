# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Rank the first TT/HF free-running divergence under the identical HF prefix."""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def main():
    torch.set_num_threads(8)
    root = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/full_model")
    data = json.loads((root / "qualitative_tt.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(data["hf_model"], revision=data["revision"]).eval()
    rows = []
    for row in data["prompts"]:
        first = next((i for i, (a, b) in enumerate(zip(row["hf_tokens"], row["tt_tokens"])) if a != b), None)
        record = dict(id=row["id"], first_divergence=first)
        if first is not None:
            ids = row["prompt_tokens"] + row["hf_tokens"][:first]
            with torch.inference_mode():
                logits = model(torch.tensor([ids])).logits[0, -1].float()
            token = row["tt_tokens"][first]
            record.update(
                tt_token=token, hf_token=row["hf_tokens"][first], hf_rank=int((logits > logits[token]).sum()) + 1
            )
        rows.append(record)
    (root / "qualitative_divergence.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(rows, flush=True)


if __name__ == "__main__":
    main()
