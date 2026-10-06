# SPDX-License-Identifier: Apache-2.0
"""CPU-only pinned HF control for the selected model's unfinished haiku draft."""

import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..tt.checkpoint import MODEL_ID, REVISION, SNAPSHOT
from .hf_model import Kolibri1ForCausalLM


def main():
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1] / "doc/datatype_sweep"
    selected = json.loads((root / "selected_default/qualitative_tt.json").read_text())[0]
    previous_hf = json.loads((root / "selected_default/qualitative_hf.json").read_text())[0]
    assert selected["id"] == previous_hf["id"] == 0
    assert selected["prompt_ids"] == previous_hf["prompt_ids"]
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    prefix_text = selected["completion"].split("Code learns to see", 1)[0] + "Code learns to see"
    matching = [
        i
        for i in range(1, len(selected["generated_ids"]) + 1)
        if tokenizer.decode(selected["generated_ids"][:i], skip_special_tokens=False) == prefix_text
    ]
    assert len(matching) == 1, "Require an exact original-token boundary before the draft's claimed count"
    draft_prefix = selected["generated_ids"][: matching[0]]
    model = Kolibri1ForCausalLM.from_pretrained(SNAPSHOT).eval()
    result = dict(
        model_id=MODEL_ID,
        revision=REVISION,
        backend="CPU PyTorch/HF reference; no TTNN import/device use",
        prompt_ids=selected["prompt_ids"],
        selected_output_evidence="selected_default/qualitative_tt.json, id0",
        issue="Draft says 'Code learns to see (5)' although the line has four syllables",
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    out = root / "qualitative_control.json"
    started = time.monotonic()
    inputs = torch.tensor([selected["prompt_ids"]])
    generated = model.generate(inputs, max_new_tokens=256)[0, inputs.shape[1] :].tolist()
    assert generated[: len(previous_hf["generated_ids"])] == previous_hf["generated_ids"]
    result["unforced_256"] = dict(
        generated_ids=generated,
        completion=tokenizer.decode(generated, skip_special_tokens=False),
        reproduces_original_hf_128=True,
        seconds=time.monotonic() - started,
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    started = time.monotonic()
    inputs = torch.tensor([selected["prompt_ids"] + draft_prefix])
    continuation = model.generate(inputs, max_new_tokens=64)[0, inputs.shape[1] :].tolist()
    result["same_draft_prefix"] = dict(
        prefix_generated_ids=draft_prefix,
        prefix_completion=prefix_text,
        generated_ids=continuation,
        completion=tokenizer.decode(continuation, skip_special_tokens=False),
        selected_continuation=selected["generated_ids"][len(draft_prefix) :],
        seconds=time.monotonic() - started,
        interpretation_scope="Conditional control at the observed draft; this is not an unforced HF trajectory or an accuracy gate.",
    )
    result["status"] = "complete"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
