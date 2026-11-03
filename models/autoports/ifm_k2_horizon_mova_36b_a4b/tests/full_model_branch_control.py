# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU HF control for an alternative first-token qualitative branch."""
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .full_model_qualitative import MODEL, REVISION, ROOT


def main():
    torch.set_num_threads(1)
    source = ROOT / "doc/full_model/autoregressive/qualitative_extended_prompt_3/autoregressive_meta.json"
    data = json.loads(source.read_text())
    ids = data["prompt_token_ids"]
    forced = data["tt"]["token_ids"][:1]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()
    began = time.perf_counter()
    with torch.inference_mode():
        logits = model(torch.tensor([ids])).logits[0, -1].float()
        values, indices = logits.topk(20)
        rank = int((logits > logits[forced[0]]).sum()) + 1
        result = model.generate(
            torch.tensor([ids + forced]), max_new_tokens=511, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )[0, len(ids) :].tolist()
    report = dict(
        model=MODEL,
        revision=REVISION,
        source=str(source),
        prompt_token_ids=ids,
        forced_prefix_tokens=forced,
        forced_prefix_text=tokenizer.decode(forced),
        first_token_hf_rank=rank,
        first_token_hf_logit=float(logits[forced[0]]),
        first_token_top20=[
            dict(id=int(i), text=tokenizer.decode([int(i)]), logit=float(v)) for i, v in zip(indices, values)
        ],
        token_ids=result,
        text=tokenizer.decode(result, skip_special_tokens=False),
        seconds=time.perf_counter() - began,
    )
    dest = ROOT / "doc/full_model/qualitative_branch_control.json"
    dest.write_text(json.dumps(report, indent=2) + "\n")
    dest.with_suffix(".txt").write_text(report["text"])
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
