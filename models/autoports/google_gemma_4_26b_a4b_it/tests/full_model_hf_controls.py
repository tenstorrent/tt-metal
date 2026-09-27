# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prompt-correct HF controls for the shared readiness qualitative suite."""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
ROOT = Path("models/autoports/google_gemma_4_26b_a4b_it/doc/full_model")
SUITE = Path("models/common/readiness_check/vllm_prompts.txt")


def main():
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION).eval()
    rows = []
    for i, prompt in enumerate(SUITE.read_text().strip().split("\n\n")):
        messages = [{"role": "user", "content": prompt}]
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        with torch.inference_mode():
            out = model.generate(torch.tensor([tokens]), max_new_tokens=128, do_sample=False)
        generated = out[0, len(tokens) :].tolist()
        rows.append(
            dict(
                id=f"shared_{i}",
                messages=messages,
                rendered_prompt=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                prompt_tokens=tokens,
                hf_tokens=generated,
                hf_completion=tokenizer.decode(generated, skip_special_tokens=False),
            )
        )
        report = dict(
            hf_model=MODEL,
            revision=REVISION,
            tokenizer=type(tokenizer).__name__,
            chat_template_present=bool(tokenizer.chat_template),
            prompt_mode="chat",
            prompt_source=str(SUITE),
            max_new_tokens=128,
            do_sample=False,
            dtype=str(model.dtype),
            prompts=rows,
        )
        (ROOT / "qualitative_hf.json").write_text(json.dumps(report, indent=2) + "\n")
        print(rows[-1]["id"], rows[-1]["hf_completion"], flush=True)


if __name__ == "__main__":
    main()
