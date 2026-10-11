# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fresh chat-template HF controls for the shared six-prompt qualitative suite."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import transformers

from models.autoports.meta_models_muse_glimmer_30b.tests.generate_full_reference import (
    MODEL_ID,
    REVISION,
    ROOT,
    ExactModel,
    ExactTokenizer,
)

if __name__ == "__main__":
    torch.set_num_threads(16)
    tokenizer = ExactTokenizer.from_pretrained(MODEL_ID)
    model = ExactModel.from_pretrained(MODEL_ID).eval()
    suite = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime/readiness_check/vllm_prompts.txt"
    prompts = [text.strip() for text in suite.read_text().split("\n\n") if text.strip()]
    destination = ROOT / "doc/full_model"
    metadata = dict(
        model_id=MODEL_ID,
        revision=REVISION,
        tokenizer=type(tokenizer).__name__,
        transformers_version=transformers.__version__,
        chat_template_present=bool(tokenizer.chat_template),
        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        mode="chat",
        render="apply_chat_template(tokenize=True,add_generation_prompt=True,return_dict=False)",
        prompt_source=str(suite),
        prompt_source_sha256=hashlib.sha256(suite.read_bytes()).hexdigest(),
        max_new_tokens=128,
        do_sample=False,
        command=sys.argv,
        dtype="bfloat16",
        device="cpu",
    )
    (destination / "qualitative_prompt_format.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = []
    for index, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        print("HF_QUALITATIVE_START", index, len(ids), flush=True)
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                torch.tensor([ids]),
                max_new_tokens=128,
                do_sample=False,
                attention_mask=torch.ones((1, len(ids)), dtype=torch.int64),
                pad_token_id=tokenizer.eos_token_id,
            )
        tokens = output[0, len(ids) :].tolist()
        result = dict(
            id=index,
            messages=messages,
            prompt_token_ids=ids,
            rendered_prompt=tokenizer.decode(ids, skip_special_tokens=False),
            generated_token_ids=tokens,
            completion=tokenizer.decode(tokens, skip_special_tokens=False),
            generated_count=len(tokens),
            elapsed_seconds=time.perf_counter() - started,
        )
        results.append(result)
        (destination / "qualitative_hf_outputs.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n"
        )
        print("HF_QUALITATIVE_DONE", index, result["completion"], flush=True)
