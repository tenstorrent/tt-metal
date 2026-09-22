# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Batch the shared HF controls so every decode step reads each weight once."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from models.autoports.qwen_qwen3_8_27b.tests.hf_reference import MODEL_ID, REVISION, load_hf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gen-len", type=int, default=128)
    p.add_argument("--prompt-ids", type=int, nargs="+")
    a = p.parse_args()
    torch.set_num_threads(8)
    snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = load_hf(snapshot)
    source = Path("models/common/readiness_check/vllm_prompts.txt")
    prompts = [s.strip() for s in source.read_text().split("\n\n") if s.strip()]
    prompt_ids = a.prompt_ids if a.prompt_ids is not None else list(range(len(prompts)))
    prompts = [prompts[i] for i in prompt_ids]
    rendered = [
        tokenizer.apply_chat_template([{"role": "user", "content": s}], tokenize=False, add_generation_prompt=True)
        for s in prompts
    ]
    ids = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in rendered]
    width = max(map(len, ids))
    tokens = torch.zeros(len(ids), width, dtype=torch.long)
    mask = torch.zeros_like(tokens)
    for i, row in enumerate(ids):
        tokens[i, -len(row) :] = torch.tensor(row)
        mask[i, -len(row) :] = 1
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    cache = None
    generated = []
    stop = model.generation_config.eos_token_id
    stops = set(stop if isinstance(stop, list) else [stop])
    finished = torch.zeros(len(ids), dtype=torch.bool)
    begin = time.perf_counter()
    with torch.no_grad():
        for step in range(a.gen_len):
            out = model(
                input_ids=tokens,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            cache = out.past_key_values
            tokens = out.logits[:, -1].argmax(-1)[:, None]
            generated.append(tokens[:, 0])
            finished |= torch.tensor([int(t) in stops for t in tokens[:, 0]])
            positions = positions[:, -1:] + 1
            mask = torch.cat([mask, torch.ones(len(ids), 1, dtype=torch.long)], dim=1)
            if step % 10 == 0:
                print("HF_BATCH_STEP", step, time.perf_counter() - begin, flush=True)
            if finished.all():
                break
    generated = torch.stack(generated, dim=1).tolist()
    rows = []
    for i, row in enumerate(generated):
        end = next((j + 1 for j, t in enumerate(row) if t in stops), len(row))
        row = row[:end]
        rows.append(
            dict(
                prompt_id=prompt_ids[i],
                prompt=prompts[i],
                rendered=rendered[i],
                prompt_tokens=ids[i],
                tokens=row,
                text=tokenizer.decode(row),
            )
        )
    result = dict(
        hf_model_id=MODEL_ID,
        revision=REVISION,
        prompt_mode="chat",
        tokenizer=type(tokenizer).__name__,
        chat_template=True,
        prompt_source=str(source),
        prompt_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        max_new_tokens=a.gen_len,
        do_sample=False,
        control_batch_size=len(ids),
        prompt_ids=prompt_ids,
        executed_steps=len(generated[0]),
        left_padding=True,
        command=" ".join(__import__("sys").argv),
        seconds=time.perf_counter() - begin,
        outputs=rows,
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(row["prompt_id"], row["text"], flush=True)


if __name__ == "__main__":
    main()
