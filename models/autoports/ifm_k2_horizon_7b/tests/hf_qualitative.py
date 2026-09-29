"""Pinned HF controls for the shared qualitative suite, without TT device use."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "IFM/K2-Horizon-7B"
REVISION = "036114ce8d46c32b24c15423211069abb9c5d25e"
DOC = Path("models/autoports/ifm_k2_horizon_7b/doc/full_model")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--suffix", default="")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--threads", type=int, default=16)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    suite = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime/readiness_check/vllm_prompts.txt"
    prompts = suite.read_text().strip().split("\n\n")
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True).eval()
    records = []
    prepared = []
    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        prepared.append(
            {
                "id": f"shared_{i}",
                "messages": messages,
                "rendered_prompt": tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                "prompt_token_ids": tok.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_dict=False
                ),
            }
        )
    for offset in range(0, len(prepared), args.batch):
        group = prepared[offset : offset + args.batch]
        width = max(len(record["prompt_token_ids"]) for record in group)
        batch_ids = torch.full((len(group), width), tok.eos_token_id, dtype=torch.int64)
        mask = torch.zeros_like(batch_ids)
        for row, record in enumerate(group):
            ids = record["prompt_token_ids"]
            batch_ids[row, -len(ids) :] = torch.tensor(ids)
            mask[row, -len(ids) :] = 1
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                batch_ids,
                attention_mask=mask,
                max_new_tokens=args.steps,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )[:, width:]
        elapsed = time.perf_counter() - start
        for record, output in zip(group, outputs.tolist()):
            if tok.eos_token_id in output:
                output = output[: output.index(tok.eos_token_id) + 1]
            record.update(
                hf_token_ids=output,
                hf_text=tok.decode(output, skip_special_tokens=False),
                seconds=elapsed,
                generation_batch_size=len(group),
                padding="left, masked",
            )
            records.append(record)
            print(json.dumps(record), flush=True)
        (DOC / f"hf_qualitative{args.suffix}.json").write_text(json.dumps(records, indent=2) + "\n")
    (DOC / f"qualitative_prompt_format{args.suffix}.json").write_text(
        json.dumps(
            {
                "hf_model": MODEL,
                "revision": REVISION,
                "tokenizer_class": type(tok).__name__,
                "chat_template_present": bool(tok.chat_template),
                "prompt_mode": "chat",
                "rendering": "apply_chat_template(add_generation_prompt=True, return_dict=False)",
                "source": str(suite),
                "generation": {
                    "max_new_tokens": args.steps,
                    "do_sample": False,
                    "batch_size": args.batch,
                    "padding": "left, masked",
                },
                "command": f"python -m models.autoports.ifm_k2_horizon_7b.tests.hf_qualitative --steps {args.steps} --suffix={args.suffix} --batch {args.batch} --threads {args.threads}",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
