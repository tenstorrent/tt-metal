# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""HF (CPU) reference for ``test_long_trace_fidelity.py``: a real thinking trace
served by the TT stack is teacher-forced through the Hugging Face model and
its per-position top-1 / top-5 are stored.

    HF_MODEL=google/gemma-4-12B-it python generate_trace_reference.py \\
        --samples <lm-eval samples_r1_gpqa_diamond_*.jsonl> --doc 129

The samples file must carry ``reasoning_content`` (run lm-eval with
LM_EVAL_PRESERVE_REASONING=1). The sequence is the chat-templated prompt with
thinking enabled, then ``<|channel>thought\\n`` + trace + ``<channel|>`` + answer,
exactly as the server generated it. ``score_start`` is the first generated
token, so the prompt is prefilled and every generated token is scored.
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _flat(x):
    while isinstance(x, list):
        x = x[0] if x else ""
    return x if isinstance(x, str) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--doc", type=int, required=True)
    ap.add_argument("--model", default=os.environ.get("HF_MODEL", "google/gemma-4-12B-it"))
    ap.add_argument("--max-tokens", type=int, default=8192, help="truncate the generated part to this many tokens")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = {json.loads(l)["doc_id"]: json.loads(l) for l in open(args.samples)}
    r = rows[args.doc]
    a = r["arguments"]["gen_args_0"]["arg_0"]
    msgs = json.loads(a[0] if isinstance(a, list) else a)
    reasoning, answer = _flat(r.get("reasoning_content")), _flat(r["resps"][0][0])
    tok = AutoTokenizer.from_pretrained(args.model)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    gen = "<|channel>thought\n" + reasoning + "<channel|>" + answer
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    g_ids = tok(gen, add_special_tokens=False)["input_ids"][: args.max_tokens]
    ids = torch.tensor([p_ids + g_ids])
    torch.set_num_threads(max(8, os.cpu_count() - 8))
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0].float()  # [N+1, vocab]
    top5 = torch.topk(logits[:-1], k=5, dim=-1).indices.to(torch.int32)  # logits at p predict p + 1
    name = os.path.basename(os.path.normpath(args.model))
    out = args.out or os.path.join(os.path.dirname(__file__), "reference_outputs", f"{name}_trace.refpt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(
        {
            "reference_tokens": ids.to(torch.int64),
            "top1_tokens": top5[:, 0].to(torch.int64),
            "top5_tokens": top5,
            "score_start": len(p_ids),
            "meta": {
                "model": args.model,
                "source": f"{os.path.basename(args.samples)} doc {args.doc}",
                "gold": r.get("target"),
                "tt_exact_match": r.get("exact_match"),
                "prompt_tokens": len(p_ids),
                "generated_tokens": len(g_ids),
            },
        },
        out,
    )
    print(f"wrote {out}: prompt {len(p_ids)} tok + generated {len(g_ids)} tok")


if __name__ == "__main__":
    main()
