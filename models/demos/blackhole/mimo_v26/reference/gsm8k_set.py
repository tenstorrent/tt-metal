"""Build the GSM8K accuracy prompt sets (first N rows of the GSM8K test split, same rows/order as the DSV4 sets gsm8k_o0..o448) with the MiMo chat template
applied, plus gold answers. No model run.

    python -m models.demos.blackhole.mimo_v26.reference.gsm8k_set --n 128 --out /mnt/tt-data/ssinghal/mimo-dumps/gsm8k/gsm8k_128.json

Output json: {"n", "sha256" (of the json-encoded list of templated prompt strings), "dataset": path, "items": [{"idx","question","prompt" (templated text),
"prompt_ids" (tokens), "gold" (normalised final number)}]}. Scoring: score_gsm8k.py (extraction rule identical to DSV4 reference/e2e_analyze.py extract/norm).
"""
import argparse
import hashlib
import json
import os

from .score_gsm8k import norm

GSM8K = "/mnt/tt-data/ssinghal/datasets/gsm8k_test.jsonl"


def build(n, tok=None, enable_thinking=None):
    from .prompts import tokenizer

    tok = tok or tokenizer()
    rows = [json.loads(line) for line in open(GSM8K)][:n]
    items = []
    for i, r in enumerate(rows):
        txt = tok.apply_chat_template(
            [{"role": "user", "content": r["question"]}],
            tokenize=False,
            add_generation_prompt=True,
            **({} if enable_thinking is None else {"enable_thinking": enable_thinking}),
        )
        items.append(
            {
                "idx": i,
                "question": r["question"],
                "prompt": txt,
                "prompt_ids": tok(txt, add_special_tokens=False)["input_ids"],
                "gold": norm(r["answer"].split("####")[-1].strip()),
            }
        )
    sha = hashlib.sha256(json.dumps([it["prompt"] for it in items], ensure_ascii=False).encode()).hexdigest()
    return {"n": n, "enable_thinking": enable_thinking, "sha256": sha, "dataset": GSM8K, "items": items}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--no-thinking",
        action="store_true",
        help="enable_thinking=False: the template appends <think></think> after the assistant tag (GSM8K gate decision)",
    )
    a = ap.parse_args()
    d = build(a.n, enable_thinking=False if a.no_thinking else None)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(d, open(a.out, "w"), ensure_ascii=False)
    print(a.out, "n", d["n"], "sha256", d["sha256"], "max prompt tokens", max(len(i["prompt_ids"]) for i in d["items"]))
