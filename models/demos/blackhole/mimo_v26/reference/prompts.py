"""Real-text prompts for the dumps: GSM8K test questions in the checkpoint's chat template (+ gold solution as continuation text so every row
has exactly S real tokens), and one long book-style text. Deterministic; sha256 of the token matrix is logged."""
import hashlib
import json
import os

import torch

GSM8K = "/mnt/tt-data/ssinghal/datasets/gsm8k_test.jsonl"


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        os.environ.get("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL"), trust_remote_code=True
    )


def gsm8k_tokens(B, S, start=0, tok=None, steps=0):
    """Returns (tokens [B,S], texts) or, with steps>0, (tokens, texts, dec_tokens [B,steps]) = the next real tokens (teacher-forced decode)."""
    tok = tok or tokenizer()
    rows = [json.loads(l) for l in open(GSM8K)]
    out, texts, dec = [], [], []
    i = start
    while len(out) < B:
        r = rows[i]
        i += 1
        txt = tok.apply_chat_template(
            [{"role": "user", "content": r["question"]}], tokenize=False, add_generation_prompt=True
        )
        txt = txt + r["answer"]  # gold chain-of-thought as real continuation text
        ids = tok(txt, add_special_tokens=False)["input_ids"]
        if len(ids) < S + steps:
            continue
        out.append(ids[:S])
        dec.append(ids[S : S + steps])
        texts.append(txt)
    t = torch.tensor(out)
    if steps:
        return t, texts, torch.tensor(dec)
    return t, texts


def sha256_tokens(t):
    return hashlib.sha256(t.to(torch.int64).numpy().tobytes()).hexdigest()
