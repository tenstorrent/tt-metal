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


BOOK = "/mnt/tt-data/ssinghal/tests/tt-metal/models/tt_transformers/demo/context_cache/4c1a705addfd44fe41b8ce06d83e3fd5"  # Project Gutenberg Frankenstein (public domain), read-only
BOOK_SHA256 = "7810cd483cffcf2cc8a1d8f0d5807931e69d4f48cd14149b8c76f88af82fead3"
BOOK_QUESTION = "\n\nExplicitly state the quotes directly taken from the book inside double quotes literally, and summarise the story."


def book_tokens(B, S, start=0, tok=None, steps=0):
    """B rows of EXACTLY S real tokens: chat template [user: <book window> + question] + generation prompt; windows are disjoint slices of the book text
    (after the Gutenberg START marker), window b starts at book token start + b*S. With steps>0 also returns the next `steps` book tokens per row (teacher-forced
    decode tokens)."""
    tok = tok or tokenizer()
    text = open(BOOK).read()
    text = text[text.index("*** START OF THE PROJECT GUTENBERG EBOOK") :]
    body = tok(text, add_special_tokens=False)["input_ids"]
    probe = tok.apply_chat_template([{"role": "user", "content": "XQXQ"}], tokenize=False, add_generation_prompt=True)
    pre_txt, post_txt = probe.split("XQXQ")
    pre = tok(pre_txt, add_special_tokens=False)["input_ids"]
    post = tok(BOOK_QUESTION + post_txt, add_special_tokens=False)["input_ids"]
    n = S - len(pre) - len(post)
    rows, dec, texts = [], [], []
    for b in range(B):
        o = start + b * S
        rows.append(pre + body[o : o + n] + post)
        dec.append(body[o + n : o + n + steps])
        texts.append(tok.decode(rows[-1]))
    t = torch.tensor(rows)
    assert t.shape == (B, S)
    return (t, texts, torch.tensor(dec)) if steps else (t, texts)
