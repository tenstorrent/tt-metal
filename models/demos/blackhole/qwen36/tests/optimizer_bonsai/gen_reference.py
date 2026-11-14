# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU HF reference for the Bonsai optimizer gate (test_bonsai_pcc.py).

Sequence: the start of Frankenstein (Project Gutenberg pg84, the text the qwen36 demo's long prompts use), tokenized
as plain text: the first ISL tokens are the prompt, the next GEN tokens the teacher-forced continuation.
Logits: one bf16 forward over prompt + continuation[:-1] on the CPU (the 27B checkpoint does not fit this host in
fp32); row i is the distribution at position ISL-1+i, which predicts continuation[i].

Run once on the host, from the tt-metal root, with HF_MODEL naming the converted dense checkpoint:
    python models/demos/blackhole/qwen36/tests/optimizer_bonsai/gen_reference.py
"""

from __future__ import annotations

import os
import time

import requests
import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from models.demos.blackhole.qwen36.tests.optimizer_bonsai.bonsai_support import GEN, ISL, REFERENCE, TEXT_URL


def main():
    model_dir = os.environ["HF_MODEL"]
    text_path = REFERENCE.parent / "pg84.txt"
    if not text_path.is_file():
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(requests.get(TEXT_URL, timeout=120).text)
    text = text_path.read_text()
    start = text.find("*** START OF")
    text = text[text.find("\n", start) + 1 :] if start >= 0 else text

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][: ISL + GEN]
    assert len(ids) == ISL + GEN, len(ids)
    prompt, cont = ids[:ISL], ids[ISL:]
    print(f"prompt {ISL} tokens, continuation {GEN}: {tokenizer.decode(cont)[:200]!r}", flush=True)

    t0 = time.time()
    config = Qwen3_5TextConfig.from_pretrained(model_dir)
    model = Qwen3_5ForCausalLM.from_pretrained(model_dir, config=config, dtype=torch.bfloat16).eval()
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    with torch.no_grad():
        t1 = time.time()
        logits = model(torch.tensor([prompt + cont[:-1]]), logits_to_keep=GEN).logits[0].float()
    assert logits.shape[0] == GEN, logits.shape
    agree = int((logits.argmax(-1) == torch.tensor(cont)).sum())
    print(f"forward in {time.time() - t1:.0f} s; reference argmax == text at {agree}/{GEN} positions", flush=True)

    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model_dir,
            "text_url": TEXT_URL,
            "prompt_ids": prompt,
            "continuation_ids": cont,
            "logits": logits.contiguous(),
            "dtype": "bfloat16 forward, logits as float32",
        },
        REFERENCE,
    )
    print(f"saved {REFERENCE}", flush=True)


if __name__ == "__main__":
    main()
