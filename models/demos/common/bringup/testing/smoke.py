# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The intake smoke on the device: the spec's ``intake.smoke`` prompt (the model card's usage example, checked on the
CPU by R.1) through the bring-up device model with its final norm and LM head, greedy, a few tokens. Prefill only:
each new token is appended and the chunk prefilled again (prompts are short, one chunk). The answer must contain
``expect``. Runs on the shipped defaults (an end-to-end check, not a component test).

Metrics: smoke_device_ok, smoke_device_tokens."""

from __future__ import annotations

import time

import torch

from models.demos.common.bringup.core import metrics


def prompt_ids(s, tok) -> list[int]:
    from models.demos.common.bringup.reference.prompt import template_kwargs

    text = s.data["intake"]["smoke"]["prompt"]
    if getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, **template_kwargs(s)
        )
    return tok(text, add_special_tokens=False)["input_ids"]


def tokenizer(s):
    from transformers import AutoTokenizer

    from models.demos.common.bringup.reference.golden import hf_path

    return AutoTokenizer.from_pretrained(str(hf_path(s)), trust_remote_code=bool(s.get("hf.trust_remote_code", False)))


def run_smoke(s, mesh, max_new: int = 8) -> dict:
    tok = tokenizer(s)
    ids = prompt_ids(s, tok)
    chunk = int(s.get("target.chunk"))
    assert len(ids) + max_new <= chunk, f"smoke prompt of {len(ids)} tokens does not fit one {chunk}-token chunk"
    layers = s.layers()
    assert layers[-1] == s.num_layers - 1 and layers[0] == 0, "the smoke needs the whole model (embedding to LM head)"
    model = s.hooks().device_model(mesh, s, layers, lm_head=True)
    eos = {t for t in (tok.eos_token_id,) if t is not None}
    out = []
    for step in range(max_new):
        seq = ids + out
        tokens = torch.tensor(seq + [0] * (chunk - len(seq)), dtype=torch.long)  # pad after the prompt: causal
        state = model.new_state(chunk)
        t0 = time.time()
        h = model.embed(tokens)
        for i in layers:
            h2 = model.layer(i, h, 0, state)
            model.free(h)
            h = h2
        hidden = model.final_norm(h)
        model.free(h)
        nxt = int(model.logits(hidden, [len(seq) - 1]).float()[0].argmax())
        model.free(hidden)
        print(f"smoke step {step}: token {nxt} {tok.decode([nxt])!r} ({time.time() - t0:.1f}s)", flush=True)
        if nxt in eos:
            break
        out.append(nxt)
        text = tok.decode(out)
        if s.data["intake"]["smoke"]["expect"].lower() in text.lower() and step >= 1:
            break
    text = tok.decode(out)
    ok = s.data["intake"]["smoke"]["expect"].lower() in text.lower()
    print(
        f"smoke on device: {s.data['intake']['smoke']['prompt']!r} -> {text!r} (expect "
        f"{s.data['intake']['smoke']['expect']!r}): {'ok' if ok else 'FAIL'}"
    )
    metrics.record("smoke_device_ok", int(ok))
    metrics.record("smoke_device_tokens", len(out))
    return {"ok": ok, "text": text}
