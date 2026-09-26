# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Intake gate, part 2: HF itself behaves sensibly on the inputs the bring-up will use.

    python -m models.demos.common.bringup.intake.check_hf_sanity --spec S

1. Revision: the local checkpoint's commit (from the HF download metadata) is recorded; if the spec pins
   ``hf.revision``, it must match (``revision_ok``).
2. Smoke: ``intake.smoke: {prompt, expect}`` (the model card's usage example) is run through the chat template if the
   tokenizer has one, greedy, and the answer must contain ``expect`` (``smoke_ok``).
3. Plausibility: HF top-1 next-token accuracy on the first ``intake.accuracy_tokens`` (default 2048) tokens of the
   canonical prompt (``text_top1_acc``). An instruction-tuned model fed raw text scores far below a real model
   reciting a book (Gemma-4-it: 18% raw vs 67% as a model turn); the ledger gates it at ``text.min_top1`` (0.4).

bf16 on CPU; the model is loaded once. Hooks the model may provide: ``hf_model(spec, num_layers)``.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference import prompt
from models.demos.common.bringup.reference.golden import hf_path, load_spec, tokenizer


def local_revision(hf_dir: str) -> str | None:
    """Commit sha of a snapshot_download(local_dir=...) checkout, from its download metadata, or of a hub-cache path."""
    meta = Path(hf_dir) / ".cache" / "huggingface" / "download"
    for f in sorted(meta.glob("*.metadata")) if meta.exists() else []:
        first = f.read_text().splitlines()[:1]
        if first and re.fullmatch(r"[0-9a-f]{40}", first[0].strip()):
            return first[0].strip()
    m = re.search(r"/snapshots/([0-9a-f]{40})", str(Path(hf_dir).resolve()))
    return m.group(1) if m else None


def load_model(spec):
    hooks = spec.hooks()
    if hasattr(hooks, "hf_model"):
        return hooks.hf_model(spec, None)
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        hf_path(spec), dtype=torch.bfloat16, trust_remote_code=bool(spec.get("hf.trust_remote_code"))
    ).eval()


def smoke(model, tok, prompt_text: str, n: int = 24, template_kwargs: dict | None = None) -> str:
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=False,
            **(template_kwargs or {}),
        )
    else:
        text = prompt_text
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]])
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False)
    return tok.decode(out[0][ids.shape[1] :])


def next_token_acc(model, ids: torch.Tensor) -> float:
    with torch.no_grad():
        logits = model(ids[None]).logits[0]
    return (logits[:-1].argmax(-1) == ids[1:]).float().mean().item()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    a = ap.parse_args(argv)
    torch.set_num_threads(os.cpu_count())
    spec = load_spec(a.spec)
    tok = tokenizer(spec)

    try:
        rev = local_revision(hf_path(spec))
    except Exception:
        rev = None
    pinned = spec.get("hf.revision")
    rev_ok = int(pinned is None or rev == pinned)
    print(f"checkpoint revision {rev} (spec pins {pinned})")
    metrics.record("revision_ok", rev_ok)
    metrics.record("revision_pinned", int(pinned is not None))

    model = load_model(spec)
    sm = spec.get("intake.smoke")
    if sm:
        answer = smoke(model, tok, sm["prompt"], template_kwargs=prompt.template_kwargs(spec))
        ok = sm["expect"].lower() in answer.lower()
        print(f"smoke: {sm['prompt']!r} -> {answer!r} (expect {sm['expect']!r}): {'ok' if ok else 'FAIL'}")
        metrics.record("smoke_ok", int(ok))
    rec = prompt.load(spec)
    n = min(int(spec.get("intake.accuracy_tokens", 2048)), rec["n"] if rec else prompt.default_length(spec))
    acc = next_token_acc(model, prompt.tokens(spec, n, tok))
    print(f"HF top-1 next-token accuracy on the first {n} prompt tokens: {acc:.3f}")
    metrics.record("text_top1_acc", acc)


if __name__ == "__main__":
    main()
