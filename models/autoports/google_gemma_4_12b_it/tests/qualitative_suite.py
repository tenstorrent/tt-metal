# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared qualitative prompt suite (the plugin's ``vllm_prompts.txt``) in chat format: HF control and TT generator.

``hf`` (CPU, no device): renders every prompt with the checkpoint's chat template (``add_generation_prompt=True``),
writes the prompt-format metadata, rendered prompts and HF greedy completions.
``tt`` is called from ``readiness_run.py --checks qual`` with the live generator: the same token ids through
``generator.generate`` as one batch (one decode row per prompt) and one prompt at a time, greedy, on-device sampling.

  python models/autoports/google_gemma_4_12b_it/tests/qualitative_suite.py hf --max-new-tokens 128
"""

import argparse
import json
import os
from pathlib import Path

import torch

# stage evidence directory (GEMMA4_STAGE_DOC=doc/<stage>, default the full-model stage)
OUT = Path(__file__).resolve().parents[1] / os.environ.get("GEMMA4_STAGE_DOC", "doc/full_model") / "qualitative"
HF_MODEL = "google/gemma-4-12B-it"
HF_REVISION = os.environ.get("HF_REVISION", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7")


def prompt_source():
    root = os.environ.get("TT_MODEL_BRINGUP_ROOT")
    assert root, "TT_MODEL_BRINGUP_ROOT is not set"
    return Path(root) / "runtime/readiness_check/vllm_prompts.txt"


def load_prompts():
    text = prompt_source().read_text(encoding="utf-8")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def render(tokenizer, prompts):
    out = []
    for i, p in enumerate(prompts):
        msgs = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False)
        out.append({"id": f"p{i}", "prompt": p, "rendered": text, "token_ids": [int(t) for t in ids]})
    return out


def hf_main(max_new_tokens):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(16)
    tok = AutoTokenizer.from_pretrained(HF_MODEL, revision=HF_REVISION)
    OUT.mkdir(parents=True, exist_ok=True)
    items = render(tok, load_prompts())
    meta = {
        "hf_model": HF_MODEL,
        "revision": HF_REVISION,
        "tokenizer_class": type(tok).__name__,
        "chat_template_present": bool(getattr(tok, "chat_template", None)),
        "prompt_mode": "chat",
        "rendering": "tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True)",
        "generation": {"do_sample": False, "max_new_tokens": max_new_tokens, "stop": "eos ids of generation_config"},
        "prompt_source": str(prompt_source()),
    }
    (OUT / "qualitative_prompt_format.json").write_text(json.dumps(meta, indent=1))
    (OUT / "rendered_prompts.json").write_text(json.dumps(items, indent=1))
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, revision=HF_REVISION, dtype=torch.bfloat16).eval()
    res = []
    for it in items:
        with torch.no_grad():
            out = model.generate(
                torch.tensor([it["token_ids"]]), max_new_tokens=max_new_tokens, do_sample=False, num_beams=1
            )
        ids = out[0, len(it["token_ids"]) :].tolist()
        res.append({"id": it["id"], "token_ids": ids, "text": tok.decode(ids, skip_special_tokens=False)})
        print(it["id"], repr(res[-1]["text"][:200]), flush=True)
    (OUT / "hf_outputs.json").write_text(json.dumps(res, indent=1))


def tt_run(gen, max_new_tokens=128):
    """Run the suite through a live generator; returns and writes the TT outputs (batch and one-at-a-time)."""
    items = json.loads((OUT / "rendered_prompts.json").read_text())
    hf = {r["id"]: r for r in json.loads((OUT / "hf_outputs.json").read_text())}
    tok = gen.tokenizer
    batch = gen.generate([it["token_ids"] for it in items], max_new_tokens)
    gen.reset()
    res = []
    for it, b_ids in zip(items, batch):
        single = gen.generate(it["token_ids"], max_new_tokens)
        gen.reset()
        h = hf[it["id"]]["token_ids"]
        first_div = next((i for i, (x, y) in enumerate(zip(single, h)) if x != y), min(len(single), len(h)))
        res.append(
            {
                "id": it["id"],
                "prompt": it["prompt"],
                "tt_token_ids": single,
                "tt_text": tok.decode(single, skip_special_tokens=False),
                "tt_batch_token_ids": b_ids,
                "batch_equals_single": b_ids == single,
                "hf_text": hf[it["id"]]["text"],
                "first_divergence_vs_hf": first_div,
                "tokens_tt_hf": [len(single), len(h)],
            }
        )
        print(it["id"], "first divergence", first_div, "batch==single", b_ids == single, flush=True)
    (OUT / "tt_outputs.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hf"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    a = ap.parse_args()
    hf_main(a.max_new_tokens)
