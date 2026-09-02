# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared qualitative prompt suite through HF and the TT full model.

Implements the ``$qualitative-check`` contract for GLM-4.7-Flash: the
checkpoint ships a chat template, so every prompt is rendered with
``tokenizer.apply_chat_template(..., add_generation_prompt=True)`` and the HF
control uses the identical token ids. Prompt-format metadata, rendered prompts,
prompt token ids, HF control outputs and TT outputs all land under
``doc/full_model/qualitative/``.

    python models/autoports/zai_org_glm_4_7_flash/tests/run_qualitative_suite.py \\
        --max-new-tokens 128

``--skip-hf`` reuses a previous HF control run.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import build_generator
from models.autoports.zai_org_glm_4_7_flash.tt.model import resolve_checkpoint_dir, source_manifest

MODEL_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = MODEL_DIR / "doc" / "full_model" / "qualitative"
PROMPT_FILE = Path("models/common/readiness_check/vllm_prompts.txt")
HF_MODEL_ID = "zai-org/GLM-4.7-Flash"


def load_prompts():
    text = PROMPT_FILE.read_text(encoding="utf-8")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def render(tokenizer, prompt):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
    )
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return text, [int(i) for i in ids]


def hf_control(snapshot, rendered, max_new_tokens):
    from transformers import AutoModelForCausalLM

    print("loading HF reference on cpu...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True).eval()
    outs = []
    for i, (_text, ids) in enumerate(rendered):
        t0 = time.perf_counter()
        input_ids = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=model.config.pad_token_id,
            )
        gen = out[0, input_ids.shape[1] :].tolist()
        outs.append(gen)
        print(f"  hf prompt {i}: {len(gen)} tokens in {time.perf_counter() - t0:.1f}s", flush=True)
    del model
    gc.collect()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--skip-hf", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    snapshot = resolve_checkpoint_dir()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    prompts = load_prompts()
    rendered = [render(tokenizer, p) for p in prompts]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fmt = {
        "source_manifest": source_manifest([__file__]),
        "hf_model_id": HF_MODEL_ID,
        "hf_snapshot_revision": snapshot.name,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_present": (snapshot / "chat_template.jinja").is_file()
        or bool(getattr(tokenizer, "chat_template", None)),
        "prompt_mode": "chat",
        "rendering": "tokenizer.apply_chat_template(messages, add_generation_prompt=True)",
        "prompt_source": str(PROMPT_FILE),
        "num_prompts": len(prompts),
        "generation": {"greedy": True, "max_new_tokens": args.max_new_tokens},
        "note": (
            "GLM-4.7-Flash is a reasoning/instruct checkpoint: the chat template ends in "
            "<|assistant|><think>, so completions open with a reasoning trace before the answer. "
            "Raw-completion prompts would not be a valid quality verdict for this checkpoint."
        ),
    }
    (OUT_DIR / "qualitative_prompt_format.json").write_text(json.dumps(fmt, indent=2) + "\n")

    hf_path = OUT_DIR / "hf_control.json"
    if args.skip_hf and hf_path.is_file():
        hf_tokens = json.loads(hf_path.read_text())["completions"]
    else:
        hf_tokens = hf_control(snapshot, rendered, args.max_new_tokens)
        hf_path.write_text(
            json.dumps(
                {
                    "hf_model_id": HF_MODEL_ID,
                    "max_new_tokens": args.max_new_tokens,
                    "completions": hf_tokens,
                    "texts": [tokenizer.decode(t, skip_special_tokens=False) for t in hf_tokens],
                },
                indent=2,
            )
            + "\n"
        )

    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=350_000_000)
    try:
        gen = build_generator(MODEL_DIR, dev, tokenizer=tokenizer)
        tt_tokens = []
        for i, (_text, ids) in enumerate(rendered):
            gen.reset()
            t0 = time.perf_counter()
            out = gen.generate(ids, args.max_new_tokens, enable_trace=True)
            tt_tokens.append(out)
            print(f"  tt prompt {i}: {len(out)} tokens in {time.perf_counter() - t0:.1f}s", flush=True)
        gen.teardown()
    finally:
        ttnn.close_mesh_device(dev)

    records = []
    for i, (prompt, (text, ids)) in enumerate(zip(prompts, rendered)):
        records.append(
            {
                "id": i,
                "prompt": prompt,
                "rendered_prompt": text,
                "prompt_token_ids": ids,
                "hf_completion": tokenizer.decode(hf_tokens[i], skip_special_tokens=False),
                "tt_completion": tokenizer.decode(tt_tokens[i], skip_special_tokens=False),
                "hf_token_ids": hf_tokens[i],
                "tt_token_ids": tt_tokens[i],
                "exact_prefix_match_tokens": _common_prefix(hf_tokens[i], tt_tokens[i]),
            }
        )
    (OUT_DIR / "qualitative_outputs.json").write_text(json.dumps(records, indent=2) + "\n")

    lines = []
    for rec in records:
        lines.append("=" * 100)
        lines.append(f"[{rec['id']}] {rec['prompt']}")
        lines.append(
            f"    (greedy prefix agreement: {rec['exact_prefix_match_tokens']}/{len(rec['hf_token_ids'])} tokens)"
        )
        lines.append("-" * 40 + " HF " + "-" * 40)
        lines.append(rec["hf_completion"])
        lines.append("-" * 40 + " TT " + "-" * 40)
        lines.append(rec["tt_completion"])
        lines.append("")
    (OUT_DIR / "qualitative_side_by_side.txt").write_text("\n".join(lines))
    print(f"wrote {OUT_DIR}")
    for rec in records:
        print(
            f"  prompt {rec['id']}: greedy prefix agreement {rec['exact_prefix_match_tokens']}/{len(rec['hf_token_ids'])}"
        )


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


if __name__ == "__main__":
    main()
