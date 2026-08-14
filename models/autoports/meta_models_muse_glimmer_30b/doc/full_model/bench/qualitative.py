# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared qualitative prompt suite through the TT full model and the HF control.

``$qualitative-check`` requires the prompt format the checkpoint declares.  This
tokenizer has a non-empty chat template, so the model is chat/instruct and every
prompt in the shared suite (``models/common/readiness_check/vllm_prompts.txt``) is
rendered with ``apply_chat_template(add_generation_prompt=True)``.  A raw
continuation arm is available (``--mode completion``) but only as labelled stress
coverage, never as the verdict.

The two arms are separate invocations on purpose: the HF control is a 56 GB CPU
model and the TT arm holds the mesh, and neither should be resident while the
other runs.

Artifacts written under ``doc/full_model/qualitative/``:

* ``qualitative_prompt_format.json`` -- the prompt-format decision and its evidence;
* ``qualitative_prompts.json`` -- rendered prompt text and token ids per prompt id;
* ``qualitative_hf.json`` / ``qualitative_tt.json`` -- completions per prompt id.

Usage::

    python doc/full_model/bench/qualitative.py --arm hf --max-new-tokens 128
    python doc/full_model/bench/qualitative.py --arm tt --max-new-tokens 128
    python doc/full_model/bench/qualitative.py --arm compare
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]  # models/autoports/<model>/
REPO = ROOT.parents[2]  # the tt-metal checkout
sys.path.insert(0, str(REPO))

SUITE = REPO / "models/common/readiness_check/vllm_prompts.txt"
OUT = ROOT / "doc/full_model/qualitative"


def say(*args) -> None:
    print(*args, flush=True)


def load_suite() -> list[str]:
    text = SUITE.read_text(encoding="utf-8")
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def resolve_snapshot(model_id: str) -> str:
    repo = pathlib.Path(__import__("huggingface_hub").constants.HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    for index_path in sorted(repo.glob("snapshots/*/model.safetensors.index.json")):
        snapshot = index_path.parent
        shards = set(json.loads(index_path.read_text())["weight_map"].values())
        if all((snapshot / shard).exists() for shard in shards):
            return str(snapshot)
    raise FileNotFoundError(f"no complete snapshot of {model_id}")


def render(tokenizer, prompts: list[str], *, mode: str) -> list[dict]:
    rendered = []
    for index, prompt in enumerate(prompts):
        if mode == "chat":
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
            )
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            if len(ids) and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
            ids = [int(i) for i in ids]
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
            )
        else:
            ids = [int(i) for i in tokenizer.encode(prompt, add_special_tokens=True)]
            text = prompt
        rendered.append({"id": f"p{index}", "prompt": prompt, "rendered": text, "token_ids": ids})
    return rendered


def compare(mode: str) -> int:
    """Mechanical HF-vs-TT comparison, so the verdict cites numbers not impressions.

    Four metrics per prompt, the first three being the ones a decode-loop bug moves:

    * **adjacent token duplication** -- the signature of stale token/position
      feedback (``check_degenerate_output.py`` calls >10 % mechanical);
    * **trigram loop coverage** -- phrase looping, which greedy decoding does
      legitimately produce, so it is advisory;
    * **non-ASCII fraction** -- wrong-language drift;
    * **first divergence from the HF control**, in tokens. Greedy TT and greedy HF
      diverge eventually (bf16 vs bf16-on-a-different-reduction-order), and *where*
      is the interesting number: divergence at token 0-2 is a wrapper bug, late
      divergence with both texts coherent is ordinary numerics.
    """
    hf_path = OUT / f"qualitative_hf_{mode}.json"
    tt_path = OUT / f"qualitative_tt_{mode}.json"
    if not (hf_path.is_file() and tt_path.is_file()):
        print(f"need both {hf_path.name} and {tt_path.name}", flush=True)
        return 2
    hf = {item["id"]: item for item in json.loads(hf_path.read_text())}
    tt = {item["id"]: item for item in json.loads(tt_path.read_text())}

    def adjacent_dup(tokens):
        if len(tokens) < 2:
            return 0.0
        return sum(1 for a, b in zip(tokens, tokens[1:]) if a == b) / (len(tokens) - 1)

    def trigram_loop(tokens):
        if len(tokens) < 3:
            return 0.0
        counts: dict = {}
        for index in range(len(tokens) - 2):
            gram = tuple(tokens[index : index + 3])
            counts[gram] = counts.get(gram, 0) + 1
        top = max(counts, key=counts.get)
        covered = 0
        index = 0
        while index <= len(tokens) - 3:
            if tuple(tokens[index : index + 3]) == top:
                covered += 3
                index += 3
            else:
                index += 1
        return covered / len(tokens)

    def non_ascii(text):
        return sum(1 for ch in text if ord(ch) > 127) / max(len(text), 1)

    rows = []
    for key in sorted(tt):
        tt_tokens, hf_tokens = tt[key]["token_ids"], hf.get(key, {}).get("token_ids", [])
        first_diff = next((i for i, (a, b) in enumerate(zip(tt_tokens, hf_tokens)) if a != b), None)
        row = {
            "id": key,
            "tt_tokens": len(tt_tokens),
            "tt_adjacent_dup": round(adjacent_dup(tt_tokens), 4),
            "hf_adjacent_dup": round(adjacent_dup(hf_tokens), 4),
            "tt_trigram_loop": round(trigram_loop(tt_tokens), 4),
            "hf_trigram_loop": round(trigram_loop(hf_tokens), 4),
            "tt_non_ascii": round(non_ascii(tt[key]["completion"]), 4),
            "hf_non_ascii": round(non_ascii(hf.get(key, {}).get("completion", "")), 4),
            "first_divergence_from_hf": -1 if first_diff is None else first_diff,
            "exact_match": tt_tokens == hf_tokens,
        }
        rows.append(row)
        say(f"CMP {row}")
    (OUT / f"qualitative_comparison_{mode}.json").write_text(json.dumps(rows, indent=2) + "\n")
    worst_dup = max(row["tt_adjacent_dup"] for row in rows)
    say(f"CMP_OK prompts={len(rows)} worst_tt_adjacent_dup={worst_dup} (critical threshold 0.10)")
    return 0 if worst_dup <= 0.10 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("hf", "tt", "compare"), required=True)
    parser.add_argument("--mode", choices=("chat", "completion"), default="chat")
    parser.add_argument("--hf-model", default="meta-models/Muse-Glimmer-30B")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=131072)
    parser.add_argument("--layers", default="all")
    args = parser.parse_args()

    if args.arm == "compare":
        return compare(args.mode)

    from transformers import AutoTokenizer

    snapshot = resolve_snapshot(args.hf_model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    prompts = load_suite()
    rendered = render(tokenizer, prompts, mode=args.mode)
    OUT.mkdir(parents=True, exist_ok=True)

    (OUT / "qualitative_prompt_format.json").write_text(
        json.dumps(
            {
                "hf_model": args.hf_model,
                "hf_revision": pathlib.Path(snapshot).name,
                "tokenizer_class": type(tokenizer).__name__,
                "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
                "prompt_mode": args.mode,
                "prompt_mode_reason": (
                    "the tokenizer declares a non-empty chat_template, so the checkpoint is "
                    "chat/instruct and the suite is rendered with apply_chat_template"
                ),
                "rendering_method": "tokenizer.apply_chat_template(add_generation_prompt=True)",
                "prompt_source": str(SUITE.relative_to(REPO)),
                "generation": {"greedy": True, "max_new_tokens": args.max_new_tokens},
                "chat_template_note": (
                    "the rendered system message embeds the current date, so the exact prompt text is "
                    "date-dependent; the token ids recorded in qualitative_prompts.json are what both arms ran"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    (OUT / "qualitative_prompts.json").write_text(json.dumps(rendered, indent=2) + "\n")

    results = []
    if args.arm == "hf":
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.muse_glimmer.configuration_muse_glimmer import MuseGlimmerConfig
        from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerForConditionalGeneration

        AutoModelForCausalLM._model_mapping._extra_content[MuseGlimmerConfig] = MuseGlimmerForConditionalGeneration
        model = (
            AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16, local_files_only=True).eval().to("cpu")
        )
        for item in rendered:
            started = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    torch.tensor([item["token_ids"]], dtype=torch.long),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            generated = out[0, len(item["token_ids"]) :].tolist()
            results.append(
                {
                    "id": item["id"],
                    "prompt": item["prompt"],
                    "token_ids": [int(t) for t in generated],
                    "completion": tokenizer.decode(generated, skip_special_tokens=False),
                    "seconds": round(time.perf_counter() - started, 1),
                }
            )
            say(f"QUAL hf {item['id']} {len(generated)} tokens in {results[-1]['seconds']}s")
            say(f"QUAL hf {item['id']} :: {results[-1]['completion'][:300]!r}")
        (OUT / f"qualitative_hf_{args.mode}.json").write_text(json.dumps(results, indent=2) + "\n")
    else:
        from models.autoports.meta_models_muse_glimmer_30b.tt.generator import (
            DEFAULT_TRACE_REGION_SIZE,
            build_generator,
        )
        from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
            close_multichip_mesh,
            open_multichip_mesh,
        )

        layer_indices = None if args.layers == "all" else [int(i) for i in args.layers.split(",")]
        mesh = open_multichip_mesh(trace_region_size=DEFAULT_TRACE_REGION_SIZE)
        generator = None
        try:
            generator = build_generator(
                ROOT, mesh, max_seq_len=args.max_seq_len, max_batch_size=1, layer_indices=layer_indices
            )
            for position, item in enumerate(rendered):
                if position:
                    generator.reset()
                started = time.perf_counter()
                generated = generator.generate(
                    prompt_token_ids=item["token_ids"],
                    max_new_tokens=args.max_new_tokens,
                    enable_trace=True,
                )
                results.append(
                    {
                        "id": item["id"],
                        "prompt": item["prompt"],
                        "token_ids": [int(t) for t in generated],
                        "completion": tokenizer.decode(generated, skip_special_tokens=False),
                        "seconds": round(time.perf_counter() - started, 1),
                        "counters": dict(generator.counters),
                    }
                )
                say(f"QUAL tt {item['id']} {len(generated)} tokens in {results[-1]['seconds']}s")
                say(f"QUAL tt {item['id']} :: {results[-1]['completion'][:300]!r}")
            (OUT / f"qualitative_tt_{args.mode}.json").write_text(json.dumps(results, indent=2) + "\n")
        finally:
            if generator is not None:
                generator.teardown()
            close_multichip_mesh(mesh)

    say(f"QUAL_OK arm={args.arm} mode={args.mode} prompts={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
