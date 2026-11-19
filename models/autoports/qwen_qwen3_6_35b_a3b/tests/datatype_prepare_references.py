# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prepare pinned CPU HF natural-text reference and P1 reasoning answers.

This helper never imports TTNN. Run it serially with performance measurements.
The natural-text fixture deliberately uses raw literary continuation as stress
coverage; reasoning uses the checkpoint's chat template with thinking enabled.
"""

import argparse
import bz2
import json
import os
import sys
import time
from pathlib import Path

MODEL = "Qwen/Qwen3.6-35B-A3B"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("natural", "reasoning", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc/datatype_sweep/references")
    parser.add_argument("--book", type=Path, default=REPO / "models/ttt_compat/tests/tale-of-two-cities.txt.bz2")
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--natural-positions", type=int, default=768)
    parser.add_argument("--num-texts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--min-reasoning-positions", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--prepare-only", action="store_true", help="Render prompts, validate paths; do not load HF weights"
    )
    args = parser.parse_args()
    assert args.prompt_length > 0 and args.natural_positions >= 300
    assert args.max_new_tokens >= 6144, "Do not lower the P1 answer cap"
    assert args.min_reasoning_positions >= 1000
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    snapshot = Path(os.environ["HF_HOME"]) / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots" / REVISION
    assert (snapshot / "model.safetensors.index.json").is_file(), snapshot
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    assert tokenizer.chat_template
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    common = dict(
        hf_model_id=MODEL,
        revision=REVISION,
        checkpoint=str(snapshot),
        tokenizer_class=type(tokenizer).__name__,
        chat_template_present=True,
        torch=torch.__version__,
        transformers=transformers.__version__,
        device="cpu",
        threads=args.threads,
        command=sys.argv,
    )
    natural = None
    if args.mode in ("natural", "both"):
        book = bz2.decompress(args.book.read_bytes()).decode("utf-8")
        # Skip Gutenberg front matter when the standard title marker exists.
        marker = "It was the best of times"
        start = book.find(marker)
        assert start >= 0, "Expected the opening novel text in the readiness book"
        text = book[start:]
        ids = tokenizer.encode(text, add_special_tokens=False)
        n = args.prompt_length + args.natural_positions
        assert len(ids) >= n
        natural = torch.tensor([ids[:n]], dtype=torch.long)
        natural_meta = dict(
            **common,
            prompt_mode="raw literary continuation stress; not a chat quality verdict",
            source=str(args.book),
            source_start_character=start,
            prompt_length=args.prompt_length,
            positions=args.natural_positions,
            continuation_source="real next tokens in the original book; not HF generated",
            prompt_token_ids=ids[: args.prompt_length],
            completion_token_ids=ids[args.prompt_length : n],
        )
        save_json(out / "natural_text.provenance.json", natural_meta)
    settings = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
        use_cache=True,
    )
    prompts = []
    if args.mode in ("reasoning", "both"):
        suite = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime/readiness_check/reasoning_prompts.json"
        prompt_data = json.loads(suite.read_text())["prompts"]
        assert 1 <= args.num_texts <= len(prompt_data)
        for index, q in enumerate(prompt_data[: args.num_texts]):
            rendered = tokenizer.apply_chat_template(
                q["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True, thinking=True
            )
            prompts.append(
                dict(
                    id=q["id"],
                    seed=index,
                    messages=q["messages"],
                    rendered_prompt=rendered,
                    prompt_token_ids=tokenizer.encode(rendered, add_special_tokens=False),
                )
            )
        metadata = dict(
            **common,
            prompt_mode="chat",
            prompt_source=str(suite),
            chat_template_kwargs={"enable_thinking": True, "thinking": True},
            generation=settings,
            presence_penalty=1.5,
            presence_penalty_scope="generated tokens only",
            prompts=prompts,
        )
        path = out / "reasoning.provenance.json"
        if path.exists():
            old = json.loads(path.read_text())
            for key in ("hf_model_id", "revision", "chat_template_kwargs", "generation", "presence_penalty", "prompts"):
                assert (
                    old[key] == metadata[key]
                ), f"Existing reasoning controls differ in {key}; use a new output directory"
        save_json(path, metadata)
    if args.prepare_only:
        print(f"PREPARED: natural={natural is not None}, reasoning_prompts={len(prompts)}; no model weights loaded")
        return
    print(f"Loading pinned CPU HF weights from {snapshot}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, dtype=torch.bfloat16).eval()
    assert next(model.parameters()).device.type == "cpu"
    print(f"HF class={type(model).__name__}, dtype={model.dtype}", flush=True)
    if natural is not None:
        from readiness_check.schema import Reference, ReferenceEntry, save_reference

        before = time.monotonic()
        with torch.inference_mode():
            # Full-sequence HF forward matches the main readiness reference.
            logits = model(input_ids=natural, use_cache=False).logits
            prediction = logits[0, args.prompt_length - 1 : -1]
            assert prediction.shape[0] == args.natural_positions
            topk = prediction.topk(100, dim=-1).indices.to(torch.int32).cpu()
        entry = ReferenceEntry(
            prompt_text=tokenizer.decode(natural[0, : args.prompt_length]),
            prompt_tokens=natural[:, : args.prompt_length],
            generated_tokens=natural[:, args.prompt_length :],
            topk_tokens=topk,
            tf_prompt_len=args.prompt_length,
        )
        save_reference(
            Reference(
                k=100,
                hf_model_id=MODEL,
                entries=[entry],
                token_ids_meta={
                    "bos_id": tokenizer.bos_token_id,
                    "eos_id": tokenizer.eos_token_id,
                    "pad_id": tokenizer.pad_token_id,
                },
            ),
            out / "natural_text.refpt",
        )
        natural_meta.update(
            elapsed_s=time.monotonic() - before, model_class=type(model).__name__, dtype=str(model.dtype)
        )
        save_json(out / "natural_text.provenance.json", natural_meta)
        del logits, prediction
        print(f"NATURAL_REFERENCE_OK positions={args.natural_positions}", flush=True)

    class PresencePenalty(LogitsProcessor):
        def __init__(self, prompt_len):
            self.prompt_len = prompt_len

        def __call__(self, input_ids, scores):
            for row in range(input_ids.shape[0]):
                scores[row, input_ids[row, self.prompt_len :].unique()] -= 1.5
            return scores

    text_path = out / "reasoning_text.jsonl"
    rows = [json.loads(line) for line in text_path.read_text().splitlines() if line] if text_path.exists() else []
    done = {row["prompt_id"] for row in rows}
    assert len(done) == len(rows), "Duplicate reasoning answer id"
    for q in prompts:
        if q["id"] in done:
            continue
        torch.manual_seed(q["seed"])
        ids = torch.tensor([q["prompt_token_ids"]], dtype=torch.long)
        before = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                logits_processor=LogitsProcessorList([PresencePenalty(ids.shape[1])]),
                **settings,
            )
        completion = generated[0, ids.shape[1] :].tolist()
        row = dict(
            id=q["id"],
            prompt_id=q["id"],
            prompt_token_ids=q["prompt_token_ids"],
            completion_token_ids=completion,
            messages=q["messages"],
            response=tokenizer.decode(completion, skip_special_tokens=False),
            seed=q["seed"],
            elapsed_s=time.monotonic() - before,
            reached_cap=len(completion) == args.max_new_tokens,
        )
        rows.append(row)
        text_path.write_text("".join(json.dumps(item) + "\n" for item in rows))
        print(f"REASONING {q['id']} tokens={len(completion)} seconds={row['elapsed_s']:.2f}", flush=True)
    if prompts:
        total = sum(len(row["completion_token_ids"]) for row in rows)
        assert total >= args.min_reasoning_positions, f"Only {total} positions; increase --num-texts"
        print(f"REASONING_TEXT_OK answers={len(rows)} positions={total} path={text_path}", flush=True)


if __name__ == "__main__":
    main()
