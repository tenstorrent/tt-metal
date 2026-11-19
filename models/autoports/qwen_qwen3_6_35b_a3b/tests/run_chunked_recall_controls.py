# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Preserve canonical recall cases and separately labeled chat/HF controls."""

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("http", "hf"), required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--plugin-root", type=Path, default=Path.cwd().parent / "vllm-tt-plugin")
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, choices=(1, 4), default=1)
    parser.add_argument("--cases", nargs="+", help="Optional exact case IDs, e.g. 0-2 1-2")
    parser.add_argument(
        "--include-solo", action="store_true", help="Also include the canonical short and long solo prompts"
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="raw")
    parser.add_argument("--thinking", choices=("default", "off"), default="default")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--sampling", choices=("greedy", "model-card"), default="greedy")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior evidence: output already exists")
    if args.backend == "hf" and args.concurrency != 1:
        parser.error("HF control is serial")
    if args.prompt_format == "raw" and args.thinking != "default":
        parser.error("Thinking override requires chat format")
    sys.path.insert(0, str(args.plugin_root.resolve()))
    from transformers import AutoTokenizer

    from tests.tt.test_chunked_prefill import CONCURRENCY, ROUNDS, _needle_prompt
    from tests.tt.utils import recalled_passphrase

    model_id = "Qwen/Qwen3.6-35B-A3B"
    revision = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True)
    cases = []
    for rnd in range(ROUNDS):
        for index in range(CONCURRENCY):
            case_id = f"{rnd}-{index}"
            phrase = f"cobalt-heron-{case_id}"
            if args.cases and case_id not in args.cases:
                continue
            prompt = _needle_prompt(args.budget, phrase, seed=0xC401 + index, nonce=f"Round {rnd} doc {index}.\n")
            cases.append(dict(id=case_id, round=rnd, passphrase=phrase, prompt=prompt))
    if args.include_solo:
        for case_id, group, words, phrase, seed, nonce in (
            ("solo", -2, args.budget, "cobalt-heron-42", 0xC401, "Solo.\n"),
            ("long", -1, 3 * args.budget, "cobalt-heron-256", 0xC402, "Long.\n"),
        ):
            if not args.cases or case_id in args.cases:
                cases.append(
                    dict(id=case_id, round=group, passphrase=phrase, prompt=_needle_prompt(words, phrase, seed, nonce))
                )
    if not cases or (args.cases and {row["id"] for row in cases} != set(args.cases)):
        parser.error("Unknown or empty case selection")
    source = args.plugin_root / "tests/tt/test_chunked_prefill.py"
    result = dict(
        status="RUNNING",
        backend=args.backend,
        concurrency=args.concurrency,
        budget=args.budget,
        model=model_id,
        revision=revision,
        prompt_mode=args.prompt_format,
        tokenizer_class=type(tokenizer).__name__,
        chat_template_present=bool(tokenizer.chat_template),
        max_tokens=args.max_tokens,
        sampling=args.sampling,
        thinking=args.thinking,
        seeds=args.seeds,
        scope="Canonical raw stress only at raw/greedy/24; other settings are separately labeled controls",
        rubric="Recall requires the complete passphrase in final content; chat also requires EOS. Mechanical scorer receives only text and expected phrase, not backend identity.",
        quality_failure_margin_percentage_points=20,
        test_source=str(source.resolve()),
        test_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixtures=cases,
        outputs=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        if args.backend == "hf":
            import torch
            from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

            torch.set_num_threads(args.threads)
            torch.set_num_interop_threads(1)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, local_files_only=True, dtype="auto"
            ).eval()
            result["dtype"] = str(model.dtype)
            result["generation_config"] = model.generation_config.to_dict()

        def run(job):
            case, seed = job
            started = time.monotonic()
            payload = dict(
                model=model_id,
                max_tokens=args.max_tokens,
                temperature=0.0 if args.sampling == "greedy" else 1.0,
                top_p=1.0 if args.sampling == "greedy" else 0.95,
                seed=None,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                logprobs=None,
            )
            if args.sampling == "model-card":
                payload.update(seed=seed, top_k=20, presence_penalty=1.5, repetition_penalty=1.0, min_p=0.0)
                if args.thinking == "off":
                    payload.update(temperature=0.7, top_p=0.8)
            if args.prompt_format == "chat":
                payload.update(messages=[dict(role="user", content=case["prompt"])], reasoning_effort="low")
                template_kwargs = {} if args.thinking == "default" else dict(enable_thinking=False)
                if template_kwargs:
                    payload["chat_template_kwargs"] = template_kwargs
                rendered = tokenizer.apply_chat_template(
                    payload["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    reasoning_effort="low",
                    **template_kwargs,
                )
            else:
                payload["prompt"] = case["prompt"]
                rendered = case["prompt"]
            prompt_tokens = tokenizer.encode(rendered, add_special_tokens=False)
            if args.backend == "http":
                import requests

                endpoint = "/v1/chat/completions" if args.prompt_format == "chat" else "/v1/completions"
                response = requests.post(args.url + endpoint, json=payload, timeout=1800)
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                output = choice["message"].get("content") or "" if args.prompt_format == "chat" else choice["text"]
                completed = choice["finish_reason"] == "stop"
            else:
                ids = torch.tensor([prompt_tokens], dtype=torch.long)
                processors = LogitsProcessorList()
                if args.sampling == "model-card":

                    class PresencePenalty(LogitsProcessor):
                        def __call__(self, history, scores):
                            seen = torch.unique(history[0, ids.shape[1] :])
                            scores[:, seen] -= 1.5
                            return scores

                    processors.append(PresencePenalty())
                torch.manual_seed(seed)
                with torch.inference_mode():
                    tokens = model.generate(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        max_new_tokens=args.max_tokens,
                        do_sample=args.sampling == "model-card",
                        logits_processor=processors,
                        **(
                            {key: payload[key] for key in ("temperature", "top_p", "top_k")}
                            if args.sampling == "model-card"
                            else {}
                        ),
                    )
                generated = tokens[0, ids.shape[1] :].tolist()
                raw_output = tokenizer.decode(generated, skip_special_tokens=False)
                output = tokenizer.decode(generated, skip_special_tokens=True)
                if args.prompt_format == "chat" and args.thinking != "off":
                    output = output.split("</think>", 1)[-1] if "</think>" in output else ""
                eos = model.generation_config.eos_token_id
                completed = bool(generated) and generated[-1] in (eos if isinstance(eos, list) else [eos])
                body = dict(prompt_tokens=len(prompt_tokens), token_ids=generated, text=raw_output, completed=completed)
            return dict(
                id=case["id"],
                seed=seed,
                request=payload,
                prompt_tokens=prompt_tokens,
                response=body,
                output=output,
                completed=completed,
                recalled=recalled_passphrase(output, case["passphrase"]) and (args.prompt_format == "raw" or completed),
                elapsed_seconds=time.monotonic() - started,
            )

        for rnd in sorted({case["round"] for case in cases}):
            group = [case for case in cases if case["round"] == rnd]
            jobs = [(case, seed) for seed in args.seeds for case in group]
            with (
                concurrent.futures.ThreadPoolExecutor(args.concurrency) if args.concurrency > 1 else nullcontext()
            ) as pool:
                for row in pool.map(run, jobs) if pool is not None else map(run, jobs):
                    result["outputs"].append(row)
                    save()
                    print(json.dumps({k: row[k] for k in ("id", "recalled", "output", "elapsed_seconds")}), flush=True)
        result["status"] = "PASS" if all(row["recalled"] for row in result["outputs"]) else "RECALL_MISS"
    except BaseException as error:
        result.update(status="ERROR", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
