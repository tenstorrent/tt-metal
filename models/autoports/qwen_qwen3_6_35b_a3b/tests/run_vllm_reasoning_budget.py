# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Matched HF/HTTP controls for shared chat tests' final-content budgets."""

import argparse
import json
import time
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.6-35B-A3B"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def shared_host_processor(prompt, seed, temperature, top_p, penalty, banned):
    """CPU-only diagnostic: HF model/cache with the exact serving host sampler.

    Return a one-hot score vector to HF's generation driver after the shared
    sampler has selected the token. This never participates in TT serving or
    performance; it removes differing RNG/filter implementations as a control
    confound while leaving HF responsible for model execution and cache state.
    """
    from vllm.v1.sample.logits_processor import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm_tt_plugin.host_sampler import create_host_sampler

    sampler = create_host_sampler()
    metadata = SamplingMetadata(
        temperature=torch.tensor([temperature]),
        all_greedy=False,
        all_random=True,
        top_p=torch.tensor([top_p]),
        top_k=torch.tensor([20], dtype=torch.int32),
        generators={0: torch.Generator(device="cpu").manual_seed(seed)},
        max_num_logprobs=None,
        # Apply presence below without accelerator pinned-memory/custom-op
        # setup. Frequency=0 and repetition=1 are identity operations here.
        no_penalties=True,
        prompt_token_ids=torch.tensor([prompt], dtype=torch.int64),
        frequency_penalties=torch.tensor([0.0]),
        presence_penalties=torch.tensor([penalty]),
        repetition_penalties=torch.tensor([1.0]),
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={0: banned},
        logitsprocs=LogitsProcessors(),
    )

    def process(input_ids, scores):
        metadata.output_token_ids = [input_ids[0, len(prompt) :].tolist()]
        values = scores.float().clone()
        values[:, torch.unique(input_ids[0, len(prompt) :])] -= penalty
        token = sampler(logits=values, sampling_metadata=metadata).sampled_token_ids
        return torch.full_like(scores, -float("inf")).scatter_(1, token.long().reshape(1, 1), 0.0)

    return process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hf", "http"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--case", choices=("bad_words", "plain", "choice"), default="bad_words")
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="default")
    parser.add_argument("--profile", choices=("shared_test", "model_card"), default="shared_test")
    parser.add_argument("--hf-sampler", choices=("transformers", "vllm_host"), default="transformers")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh control artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    sampling = args.case == "bad_words"
    if args.hf_sampler == "vllm_host" and (args.backend != "hf" or not sampling):
        raise ValueError("Shared host-sampler control requires native HF and the sampled bad_words case")
    if args.profile == "model_card" and not sampling:
        raise ValueError("The model-card sampling control is for the sampled bad_words case")
    temperature = (0.7 if args.thinking == "off" else 1.0) if args.profile == "model_card" else 1.0
    top_p = (0.8 if args.thinking == "off" else 0.95) if args.profile == "model_card" else 1.0
    penalty = 1.5 if args.profile == "model_card" else (0.5 if args.case == "choice" else 0.0)
    text = "Say hello to me" if sampling else "Reply with a short sentence for request 3."
    if args.case == "choice":
        text = "Pick one color for request 0."
    messages = [{"role": "user", "content": text}]
    template_kwargs = {} if args.thinking == "default" else {"enable_thinking": args.thinking == "on"}
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
    prompt = tokenizer.encode(rendered, add_special_tokens=False)
    bad_words = ["hello", "Hello", "hi", "Hi", "hey", "Hey"] if sampling else []
    # Match the pinned upstream SamplingParams.update_from_tokenizer rules.
    banned = []
    for word in bad_words:
        for prefix in ("", " "):
            ids = tokenizer.encode(prefix + word.lstrip(), add_special_tokens=False)
            if not prefix or (ids[0] != banned[-1][0] and len(ids) == len(banned[-1])):
                banned.append(ids)
    result = dict(
        status="RUNNING",
        backend=args.backend,
        case=args.case,
        model=MODEL,
        revision=REVISION,
        messages=messages,
        rendered_prompt=rendered,
        prompt_tokens=prompt,
        bad_words=bad_words,
        bad_words_token_ids=banned,
        max_tokens=args.max_tokens,
        chat_template_kwargs=template_kwargs,
        profile=args.profile,
        hf_sampler=args.hf_sampler if args.backend == "hf" else None,
        generation=dict(temperature=temperature if sampling else 0.0, top_p=top_p, top_k=20),
        presence_penalty=penalty,
        structured_outputs={"choice": ["red", "green", "blue", "yellow"]} if args.case == "choice" else None,
        seeds=list(range(10)),
        rubric="Compare final-content availability at the original 100-token cap and measured extended budget; inspect outputs blind before assigning a model-quality limitation. TT is worse if its failure rate exceeds HF by more than 20 percentage points.",
        outputs=[],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    model = None
    if args.backend == "hf":
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True).eval()
        result["resolved_model_dtype"] = str(model.dtype)

    def presence_penalty(input_ids, scores):
        # vLLM presence penalty applies once to each generated token type.
        scores = scores.float().clone()
        scores[:, torch.unique(input_ids[0, len(prompt) :])] -= penalty
        return scores

    for seed in range(10):
        started = time.monotonic()
        if model is not None:
            torch.manual_seed(seed)
            shared_sampling = args.hf_sampler == "vllm_host"
            processors = (
                [shared_host_processor(prompt, seed, temperature, top_p, penalty, banned)]
                if shared_sampling
                else ([presence_penalty] if penalty else None)
            )
            with torch.inference_mode():
                output = model.generate(
                    torch.tensor([prompt], dtype=torch.long),
                    max_new_tokens=args.max_tokens,
                    do_sample=sampling and not shared_sampling,
                    temperature=temperature if sampling and not shared_sampling else None,
                    top_p=top_p if sampling and not shared_sampling else None,
                    top_k=20 if sampling and not shared_sampling else None,
                    bad_words_ids=(banned or None) if not shared_sampling else None,
                    logits_processor=processors,
                )
            tokens = output[0, len(prompt) :].tolist()
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            if args.case == "choice" and tokenizer.convert_tokens_to_ids("</think>") in tokens:
                # The pinned vLLM grammar is inactive during reasoning. This
                # short HF control is valid only while that remains true.
                raise RuntimeError("Choice control reached grammar activation; a grammar-aware native run is needed")
            row = dict(seed=seed, tokens=tokens, text=text)
        else:
            payload = dict(
                model=MODEL,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=temperature if sampling else 0.0,
                top_p=top_p,
                top_k=20,
                seed=seed,
                return_token_ids=True,
                presence_penalty=result["presence_penalty"],
            )
            if bad_words:
                payload["bad_words"] = bad_words
            if template_kwargs:
                payload["chat_template_kwargs"] = template_kwargs
            if result["structured_outputs"] is not None:
                payload["structured_outputs"] = result["structured_outputs"]
            response = requests.post(
                args.url + "/v1/chat/completions",
                json=payload,
                timeout=1800,
            )
            if not response.ok:
                result.update(
                    status="FAIL", http_status=response.status_code, response=response.json(), request=payload
                )
                save()
            response.raise_for_status()
            body = response.json()
            row = dict(seed=seed, response=body)
        row["elapsed_s"] = time.monotonic() - started
        result["outputs"].append(row)
        save()
        print(f"{args.backend} seed={seed} completed in {row['elapsed_s']:.2f}s", flush=True)
    result["status"] = "PASS"
    save()


if __name__ == "__main__":
    main()
