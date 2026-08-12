#!/usr/bin/env python3
"""Hardware-backed async-overlap and logit-determinism checks for Falcon3 vLLM."""

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path


def _completion(url, model, prompt, *, max_tokens, logprobs=None):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
    }
    if logprobs is not None:
        body["logprobs"] = logprobs
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    result["elapsed_s"] = time.perf_counter() - started
    return result


def _choice(result):
    return result["choices"][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--hf-model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    target = "Explain how a solar cell converts sunlight into electricity in simple terms."
    control = _completion(args.url, args.model, target, max_tokens=300)

    def delayed(prompt, max_tokens, delay):
        time.sleep(delay)
        return _completion(args.url, args.model, prompt, max_tokens=max_tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            "target": pool.submit(delayed, target, 300, 0.0),
            "short_admission": pool.submit(delayed, "Give three names for a friendly robot.", 30, 0.20),
            "medium_admission": pool.submit(delayed, "Describe the water cycle for a young student.", 120, 0.45),
            "late_admission": pool.submit(delayed, "List two uses of copper.", 24, 1.00),
        }
        overlap = {name: future.result() for name, future in futures.items()}

    control_text = _choice(control)["text"]
    overlap_text = _choice(overlap["target"])["text"]
    if control_text != overlap_text:
        raise AssertionError("greedy target changed under async admission/removal churn")

    det_prompt = "State one practical benefit of renewable energy."
    repeat_a = _completion(args.url, args.model, det_prompt, max_tokens=8, logprobs=10)
    repeat_b = _completion(args.url, args.model, det_prompt, max_tokens=8, logprobs=10)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        decoy_a = pool.submit(delayed, "Count from one to ten.", 32, 0.0)
        decoy_b = pool.submit(delayed, "Name four geometric shapes.", 32, 0.0)

        def delayed_determinism_target():
            time.sleep(0.10)
            return _completion(args.url, args.model, det_prompt, max_tokens=8, logprobs=10)

        batched = pool.submit(delayed_determinism_target).result()
        decoy_a.result()
        decoy_b.result()

    def signature(result):
        choice = _choice(result)
        lp = choice["logprobs"]
        return {
            "text": choice["text"],
            "tokens": lp["tokens"],
            "token_logprobs": lp["token_logprobs"],
            "top_logprobs": lp["top_logprobs"],
        }

    signatures = [signature(repeat_a), signature(repeat_b), signature(batched)]
    if not (signatures[0] == signatures[1] == signatures[2]):
        raise AssertionError("logits/tokens changed across repeat or batch position")

    standalone_comparison = None
    if args.hf_model:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
        reference = AutoModelForCausalLM.from_pretrained(args.hf_model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
        input_ids = tokenizer(det_prompt, return_tensors="pt").input_ids
        with torch.no_grad():
            logits = reference(input_ids).logits[0, -1].float()
        hf_logprobs = torch.log_softmax(logits, dim=-1)
        values, indices = torch.topk(hf_logprobs, 10)
        vllm_first_token_id = tokenizer.encode(signatures[0]["text"], add_special_tokens=False)[0]
        rank = int((hf_logprobs > hf_logprobs[vllm_first_token_id]).sum().item()) + 1
        standalone_comparison = {
            "reference": "HuggingFace eager BF16",
            "vllm_first_token_id": vllm_first_token_id,
            "vllm_first_token_hf_rank": rank,
            "hf_top_token_ids": indices.tolist(),
            "hf_top_token_text": [tokenizer.decode([i]) for i in indices.tolist()],
            "hf_top_logprobs": values.tolist(),
            "top1_match": vllm_first_token_id == int(indices[0]),
        }

    artifact = {
        "model": args.model,
        "sampling": {"temperature": 0.0, "ignore_eos": True},
        "async_overlap": {
            "target_max_tokens": 300,
            "crosses_initial_rope_boundary": True,
            "admission_delays_s": [0.20, 0.45, 1.00],
            "control_elapsed_s": control["elapsed_s"],
            "overlap_elapsed_s": overlap["target"]["elapsed_s"],
            "control_output_tokens": control["usage"]["completion_tokens"],
            "overlap_output_tokens": overlap["target"]["usage"]["completion_tokens"],
            "exact_output_match": True,
            "control_completion": control_text,
            "overlap_completion": overlap_text,
            "churn": {
                name: {
                    "elapsed_s": result["elapsed_s"],
                    "output_tokens": result["usage"]["completion_tokens"],
                    "completion": _choice(result)["text"],
                }
                for name, result in overlap.items()
                if name != "target"
            },
        },
        "vllm_logit_determinism": {
            "prompt": det_prompt,
            "logprobs": 10,
            "repeat_and_cross_batch_position_exact_match": True,
            "signatures": signatures,
            "standalone_comparison": standalone_comparison,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"status": "pass", "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
