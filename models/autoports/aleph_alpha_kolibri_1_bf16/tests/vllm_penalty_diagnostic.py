# SPDX-License-Identifier: Apache-2.0
"""Focused penalty diagnostics. Run only after the active serving suite finishes."""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PROMPTS = [
    "She opened the door and",
    "The reason is that the",
    "He said that the",
    "I think the answer is maybe",
    "After a while, she",
    "It was a",
    "The book was",
    "a b c a b c a b c",
]
REVISION = "7a8f290e7858825c3cf5e4c447ba68345de9f1d3"


def token_id(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.startswith("token_id:"):
        return int(value.split(":", 1)[1])
    raise ValueError(f"Expected numeric token-id representation, got {value!r}")


def analyze(choice, tokenizer, presence, frequency, requested_topk):
    ids = choice.get("token_ids")
    logs = choice.get("logprobs") or {}
    if ids is None and logs.get("tokens"):
        ids = [token_id(value) for value in logs["tokens"]]
    result = {
        "text": choice.get("text", ""),
        "character_a_count": choice.get("text", "").count("a"),
        "token_ids": ids,
    }
    if ids is None:
        result["status"] = "missing-token-ids"
        return result
    ids = [int(value) for value in ids]
    counts = Counter(ids)
    result["token_histogram"] = [
        {"id": ident, "count": count, "decoded_piece": tokenizer.decode([ident], skip_special_tokens=False)}
        for ident, count in counts.most_common()
    ]
    result["repeated_token_occurrences"] = len(ids) - len(counts)
    result["steps"] = []
    history = Counter()
    top_rows = logs.get("top_logprobs") or []
    selected_logprobs = logs.get("token_logprobs") or []
    for step, ident in enumerate(ids):
        record = {"step": step, "selected": ident, "prior_count": history[ident]}
        if step < len(top_rows) and top_rows[step]:
            raw = {token_id(k): float(v) for k, v in top_rows[step].items()}
            if ident not in raw and step < len(selected_logprobs):
                raw[ident] = float(selected_logprobs[step])
            adjusted = {
                tok: score - presence * (history[tok] > 0) - frequency * history[tok] for tok, score in raw.items()
            }
            best = max(adjusted, key=adjusted.get)
            chosen = adjusted[ident]
            ranked_raw = sorted(raw.values(), reverse=True)
            # vLLM returns raw top-K plus the selected token. Every omitted token
            # is <= the K-th largest raw score. Nonnegative penalties only lower
            # omitted scores, giving a valid sufficient global-argmax certificate.
            bound = ranked_raw[requested_topk - 1] if len(ranked_raw) >= requested_topk else None
            gap = adjusted[best] - chosen
            record.update(
                raw_logprobs=raw,
                adjusted_logprobs=adjusted,
                best_known_token=best,
                selected_adjusted_gap=gap,
                known_candidate_violation=gap > 1e-5,
                omitted_raw_upper_bound=bound,
                global_argmax_certified=bound is not None and chosen >= bound - 1e-5 and gap <= 1e-5,
            )
        else:
            record["status"] = "no-logprobs"
        result["steps"].append(record)
        history[ident] += 1
    result["known_candidate_violations"] = sum(x.get("known_candidate_violation", False) for x in result["steps"])
    result["certified_steps"] = sum(x.get("global_argmax_certified", False) for x in result["steps"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--modes", nargs="+", choices=("exact", "diagnostic", "chat"), default=["exact", "diagnostic", "chat"]
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-logprobs", type=int, default=20)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "diagnostic-not-a-gate-waiver",
        "revision": REVISION,
        "model": args.model,
        "checkpoint": args.checkpoint,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_present": bool(tokenizer.chat_template),
        "endpoint": args.url + "/v1/completions",
        "prompt_source": "vllm-tt-plugin/tests/tt/test_tt_penalties.py::TestPresencePenalty.PRESENCE_PROMPTS",
        "modes": args.modes,
        "groups": [],
        "note": "Exact mode adds token-ID reporting only. Diagnostic/chat modes request raw top-logprobs, forcing host compatibility on this four-chip target.",
    }

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    def send(payload):
        started = time.time()
        response = requests.post(args.url + "/v1/completions", json=payload, timeout=900)
        record = {"payload": payload, "http_status": response.status_code, "elapsed_seconds": time.time() - started}
        try:
            record["response"] = response.json()
        except ValueError:
            record["response_text"] = response.text
        return record

    def batch(label, mode, prompt, coefficients):
        payloads = []
        for presence, frequency, max_tokens in coefficients:
            body = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": None,
                "presence_penalty": presence,
                "frequency_penalty": frequency,
                "return_token_ids": True,
            }
            if mode != "exact":
                body.update(logprobs=args.top_logprobs, return_tokens_as_token_ids=True)
            payloads.append(body)
        with ThreadPoolExecutor(max_workers=len(payloads)) as pool:
            records = list(pool.map(send, payloads))
        group = {"label": label, "mode": mode, "records": records}
        result["groups"].append(group)
        save()  # Preserve responses even if analysis finds an unexpected schema.
        for record in records:
            for choice in record.get("response", {}).get("choices", []):
                record.setdefault("analysis", []).append(
                    analyze(
                        choice,
                        tokenizer,
                        record["payload"]["presence_penalty"],
                        record["payload"]["frequency_penalty"],
                        args.top_logprobs,
                    )
                )
                if record["payload"]["presence_penalty"] == 0 and record["payload"]["frequency_penalty"] == 0:
                    record.setdefault("presence_2_counterfactual", []).append(
                        analyze(choice, tokenizer, 2.0, 0.0, args.top_logprobs)
                    )
        save()
        print("RECORDED", mode, label, flush=True)
        return records

    def text_of(record):
        return record["response"]["choices"][0]["text"]

    for mode in args.modes:
        if mode == "chat" and not tokenizer.chat_template:
            raise ValueError("Chat control requested but checkpoint has no chat template")
        changed = noise = 0
        for index, raw_prompt in enumerate(PROMPTS):
            prompt = raw_prompt
            if mode == "chat":
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": raw_prompt}], tokenize=False, add_generation_prompt=True
                )
            pair = batch(f"presence-{index}-penalty", mode, prompt, [(0.0, 0.0, 24), (2.0, 0.0, 24)])
            control = batch(f"presence-{index}-control", mode, prompt, [(0.0, 0.0, 24)] * 2)
            changed += text_of(pair[0]) != text_of(pair[1])
            noise += text_of(control[0]) != text_of(control[1])
        raw_prompt = "a a a a a a a a a"
        prompt = (
            raw_prompt
            if mode != "chat"
            else tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}], tokenize=False, add_generation_prompt=True
            )
        )
        records = batch(
            "frequency-mixed", mode, prompt, [(0.0, 0.0 if row % 2 == 0 else 2.0, 15) for row in range(args.batch_size)]
        )
        result.setdefault("summaries", {})[mode] = {
            "presence_changed": changed,
            "presence_slot_noise": noise,
            "original_presence_threshold_satisfied": changed >= 4 and changed > noise,
            "frequency_baseline_character_a_count": text_of(records[0]).count("a"),
            "frequency_penalized_character_a_count": text_of(records[1]).count("a"),
            "original_frequency_character_assertion_satisfied": text_of(records[0]).count("a")
            > text_of(records[1]).count("a"),
        }
        save()
    result["collection_complete"] = True
    save()


if __name__ == "__main__":
    main()
