"""Collect original constraint requests and validate returned token IDs.

Run against an already-running server. Each raw response is saved as it arrives;
decoded lexical matches are diagnostics, never substitutes for token checks.
"""

import argparse
import concurrent.futures
import json
import string
import time
from pathlib import Path

import requests

BAD_WORDS = ["hello", "Hello", "hi", "Hi", "hey", "Hey"]


def sequence_matches(tokens, sequences):
    return [
        {"sequence": sequence, "offsets": offsets}
        for sequence in sequences
        if (
            offsets := [
                start
                for start in range(len(tokens) - len(sequence) + 1)
                if tokens[start : start + len(sequence)] == sequence
            ]
        )
    ]


def check_response(case, forbidden_sequences):
    failures = []
    response = case.get("response")
    if case.get("error") or case.get("http_status") != 200:
        return {"passed": False, "failures": [case.get("error", "HTTP request failed")]}
    if not isinstance(response, dict) or response.get("error"):
        return {"passed": False, "failures": ["Expected a successful JSON response"]}
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return {"passed": False, "failures": ["Expected exactly one completion choice"]}
    choice = choices[0]
    tokens = choice.get("token_ids")
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
        return {"passed": False, "failures": ["Expected nonempty returned integer token_ids"]}
    request = case["request"]
    if not 1 <= len(tokens) <= request["max_tokens"]:
        failures.append("Returned token count is outside the requested generation bound")
    usage_count = (response.get("usage") or {}).get("completion_tokens")
    if usage_count != len(tokens):
        failures.append(f"usage.completion_tokens={usage_count} differs from {len(tokens)} returned IDs")
    text = choice.get("message", {}).get("content") if case["chat"] else choice.get("text")
    if not isinstance(text, str):
        failures.append("Expected string completion text (empty text is valid)")
    checks = {
        "token_count": len(tokens),
        "usage_completion_tokens": usage_count,
        "finish_reason": choice.get("finish_reason"),
        "stop_reason": choice.get("stop_reason"),
        "visible_text_empty": text == "",
    }
    if "allowed_token_ids" in request:
        allowed = set(request["allowed_token_ids"])
        outside = [{"offset": index, "token_id": token} for index, token in enumerate(tokens) if token not in allowed]
        checks["outside_whitelist"] = outside
        if outside:
            failures.append(f"Emitted {len(outside)} token(s) outside the whitelist")
    if request.get("bad_words"):
        violations = sequence_matches(tokens, forbidden_sequences)
        checks["prohibited_subsequence_matches"] = violations
        if violations:
            failures.append("Returned IDs contain a prohibited bad-word token sequence")
        # Reproduce the shared test's lexical assertion for diagnosis only.
        punct = string.punctuation.replace(">", "")
        lexical_hits = [
            {"text_piece": piece, "stripped_word": piece.strip(punct)}
            for piece in (text or "").split()
            if piece.strip(punct) in BAD_WORDS
        ]
        checks["punctuation_stripped_lexical_hits"] = lexical_hits
        checks["lexical_false_positive"] = bool(lexical_hits) and not violations
    checks.update(passed=not failures, failures=failures)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="IFM/K2-Horizon-7B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--bias-control", action="store_true", help="Also run a paired single-token ban/bias control")
    parser.add_argument(
        "--multi-token-control", action="store_true", help="Also test a two-token ban with no penalties"
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    result = {
        "server_url": base_url,
        "model": args.model,
        "status": "collecting",
        "tokenization": [],
        "forbidden_sequences": [],
        "cases": [],
        "bias_control": {"requested": args.bias_control},
        "multi_token_control": {"requested": args.multi_token_control},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(args.output)

    def post(endpoint, payload):
        started = time.monotonic()
        record = {"endpoint": endpoint, "request": payload}
        try:
            response = requests.post(base_url + endpoint, json=payload, timeout=args.timeout)
            record["http_status"] = response.status_code
            try:
                record["response"] = response.json()
            except ValueError:
                record["response_text"] = response.text
                record["error"] = "Response was not JSON"
        except requests.RequestException as error:
            record["error"] = str(error)
        record["elapsed_seconds"] = time.monotonic() - started
        return record

    def tokenize(text):
        record = post(
            "/tokenize",
            {
                "model": args.model,
                "prompt": text,
                "add_special_tokens": False,
                "return_token_strs": True,
            },
        )
        result["tokenization"].append(record)
        save()
        response = record.get("response")
        if record.get("http_status") != 200 or not isinstance(response, dict):
            raise AssertionError(f"Server tokenization failed for {text!r}")
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int for token in tokens):
            raise AssertionError(f"Server returned invalid tokenization for {text!r}")
        return tokens

    def collect(cases):
        start = len(result["cases"])
        result["cases"].extend(cases)
        save()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cases)) as pool:
            futures = {
                pool.submit(post, "/v1/chat/completions" if case["chat"] else "/v1/completions", case["request"]): index
                for index, case in enumerate(cases, start)
            }
            for future in concurrent.futures.as_completed(futures):
                case = result["cases"][futures[future]]
                case.update(future.result())
                # Preserve raw evidence even if checking it exposes a malformed response.
                save()
                case["checks"] = check_response(case, case.get("forbidden_sequences", result["forbidden_sequences"]))
                save()
                print(f"CONSTRAINT_CASE {case['id']} passed={case['checks']['passed']}", flush=True)

    save()
    try:
        # Match SamplingParams.update_from_tokenizer exactly, including its
        # conditional inclusion of a leading-space variant.
        for word in BAD_WORDS:
            plain = tokenize(word.lstrip())
            spaced = tokenize(" " + word.lstrip())
            result["forbidden_sequences"].append(plain)
            if spaced[0] != plain[0] and len(spaced) == len(plain):
                result["forbidden_sequences"].append(spaced)
            save()

        bad_cases = [
            {
                "id": f"bad_words_seed_{seed}",
                "chat": True,
                "request": {
                    "model": args.model,
                    "messages": [{"role": "user", "content": "Say hello to me"}],
                    "max_tokens": 100,
                    "bad_words": BAD_WORDS,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "seed": seed,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "return_token_ids": True,
                },
            }
            for seed in range(5)
        ]
        collect(bad_cases)
        allowed_cases = [
            {
                "id": f"allowed_{start}_{start + 2}",
                "chat": False,
                "request": {
                    "model": args.model,
                    "prompt": "Allowed: ",
                    "max_tokens": 10,
                    "allowed_token_ids": list(range(start, start + 3)),
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "return_token_ids": True,
                },
            }
            for start in (1, 4, 7, 10, 13)
        ]
        collect(allowed_cases)

        if args.bias_control:
            singletons = {sequence[0] for sequence in result["forbidden_sequences"] if len(sequence) == 1}
            alternatives = [token for token in tokenize("0") if token not in singletons]
            if not singletons or not alternatives:
                raise AssertionError("Requested bias control needs a single-token forbidden word and an alternative")
            candidate, alternative = min(singletons), alternatives[0]
            base = {
                **bad_cases[0]["request"],
                "max_tokens": 1,
                "temperature": 0.0,
                "allowed_token_ids": [candidate, alternative],
                "logit_bias": {str(candidate): 100.0},
            }
            unbanned = {key: value for key, value in base.items() if key != "bad_words"}
            collect(
                [
                    {"id": "bias_unbanned", "chat": True, "request": unbanned},
                    {"id": "bias_banned", "chat": True, "request": base},
                ]
            )
            controls = result["cases"][-2:]
            control_pass = all(case["checks"]["passed"] for case in controls)
            if control_pass:
                control_pass = controls[0]["response"]["choices"][0]["token_ids"] == [candidate] and controls[1][
                    "response"
                ]["choices"][0]["token_ids"] == [alternative]
            result["bias_control"].update(
                candidate=candidate,
                alternative=alternative,
                passed=control_pass,
            )
            save()

        if args.multi_token_control:
            # Find a server-encoded phrase whose included space/plain variant
            # is the same token twice. Biasing that token makes the first output
            # legal and the second complete the banned sequence. Do not infer
            # this sequence by re-encoding a generated response.
            repeated = None
            for word in ("hi", "hello", "yes", "no", "cat", "a", "0"):
                phrase = f"{word} {word}"
                plain, spaced = tokenize(phrase), tokenize(" " + phrase)
                sequences = [plain]
                if spaced[0] != plain[0] and len(spaced) == len(plain):
                    sequences.append(spaced)
                repeated = next(
                    (sequence for sequence in sequences if len(sequence) == 2 and sequence[0] == sequence[1]),
                    None,
                )
                if repeated is not None:
                    break
            if repeated is None:
                raise AssertionError("No repeated two-token bad-word control found for the serving tokenizer")
            candidate = repeated[0]
            alternative = next(
                (token for token in tokenize("0") if token != candidate),
                None,
            )
            if alternative is None:
                alternative = next(
                    (token for token in tokenize("1") if token != candidate),
                    None,
                )
            if alternative is None:
                raise AssertionError("No distinct alternative token found for the multi-token control")
            base = {
                **bad_cases[0]["request"],
                "max_tokens": 2,
                "temperature": 0.0,
                "bad_words": [phrase],
                "allowed_token_ids": [candidate, alternative],
                "logit_bias": {str(candidate): 100.0, str(alternative): -100.0},
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
            }
            unbanned = {key: value for key, value in base.items() if key != "bad_words"}
            result["multi_token_control"].update(
                phrase=phrase,
                forbidden_sequences=sequences,
                candidate=candidate,
                alternative=alternative,
            )
            save()
            collect(
                [
                    {"id": "multi_token_unbanned", "chat": True, "request": unbanned},
                    {"id": "multi_token_banned", "chat": True, "request": base, "forbidden_sequences": sequences},
                ]
            )
            controls = result["cases"][-2:]
            control_pass = all(case["checks"]["passed"] for case in controls)
            if control_pass:
                control_pass = controls[0]["response"]["choices"][0]["token_ids"] == [
                    candidate,
                    candidate,
                ] and controls[1]["response"]["choices"][0]["token_ids"] == [candidate, alternative]
            result["multi_token_control"]["passed"] = control_pass
            save()

        failures = [
            {"id": case["id"], "failures": case["checks"]["failures"]}
            for case in result["cases"]
            if not case["checks"]["passed"]
        ]
        if args.bias_control and not result["bias_control"]["passed"]:
            failures.append(
                {"id": "bias_control", "failures": ["Unbanned/banned paired control did not isolate the filter"]}
            )
        if args.multi_token_control and not result["multi_token_control"]["passed"]:
            failures.append(
                {
                    "id": "multi_token_control",
                    "failures": ["Expected repeated unbanned token and banned second token with zero penalties"],
                }
            )
        result.update(status="failed" if failures else "passed", passed=not failures, failures=failures)
        save()
        if failures:
            raise AssertionError(json.dumps(failures))
        print("CONSTRAINT_TOKEN_CHECKS_PASS", args.output)
    except BaseException as error:
        result.update(status="failed", passed=False, error=f"{type(error).__name__}: {error}")
        save()
        raise


if __name__ == "__main__":
    main()
