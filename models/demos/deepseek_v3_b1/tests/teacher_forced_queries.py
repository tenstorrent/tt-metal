#!/usr/bin/env python3
"""Run teacher-forced one-token chat-completions queries."""

import argparse
import json
import os
import sys
import urllib.error
from pathlib import Path
from typing import Optional, Sequence

from transformers import AutoTokenizer
from query_tenstorrent_console import send_chat_completion_request

DEFAULT_URL = "https://console.tenstorrent.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-0528"
DEFAULT_MODEL_PATH = os.getenv("DEEPSEEK_V3_HF_MODEL", DEFAULT_MODEL)
API_KEY_ENV_VARS = ("TENSTORRENT_API_KEY", "OPENAI_API_KEY")
DEFAULT_PROMPT = (
    "Solve this step by step: Design a cache for sharded LLM weights across multiple hosts with local NVMe and "
    "shared NFS. Minimize startup latency, avoid duplicate reads, support cache invalidation, and explain tradeoffs "
    "between per-host caches, content-addressable storage, and pre-sharded artifacts."
)
DEFAULT_REFERENCE_TOKENS_FILE = Path("models/demos/deepseek_v3_b1/tests/output1_reference.txt")


def load_reference_tokens(path: Path, model_path: str) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    token_ids = tokenizer.encode(path.read_text(encoding="utf-8"), add_special_tokens=False)
    return [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]


def find_api_key(explicit_api_key: Optional[str]) -> Optional[str]:
    if explicit_api_key:
        return explicit_api_key
    for env_var in API_KEY_ENV_VARS:
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key
    return None


def evaluate_step(index: int, reference_token: str, generated_token: Optional[str]) -> dict:
    return {
        "index": index,
        "reference_token": reference_token,
        "generated_token": generated_token,
        "top_1": generated_token == reference_token,
    }


def run_teacher_forced_queries(
    api_key: Optional[str],
    reference_tokens: Sequence[str],
    url: str = DEFAULT_URL,
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    max_steps: Optional[int] = 6,
    timeout: int = 60,
) -> list[dict]:
    session_id = None
    pending_generated_token = None
    results = []
    tokens = reference_tokens[:max_steps] if max_steps is not None else reference_tokens
    reference_index = 0

    def append_result(
        reference_index: int, reference_token: str, generated_token: Optional[str]
    ) -> int:
        results.append(
            evaluate_step(index=reference_index, reference_token=reference_token, generated_token=generated_token)
        )
        return reference_index + 1

    while reference_index < len(tokens):
        reference_token = tokens[reference_index]
        if pending_generated_token is not None:
            generated_token = pending_generated_token
            pending_generated_token = None
            reference_index = append_result(reference_index, reference_token, generated_token)
            continue

        teacher_index = reference_index - 1
        while teacher_index >= 0 and tokens[teacher_index] in ("<think>", "</think>"):
            teacher_index -= 1
        if teacher_index < 0:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": tokens[teacher_index]}]

        payload = {
            "max_tokens": 2,
            "model": model,
            "messages": messages,
            # The server only supports top-1 today; keep this field so top-k can grow here later.
            "logprobs": 1,
        }
        if session_id:
            payload["session_id"] = session_id

        print(f"payload: {payload}")
        response = send_chat_completion_request(
            payload=payload,
            api_key=api_key,
            url=url,
            timeout=timeout,
        )
        print(f"response: {response}")
        if isinstance(response, Exception):
            raise response

        if session_id is None:
            session_id = response["usage"]["sessionId"]

        choices = response.get("choices") or []
        message = choices[0].get("message") if choices else {}
        message = message or {}
        content = message.get("content")
        reasoning = message.get("reasoning")
        if reference_token == "<think>" and reasoning:
            generated_token = "<think>"
            pending_generated_token = reasoning
        elif reference_token == "</think>" and content:
            generated_token = "</think>"
            pending_generated_token = content
        else:
            generated_token = content or reasoning

        if generated_token is None:
            raise RuntimeError("response did not include content or reasoning")

        reference_index = append_result(reference_index, reference_token, generated_token)

    return results


def summarize_results(results: Sequence[dict]) -> dict:
    hits = sum(1 for result in results if result["top_1"])
    return {
        "steps": len(results),
        "top_1": {
            "hits": hits,
            "accuracy": hits / len(results) if results else None,
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher-forced one-token queries.")
    parser.add_argument(
        "reference_tokens_file",
        nargs="?",
        type=Path,
        default=DEFAULT_REFERENCE_TOKENS_FILE,
        help=f"Reference text file. Defaults to {DEFAULT_REFERENCE_TOKENS_FILE}.",
    )
    parser.add_argument("--api-key", default=None, help="API key. Defaults to TENSTORRENT_API_KEY or OPENAI_API_KEY.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Endpoint URL. Defaults to {DEFAULT_URL}.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name. Defaults to {DEFAULT_MODEL}.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="HF model path containing tokenizer files.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Initial user prompt.")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum reference tokens to evaluate.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    return parser.parse_args(argv)


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    api_key = find_api_key(args.api_key)
    if not api_key:
        sys.stderr.write("error: provide --api-key, TENSTORRENT_API_KEY, or OPENAI_API_KEY\n")
        return 2

    try:
        reference_tokens = load_reference_tokens(args.reference_tokens_file, args.model_path)
        results = run_teacher_forced_queries(
            api_key=api_key,
            reference_tokens=reference_tokens,
            url=args.url,
            model=args.model,
            prompt=args.prompt,
            max_steps=args.max_steps,
            timeout=args.timeout,
        )
    except (OSError, ValueError, urllib.error.URLError) as error:
        sys.stderr.write(f"error: {error}\n")
        return 2

    report = {
        "summary": summarize_results(results),
        "steps": results,
    }
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
