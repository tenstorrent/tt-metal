#!/usr/bin/env python3
"""Send a test chat-completions request through the Tenstorrent Console API."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Optional, Sequence, Union


DEFAULT_URL = "https://console.tenstorrent.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-0528"
DEFAULT_PROMPT = "Is Gdansk University of Technology a good school?"
API_KEY_ENV_VARS = ("TENSTORRENT_API_KEY", "OPENAI_API_KEY")


def send_chat_completion_request(
    payload: dict,
    api_key: Optional[str] = None,
    url: str = DEFAULT_URL,
    timeout: int = 60,
) -> Union[dict, Exception]:
    for env_var in API_KEY_ENV_VARS:
        if api_key:
            break
        api_key = os.environ.get(env_var)

    if not api_key:
        return ValueError("provide --api-key, TENSTORRENT_API_KEY, or OPENAI_API_KEY")

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error
    except urllib.error.URLError as error:
        return error

    return json.loads(response_body)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the Tenstorrent Console chat endpoint.")
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key. Defaults to TENSTORRENT_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Endpoint URL. Defaults to {DEFAULT_URL}.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name. Defaults to {DEFAULT_MODEL}.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="User prompt to send.")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate.")
    return parser.parse_args(argv)


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = {
        "max_tokens": args.max_tokens,
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
    }
    result = send_chat_completion_request(
        payload=payload,
        api_key=args.api_key,
        url=args.url,
    )
    if isinstance(result, ValueError):
        sys.stderr.write(f"error: {result}\n")
        return 2
    if isinstance(result, urllib.error.HTTPError):
        sys.stderr.write(f"HTTP {result.code}: {result.reason}\n")
        sys.stderr.write(result.read().decode("utf-8", errors="replace"))
        sys.stderr.write("\n")
        return 1
    if isinstance(result, urllib.error.URLError):
        sys.stderr.write(f"request failed: {result.reason}\n")
        return 1

    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    return run_cli()


if __name__ == "__main__":
    raise SystemExit(main())
