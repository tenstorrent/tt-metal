# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Request harness for the Gemma 4 prefill HTTP service.

Example:
    python models/demos/gemma4/demo/prefill_harness.py \
        --prompt "Explain sliding-window attention in one paragraph."

    python models/demos/gemma4/demo/prefill_harness.py \
        --preset gutenberg-2600 --context-len 131072
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

PRESET_CACHE_DIR = Path("models/tt_transformers/demo/context_cache")
PROMPT_TOKEN_RESERVE = 64
_OVERSIZED_PROMPT_PATTERN = re.compile(r"prompt has (?P<tokens>\d+) tokens")


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


@dataclass(frozen=True)
class PromptPreset:
    name: str
    tokens: int
    url: str


PROMPT_PRESETS = {
    preset.name: preset
    for preset in (
        PromptPreset("gutenberg-135", 798_553, "https://www.gutenberg.org/cache/epub/135/pg135.txt"),
        PromptPreset("gutenberg-2600", 774_046, "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"),
        PromptPreset("gutenberg-1184", 652_941, "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"),
        PromptPreset("gutenberg-996", 564_232, "https://www.gutenberg.org/cache/epub/996/pg996.txt"),
        PromptPreset("gutenberg-1023", 490_836, "https://www.gutenberg.org/cache/epub/1023/pg1023.txt"),
        PromptPreset("gutenberg-1399", 489_021, "https://www.gutenberg.org/cache/epub/1399/pg1399.txt"),
        PromptPreset("gutenberg-145", 439_721, "https://www.gutenberg.org/cache/epub/145/pg145.txt"),
        PromptPreset("gutenberg-2701", 304_337, "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"),
    )
}


def _preset_cache_path(preset: PromptPreset, cache_dir: Path) -> Path:
    return cache_dir / hashlib.md5(preset.url.encode()).hexdigest()


def load_preset_text(
    preset: PromptPreset,
    *,
    cache_dir: Path = PRESET_CACHE_DIR,
    refresh: bool = False,
) -> str:
    """Load a Gutenberg text, sharing the context cache used by other demos."""
    cache_path = _preset_cache_path(preset, cache_dir)
    if cache_path.exists() and not refresh:
        return cache_path.read_text()

    request = urllib.request.Request(preset.url, headers={"User-Agent": "tt-metal-gemma4-prefill-demo/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8-sig")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return text


def preset_prompt_prefix(text: str, preset: PromptPreset, context_len: int) -> str:
    """Estimate a fitting prefix from the supplied whole-document token count."""
    if context_len <= PROMPT_TOKEN_RESERVE:
        raise ValueError(f"context_len must be greater than the {PROMPT_TOKEN_RESERVE}-token prompt reserve")
    target_tokens = min(preset.tokens, context_len - PROMPT_TOKEN_RESERVE)
    if target_tokens == preset.tokens:
        return text
    prefix_characters = max(1, int(len(text) * target_tokens / preset.tokens))
    return text[:prefix_characters]


def _reported_prompt_tokens(result: dict) -> int | None:
    prompt_tokens = result.get("prompt_tokens")
    if isinstance(prompt_tokens, int):
        return prompt_tokens
    match = _OVERSIZED_PROMPT_PATTERN.search(str(result.get("error", "")))
    return int(match.group("tokens")) if match else None


def submit_preset_prefill(
    service_url: str,
    prompt: str,
    request_id: str,
    *,
    preset: PromptPreset,
    context_len: int,
    source_characters: int | None = None,
    timeout: float = 600.0,
    max_attempts: int = 4,
) -> dict:
    """Submit an estimated preset prefix, shrinking only after server rejection."""
    source_characters = source_characters or len(prompt)
    for attempt in range(1, max_attempts + 1):
        result = submit_prefill(service_url, prompt, request_id, timeout)
        prompt_tokens = _reported_prompt_tokens(result)
        fits = result.get("status") == "prefilled" and prompt_tokens is not None
        if fits:
            result.update(
                {
                    "preset": preset.name,
                    "preset_url": preset.url,
                    "preset_total_tokens": preset.tokens,
                    "requested_context_len": context_len,
                    "context_overflow_tokens": max(0, prompt_tokens - context_len),
                    "prefix_characters": len(prompt),
                    "prefix_fraction": round(len(prompt) / source_characters, 6),
                    "fit_attempts": attempt,
                }
            )
            return result

        can_shrink = (
            result.get("status") != "prefilled"
            and prompt_tokens is not None
            and prompt_tokens > context_len
            and len(prompt) > 1
        )
        if not can_shrink:
            return result
        # Keep a one-percent buffer so tokenizer density fluctuations do not
        # require another full prefill in the common case.
        scale = min(0.99, 0.99 * context_len / prompt_tokens)
        prompt = prompt[: max(1, int(len(prompt) * scale))]

    return {
        "request_id": request_id,
        "status": "error",
        "error": f"preset prefix did not fit context_len={context_len} after {max_attempts} attempts",
        "preset": preset.name,
    }


def _request_json(url: str, *, payload: dict | None = None, timeout: float = 600.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers, method="GET" if data is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            details = json.loads(exc.read())
        except (json.JSONDecodeError, UnicodeDecodeError):
            details = {"status": "error", "error": str(exc)}
        details["http_status"] = exc.code
        return details


def submit_prefill(service_url: str, prompt: str, request_id: str, timeout: float = 600.0) -> dict:
    started = time.perf_counter()
    result = _request_json(
        f"{service_url.rstrip('/')}/prefill",
        payload={"request_id": request_id, "prompt": prompt},
        timeout=timeout,
    )
    result["client_time_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080", help="prefill service base URL")
    parser.add_argument("--prompt", action="append", default=[], help="prompt text; may be repeated")
    parser.add_argument(
        "--preset",
        action="append",
        choices=sorted(PROMPT_PRESETS),
        default=[],
        help="cached Gutenberg prompt preset; may be repeated",
    )
    parser.add_argument("--list-presets", action="store_true", help="list prompt presets and exit")
    parser.add_argument(
        "--context-len",
        type=int,
        help="target preset context length in tokens (default: service max_context_len)",
    )
    parser.add_argument(
        "--preset-cache-dir",
        type=Path,
        default=PRESET_CACHE_DIR,
        help="download cache shared with the tt_transformers demos",
    )
    parser.add_argument("--refresh-presets", action="store_true", help="redownload selected presets")
    parser.add_argument(
        "--prompt-file",
        action="append",
        type=Path,
        default=[],
        help="file whose complete contents form one prompt; may be repeated",
    )
    parser.add_argument("--repeat", type=int, default=1, help="submit each prompt this many times")
    parser.add_argument("--concurrency", type=int, default=1, help="number of client request threads")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-request timeout in seconds")
    parser.add_argument("--skip-health-check", action="store_true", help="submit without first calling /health")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _create_parser().parse_args(argv)
    if args.list_presets:
        for preset in PROMPT_PRESETS.values():
            print(f"{preset.name}\t{preset.tokens}\t{preset.url}")
        return 0
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive")
    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be positive")

    health = None
    if not args.skip_health_check:
        health = _request_json(f"{args.url.rstrip('/')}/health", timeout=args.timeout)
        _print_json({"health": health})
        if health.get("status") != "ready":
            return 1

    preset_context_len = args.context_len
    if args.preset:
        if preset_context_len is None and health is not None:
            preset_context_len = health.get("max_context_len")
        if not isinstance(preset_context_len, int) or preset_context_len <= 0:
            raise SystemExit("preset requests need --context-len or a /health response with max_context_len")
        if health is not None and preset_context_len > health.get("max_context_len", preset_context_len):
            raise SystemExit(
                f"--context-len={preset_context_len} exceeds service max_context_len={health['max_context_len']}"
            )

    prompt_specs = [(prompt, None, None) for prompt in args.prompt]
    prompt_specs.extend((path.read_text(), None, None) for path in args.prompt_file)
    for preset_name in args.preset:
        preset = PROMPT_PRESETS[preset_name]
        text = load_preset_text(preset, cache_dir=args.preset_cache_dir, refresh=args.refresh_presets)
        prompt = preset_prompt_prefix(text, preset, preset_context_len)
        prompt_specs.append((prompt, preset, len(text)))
    if not prompt_specs:
        prompt_specs = [("Explain sliding-window attention in one paragraph.", None, None)]

    jobs = []
    request_number = 0
    for _ in range(args.repeat):
        for prompt, preset, source_characters in prompt_specs:
            request_number += 1
            request_id = f"{preset.name if preset else 'request'}-{request_number:04d}"
            jobs.append((prompt, request_id, preset, source_characters))

    failed = False
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        for prompt, request_id, preset, source_characters in jobs:
            if preset is None:
                future = executor.submit(submit_prefill, args.url, prompt, request_id, args.timeout)
            else:
                future = executor.submit(
                    submit_preset_prefill,
                    args.url,
                    prompt,
                    request_id,
                    preset=preset,
                    context_len=preset_context_len,
                    source_characters=source_characters,
                    timeout=args.timeout,
                )
            futures[future] = request_id
        for future in as_completed(futures):
            request_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"request_id": request_id, "status": "error", "error": str(exc)}
            failed |= result.get("status") != "prefilled"
            _print_json(result)

    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
