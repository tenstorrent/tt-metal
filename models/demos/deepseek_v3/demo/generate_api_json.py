"""
Generate responses for prompt JSON using an OpenAI-compatible API.

Input can be [{"prompt": "..."}] or {"prompts": [...]}. Results are saved
incrementally so interrupted runs can resume.

Example:
  python tt-metal/models/demos/deepseek_v3/demo/generate_api_json.py --prompts-json /path/to/prompts.json --output results.json --api-key "$TOGETHER_API_KEY"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing, nullcontext
from pathlib import Path
from typing import Any, Callable

try:
    from openai import OpenAI
except ImportError:
    print("Please install the OpenAI SDK:  pip install openai", file=sys.stderr)
    raise

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def load_prompts(json_path: Path, n: int) -> list[str]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "prompts" in data:
        items = data["prompts"]
    else:
        raise ValueError(f"JSON must be a list or have a 'prompts' key: {json_path}")

    prompts: list[str] = []
    for item in items:
        if len(prompts) >= n:
            break
        if isinstance(item, dict) and "prompt" in item:
            prompt_text = str(item["prompt"]).strip()
            if prompt_text:
                prompts.append(prompt_text)
    if len(prompts) < n:
        raise ValueError(f"Requested {n} prompts but only found {len(prompts)} in {json_path}")
    return prompts


def load_existing(
    output_path: Path,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any] | None]:
    """Load existing output keyed by prompt index."""
    if not output_path.exists():
        return {}, None
    try:
        with output_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        entries = {int(entry["prompt_idx"]): entry for entry in payload.get("entries", [])}
        header = {key: value for key, value in payload.items() if key not in {"entries", "num_entries"}}
        return entries, header
    except Exception:
        print(f"[warn] Couldn't read existing {output_path}; starting fresh.", file=sys.stderr)
        return {}, None


def save_results(
    output_path: Path,
    header: dict[str, Any],
    entries_by_idx: dict[int, dict[str, Any]],
) -> None:
    """Save results atomically."""
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    ordered_entries = [entries_by_idx[index] for index in sorted(entries_by_idx)]
    payload = {**header, "num_entries": len(ordered_entries), "entries": ordered_entries}
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    os.replace(tmp_path, output_path)


def call_one(
    client: OpenAI,
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_logprobs: int,
) -> dict:
    """
    Call the API for one prompt. Returns a dict with the content plus optional
    logprob-derived top-k tokens per generated position.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if top_logprobs > 0:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = top_logprobs

    start_time = time.time()
    resp = client.chat.completions.create(**kwargs)
    elapsed = time.time() - start_time

    choice = resp.choices[0]
    out: dict[str, Any] = {
        "generated_text": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "elapsed_seconds": elapsed,
    }
    usage = getattr(resp, "usage", None)
    if usage is not None:
        out["usage"] = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    if top_logprobs > 0 and choice.logprobs is not None:
        steps = []
        for tok in choice.logprobs.content or []:
            top_candidates = []
            for t in tok.top_logprobs or []:
                top_candidates.append(
                    {
                        "token": t.token,
                        "logprob": float(t.logprob),
                        "bytes": list(t.bytes) if getattr(t, "bytes", None) else None,
                    }
                )

            steps.append(
                {
                    "token": tok.token,
                    "logprob": float(tok.logprob),
                    "bytes": list(tok.bytes) if getattr(tok, "bytes", None) else None,
                    "top": top_candidates,
                }
            )
        out["top_logprobs_per_step"] = steps
    return out


def call_with_retries(
    fn: Callable[[], dict[str, Any]],
    *,
    retries: int = 5,
    backoff: float = 2.0,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as error:
            last_error = error
            wait = backoff**attempt
            print(
                f"[warn] attempt {attempt + 1}/{retries} failed ({error!r}); sleeping {wait:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"All retries exhausted: {last_error!r}")


def run_batch(
    prompts_json: Path,
    output: Path,
    *,
    num_prompts: int,
    max_new_tokens: int,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    concurrency: int,
    capture_top_logprobs: int,
) -> None:
    if capture_top_logprobs > 1 and temperature <= 0:
        print(
            "[warn] temperature=0 often yields degenerate top-k logprobs "
            "(chosen token + -9999.0 sentinel fillers). Use --temperature 0.2+ "
            "for more informative alternatives.",
            file=sys.stderr,
        )
    prompts = load_prompts(prompts_json, num_prompts)
    print(f"Loaded {len(prompts)} prompts from {prompts_json}")

    existing, existing_header = load_existing(output)
    todo_indices = [i for i in range(len(prompts)) if i not in existing]
    print(f"{len(existing)} already done, {len(todo_indices)} to go.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    header = {
        "format_version": "deepseek_api_batch_v1",
        "model": model,
        "base_url": base_url,
        "source_prompts_json": str(prompts_json),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "capture_top_logprobs": capture_top_logprobs,
    }
    if existing_header:
        compare_keys = (
            "model",
            "base_url",
            "max_new_tokens",
            "temperature",
            "capture_top_logprobs",
        )
        mismatched = [k for k in compare_keys if k in existing_header and existing_header.get(k) != header.get(k)]
        if mismatched:
            details = ", ".join(f"{k}: old={existing_header.get(k)!r} new={header.get(k)!r}" for k in mismatched)
            print(
                "[warn] Existing output metadata differs from current args; " f"resume may mix settings ({details}).",
                file=sys.stderr,
            )

    if not todo_indices:
        save_results(output, header, existing)
        print(f"All {len(prompts)} prompts already present in {output}")
        return

    progress_scope = (
        closing(tqdm(total=len(todo_indices), desc="API calls", unit="prompt")) if tqdm else nullcontext(None)
    )

    def worker(idx: int) -> tuple[int, dict]:
        entry = call_with_retries(
            lambda: call_one(
                client,
                model,
                prompts[idx],
                max_new_tokens,
                temperature,
                capture_top_logprobs,
            )
        )
        entry["prompt_idx"] = idx
        entry["prompt"] = prompts[idx]
        return idx, entry

    results_by_idx = dict(existing)
    run_error: BaseException | None = None
    with progress_scope as progress:
        try:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {pool.submit(worker, i): i for i in todo_indices}
                save_every = max(1, min(20, len(todo_indices) // 10 or 1))
                done_since_save = 0
                for fut in as_completed(futures):
                    try:
                        idx, entry = fut.result()
                    except Exception as error:
                        print(f"[error] prompt {futures[fut]}: {error!r}", file=sys.stderr)
                        continue
                    results_by_idx[idx] = entry
                    done_since_save += 1
                    if progress is not None:
                        progress.update(1)
                    if done_since_save >= save_every:
                        save_results(output, header, results_by_idx)
                        done_since_save = 0
        except BaseException as error:
            run_error = error

    save_results(output, header, results_by_idx)
    if run_error is not None:
        raise run_error

    if capture_top_logprobs > 0:
        total_steps = 0
        sparse_steps = 0
        for entry in results_by_idx.values():
            for step in entry.get("top_logprobs_per_step") or []:
                total_steps += 1
                if len(step.get("top", [])) <= 1:
                    sparse_steps += 1

        if total_steps > 0 and (sparse_steps / total_steps) > 0.9:
            print(
                "[warn] More than 90% of generation steps have <=1 returned "
                "top candidate. Increase --temperature for richer top-k values.",
                file=sys.stderr,
            )

    print(f"Wrote {len(results_by_idx)} entries to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DeepSeek API caller with resumable JSON output.")
    parser.add_argument("--prompts-json", type=Path, help="Path to prompts JSON (AIME/GPQA style).")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results.json"),
        help="Path to the JSON result file (read + written; resumable).",
    )
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1", help="Model name on the provider.")
    parser.add_argument(
        "--base-url",
        default="https://api.together.xyz/v1",
        help="Any OpenAI-compatible endpoint (Together, OpenRouter, Fireworks, DeepSeek, etc.)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("TOGETHER_API_KEY", ""),
        help="API key. Defaults to TOGETHER_API_KEY.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=("Sampling temperature. Defaults to 1.0 when --capture-top-logprobs > 0, else 0.0."),
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel API calls.")
    parser.add_argument(
        "--capture-top-logprobs",
        type=int,
        default=5,
        help=("How many candidates to return at each generated position (max 20). Set to 0 to disable."),
    )
    args = parser.parse_args()

    if not args.prompts_json:
        parser.error("--prompts-json is required.")
    if not args.api_key:
        parser.error("No API key. Set TOGETHER_API_KEY or pass --api-key.")

    if args.temperature is None:
        if args.capture_top_logprobs > 0:
            args.temperature = 1.0
            print(
                "[info] --temperature not set; defaulting to 1.0 because "
                "--capture-top-logprobs is enabled. (temperature=0 causes the "
                "API to return post-temperature logprobs where the top-1 gets "
                "logprob=0.0 and everything else is -9999.0.)",
                file=sys.stderr,
            )
        else:
            args.temperature = 0.0

    if args.capture_top_logprobs > 0 and args.temperature == 0.0:
        print(
            "[warn] temperature=0.0 with logprob capture will produce useless "
            "top-K logprobs (all -9999.0 except the greedy pick). Consider "
            "using --temperature 1.0 for meaningful logprobs.",
            file=sys.stderr,
        )

    run_batch(
        args.prompts_json,
        args.output,
        num_prompts=args.num_prompts,
        max_new_tokens=args.max_new_tokens,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        concurrency=args.concurrency,
        capture_top_logprobs=args.capture_top_logprobs,
    )


if __name__ == "__main__":
    main()
