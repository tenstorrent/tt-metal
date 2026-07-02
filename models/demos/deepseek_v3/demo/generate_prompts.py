# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
deepseek_api_batch.py
=====================

The sane way to do this: skip local weight loading entirely. Read prompts
from the same JSON format the Tenstorrent script uses, call the DeepSeek
API for each one, save results as you go (resumable), and optionally emit
a .refpt-compatible file.

Prompt JSON format (same as demo_aime24_gpqa_short.json):
    either a plain list: [{"prompt": "..."}, ...]
    or a dict with key:  {"prompts": [{"prompt": "..."}, ...]}

Quick start
-----------
    export DEEPSEEK_API_KEY=sk-...
    pip install openai tqdm

    # plain: one prompt/response JSON per entry
    python deepseek_api_batch.py \
        --prompts-json demo_aime24_gpqa_short.json \
        --output results.json \
        --num-prompts 256 \
        --max-new-tokens 128

    # with top-5 token captures (top_logprobs):
    python deepseek_api_batch.py \
        --prompts-json demo_aime24_gpqa_short.json \
        --output results.json \
        --num-prompts 256 \
        --max-new-tokens 128 \
        --capture-top-logprobs 5

    # build a .refpt-compatible file from results:
    python deepseek_api_batch.py --make-refpt \
        --results results.json --refpt-output teacher.refpt

Notes
-----
* Results JSON is written compactly (no indentation). `top_logprobs_per_step`
  uses v2 encoding: each `top` candidate is `[token, logprob]` plus an optional
  third element for raw `bytes` only when it differs from UTF-8(`token`).
  `make-refpt` accepts both this and the older verbose per-candidate objects.
* `deepseek-chat` supports logprobs (up to 20). `deepseek-reasoner` does not.
* Any other OpenAI-compatible endpoint works: pass --base-url and --api-key.
  e.g. OpenRouter, Together, Fireworks, or your own local server.
* IMPORTANT: the DeepSeek API returns **post-temperature** logprobs.
  With temperature=0 the distribution collapses: top-1 gets logprob 0.0
  and every other candidate is -9999.0. Use temperature=1.0 (now the
  default when capturing logprobs) to get the model's actual distribution.
"""

from __future__ import annotations

# Standard library imports used for CLI parsing, JSON I/O, filesystem ops,
# timing, and simple concurrency.
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Import the OpenAI-compatible client. DeepSeek's API is compatible with this SDK.
try:
    from openai import OpenAI
except ImportError:
    # Surface an actionable install hint, then re-raise to fail fast.
    print("Please install the OpenAI SDK:  pip install openai", file=sys.stderr)
    raise

# tqdm is optional; if missing we just run without a progress bar.
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None  # type: ignore

# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_prompts(json_path: Path, n: int) -> list[str]:
    # Load and parse the JSON file containing prompts.
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Support either a plain list or {"prompts": [...]}.
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "prompts" in data:
        items = data["prompts"]
    else:
        raise ValueError(f"JSON must be a list or have a 'prompts' key: {json_path}")

    # Collect the first `n` non-empty prompts.
    prompts: list[str] = []
    for item in items:
        if len(prompts) >= n:
            break
        # Each item is expected to be an object with a "prompt" field.
        if isinstance(item, dict) and "prompt" in item:
            p = str(item["prompt"]).strip()
            if p:
                prompts.append(p)
    # Fail loudly if the dataset does not contain enough prompts.
    if len(prompts) < n:
        raise ValueError(f"Requested {n} prompts but only found {len(prompts)} in {json_path}")
    return prompts


# ---------------------------------------------------------------------------
# Resumable result store
# ---------------------------------------------------------------------------


def load_existing(output_path: Path) -> tuple[dict[int, dict], dict[str, Any] | None]:
    """Return ({prompt_idx: entry}, existing header). Tolerates missing/corrupt files."""
    # No prior output means start from scratch.
    if not output_path.exists():
        return {}, None
    try:
        # Read previously saved payload.
        with output_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        # Index entries by prompt_idx for O(1) resume checks.
        entries = {int(e["prompt_idx"]): e for e in payload.get("entries", [])}
        # Keep non-entry metadata so we can compare settings when resuming.
        header = {k: v for k, v in payload.items() if k not in {"entries", "num_entries"}}
        return entries, header
    except Exception:
        # Corrupt/partial output should not kill the whole run.
        print(f"[warn] Couldn't read existing {output_path}; starting fresh.", file=sys.stderr)
        return {}, None


def save_results(output_path: Path, header: dict, entries_by_idx: dict[int, dict]) -> None:
    """Atomic save: write to .tmp, then rename."""
    # Write to a temporary path first so interrupted writes do not corrupt output.
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    # Stable ordering makes diffs cleaner and resumability deterministic.
    ordered = [entries_by_idx[i] for i in sorted(entries_by_idx)]
    payload = {**header, "num_entries": len(ordered), "entries": ordered}
    with tmp.open("w", encoding="utf-8") as f:
        # Compact serialization: logprob payloads are huge; avoid indent/whitespace.
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    # Atomic replace on the same filesystem.
    os.replace(tmp, output_path)


# ---------------------------------------------------------------------------
# Logprob JSON compaction (smaller files, same information for refpt export)
# ---------------------------------------------------------------------------


def _utf8_code_units(token_str: str) -> list[int] | None:
    """UTF-8 bytes of token string; None if encoding fails."""
    try:
        return list(token_str.encode("utf-8"))
    except Exception:
        return None


def _bytes_to_store(token_str: str, api_bytes: list[int] | None) -> list[int] | None:
    """
    Persist raw API bytes only when they are not exactly the UTF-8 encoding of
    `token_str`. Omitting the common case removes the largest JSON cost.
    """
    if not api_bytes:
        return None
    enc = _utf8_code_units(token_str)
    if enc is not None and enc == api_bytes:
        return None
    return api_bytes


def _round_lp(x: float) -> float:
    # Enough precision for top-k ranking / probability math; shrinks JSON a lot.
    return round(float(x), 5)


def _expand_top_candidate(obj: Any) -> dict[str, Any]:
    """Normalize one top-k entry from legacy dict or compact [token, lp, bytes?]."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        out: dict[str, Any] = {"token": obj[0], "logprob": float(obj[1])}
        if len(obj) >= 3 and obj[2] is not None:
            out["bytes"] = obj[2]
        return out
    raise TypeError(f"Bad top_logprob candidate: {type(obj)!r}")


def iter_step_top_candidates(step: dict[str, Any]) -> list[dict[str, Any]]:
    """Top-k list for one generation step (legacy or compact on-disk format)."""
    raw = step.get("top") or []
    return [_expand_top_candidate(x) for x in raw]


# ---------------------------------------------------------------------------
# Single-prompt worker
# ---------------------------------------------------------------------------


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
    # Build chat completion request arguments.
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": False,
    }
    # Ask API for token-level alternatives only if requested.
    if top_logprobs > 0:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = top_logprobs

    # Time each API call for per-entry latency visibility.
    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    elapsed = time.time() - t0

    # We only request one completion, so read choices[0].
    choice = resp.choices[0]
    out: dict[str, Any] = {
        "generated_text": choice.message.content or "",
        "finish_reason": choice.finish_reason,
        "elapsed_seconds": round(elapsed, 4),
    }
    # Include token usage metrics when the endpoint returns them.
    if getattr(resp, "usage", None) is not None:
        out["usage"] = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }

    # Per-token captures -------------------------------------------------------
    # For each generated position we store:
    #   token / logprob / bytes?    -> the chosen token at this step (bytes only
    #                                  if not derivable as UTF-8(token))
    #   top[]                       -> top-K as compact [token, logprob] or
    #                                  [token, logprob, bytes] when bytes differ
    #                                  from UTF-8(token) (API edge cases).
    if top_logprobs > 0 and choice.logprobs is not None:
        # One step per generated token.
        steps = []
        for tok in choice.logprobs.content or []:
            top_candidates: list[list[Any]] = []
            for t in tok.top_logprobs or []:
                api_b = list(t.bytes) if getattr(t, "bytes", None) else None
                row: list[Any] = [t.token, _round_lp(t.logprob)]
                stored = _bytes_to_store(t.token, api_b)
                if stored is not None:
                    row.append(stored)
                top_candidates.append(row)

            step_obj: dict[str, Any] = {
                "token": tok.token,
                "logprob": _round_lp(tok.logprob),
                "top": top_candidates,
            }
            chosen_b = list(tok.bytes) if getattr(tok, "bytes", None) else None
            b_out = _bytes_to_store(tok.token, chosen_b)
            if b_out is not None:
                step_obj["bytes"] = b_out
            steps.append(step_obj)
        out["top_logprobs_per_step"] = steps
    return out


def call_with_retries(fn, *, retries: int = 5, backoff: float = 2.0):
    # Keep the last exception so we can surface it after final retry.
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            # Exponential backoff: 1, 2, 4, 8, ...
            wait = backoff**attempt
            print(f"[warn] attempt {attempt+1}/{retries} failed ({e!r}); sleeping {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
    # Wrap in a clear terminal error after all attempts fail.
    raise RuntimeError(f"All retries exhausted: {last!r}")


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------


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
    # Warn if settings likely produce low-information top-k alternatives.
    if capture_top_logprobs > 1 and temperature <= 0:
        print(
            "[warn] temperature=0 often yields degenerate top-k logprobs "
            "(chosen token + -9999.0 sentinel fillers). Use --temperature 0.2+ "
            "for more informative alternatives.",
            file=sys.stderr,
        )
    # Read exactly `num_prompts` prompts from input JSON.
    prompts = load_prompts(prompts_json, num_prompts)
    print(f"Loaded {len(prompts)} prompts from {prompts_json}")

    # Load any prior progress to support resume.
    existing, existing_header = load_existing(output)
    # Only schedule prompts that are not already in saved output.
    todo_indices = [i for i in range(len(prompts)) if i not in existing]
    print(f"{len(existing)} already done, {len(todo_indices)} to go.")

    # Create API client once and share across worker threads.
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Persist run metadata with results for traceability/resume safety checks.
    header = {
        # v2: compact top_logprobs_per_step rows + compact JSON on disk.
        "format_version": "deepseek_api_batch_v2",
        "model": model,
        "base_url": base_url,
        "source_prompts_json": str(prompts_json),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "capture_top_logprobs": capture_top_logprobs,
    }
    if existing_header:
        # Flag major setting drifts when appending to old outputs.
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
        # Nothing new to call.
        save_results(output, header, existing)
        print(f"All {len(prompts)} prompts already present in {output}")
        return

    # Show progress only when tqdm is available.
    progress = tqdm(total=len(todo_indices), desc="API calls", unit="prompt") if tqdm else None

    def worker(idx: int) -> tuple[int, dict]:
        # Retry transient network/rate-limit failures for robustness.
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
        # Keep source mapping in each row for easier downstream joins/debugging.
        entry["prompt_idx"] = idx
        entry["prompt"] = prompts[idx]
        return idx, entry

    # Start with existing results so resume is additive.
    results_by_idx = dict(existing)
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            # Submit one future per remaining prompt.
            futures = {pool.submit(worker, i): i for i in todo_indices}
            # Save progress periodically (every N results) so we can resume.
            save_every = max(1, min(20, len(todo_indices) // 10 or 1))
            done_since_save = 0
            for fut in as_completed(futures):
                try:
                    # Merge completed result into in-memory index.
                    idx, entry = fut.result()
                    results_by_idx[idx] = entry
                    done_since_save += 1
                    if progress is not None:
                        progress.update(1)
                    if done_since_save >= save_every:
                        # Periodic checkpoint keeps long runs resumable.
                        save_results(output, header, results_by_idx)
                        done_since_save = 0
                except Exception as e:
                    # Keep run alive even if one prompt ultimately fails.
                    print(f"[error] prompt {futures[fut]}: {e!r}", file=sys.stderr)
    finally:
        if progress is not None:
            progress.close()
        # Final save in finally to preserve all completed work.
        save_results(output, header, results_by_idx)

    if capture_top_logprobs > 0:
        # Compute a health signal for top-k richness.
        total_steps = 0
        sparse_steps = 0
        for entry in results_by_idx.values():
            for step in entry.get("top_logprobs_per_step") or []:
                total_steps += 1
                if len(iter_step_top_candidates(step)) <= 1:
                    sparse_steps += 1

        # Warn if nearly all steps have only one viable candidate.
        if total_steps > 0 and (sparse_steps / total_steps) > 0.9:
            print(
                "[warn] More than 90% of generation steps have <=1 returned "
                "top candidate. Increase --temperature for richer top-k values.",
                file=sys.stderr,
            )

    print(f"Wrote {len(results_by_idx)} entries to {output}")


# ---------------------------------------------------------------------------
# Optional: build a .refpt-compatible torch file from the API results.
# Only fills in what the API can actually provide.
# ---------------------------------------------------------------------------


def _api_token_to_id(tokenizer, token_str: str, token_bytes: list[int] | None) -> int:
    """
    Best-effort mapping of an API-returned token back to a single token ID.
    The API returns tokens as decoded strings (plus the raw UTF-8 bytes), not
    IDs. We try encoding the string first; if that produces != 1 id we retry
    with the bytes-derived string; as a last resort we take the first id.
    """

    def _try(text: str) -> int | None:
        try:
            # Convert text to token IDs under this tokenizer.
            ids = tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return None
        # Accept only exact one-token matches (ideal case).
        if len(ids) == 1:
            return ids[0]
        return None

    # First pass: use API string token directly.
    out = _try(token_str)
    if out is not None:
        return out
    if token_bytes:
        try:
            # Fallback: decode raw bytes and retry one-token mapping.
            decoded = bytes(token_bytes).decode("utf-8", errors="replace")
        except Exception:
            decoded = None
        if decoded is not None:
            out = _try(decoded)
            if out is not None:
                return out
    # Final fallback — grab the first id of whatever (multi-id) encoding we get.
    try:
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        return ids[0] if ids else 0
    except Exception:
        return 0


def make_refpt(results_path: Path, refpt_output: Path, *, model_path: str | None = None) -> None:
    # Lazy import keeps API-call path lightweight when torch is not installed.
    import torch  # local import so the main path doesn't require torch
    from transformers import AutoTokenizer  # noqa

    # Load previously captured JSON results.
    with results_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    entries_in = payload.get("entries", [])
    if not entries_in:
        raise ValueError(f"No entries in {results_path}")

    # Tokenizer is needed to convert text -> token IDs (the API returns text).
    tok_path = model_path or os.getenv("DEEPSEEK_V3_HF_MODEL")
    if not tok_path:
        raise RuntimeError(
            "To build a .refpt you need a local DeepSeek V3 tokenizer directory. "
            "Pass --model-path or set DEEPSEEK_V3_HF_MODEL. (You don't need model "
            "weights — just the tokenizer files: tokenizer.json etc.)"
        )
    # Load tokenizer only (no model weights) for ID conversions.
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    # Cache special token IDs in output metadata.
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id
    pad_id = tokenizer.pad_token_id

    refpt_entries = []
    for entry in entries_in:
        # Pull prompt/response text from API results.
        prompt_text = entry["prompt"]
        gen_text = entry["generated_text"]

        # Rebuild prompt-side tokenization using chat template.
        raw_prompt_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=True,
        )
        # Tokenize generated continuation exactly as plain text.
        gen_tokens = tokenizer.encode(gen_text, add_special_tokens=False)
        # Build full sequence used by the reference format.
        full_tokens = list(raw_prompt_tokens) + list(gen_tokens)
        prompt_len = len(raw_prompt_tokens)
        total_len = len(full_tokens)

        # Per-position tensors. Zeros for prompt positions (no prediction
        # captured there). For generated positions we fill in:
        #   top1_tokens[i]     = the chosen token's id
        #   top5_tokens[i, :]  = top-5 candidate ids, ordered by prob desc
        #   top5_logprobs[i,:] = their logprobs (for downstream probability math)
        top1 = torch.zeros(total_len, dtype=torch.int32)
        top5 = torch.zeros(total_len, 5, dtype=torch.int32)
        top5_lp = torch.full((total_len, 5), float("-inf"), dtype=torch.float32)

        # Fill generated region from stored per-step top-logprob payload.
        steps = entry.get("top_logprobs_per_step") or []
        for i, step in enumerate(steps):
            pos = prompt_len + i
            # Guard against mismatches between text tokenization and step count.
            if pos >= total_len:
                break

            # Top-1 (the chosen/sampled token)
            chosen_id = _api_token_to_id(tokenizer, step["token"], step.get("bytes"))
            top1[pos] = chosen_id

            # Top-5 candidates (API-ordered by logprob desc)
            tops = iter_step_top_candidates(step)[:5]
            ids: list[int] = []
            lps: list[float] = []
            for t in tops:
                ids.append(_api_token_to_id(tokenizer, t["token"], t.get("bytes")))
                lps.append(float(t["logprob"]))
            # Refpt expects fixed top-5 width, so right-pad with zeros/-inf.
            while len(ids) < 5:
                ids.append(0)
                lps.append(float("-inf"))
            top5[pos] = torch.tensor(ids[:5], dtype=torch.int32)
            top5_lp[pos] = torch.tensor(lps[:5], dtype=torch.float32)

        # Emit one refpt entry per original prompt.
        refpt_entries.append(
            {
                "prompt_idx": int(entry["prompt_idx"]),
                "prompt": prompt_text,
                "decoded_generated_text": gen_text,
                "reference_tokens": torch.tensor([full_tokens], dtype=torch.int32),
                "prompt_tokens": torch.tensor([raw_prompt_tokens], dtype=torch.int32),
                "generated_tokens": torch.tensor([gen_tokens], dtype=torch.int32),
                "top1_tokens": top1,  # [total_len]
                "top5_tokens": top5,  # [total_len, 5]
                "top5_logprobs": top5_lp,  # [total_len, 5]
                "tf_prompt_len": prompt_len,
                "max_new_tokens": payload.get("max_new_tokens"),
                "token_ids_meta": {"bos_id": bos_id, "eos_id": eos_id, "pad_id": pad_id},
            }
        )

    result = {
        "format_version": "multi_prompt_v1",
        "num_prompts": len(refpt_entries),
        "max_new_tokens": payload.get("max_new_tokens"),
        "model_path": tok_path,
        "source_prompts_json": payload.get("source_prompts_json"),
        "token_ids_meta": {"bos_id": bos_id, "eos_id": eos_id, "pad_id": pad_id},
        "entries": refpt_entries,
        "_notes": (
            "Built from DeepSeek API outputs. Per generated position: "
            "top1_tokens=chosen token id, top5_tokens=top-5 candidate ids (with "
            "top5_logprobs), derived from top_logprobs=5. "
            "teacher_mass_candidates and topk_candidates (k>20) are unavailable "
            "— the API does not expose the full probability distribution."
        ),
    }
    # Mirror the original's convenience fields pointing at entries[0]
    if refpt_entries:
        e0 = refpt_entries[0]
        result.update(
            {
                k: e0[k]
                for k in (
                    "reference_tokens",
                    "prompt_tokens",
                    "generated_tokens",
                    "top1_tokens",
                    "top5_tokens",
                    "top5_logprobs",
                    "tf_prompt_len",
                    "prompt",
                    "decoded_generated_text",
                )
            }
        )

    # Ensure output directory exists, then write torch payload.
    refpt_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, refpt_output)
    print(f"Saved {len(refpt_entries)} entries to {refpt_output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    # Define CLI interface and defaults.
    p = argparse.ArgumentParser(description="Batch DeepSeek API caller with resume + optional refpt export.")
    sub_action = p.add_mutually_exclusive_group()
    sub_action.add_argument(
        "--make-refpt", action="store_true", help="Skip API calls; build a .refpt file from an existing results JSON."
    )

    # --- common ---
    p.add_argument("--prompts-json", type=Path, help="Path to prompts JSON (AIME/GPQA style).")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results.json"),
        help="Path to the JSON result file (read + written; resumable).",
    )
    p.add_argument("--num-prompts", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-R1", help="Model name on the inference provider.")
    p.add_argument(
        "--base-url",
        default="https://api.together.xyz/v1",
        help="Any OpenAI-compatible endpoint (Together, OpenRouter, Fireworks, DeepSeek, etc.)",
    )
    p.add_argument(
        "--api-key", default=os.getenv("TOGETHER_API_KEY", ""), help="API key. Defaults to TOGETHER_API_KEY env var."
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Defaults to 1.0 when --capture-top-logprobs > 0 "
        "(needed for meaningful logprobs), else 0.0.",
    )
    p.add_argument("--concurrency", type=int, default=8, help="Parallel API calls. Bump up if you're rate-limit-free.")
    p.add_argument(
        "--capture-top-logprobs",
        type=int,
        default=5,
        help="How many candidates to return at each generated position "
        "(max 20, API limit). Default 5. The chosen (top-1) token is "
        "always captured separately. Set to 0 to disable.",
    )
    # --- refpt build ---
    p.add_argument("--results", type=Path, help="(With --make-refpt) input results JSON.")
    p.add_argument(
        "--refpt-output", type=Path, default=Path("teacher.refpt"), help="(With --make-refpt) output .refpt path."
    )
    p.add_argument(
        "--model-path", type=str, default=None, help="(With --make-refpt) path to a local DeepSeek tokenizer directory."
    )

    # Parse command-line arguments once.
    args = p.parse_args()

    if args.make_refpt:
        # Refpt mode bypasses API calls entirely.
        if not args.results:
            p.error("--make-refpt requires --results <path>")
        make_refpt(args.results, args.refpt_output, model_path=args.model_path)
        return

    # API mode requires prompts and credentials.
    if not args.prompts_json:
        p.error("--prompts-json is required (or use --make-refpt).")
    if not args.api_key:
        p.error("No API key. Set DEEPSEEK_API_KEY or pass --api-key.")

    # Choose context-aware default temperature when user omits it.
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

    # Remind users that temp=0 defeats meaningful top-k diversity.
    if args.capture_top_logprobs > 0 and args.temperature == 0.0:
        print(
            "[warn] temperature=0.0 with logprob capture will produce useless "
            "top-K logprobs (all -9999.0 except the greedy pick). Consider "
            "using --temperature 1.0 for meaningful logprobs.",
            file=sys.stderr,
        )

    # Run API batch workflow.
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
    # Standard Python entrypoint.
    main()
