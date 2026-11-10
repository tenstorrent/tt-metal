# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""LongBench-v2 against a running ``demo/server.py``, or against the official DeepSeek API, over HTTP only.

The serving twin of ``tests/prefill/test_prefill_decode_demo.py``: it asks the same questions of the same model,
in the same thinking template and effort, and grades the answers the same way -- but every question is one
streamed ``POST /v1/chat/completions``, so this script never touches tt-metal or the card. Run the server (see
``demo/server.py``), then::

    python models/experimental/deepseek_v41_flash/demo/longbench_client.py --base-url http://127.0.0.1:8000

For the official API, pass a key -- ``--api-key`` or ``DEEPSEEK_API_KEY`` -- and the run goes to
``https://api.deepseek.com`` instead (``--base-url`` overrides it); the client then authenticates with
``Authorization: Bearer <key>``, and the defaults it does not share with the server follow (``--model
deepseek-flash``, the checkpoint's published catalog id, with ``--max-tokens 8192`` and a context of 131072)::

    DEEPSEEK_API_KEY=sk-... python models/experimental/deepseek_v41_flash/demo/longbench_client.py --count 20

What does not carry over to the API is its own: ``thinking`` is not a request field there (the model id picks the
mode, so ``deepseek-flash`` is the served model and ``--reasoning-effort`` -- a served model's field -- is not
sent), and there are no server-side sessions to delete, so a question's ``user`` only labels the request, and
``cached_tokens`` is the API's own accounting. The prompt-length estimate still renders the checkpoint's thinking
template, as the server would, so the fit check errs on the long side against the API's own rendering.

Like the demo it takes the first ``--count`` (20) ``length == "short"`` questions of ``data.json`` in file order
(``--indices`` picks exactly some, ``--seed`` draws at random), prefills each as one prompt and asks for the
answer. ``--longest-first`` keeps that selection and asks it longest prompt first. The prompt is the demo's LongBench-v2 zero-shot user turn (:func:`_longbench_prompt`), rendered by the
server from the chat messages, with ``thinking: true`` and ``reasoning_effort: "max"`` as
``_tokenize_chat(..., "thinking", "max")`` renders it in the demo; the letter is read out of the reply with the
demo's patterns (:func:`_extract_choice`).

What the demo measures on the device, this reports from the wire, per question:

* ``ttft``: request sent to the first streamed token -- the prefill and the first decode step;
* ``prefill tok/s``: ``prompt_tokens / ttft``, so it includes that first step (as ``demo/test_server_perf.py``
  counts it);
* ``tok/s``: ``(completion_tokens - 1) / (last token - first token)``, the reply's decode rate.

``prompt_tokens`` / ``cached_tokens`` / ``completion_tokens`` / ``reasoning_tokens`` are the server's own
``usage`` (``stream_options.include_usage``), so no re-tokenizing of the reply is needed. A server started for
this should be a fresh one: ``cached_tokens`` is 0 only if no earlier question already left that prefix in a
slot, and a nonzero value means the TTFT skipped a prefill.

Every question gets a ``user`` of its own (a unique key), so it starts a new conversation instead of continuing
another's; its session is deleted afterwards (``--keep-sessions`` leaves them) to give its KV pages back. The
tokenizer, needed only to count a question's prompt tokens before sending it, comes from the checkpoint snapshot
(``--tokenizer-dir``); without one the lengths are estimated at ~3.5 characters a token, and the count still
decides what fits.

A question is skipped, as the demo skips it, when its prompt does not fit the context window (from
``GET /v1/models``, or ``--context``); each reply is capped at ``--max-tokens`` (16384, the demo's
``DEEPSEEK_V4_MAX_NEW_TOKENS``; 8192 on the official API), and less when the prompt leaves less room. The
questions LongBench-v2 marks ``medium``/``long`` (``--length``) and the data file (``--longbench``, else
``DEEPSEEK_V4_LONGBENCH``, else ``zai-org/LongBench-v2``'s ``data.json`` in the Hugging Face cache, else
``~/smanoj/data.json``) are the demo's.

Sampling: no ``temperature`` is sent unless ``--temperature`` is given, so the server's own applies -- the
default server samples on the device (``--device-sampling``), which rejects a request's ``temperature`` outright.
For a greedy run, start the server with ``--no-device-sampling`` and pass ``--temperature 0``. The official API
takes a request's ``temperature`` too.

A few questions at once (``--concurrency``; the server runs up to one per stage, eight by default) cut the wall
time, each on a client and a session of its own. ``--json`` writes the per-question records out for a sweep
script to compare.

Known-correct regression, the twin of ``test_prefill_decode_longbench_known_correct``: pass the indices the demo
answered correctly and require them all::

    python models/experimental/deepseek_v41_flash/demo/longbench_client.py --indices 1,6,21 --exit-on-wrong

(V4.1's ``_KNOWN_CORRECT_LONGBENCH`` is still empty -- V4's list does not carry over, its passages are past
V4.1's context -- so the indices are the run's own.)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from models.experimental.deepseek_v4_flash.demo.client import Client, OpenAIError
from models.experimental.deepseek_v41_flash import encoding_dsv41

_DEFAULT_BASE_URL = "http://127.0.0.1:8000"
_DEFAULT_MODEL = "deepseek-v4.1-flash"
# The official API: a key in ``--api-key`` (or DEEPSEEK_API_KEY) points the run here instead of at the server.
_DEFAULT_API_BASE_URL = "https://api.deepseek.com"
# V4.1 Flash 1017's published catalog id (``server.py``'s ``_MODEL_NAME``); the API also lists ``deepseek-v4-pro``.
# Its models pick the mode themselves, so there is no ``thinking`` / ``reasoning_effort`` to send.
_DEFAULT_API_MODEL = "deepseek-flash"
_DEFAULT_MODEL_DIR = "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017"
_DEFAULT_REASONING_EFFORT = "max"  # the demo's effort, and V4.1's highest
_DEFAULT_MAX_TOKENS = 16384  # the demo's DEEPSEEK_V4_MAX_NEW_TOKENS
_DEFAULT_API_MAX_TOKENS = 8192  # per reply, a cap the API takes for either listed model; raise with --max-tokens
_DEFAULT_API_CONTEXT = 131072  # the official API's window, which its /models does not report
_CHARS_PER_TOKEN = 3.5  # only for prompts sent without a tokenizer at hand


# --------------------------------------------------------------------------------------------------------------- #
# LongBench
# --------------------------------------------------------------------------------------------------------------- #
def _longbench_prompt(entry: dict) -> str:
    """The LongBench-v2 zero-shot user turn: the passage, then the question and four choices.

    As ``tests/prefill/test_prefill_decode_demo.py`` renders it, so the questions are the ones the demo asks."""
    return (
        "Please read the following text and answer the question below.\n\n"
        "<text>\n"
        f"{entry['context']}\n"
        "</text>\n\n"
        f"What is the correct answer to this question: {entry['question']}\n"
        "Choices:\n"
        f"(A) {entry['choice_A']}\n"
        f"(B) {entry['choice_B']}\n"
        f"(C) {entry['choice_C']}\n"
        f"(D) {entry['choice_D']}\n\n"
        'Format your response as follows: "The correct answer is (insert answer here)".'
    )


def _extract_choice(generated: str) -> str | None:
    """The letter the model committed to, with the demo's LongBench-v2 answer patterns."""
    text = generated.replace("*", "")
    for pattern in (
        r"The correct answer is \(([A-D])\)",
        r"The correct answer is ([A-D])",
        r"\(([A-D])\)",
        r"\b([A-D])\b",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None


@dataclass
class _Question:
    index: int  # its index in data.json
    name: str  # ``longbench[<index>]``, as the demo names it
    prompt: str
    expected: str  # the answer letter
    prompt_tokens: int = 0  # counted with the tokenizer, or estimated


def _pick_longbench(
    path: Path,
    count: int,
    seed: int | None,
    indices: list[int] | None,
    length: str,
    tokenizer,
    effort: str | None = _DEFAULT_REASONING_EFFORT,
) -> list[_Question]:
    """The questions to ask: ``count`` of ``length`` drawn at random with ``seed`` (in file order when ``seed`` is
    None), or exactly ``indices``. ``length == "any"`` takes every question, whatever LongBench calls it."""
    data = json.loads(path.read_text())
    if indices is None:
        pool = [i for i, entry in enumerate(data) if length == "any" or entry.get("length") == length]
        if seed is not None:
            random.Random(seed).shuffle(pool)
        count = min(count, len(pool)) if count > 0 else len(pool)
    else:
        for i in indices:
            if not 0 <= i < len(data):
                raise IndexError(f"{path} has {len(data)} questions, index {i} is out of range")
            if length != "any" and data[i].get("length") != length:
                raise ValueError(f"longbench[{i}] is a {data[i].get('length')!r} question, not a {length!r} one")
        pool, count = list(indices), len(indices)
    print(
        f"{path}: {len(data)} questions; drawing {count} of {len(pool)} "
        + (f"given: {indices}" if indices is not None else "in file order" if seed is None else f"with seed {seed}"),
        flush=True,
    )
    questions = []
    for i in pool:
        entry = data[i]
        prompt = _longbench_prompt(entry)
        question = _Question(i, f"longbench[{i}]", prompt, str(entry["answer"]).strip())
        question.prompt_tokens = _count_prompt_tokens(tokenizer, prompt, effort)
        print(
            f"  {question.name} {entry['_id']} ({entry['domain']} / {entry['sub_domain']}, {entry['length']}, "
            f"~{question.prompt_tokens} prompt tokens, answer {question.expected}): {entry['question']}",
            flush=True,
        )
        questions.append(question)
        if len(questions) == count:
            break
    if not questions:
        raise ValueError(f"no {length!r} question in {path}")
    return questions


def _count_prompt_tokens(tokenizer, prompt: str, effort: str | None) -> int:
    """The prompt's tokens as the server renders it: the chat template around the one user turn, in thinking mode
    at ``effort`` (the demo's ``_tokenize_chat(tokenizer, text, "thinking", "max")``)."""
    if tokenizer is None:
        return round(len(prompt) / _CHARS_PER_TOKEN)
    text = encoding_dsv41.encode_messages(
        [{"role": "user", "content": prompt}], "thinking", reasoning_effort=effort or None
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _load_tokenizer(directory: str | None):
    """The checkpoint's tokenizer, to count a question's prompt tokens; None (lengths estimated) when there is no
    snapshot at ``directory`` or ``transformers`` is not installed."""
    if not directory:
        return None
    try:
        from transformers import AutoTokenizer

        from models.experimental.deepseek_v4_flash.tt.weight_loader import resolve_snapshot_dir

        return AutoTokenizer.from_pretrained(resolve_snapshot_dir(Path(directory).expanduser()))
    except Exception as e:  # noqa: BLE001 - any missing snapshot / package leaves the estimate
        print(f"warning: no tokenizer from {directory} ({e}); prompt lengths are estimated", file=sys.stderr)
        return None


# --------------------------------------------------------------------------------------------------------------- #
# one question
# --------------------------------------------------------------------------------------------------------------- #
@dataclass
class _Result:
    """One question's verdict and what the wire said about it."""

    index: int
    name: str
    expected: str | None = None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    ttft_s: float | None = None
    decode_s: float | None = None
    e2e_s: float | None = None
    text: str = ""
    reasoning: str = ""
    choice: str | None = None
    finish_reason: str | None = None
    error: str | None = None

    @property
    def correct(self) -> bool:
        return self.expected is not None and self.choice == self.expected

    @property
    def prefill_tps(self) -> float | None:
        return self.prompt_tokens / self.ttft_s if self.ttft_s else None

    @property
    def decode_tps(self) -> float | None:
        """The reply's decode rate: every token after the first over the time from the first to the last."""
        if not self.decode_s or self.completion_tokens < 2:
            return None
        return (self.completion_tokens - 1) / self.decode_s

    def record(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "expected": self.expected,
            "choice": self.choice,
            "correct": self.correct,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "ttft_s": self.ttft_s,
            "decode_s": self.decode_s,
            "e2e_s": self.e2e_s,
            "prefill_tps": self.prefill_tps,
            "decode_tps": self.decode_tps,
            "finish_reason": self.finish_reason,
            "error": self.error,
            "text": self.text,
            "reasoning": self.reasoning,
        }


def _run_one(client: Client, question: _Question, args, user: str, max_tokens: int) -> _Result:
    """Stream one question's reply on ``client`` (as ``user``, a session of its own), then delete its session.

    The request is the demo's: the one user turn, thinking mode at ``--reasoning-effort``. The reply's reasoning
    block and text are kept apart, as ``DeepSeekV4Model``'s generation would have them in one string; the answer
    letter is read from the text, and only if there is none, from the reasoning.

    To the official API (``client.api_key``) the fields are the ones it takes, which differ: ``thinking`` is not a
    field -- the model id picks the mode -- and ``reasoning_effort`` is not either, so a thinking model is asked
    for at ``--reasoning-effort`` or above through ``--model`` (the flag still counts the prompt the way the server
    renders it). A ``user`` key only labels the request there; the conversation is the messages it sends."""
    params = {"user": user, "max_tokens": max_tokens, "stream_options": {"include_usage": True}}
    if client.api_key is None:
        # The server has a thinking template per session; the official API's models are thinking or not by model.
        params["thinking"] = args.thinking
        if args.thinking and args.reasoning_effort:
            params["reasoning_effort"] = args.reasoning_effort
    if args.temperature is not None:
        params["temperature"] = args.temperature
    if args.top_p is not None:
        params["top_p"] = args.top_p

    result = _Result(index=question.index, name=question.name, expected=question.expected)
    result.prompt_tokens = question.prompt_tokens
    usage: dict = {}
    pieces = 0
    t_first = t_last = None
    try:
        t0 = time.perf_counter()
        for chunk in client.chat([{"role": "user", "content": question.prompt}], stream=True, **params):
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or ():
                delta = choice.get("delta") or {}
                reasoning, content = delta.get("reasoning_content") or "", delta.get("content") or ""
                if choice.get("finish_reason") is not None:
                    result.finish_reason = choice["finish_reason"]
                if not (reasoning or content):
                    continue
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                pieces += 1
                result.reasoning += reasoning
                result.text += content
    except OpenAIError as e:
        result.error = str(e)
    finally:
        if not args.keep_sessions:
            try:
                client.reset_session(user)
            except OpenAIError:
                pass  # the turn failed before it made a session, or the server is gone: nothing to give back

    if result.error is None and t_first is None:
        result.error = "the server streamed no tokens"
    if usage:
        result.prompt_tokens = usage.get("prompt_tokens", result.prompt_tokens)
        result.cached_tokens = usage.get(
            "prompt_cache_hit_tokens", (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        )
        result.completion_tokens = usage.get("completion_tokens", pieces)
        result.reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    else:
        result.completion_tokens = pieces
    if t_first is not None:
        result.ttft_s = t_first - t0
        result.decode_s = t_last - t_first
        result.e2e_s = t_last - t0
        result.choice = _extract_choice(result.text) or _extract_choice(result.reasoning)
    return result


def _verdict(result: _Result) -> str:
    if result.error is not None:
        return f"ERROR {result.error}"
    if result.expected is None:
        return "ungraded"
    return "correct" if result.correct else "WRONG"


# --------------------------------------------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------------------------------------------- #
def _run_all(args, questions: list[_Question], context_window: int | None) -> list[_Result]:
    """Ask every question (``--concurrency`` at once), printing each verdict and the running tally as it lands."""
    results: list[_Result | None] = [None] * len(questions)
    lock = threading.Lock()
    done = 0
    graded = 0
    correct = 0
    total = len(questions)

    def ask(position: int, question: _Question) -> None:
        nonlocal done, graded, correct
        # A session and an HTTP connection of its own, so concurrent questions share no state.
        client = Client(args.base_url, model=args.model, timeout=args.timeout, api_key=args.api_key)
        max_tokens = args.max_tokens
        if context_window:
            max_tokens = max(1, min(max_tokens, context_window - question.prompt_tokens - 1))
        try:
            result = _run_one(client, question, args, f"{args.user_prefix}-{uuid.uuid4().hex[:12]}", max_tokens)
        except Exception as e:  # noqa: BLE001 - one bad question must not end the others' run
            result = _Result(index=question.index, name=question.name, expected=question.expected, error=repr(e))
            result.prompt_tokens = question.prompt_tokens
        with lock:
            results[position] = result
            done += 1
            if result.expected is not None:
                graded += 1
                correct += result.correct
            rate = f"{result.decode_tps:.2f} tok/s" if result.decode_tps else "-"
            ttft = f"{result.ttft_s:.2f}s" if result.ttft_s is not None else "-"
            print(
                f"[{done}/{total}] {result.name}: {result.prompt_tokens} prompt tokens "
                f"({result.cached_tokens} cached), {result.completion_tokens} out ({result.reasoning_tokens} "
                f"reasoning), TTFT {ttft}, {rate} -> expected {result.expected}, model "
                f"{result.choice or '?'} {_verdict(result)}"
                + (f"  (running: {correct}/{graded} correct)" if graded else ""),
                flush=True,
            )

    workers = max(1, min(args.concurrency, total))
    if workers == 1:
        for position, question in enumerate(questions):
            ask(position, question)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="longbench") as pool:
            for position, question in enumerate(questions):
                pool.submit(ask, position, question)
    return [r for r in results if r is not None]


def _print_correctness(results: list[_Result]) -> int:
    """Print each graded question's verdict and the accuracy; returns the number correct."""
    graded = [r for r in results if r.expected is not None]
    correct = sum(r.correct for r in graded)
    lines = ["", f"=== correctness ({len(graded)} question(s)) ==="]
    for i, r in enumerate(graded, 1):
        lines.append(
            f"{i:>3}. {r.name:<18} expected {r.expected}  model {r.choice or '?'}  "
            f"{'correct' if r.correct else _verdict(r)}"
        )
    if graded:
        lines.append(f"accuracy: {correct}/{len(graded)} ({correct / len(graded):.0%})")
    print("\n".join(lines), flush=True)
    return correct


def _print_perf(results: list[_Result]) -> None:
    """One line per question: its prompt, its prefill and decode rates, its answer; then the totals."""
    lines = [
        "",
        "=== end-to-end perf (measured at the client, from the stream) ===",
        (
            f"{'prompt':<18} {'tokens':>7} {'cached':>7} {'TTFT s':>8} {'prefill t/s':>12} {'out':>6} "
            f"{'tok/s':>7} {'e2e s':>7} {'answer':>9}"
        ),
    ]
    for r in results:
        ttft = f"{r.ttft_s:.2f}" if r.ttft_s is not None else "-"
        prefill = f"{r.prefill_tps:.0f}" if r.prefill_tps else "-"
        decode = f"{r.decode_tps:.2f}" if r.decode_tps else "-"
        e2e = f"{r.e2e_s:.1f}" if r.e2e_s is not None else "-"
        answer = "" if r.expected is None else f"{r.choice or '?'}/{r.expected}{'' if r.correct else ' x'}"
        lines.append(
            f"{r.name:<18} {r.prompt_tokens:>7} {r.cached_tokens:>7} {ttft:>8} {prefill:>12} "
            f"{r.completion_tokens:>6} {decode:>7} {e2e:>7} {answer:>9}"
        )
    rates = [r.decode_tps for r in results if r.decode_tps]
    ttfts = [r.ttft_s for r in results if r.ttft_s is not None]
    if ttfts:
        lines.append(f"TTFT, mean             : {sum(ttfts) / len(ttfts):.2f}s")
    if rates:
        lines.append(f"decode, mean           : {sum(rates) / len(rates):.2f} tok/s per user")
    timed = [r for r in results if r.ttft_s is not None]
    if timed:
        prompt_tokens = sum(r.prompt_tokens for r in timed)
        prompt_seconds = sum(r.ttft_s for r in timed)
        lines.append(f"prefill total          : {prompt_tokens} prompt tokens in {prompt_seconds:.1f}s of TTFT")
    print("\n".join(lines), flush=True)


# --------------------------------------------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------------------------------------------- #
def _longbench_path(spec: str | None) -> Path | None:
    """``--longbench``, else ``DEEPSEEK_V4_LONGBENCH``, else ``zai-org/LongBench-v2``'s ``data.json`` in the
    Hugging Face cache, else ``~/smanoj/data.json``: the demo's order."""
    if spec:
        return Path(spec).expanduser()
    if os.environ.get("DEEPSEEK_V4_LONGBENCH"):
        return Path(os.environ["DEEPSEEK_V4_LONGBENCH"]).expanduser()
    cached = None
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache("zai-org/LongBench-v2", "data.json", repo_type="dataset")
    except Exception:  # noqa: BLE001 - an old huggingface_hub has no cache lookup (see _load_tokenizer)
        cached = None
    if isinstance(cached, str):
        return Path(cached)
    default = Path(os.path.expanduser("~/smanoj/data.json"))
    return default if default.is_file() else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY"),
        help="key of the official API (default: DEEPSEEK_API_KEY); given, the run goes to the official API "
        "(https://api.deepseek.com) instead of the local server, and --base-url / --model / --max-tokens / "
        "--context take their official defaults unless set",
    )
    p.add_argument("--base-url", default=os.environ.get("DEEPSEEK_V4_BASE_URL", _DEFAULT_BASE_URL), help="server URL")
    p.add_argument("--model", default=os.environ.get("DEEPSEEK_V4_MODEL_ID", _DEFAULT_MODEL), help="model id")
    p.add_argument("--timeout", type=float, default=7200, help="per-request timeout in seconds")
    p.add_argument(
        "--longbench",
        help="data.json to ask (default: DEEPSEEK_V4_LONGBENCH, else the LongBench-v2 cache, else ~/smanoj/data.json)",
    )
    p.add_argument("--count", type=int, default=20, help="questions to ask, 0 for all that qualify (default: 20)")
    p.add_argument("--indices", help="exactly these data.json indices, comma-separated (e.g. 1,6,21)")
    p.add_argument("--seed", type=int, default=None, help="draw the questions at random with this seed, not in order")
    p.add_argument(
        "--longest-first",
        action="store_true",
        help="ask the selected questions longest prompt first (by counted prompt tokens, descending)",
    )
    p.add_argument(
        "--length",
        default="short",
        choices=("short", "medium", "long", "any"),
        help="LongBench length of the questions to ask (default: short, as the demo)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=f"cap on each reply (default: {_DEFAULT_MAX_TOKENS}, the demo's max new tokens; on the official API, "
        f"{_DEFAULT_API_MAX_TOKENS})",
    )
    p.add_argument(
        "--max-prompt-tokens", type=int, default=0, help="skip questions with longer prompts (0: only the context)"
    )
    p.add_argument(
        "--context",
        type=int,
        default=0,
        help="the context window, when GET /v1/models does not say (0: ask the server; the official API's is "
        f"{_DEFAULT_API_CONTEXT})",
    )
    p.add_argument(
        "--tokenizer-dir",
        default=os.environ.get("DEEPSEEK_V4_MODEL_DIR", _DEFAULT_MODEL_DIR),
        help="checkpoint snapshot to count prompt tokens with; empty estimates them",
    )
    p.add_argument(
        "--reasoning-effort",
        default=_DEFAULT_REASONING_EFFORT,
        choices=("low", "medium", "high", "max", ""),
        help=f"thinking effort, sent to the server (default: {_DEFAULT_REASONING_EFFORT}, the demo's; empty sends "
        "none). The official API has no such field: the model picks the mode there",
    )
    p.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ask the server for its thinking template (default: on, as the demo); the official API picks by model",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="per-request temperature (default: none, so the server's applies; a --device-sampling server "
        "rejects a request's own, so send 0 only to a --no-device-sampling one)",
    )
    p.add_argument("--top-p", type=float, default=None, help="per-request nucleus probability")
    p.add_argument(
        "--concurrency", type=int, default=1, help="questions in flight at once (the server runs one per stage)"
    )
    p.add_argument("--user-prefix", default="longbench", help="prefix of each question's session key")
    p.add_argument("--keep-sessions", action="store_true", help="leave the sessions on the server afterwards")
    p.add_argument("--exit-on-wrong", action="store_true", help="exit 1 unless every graded question is correct")
    p.add_argument("--json", help="also write the per-question records to this file")
    args = p.parse_args(argv)
    if args.api_key:
        # The official API's defaults, for whatever was not asked for by hand (or by environment).
        if p.get_default("base_url") == args.base_url:
            args.base_url = _DEFAULT_API_BASE_URL
        if p.get_default("model") == args.model:
            args.model = _DEFAULT_API_MODEL
        if args.max_tokens is None:
            args.max_tokens = _DEFAULT_API_MAX_TOKENS
        if not args.context:
            args.context = _DEFAULT_API_CONTEXT
    if args.max_tokens is None:
        args.max_tokens = _DEFAULT_MAX_TOKENS
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    path = _longbench_path(args.longbench)
    if path is None or not path.is_file():
        print(f"LongBench data not found at {path}; pass --longbench", file=sys.stderr)
        return 1

    client = Client(args.base_url, model=args.model, timeout=args.timeout, api_key=args.api_key)
    context_window = args.context or None
    try:
        models = client.models().get("data", [])
        record = models[0] if models else {}
        context_window = context_window or record.get("context_window")
        print(
            f"{'api' if args.api_key else 'server'} {client.base_url}: model {args.model!r}, "
            f"context window {context_window}"
        )
        if args.api_key and models and args.model not in [m.get("id") for m in models]:
            print(f"warning: the API does not list {args.model!r}: {[m.get('id') for m in models]}", file=sys.stderr)
    except (OpenAIError, KeyError, IndexError) as e:
        if context_window is None:
            print(f"cannot reach {client.base_url}/models ({e}); pass --context", file=sys.stderr)
            return 1
        print(f"warning: GET /models failed ({e}); assuming a context of {context_window}", file=sys.stderr)

    indices = [int(i) for i in args.indices.split(",")] if args.indices else None
    questions = _pick_longbench(
        path,
        args.count,
        args.seed,
        indices,
        args.length,
        _load_tokenizer(args.tokenizer_dir),
        args.reasoning_effort or None,
    )

    asked = []
    for question in questions:
        if context_window and question.prompt_tokens >= context_window:
            print(
                f"{question.name} skipped: {question.prompt_tokens} prompt tokens do not fit the "
                f"{context_window}-token context",
                flush=True,
            )
            continue
        if args.max_prompt_tokens and question.prompt_tokens > args.max_prompt_tokens:
            print(
                f"{question.name} skipped: {question.prompt_tokens} prompt tokens > --max-prompt-tokens "
                f"{args.max_prompt_tokens}",
                flush=True,
            )
            continue
        asked.append(question)
    if not asked:
        print(f"no question fits {client.base_url}", file=sys.stderr)
        return 1
    if args.longest_first:
        asked.sort(key=lambda question: question.prompt_tokens, reverse=True)

    if not args.thinking:
        mode = "thinking left to the server/API"
    elif args.api_key:
        mode = f"thinking as {args.model} defaults (the API takes no effort field)"
    else:
        mode = f"thinking at effort {args.reasoning_effort or 'the server default'}"
    order = "longest prompt first" if args.longest_first else "in selection order"
    print(
        f"\nasking {len(asked)} question(s), {order}, {mode}, up to {args.max_tokens} new tokens each, "
        f"{args.concurrency} at a time",
        flush=True,
    )
    t0 = time.perf_counter()
    results = _run_all(args, asked, context_window)
    wall = time.perf_counter() - t0

    _print_perf(results)
    _print_correctness(results)
    print(f"\nwall time: {wall:.1f}s", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump([r.record() for r in results], f, indent=2)
        print(f"wrote {args.json}")

    failed = [r for r in results if r.error is not None]
    if failed:
        print(f"{len(failed)} question(s) failed: {failed[0].name}: {failed[0].error}", file=sys.stderr)
        return 1
    if args.exit_on_wrong and any(r.expected is not None and not r.correct for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
