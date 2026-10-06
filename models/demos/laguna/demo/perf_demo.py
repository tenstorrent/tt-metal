# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Laguna serving performance demo: time to first token and decode speed across input lengths.

For each serving mode (normal decode, DFlash speculative decoding) this demo starts the Laguna vLLM server with
serve_vllm.sh, waits until it is ready, prints the model's answers to two real prompts, sends random-token prompts
of each input length one request at a time (batch 1) with ``vllm bench serve``, stops the server and prints a
results table. Run it with no server running:

    python models/demos/laguna/demo/perf_demo.py                       # both modes, 128 .. 8K tokens, 1 prompt each
    python models/demos/laguna/demo/perf_demo.py --input-lens 128,16384,131072   # other lengths, up to 1M
    python models/demos/laguna/demo/perf_demo.py --modes dflash --prompts 3   # DFlash averaged over 3 prompts
    python models/demos/laguna/demo/perf_demo.py --modes normal --input-lens 128,4096

Every request is greedy (temperature 0) and generates exactly ``--output-tokens`` tokens (EOS ignored). Each input
length gets ``--prompts`` different random prompts (default 1, like tt-metal's simple_text_demo seqlen-sweep); with
more than one the table reports the mean and the per-request range.

    TTFT            time from sending the request to the first output token (prefill, plus queueing: none at batch 1)
                    Each length is first sent once as a 16-token warm-up with the same prompt, so one-time program
                    builds (DFlash builds one per new prompt length) are not timed.
    decode tok/s    1000 / time per output token after the first (per user; batch 1, so also the total)

Speculative decoding speed depends on how predictable the generated text is, so DFlash varies from prompt to prompt
much more than normal decode; pass ``--prompts 3`` or more for a steadier DFlash number. Results (table, JSON with every request) and the
server logs go to ``--output-dir`` (default: generated/laguna_perf_demo/<UTC time>/ under the repository root).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
SERVER_URL = "http://localhost:8000"  # serve_vllm.sh always serves on port 8000

# 128, then powers of two up to 8K (tt-metal's simple_text_demo sweeps 1K .. 128K; longer prefills take minutes each).
DEFAULT_INPUT_LENS = [128, 1024, 2048, 4096, 8192]

MODES = {
    # name: (description, extra serve_vllm.sh environment)
    "normal": ("normal decode", {}),
    "dflash": (
        "DFlash speculative decoding",
        {"TT_LAGUNA_DFLASH": "1", "LAGUNA_ALLOW_EXPERIMENTAL_OVERRIDES": "1"},
    ),
}


def server_is_up() -> bool:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as response:
            return response.status == 200
    except Exception:  # noqa: BLE001
        return False


def start_server(mode: str, model: str, log_dir: Path, timeout_s: int) -> None:
    env = dict(os.environ, HF_MODEL=model, LAGUNA_LOG_DIR=str(log_dir), **MODES[mode][1])
    print(f"[perf demo] starting the {MODES[mode][0]} server (logs: {log_dir}/latest.log) ...", flush=True)
    launch = subprocess.run([str(MODEL_DIR / "serve_vllm.sh")], env=env, cwd=MODEL_DIR, capture_output=True, text=True)
    if launch.returncode != 0:
        raise RuntimeError(f"serve_vllm.sh refused to start:\n{launch.stdout}\n{launch.stderr}")
    log = log_dir / "latest.log"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server_is_up():
            print("[perf demo] server ready", flush=True)
            return
        text = log.read_text(errors="ignore") if log.exists() else ""
        if "Traceback" in text or "TT_FATAL" in text:
            raise RuntimeError(f"the server failed during startup; see {log}")
        time.sleep(10)
    raise RuntimeError(f"the server was not ready after {timeout_s} s; see {log}")


def stop_server() -> None:
    subprocess.run([str(MODEL_DIR / "serve_vllm.sh"), "stop"], cwd=MODEL_DIR, capture_output=True, text=True)


def bench(model: str, input_len: int, output_len: int, prompts: int, result_dir: Path, name: str) -> dict:
    """One ``vllm bench serve`` run: ``prompts`` random-token prompts of ``input_len`` tokens, one at a time."""
    vllm = MODEL_DIR / ".venv" / "bin" / "vllm"
    if not vllm.exists():
        vllm = Path(shutil.which("vllm") or "vllm")
    command = [
        str(vllm), "bench", "serve",
        "--backend", "openai-chat", "--endpoint", "/v1/chat/completions", "--base-url", SERVER_URL,
        "--model", model, "--trust-remote-code",
        "--dataset-name", "random", "--random-input-len", str(input_len), "--random-output-len", str(output_len),
        "--num-prompts", str(prompts), "--max-concurrency", "1", "--request-rate", "inf",
        "--temperature", "0", "--ignore-eos", "--seed", "1234", "--num-warmups", "0",
        "--save-result", "--save-detailed", "--result-dir", str(result_dir), "--result-filename", f"{name}.json",
    ]  # fmt: skip
    with open(result_dir / f"{name}.log", "w") as log:
        status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd="/tmp").returncode
    if status != 0:
        raise RuntimeError(f"vllm bench serve failed for input length {input_len}; see {result_dir / (name + '.log')}")
    return json.loads((result_dir / f"{name}.json").read_text())


EXAMPLE_PROMPTS = [
    "Write a Python function that checks whether a number is prime.",
    "Explain in three sentences why the sky is blue.",
]


def run_examples(model: str, max_tokens: int) -> list[dict]:
    """Send each real prompt once (greedy, thinking off) and return the prompt, answer, token count and time."""
    examples = []
    for prompt in EXAMPLE_PROMPTS:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = urllib.request.Request(
            f"{SERVER_URL}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.time()
        with urllib.request.urlopen(request, timeout=600) as response:
            reply = json.load(response)
        examples.append(
            {
                "prompt": prompt,
                "answer": reply["choices"][0]["message"]["content"],
                "output_tokens": reply["usage"]["completion_tokens"],
                "seconds": round(time.time() - start, 2),
            }
        )
    return examples


def print_examples(mode: str, examples: list[dict]) -> None:
    for example in examples:
        print(f"\n[perf demo] {mode} | prompt: {example['prompt']}", flush=True)
        print(example["answer"].strip(), flush=True)
        print(f"[perf demo] {mode} | {example['output_tokens']} tokens in {example['seconds']:.1f} s", flush=True)


def summarize(result: dict) -> dict:
    """Mean and per-request range of TTFT and decode speed from a ``vllm bench serve --save-detailed`` result.

    ``ttfts`` are seconds per request; a request's decode speed is (output tokens - 1) / sum of its inter-token
    latencies ``itls`` (seconds). The files carry no per-request ``tpots``, so the speed is computed here."""
    ttft_s = [float(value) for value in result.get("ttfts") or [] if value] or [result["mean_ttft_ms"] / 1000.0]
    speed = [
        (int(out) - 1) / sum(gaps)
        for out, gaps in zip(result.get("output_lens") or [], result.get("itls") or [])
        if gaps and int(out) > 1 and sum(gaps) > 0
    ] or [1000.0 / result["mean_tpot_ms"]]
    inputs = [int(value) for value in result.get("input_lens") or []] or [
        round(result["total_input_tokens"] / max(1, int(result["completed"])))
    ]
    return {
        "requests": int(result["completed"]),
        "failed": int(result.get("failed", 0)),
        "input_tokens_mean": statistics.mean(inputs),
        "input_tokens_min": min(inputs),
        "input_tokens_max": max(inputs),
        "ttft_s_mean": statistics.mean(ttft_s),
        "ttft_s_min": min(ttft_s),
        "ttft_s_max": max(ttft_s),
        "decode_tok_s_mean": 1000.0 / float(result["mean_tpot_ms"]),
        "decode_tok_s_min": min(speed),
        "decode_tok_s_max": max(speed),
    }


def table(rows: dict, modes: list[str], input_lens: list[int]) -> str:
    def cell(mean, low, high, fmt):
        return fmt.format(mean) if abs(high - low) < 1e-9 else f"{fmt.format(mean)} ({fmt.format(low)}-{fmt.format(high)})"

    header = ["Input tokens (requested)", "Input tokens (actual)"]
    for mode in modes:
        header += [f"TTFT s, {mode}", f"decode tok/s, {mode}"]
    if "normal" in modes and "dflash" in modes:
        header.append("DFlash speedup")
    lines = ["| " + " | ".join(header) + " |", "|" + "---:|" * len(header)]
    for input_len in input_lens:
        line = [f"{input_len:,}"]
        measured = [rows[mode][input_len] for mode in modes if input_len in rows.get(mode, {})]
        if measured:
            low = min(m["input_tokens_min"] for m in measured)
            high = max(m["input_tokens_max"] for m in measured)
            line.append(f"{low:,}" if low == high else f"{low:,}-{high:,}")
        else:
            line.append("-")
        for mode in modes:
            s = rows.get(mode, {}).get(input_len)
            if s is None:
                line += ["-", "-"]
                continue
            line.append(cell(s["ttft_s_mean"], s["ttft_s_min"], s["ttft_s_max"], "{:.2f}"))
            line.append(cell(s["decode_tok_s_mean"], s["decode_tok_s_min"], s["decode_tok_s_max"], "{:.1f}"))
        if "normal" in modes and "dflash" in modes:
            n, d = rows.get("normal", {}).get(input_len), rows.get("dflash", {}).get(input_len)
            line.append(f"{d['decode_tok_s_mean'] / n['decode_tok_s_mean']:.2f}x" if n and d else "-")
        lines.append("| " + " | ".join(line) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", default="normal,dflash", help="comma-separated: normal, dflash (default: both)")
    parser.add_argument("--input-lens", default=None, help="comma-separated input lengths (default: 128,1024,2048,4096,8192)")
    parser.add_argument("--prompts", type=int, default=None, help="random prompts per input length (default: 1)")
    parser.add_argument("--output-tokens", type=int, default=512, help="tokens generated per request (default: 512)")
    parser.add_argument("--model", default=os.environ.get("HF_MODEL", "poolside/Laguna-S-2.1"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--use-running-server",
        action="store_true",
        help="measure the server already running on port 8000 instead of starting one per mode (one mode only)",
    )
    parser.add_argument("--startup-timeout", type=int, default=2400, help="seconds to wait for each server")
    parser.add_argument("--example-tokens", type=int, default=200, help="max tokens per example answer (0: no examples)")
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in MODES]
    if unknown:
        parser.error(f"unknown mode(s) {unknown}; choose from {sorted(MODES)}")
    if args.use_running_server and len(modes) != 1:
        parser.error("--use-running-server measures one server: pass exactly one --modes value")
    input_lens = (
        [int(value) for value in args.input_lens.split(",")]
        if args.input_lens
        else DEFAULT_INPUT_LENS
    )
    prompts = args.prompts or 1
    out = Path(args.output_dir) if args.output_dir else REPO_ROOT / "generated" / "laguna_perf_demo" / time.strftime(
        "%Y%m%dT%H%M%SZ", time.gmtime()
    )
    out.mkdir(parents=True, exist_ok=True)

    if not args.use_running_server and server_is_up():
        print(f"[perf demo] a server is already answering on {SERVER_URL}. Stop it first (serve_vllm.sh stop) or pass "
              "--use-running-server --modes <its mode>.", file=sys.stderr)  # fmt: skip
        return 2

    print(
        f"[perf demo] model {args.model} | modes {modes} | input lengths {input_lens} | {prompts} prompt(s) each | "
        f"{args.output_tokens} output tokens | batch 1 | results in {out}",
        flush=True,
    )
    rows: dict = {}
    raw: dict = {}
    examples: dict = {}
    for mode in modes:
        mode_dir = out / mode
        mode_dir.mkdir(exist_ok=True)
        try:
            if not args.use_running_server:
                start_server(mode, args.model, mode_dir / "server", args.startup_timeout)
            # One short request first so one-time setup (first-request program builds) is not in the 128 point.
            bench(args.model, 128, 16, 1, mode_dir, "warmup")
            if args.example_tokens > 0:
                examples[mode] = run_examples(args.model, args.example_tokens)
                (mode_dir / "examples.json").write_text(json.dumps(examples[mode], indent=1))
                print_examples(mode, examples[mode])
            rows[mode], raw[mode] = {}, {}
            for input_len in input_lens:
                # Warm-up at this length first: the same seed gives the same random prompts, so programs that depend
                # on the exact prompt length (DFlash compiles one per new length) are built before the measured run.
                warm = bench(args.model, input_len, 16, prompts, mode_dir, f"warmup_isl_{input_len}")
                result = bench(args.model, input_len, args.output_tokens, prompts, mode_dir, f"isl_{input_len}")
                if (warm.get("input_lens") or []) != (result.get("input_lens") or []):
                    print(f"[perf demo] warning: warm-up prompt lengths {warm.get('input_lens')} differ from the "
                          f"measured {result.get('input_lens')}", flush=True)  # fmt: skip
                summary = summarize(result)
                rows[mode][input_len], raw[mode][input_len] = summary, result
                print(
                    f"[perf demo] {mode:6s} input {input_len:>7,}: TTFT {summary['ttft_s_mean']:8.2f} s | "
                    f"decode {summary['decode_tok_s_mean']:6.1f} tok/s "
                    f"({summary['decode_tok_s_min']:.1f}-{summary['decode_tok_s_max']:.1f}) | "
                    f"{summary['requests']} ok, {summary['failed']} failed",
                    flush=True,
                )
        finally:
            if not args.use_running_server:
                stop_server()

    report = table(rows, modes, input_lens)
    if "normal" in examples and "dflash" in examples:
        same = sum(a["answer"] == b["answer"] for a, b in zip(examples["normal"], examples["dflash"]))
        print(
            f"\n[perf demo] example answers identical between normal and DFlash: {same}/{len(examples['normal'])} "
            "(greedy DFlash can differ where two tokens score almost the same)",
            flush=True,
        )
    notes = (
        f"Laguna serving performance: {args.model}, batch 1, {prompts} random-token prompt(s) per input length, "
        f"{args.output_tokens} output tokens, greedy. Values are means; ranges in parentheses are min-max over the "
        "requests. vLLM's random prompts are random token ids decoded to text and re-tokenized with the chat template, "
        "so the actual input length differs from the requested one (most at short lengths)."
    )
    (out / "results.md").write_text(notes + "\n\n" + report + "\n")
    (out / "results.json").write_text(
        json.dumps({"model": args.model, "prompts": prompts, "output_tokens": args.output_tokens, "summary": rows}, indent=1)
    )
    print("\n" + notes + "\n\n" + report + f"\n\n[perf demo] saved {out / 'results.md'} and results.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
