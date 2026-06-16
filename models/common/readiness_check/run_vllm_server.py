# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Run the vLLM server readiness check.

A single entry point that can launch the server, run any subset of checks
against it, or attach to an already-running server. Pick what to do via
``--stages``.

Stages:
  serve         Launch the vLLM server. If it's the only stage, hold it open
                until SIGINT/SIGTERM; otherwise shut it down once the trailing
                checks finish. Requires ``--mesh-device``.
  sampling      Run the canonical pytest plugin sampling tests against the
                live server. ``--sampling-profile`` selects full vs smoke.
  qualitative   Run qualitative prompts and save completions for manual
                review.
  benchmark     Fire synthetic random-token prompts at the server under
                concurrency and report TTFT, ITL, and per-user / aggregate
                throughput. Saves ``vllm_benchmark.json`` for the work log.

Default: ``--stages serve,sampling,qualitative,benchmark`` (launch, run all
checks, shut down). Full launch with the typical tuning flags:

    python -m models.common.readiness_check.run_vllm_server \\
        --model-dir models/autoports/<model_name> \\
        --hf-model <hf-model-id> \\
        --mesh-device N150 \\
        --max-num-seqs 32 \\
        --max-model-len 32768 \\
        --tt-config '{"trace_region_size": 85000000, "fabric_config": "FABRIC_1D"}'

To hold the server open without running checks (skip trace compile on
subsequent check invocations):

    python -m models.common.readiness_check.run_vllm_server \\
        --stages serve \\
        --model-dir models/autoports/<model_name> \\
        --hf-model <hf-model-id> \\
        --mesh-device N150 \\
        --max-model-len 32768 \\
        --tt-config '{"trace_region_size": 85000000, "fabric_config": "FABRIC_1D"}'

To run a single check against the running server, pass ``--server-url`` and
omit ``serve`` from the stages:

    python -m models.common.readiness_check.run_vllm_server \\
        --stages sampling \\
        --max-num-seqs 32 \\
        --sampling-profile smoke \\
        --server-url http://localhost:8000 \\
        --model-dir models/autoports/<model_name> \\
        --hf-model <hf-model-id>

To install vllm, if not already present:
1. Clone `https://github.com/tenstorrent/vllm.git`
2. Switch to the `dev` branch
3. Follow installation instructions in `plugins/vllm-tt-plugin/README.md`

Before invoking it, two things must already be true:

  1. `<model_dir>/tt/generator_vllm.py` exists and implements the model.
  2. The model architecture is registered in
     `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`
     via `_register_model_if_missing(...)`. Without that registration vLLM
     will reject the architecture at startup with
     "architecture not in TT registry".

The launch command and env vars match `.github/workflows/vllm-nightly-tests-impl.yaml`.

On-device sampling is enforced (`sample_on_device_mode: all` in the plugin
config). A model that cannot serve sampling from the device is not
production-ready, so the readiness check fails fast there rather than papering
over it with host-side sampling.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import openai
import requests

DEFAULT_PORT = 8000
DEFAULT_BLOCK_SIZE = 64
DEFAULT_MAX_NUM_SEQS = 32
# TT trace compile + weight load on first start can take 10+ minutes.
DEFAULT_SERVER_TIMEOUT_S = 1200
# Matches the nightly CI workflow.
DEFAULT_VLLM_RPC_TIMEOUT_MS = 300000

STAGE_SERVE = "serve"
STAGE_SAMPLING = "sampling"
STAGE_QUALITATIVE = "qualitative"
STAGE_BENCHMARK = "benchmark"
ALL_STAGES: tuple[str, ...] = (STAGE_SERVE, STAGE_SAMPLING, STAGE_QUALITATIVE, STAGE_BENCHMARK)
DEFAULT_STAGES: tuple[str, ...] = ALL_STAGES

# Benchmark defaults: 128-tok prompts × 128-tok outputs × 32 requests × 8
# concurrent. The output-length mirrors the per-token-decode metric in the
# productize SKILL; concurrency lets the scheduler exercise real batching
# without saturating it (default max_num_seqs = 32).
DEFAULT_BENCH_PROMPT_LEN = 128
DEFAULT_BENCH_OUTPUT_LEN = 128
DEFAULT_BENCH_NUM_REQUESTS = 32
DEFAULT_BENCH_CONCURRENCY = 8

SAMPLING_PROFILE_FULL = "full"
SAMPLING_PROFILE_SMOKE = "smoke"
DEFAULT_SAMPLING_PROFILE = SAMPLING_PROFILE_FULL
_SMOKE_SAMPLING_TESTS: tuple[str, ...] = (
    "test_request_isolation.py::TestBatchIsolation::test_mixed_params_batch",
    "test_seeding_and_variety.py::TestSeedingAndVariety::test_top1_is_greedy",
    "test_logprobs.py::TestLogprobs::test_chat_logprobs_all_vocab",
    "test_host_only_params.py::TestHostOnlyParameters::test_min_p",
)

# Always enforce on-device sampling. A ported model that cannot serve sampling
# from the device is not production-ready; the readiness check fails fast here
# rather than papering over it with host-side sampling. Users override by
# passing `--tt-config '{"sample_on_device_mode": "..."}'` if they need to.
DEFAULT_TT_CONFIG: dict[str, Any] = {"sample_on_device_mode": "all"}

# Fast-fail markers cribbed from the nightly CI workflow. vLLM writes one of
# these to the server log within seconds of an engine-core crash; polling
# /health past that point just burns time.
_FATAL_LOG_PATTERNS = (
    "EngineCore failed to start",
    "EngineCore encountered a fatal error",
    "EngineDeadError",
    "Engine core initialization failed",
    "Failed core proc",
)

_MESH_SHAPES: dict[str, tuple[int, int]] = {
    "N150": (1, 1),
    "N300": (1, 2),
    "T3K": (1, 8),
    "TG": (8, 4),
}


def _stream_choice_is_token_event(choice: Any) -> bool:
    """Return true for streamed completion chunks that represent output tokens."""
    text = getattr(choice, "text", None)
    finish_reason = getattr(choice, "finish_reason", None)
    if text is None:
        return False
    return text != "" or finish_reason is None


def _find_plugin_tests_dir() -> Path:
    """Locate `vllm-tt-plugin/tests/tt/` from the installed editable package."""
    spec = importlib.util.find_spec("vllm_tt_plugin")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "vllm_tt_plugin is not importable. Install it with "
            "`uv pip install -e <vllm-repo>/plugins/vllm-tt-plugin --no-deps`."
        )
    # spec.origin -> <plugin_root>/src/vllm_tt_plugin/__init__.py
    plugin_root = Path(spec.origin).resolve().parent.parent.parent
    tests_dir = plugin_root / "tests" / "tt"
    if not tests_dir.is_dir():
        raise RuntimeError(
            f"Expected plugin tests at {tests_dir} but did not find them. Plugin layout may have changed."
        )
    return tests_dir


def _check_port_available(port: int) -> None:
    """Fail fast if `port` is already in use — vLLM will hang otherwise."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as e:
        raise RuntimeError(f"Port {port} is already in use ({e}). Pass --port to pick a different one.") from e
    finally:
        sock.close()


def _launch_server(
    *,
    hf_model: str,
    mesh_device: str,
    max_num_seqs: int,
    block_size: int,
    port: int,
    log_file: Path,
    max_model_len: Optional[int],
    tt_config: dict[str, Any],
    additional_args: List[str],
) -> subprocess.Popen:
    """
    Launch vLLM via `python -m vllm.entrypoints.openai.api_server`.

    Mirrors `vllm-tt-plugin/examples/server_example_tt.py` (which is what the
    nightly CI runs) but inlined — the example is just argv-munging + runpy.
    """
    cmd: List[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        hf_model,
        "--block_size",
        str(block_size),
        "--max_num_seqs",
        str(max_num_seqs),
        "--port",
        str(port),
    ]
    if max_model_len is not None:
        cmd += ["--max_model_len", str(max_model_len)]
    # Pass TT plugin config as a single JSON dict so JSON quoting can't be
    # mangled by intermediate shells. The dict already has
    # `sample_on_device_mode` enforced; callers extend via `tt_config`.
    cmd += ["--plugin-config", json.dumps({"tt": tt_config})]
    cmd += additional_args

    env = {
        **os.environ,
        "MESH_DEVICE": mesh_device,
        "HF_MODEL": hf_model,
        "VLLM_RPC_TIMEOUT": str(DEFAULT_VLLM_RPC_TIMEOUT_MS),
    }

    print(f"Launching vLLM server for {hf_model} on {mesh_device}")
    print(f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"  log: {log_file}")

    log_handle = open(log_file, "wb")
    return subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=env)


def _scan_log_for_fatal(log_file: Path) -> Optional[str]:
    if not log_file.exists():
        return None
    try:
        text = log_file.read_text(errors="replace")
    except OSError:
        return None
    for pattern in _FATAL_LOG_PATTERNS:
        if pattern in text:
            return pattern
    return None


def _wait_for_server(
    *,
    proc: subprocess.Popen,
    port: int,
    log_file: Path,
    timeout_seconds: int,
) -> None:
    """Poll /health, fast-failing on launcher exit or fatal-marker in the log."""
    print(f"Waiting up to {timeout_seconds}s for server to become ready...")
    url = f"http://localhost:{port}/health"
    interval = 10
    elapsed = 0

    while elapsed < timeout_seconds:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                print(f"  Server ready after ~{elapsed}s")
                return
        except requests.exceptions.RequestException:
            pass

        if proc.poll() is not None:
            raise RuntimeError(
                f"Server launcher exited with code {proc.returncode} before becoming ready. Inspect {log_file}."
            )

        fatal = _scan_log_for_fatal(log_file)
        if fatal:
            raise RuntimeError(f"Fatal marker {fatal!r} found in {log_file}. Server cannot start.")

        time.sleep(interval)
        elapsed += interval

    raise RuntimeError(f"Server did not become ready in {timeout_seconds}s. Inspect {log_file}.")


def _probe_external_server(server_url: str) -> None:
    """Verify an externally-managed server is reachable before running checks."""
    health = f"{server_url.rstrip('/')}/health"
    try:
        resp = requests.get(health, timeout=5)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"Server not reachable at {health} ({e}). Start it first with `--stages serve` " "or check --server-url."
        ) from e
    if resp.status_code != 200:
        raise RuntimeError(f"Server at {health} returned {resp.status_code}; expected 200.")


def _run_plugin_sampling_tests(
    *,
    server_url: str,
    hf_model: str,
    log_file: Path,
    server_log: Path,
    max_num_seqs: int,
    sampling_profile: str,
) -> bool:
    """
    Invoke `vllm-tt-plugin/tests/tt/` against the live server. This is the same
    test suite the nightly CI runs (greedy determinism, seeded reproducibility,
    seed variety, logprobs, penalties, isolation, ...). Reusing it instead of
    re-implementing ad-hoc copies means our checks track the canonical ones.

    On failure, scan the server log for engine-core fatal markers and surface
    them — a mid-run crash typically shows up as cryptic empty-response errors
    in the pytest output, with the real cause buried in the server log.
    """
    tests_dir = _find_plugin_tests_dir()
    if sampling_profile == SAMPLING_PROFILE_FULL:
        test_targets: List[str] = [str(tests_dir)]
    elif sampling_profile == SAMPLING_PROFILE_SMOKE:
        test_targets = []
        for rel_nodeid in _SMOKE_SAMPLING_TESTS:
            rel_path, _, selector = rel_nodeid.partition("::")
            nodeid = str(tests_dir / rel_path)
            if selector:
                nodeid += f"::{selector}"
            test_targets.append(nodeid)
    else:
        raise RuntimeError(
            f"Unsupported sampling profile {sampling_profile!r}. "
            f"Expected one of: {SAMPLING_PROFILE_FULL}, {SAMPLING_PROFILE_SMOKE}."
        )

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *test_targets,
        "-v",
        f"--tt-server-url={server_url}",
        f"--tt-model-name={hf_model}",
        f"--tt-max-num-seqs={max_num_seqs}",
    ]
    print("\n=== Running plugin sampling tests ===")
    print(f"  profile: {sampling_profile} ({len(test_targets)} target(s))")
    print(f"  tt-max-num-seqs: {max_num_seqs}")
    print(f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"  log: {log_file}")

    with open(log_file, "wb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode == 0:
        print("  PASS")
        return True

    print(f"  FAIL (pytest exit {result.returncode}) — inspect {log_file}")
    fatal = _scan_log_for_fatal(server_log)
    if fatal:
        print(
            f"  Note: server log contains fatal marker {fatal!r} — the engine "
            "crashed mid-run. The pytest 'NoneType' / empty-response errors are "
            f"a symptom; check {server_log} around that marker for the real cause."
        )
    return False


def _run_qualitative_prompts(
    *,
    server_url: str,
    hf_model: str,
    prompts_file: Path,
    output_dir: Path,
) -> None:
    """Run prompts through the server and save completions for manual review."""
    print(f"\n=== Running qualitative prompts from {prompts_file} ===")

    if not prompts_file.exists():
        raise RuntimeError(f"Prompts file not found: {prompts_file}")

    prompts = [p.strip() for p in prompts_file.read_text().split("\n\n") if p.strip()]
    if not prompts:
        raise RuntimeError(f"No prompts found in {prompts_file}")

    print(f"  Loaded {len(prompts)} prompts")
    client = openai.OpenAI(base_url=f"{server_url.rstrip('/')}/v1", api_key="dummy")

    results: List[dict[str, Any]] = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n  Prompt {i}/{len(prompts)}: {prompt[:60]}...")

        greedy_text = (
            client.completions.create(
                model=hf_model,
                prompt=prompt,
                max_tokens=256,
                temperature=0.0,
            )
            .choices[0]
            .text
        )
        sampled_text = (
            client.completions.create(
                model=hf_model,
                prompt=prompt,
                max_tokens=256,
                temperature=0.7,
                top_p=0.9,
            )
            .choices[0]
            .text
        )

        results.append(
            {
                "prompt": prompt,
                "greedy_completion": greedy_text,
                "sampled_completion": sampled_text,
            }
        )
        print(f"    Greedy : {greedy_text[:80].replace(chr(10), ' ')!r}")
        print(f"    Sampled: {sampled_text[:80].replace(chr(10), ' ')!r}")

    output_file = output_dir / "vllm_qualitative_outputs.json"
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {len(results)} completions to {output_file}")
    print("Read both completions for each prompt and judge:")
    print("  - coherent and on-topic")
    print("  - no repetition loops or gibberish")
    print("  - greedy and sampled outputs both reasonable")


def _run_serving_benchmark(
    *,
    server_url: str,
    hf_model: str,
    output_dir: Path,
    prompt_len: int,
    output_len: int,
    num_requests: int,
    concurrency: int,
) -> None:
    """
    Measure serving-path TTFT, ITL, and throughput under concurrent load.

    Generates `num_requests` synthetic random-token prompts of length
    `prompt_len`, fires them at the server with up to `concurrency` in flight,
    streams completions, and aggregates per-request timings. Uses
    `ignore_eos=True` so every request emits exactly `output_len` tokens
    regardless of natural EOS, which keeps the workload deterministic.

    Reports TTFT P50/P99, ITL P50/P99, aggregate output throughput, mean
    per-user decode throughput, and ITL P50-implied decode throughput. Saves
    the full summary to `vllm_benchmark.json`.
    """
    import asyncio
    import random

    from transformers import AutoTokenizer

    print("\n=== Running vLLM serving benchmark ===")
    print(
        f"  prompt_len={prompt_len}, output_len={output_len}, "
        f"num_requests={num_requests}, concurrency={concurrency}"
    )
    print(f"  Loading tokenizer for {hf_model}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    rng = random.Random(0)
    vocab_size = tokenizer.vocab_size
    prompts: List[List[int]] = [[rng.randrange(0, vocab_size) for _ in range(prompt_len)] for _ in range(num_requests)]

    client = openai.AsyncOpenAI(base_url=f"{server_url.rstrip('/')}/v1", api_key="dummy")

    async def _one(token_ids: List[int]) -> dict[str, Any]:
        start = time.perf_counter()
        first: Optional[float] = None
        token_times: List[float] = []
        empty_text_token_events = 0
        non_token_stream_events = 0
        stream = await client.completions.create(
            model=hf_model,
            prompt=token_ids,
            max_tokens=output_len,
            temperature=0.0,
            stream=True,
            extra_body={"ignore_eos": True},
        )
        async for chunk in stream:
            if not chunk.choices:
                non_token_stream_events += 1
                continue
            choice = chunk.choices[0]
            if not _stream_choice_is_token_event(choice):
                non_token_stream_events += 1
                continue
            now = time.perf_counter()
            if first is None:
                first = now
            token_times.append(now)
            if getattr(choice, "text", None) == "":
                empty_text_token_events += 1
        end = time.perf_counter()
        return {
            "start": start,
            "end": end,
            "ttft_s": (first - start) if first is not None else None,
            "total_s": end - start,
            "token_count": len(token_times),
            "empty_text_token_events": empty_text_token_events,
            "non_token_stream_events": non_token_stream_events,
            "itl_s": [token_times[i] - token_times[i - 1] for i in range(1, len(token_times))],
        }

    sem = asyncio.Semaphore(concurrency)

    async def _gated(token_ids: List[int]) -> dict[str, Any]:
        async with sem:
            return await _one(token_ids)

    async def _run_all() -> List[dict[str, Any]]:
        return await asyncio.gather(*[_gated(p) for p in prompts])

    bench_start = time.perf_counter()
    results = asyncio.run(_run_all())
    elapsed = time.perf_counter() - bench_start

    def _pct(xs: List[float], p: float) -> Optional[float]:
        if not xs:
            return None
        s = sorted(xs)
        return s[max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))]

    def _ms(v: Optional[float]) -> Optional[float]:
        return v * 1000 if v is not None else None

    ttfts: List[float] = [r["ttft_s"] for r in results if r["ttft_s"] is not None]
    itls: List[float] = [t for r in results for t in r["itl_s"]]
    total_output_tokens = sum(r["token_count"] for r in results)
    requested_output_tokens = output_len * len(results)
    empty_text_token_events = sum(r["empty_text_token_events"] for r in results)
    non_token_stream_events = sum(r["non_token_stream_events"] for r in results)
    itl_p50_s = _pct(itls, 0.5)
    itl_mean_s = (sum(itls) / len(itls)) if itls else None

    per_request_tsu: List[float] = []
    for r in results:
        if r["ttft_s"] is None or r["token_count"] <= 1:
            continue
        decode_time = r["total_s"] - r["ttft_s"]
        if decode_time > 0:
            per_request_tsu.append((r["token_count"] - 1) / decode_time)

    summary = {
        "config": {
            "prompt_len": prompt_len,
            "output_len": output_len,
            "num_requests": num_requests,
            "concurrency": concurrency,
        },
        "elapsed_s": elapsed,
        "completed_requests": len(results),
        "total_output_tokens": total_output_tokens,
        "requested_output_tokens": requested_output_tokens,
        "missing_output_tokens": max(0, requested_output_tokens - total_output_tokens),
        "empty_text_token_events": empty_text_token_events,
        "non_token_stream_events": non_token_stream_events,
        "ttft_ms": {
            "p50": _ms(_pct(ttfts, 0.5)),
            "p99": _ms(_pct(ttfts, 0.99)),
            "mean": _ms(sum(ttfts) / len(ttfts)) if ttfts else None,
        },
        "itl_ms": {
            "p50": _ms(itl_p50_s),
            "p99": _ms(_pct(itls, 0.99)),
            "mean": _ms(itl_mean_s),
        },
        "itl_p50_decode_tps": (1.0 / itl_p50_s) if itl_p50_s else None,
        "itl_mean_decode_tps": (1.0 / itl_mean_s) if itl_mean_s else None,
        "output_throughput_tok_per_s": (total_output_tokens / elapsed) if elapsed > 0 else None,
        "request_throughput_per_s": (len(results) / elapsed) if elapsed > 0 else None,
        "mean_per_request_decode_tps": (sum(per_request_tsu) / len(per_request_tsu)) if per_request_tsu else None,
    }

    output_file = output_dir / "vllm_benchmark.json"
    output_file.write_text(json.dumps(summary, indent=2))

    def _fmt(v: Optional[float], unit: str) -> str:
        return f"{v:.1f}{unit}" if v is not None else "n/a"

    print("\n=== Serving benchmark summary ===")
    print(f"  Requests : {len(results)} completed in {elapsed:.1f}s")
    print(f"  TTFT     : P50={_fmt(summary['ttft_ms']['p50'], 'ms')}  P99={_fmt(summary['ttft_ms']['p99'], 'ms')}")
    print(f"  ITL      : P50={_fmt(summary['itl_ms']['p50'], 'ms')}  P99={_fmt(summary['itl_ms']['p99'], 'ms')}")
    print(f"  ITL P50  : {_fmt(summary['itl_p50_decode_tps'], ' t/s/u')} implied")
    print(f"  Output   : {_fmt(summary['output_throughput_tok_per_s'], ' tok/s')} aggregate")
    print(
        "  Tokens   : "
        f"{total_output_tokens}/{requested_output_tokens} events"
        f" ({empty_text_token_events} empty-text token events, {non_token_stream_events} non-token stream events)"
    )
    print(f"  Per-user : {_fmt(summary['mean_per_request_decode_tps'], ' t/s/u')} (mean over requests)")
    print(f"  Saved to {output_file}")


def _shutdown(proc: subprocess.Popen, log_file: Path) -> None:
    """Terminate the server; fall back to SIGKILL; show log tail."""
    if proc.poll() is not None:
        print(f"\nServer launcher already exited (code {proc.returncode}).")
    else:
        print(f"\nShutting down server (PID {proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=15)
            print("  Terminated cleanly.")
        except subprocess.TimeoutExpired:
            print("  Did not exit after SIGTERM; sending SIGKILL.")
            proc.kill()
            proc.wait()

    if log_file.exists():
        print(f"\n=== Server log tail ({log_file}, last 30 lines) ===")
        for line in log_file.read_text(errors="replace").splitlines()[-30:]:
            print(line)


def _hold_until_signal(proc: subprocess.Popen, log_file: Path) -> None:
    """Block until SIGINT/SIGTERM, or until the server exits on its own."""
    stop = False

    def _handler(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handler)

    print("Server is running. Press Ctrl-C (or send SIGTERM) to stop.")
    try:
        while not stop:
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited unexpectedly with code {proc.returncode}. Inspect {log_file}.")
            # vLLM's engine can die while the launcher keeps running; the fatal
            # markers catch that case (same scan as _wait_for_server).
            fatal = _scan_log_for_fatal(log_file)
            if fatal:
                raise RuntimeError(f"Fatal marker {fatal!r} found in {log_file}. Server engine is dead.")
            time.sleep(2)
    except KeyboardInterrupt:
        pass


def _parse_stages(raw: str) -> List[str]:
    stages = [s.strip() for s in raw.split(",") if s.strip()]
    if not stages:
        raise argparse.ArgumentTypeError("--stages must list at least one stage")
    unknown = [s for s in stages if s not in ALL_STAGES]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown stages {unknown}. Valid stages: {list(ALL_STAGES)}")
    deduped: List[str] = []
    for s in stages:
        if s not in deduped:
            deduped.append(s)
    return deduped


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the vLLM server readiness check.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--hf-model", type=str, required=True)
    parser.add_argument(
        "--stages",
        type=_parse_stages,
        default=list(DEFAULT_STAGES),
        help=("Comma-separated stages to run. Valid: " f"{','.join(ALL_STAGES)}. Default: {','.join(DEFAULT_STAGES)}."),
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help=(
            "URL of an already-running vLLM server (e.g. http://localhost:8000). "
            "Required when `serve` is not in --stages; rejected when it is."
        ),
    )
    parser.add_argument(
        "--mesh-device",
        type=str,
        default=None,
        choices=sorted(_MESH_SHAPES),
        help="Required when `serve` is in --stages; ignored otherwise.",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).parent / "vllm_prompts.txt",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    parser.add_argument(
        "--sampling-profile",
        type=str,
        choices=(SAMPLING_PROFILE_FULL, SAMPLING_PROFILE_SMOKE),
        default=DEFAULT_SAMPLING_PROFILE,
        help=(
            "Sampling test selection profile. "
            f"`{SAMPLING_PROFILE_FULL}` runs the full plugin suite; "
            f"`{SAMPLING_PROFILE_SMOKE}` runs a small integration sanity subset."
        ),
    )
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=DEFAULT_SERVER_TIMEOUT_S,
        help=f"Seconds to wait for /health when launching (default {DEFAULT_SERVER_TIMEOUT_S}).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vLLM --max_model_len. Required for some models (e.g. Llama-3.1-8B on N150 caps at 32768).",
    )
    parser.add_argument(
        "--tt-config",
        type=str,
        default="{}",
        help=(
            "JSON dict merged into the TT plugin config under the `tt` namespace. "
            f"Always extends the runner's defaults ({DEFAULT_TT_CONFIG}). "
            'Example: \'{"trace_region_size": 85000000, "fabric_config": "FABRIC_1D"}\'.'
        ),
    )
    parser.add_argument(
        "--additional-server-args",
        type=str,
        default="",
        help=(
            "Catch-all for other vLLM CLI args not covered by the typed flags. "
            'Quoted, e.g. "--async-scheduling --tokenizer X". Avoid --plugin-config / '
            "--max_model_len here; use --tt-config / --max-model-len."
        ),
    )
    parser.add_argument(
        "--benchmark-prompt-len",
        type=int,
        default=DEFAULT_BENCH_PROMPT_LEN,
        help=f"Tokens per synthetic prompt for the benchmark stage (default {DEFAULT_BENCH_PROMPT_LEN}).",
    )
    parser.add_argument(
        "--benchmark-output-len",
        type=int,
        default=DEFAULT_BENCH_OUTPUT_LEN,
        help=f"Tokens to generate per request in the benchmark stage (default {DEFAULT_BENCH_OUTPUT_LEN}).",
    )
    parser.add_argument(
        "--benchmark-num-requests",
        type=int,
        default=DEFAULT_BENCH_NUM_REQUESTS,
        help=f"Total requests sent in the benchmark stage (default {DEFAULT_BENCH_NUM_REQUESTS}).",
    )
    parser.add_argument(
        "--benchmark-concurrency",
        type=int,
        default=DEFAULT_BENCH_CONCURRENCY,
        help=f"Max in-flight requests during the benchmark (default {DEFAULT_BENCH_CONCURRENCY}).",
    )
    args = parser.parse_args()

    stages: List[str] = args.stages
    serve_locally = STAGE_SERVE in stages

    if serve_locally and args.server_url is not None:
        parser.error("--server-url is not allowed when `serve` is in --stages")
    if not serve_locally and args.server_url is None:
        parser.error("--server-url is required when `serve` is not in --stages")
    if serve_locally and args.mesh_device is None:
        parser.error("--mesh-device is required when `serve` is in --stages")

    try:
        tt_config = json.loads(args.tt_config)
    except json.JSONDecodeError as e:
        parser.error(f"--tt-config is not valid JSON: {e}")
    if not isinstance(tt_config, dict):
        parser.error(f"--tt-config must be a JSON object, got {type(tt_config).__name__}")
    merged_tt_config: dict[str, Any] = {**DEFAULT_TT_CONFIG, **tt_config}

    additional = shlex.split(args.additional_server_args) if args.additional_server_args else []

    model_dir = args.model_dir.resolve()
    output_dir = model_dir / "readiness_vllm"
    output_dir.mkdir(parents=True, exist_ok=True)
    server_log = output_dir / "server.log"
    sampling_log = output_dir / "sampling_tests.log"

    server_proc: Optional[subprocess.Popen] = None
    try:
        if serve_locally:
            _check_port_available(args.port)
            server_proc = _launch_server(
                hf_model=args.hf_model,
                mesh_device=args.mesh_device,
                max_num_seqs=args.max_num_seqs,
                block_size=args.block_size,
                port=args.port,
                log_file=server_log,
                max_model_len=args.max_model_len,
                tt_config=merged_tt_config,
                additional_args=additional,
            )
            _wait_for_server(
                proc=server_proc,
                port=args.port,
                log_file=server_log,
                timeout_seconds=args.server_timeout,
            )
            server_url = f"http://localhost:{args.port}"
        else:
            server_url = args.server_url.rstrip("/")
            _probe_external_server(server_url)

        check_stages = [s for s in stages if s != STAGE_SERVE]

        if serve_locally and not check_stages:
            print(f"\nServer ready at {server_url}")
            _hold_until_signal(server_proc, server_log)
            return

        for stage in check_stages:
            if stage == STAGE_SAMPLING:
                ok = _run_plugin_sampling_tests(
                    server_url=server_url,
                    hf_model=args.hf_model,
                    log_file=sampling_log,
                    server_log=server_log,
                    max_num_seqs=args.max_num_seqs,
                    sampling_profile=args.sampling_profile,
                )
                if not ok:
                    print("\nSampling tests failed — skipping remaining stages.")
                    sys.exit(1)
            elif stage == STAGE_QUALITATIVE:
                _run_qualitative_prompts(
                    server_url=server_url,
                    hf_model=args.hf_model,
                    prompts_file=args.prompts.resolve(),
                    output_dir=output_dir,
                )
            elif stage == STAGE_BENCHMARK:
                _run_serving_benchmark(
                    server_url=server_url,
                    hf_model=args.hf_model,
                    output_dir=output_dir,
                    prompt_len=args.benchmark_prompt_len,
                    output_len=args.benchmark_output_len,
                    num_requests=args.benchmark_num_requests,
                    concurrency=args.benchmark_concurrency,
                )

        print("\n" + "=" * 60)
        print(f"vLLM checks complete. Results in {output_dir}")
        print("=" * 60)
    finally:
        if server_proc is not None:
            _shutdown(server_proc, server_log)


if __name__ == "__main__":
    _main()
