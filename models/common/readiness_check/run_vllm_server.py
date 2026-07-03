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
  benchmark     Run ``vllm bench serve``. The primary headline profile is a
                greedy single-user 128/128/1 decode run that reports TTFT
                separately from TPOT/ITL and writes raw ``vllm_result.json``
                plus normalized ``vllm_benchmark.json``. By default the runner
                also runs the vLLM-nightly-shaped 100/100/32 serving-burst
                profile as a secondary CI/capacity artifact. Defaults to greedy
                ``--temperature 0.0``; opt into exact server-generation-config
                behavior with ``--benchmark-use-server-generation-config``.

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

To install vLLM, if not already present:
1. Clone `https://github.com/tenstorrent/vllm.git`
2. Switch to the `dev` branch
3. Follow the Tenstorrent vLLM installation instructions for that checkout.

Before invoking it, two things must already be true:

  1. `<model_dir>/tt/generator_vllm.py` exists and implements the model.
  2. The model architecture is registered in the TT vLLM platform registry.
     Without that registration vLLM will reject the architecture at startup with
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
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import requests
from transformers import AutoTokenizer

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

# Primary benchmark shape: single-user decode, used for headline t/s/u.
# We force greedy temperature for readiness t/s/u unless the caller explicitly
# asks to reproduce exact server-generation-config behavior.
DEFAULT_BENCH_PROMPT_LEN = 128
DEFAULT_BENCH_OUTPUT_LEN = 128
DEFAULT_BENCH_NUM_REQUESTS = 1
DEFAULT_BENCH_CONCURRENCY: Optional[int] = 1
DEFAULT_BENCH_TEMPERATURE = 0.0

# Secondary serving-burst profile matching `.github/workflows/vllm-nightly-tests-impl.yaml`.
DEFAULT_CI_SERVING_BENCHMARK = True
DEFAULT_CI_BENCH_PROMPT_LEN = 100
DEFAULT_CI_BENCH_OUTPUT_LEN = 100
DEFAULT_CI_BENCH_NUM_REQUESTS = 32
DEFAULT_CI_BENCH_CONCURRENCY: Optional[int] = None

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
    "P150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
}


def _find_plugin_tests_dir() -> Path:
    """Locate the TT vLLM pytest suite in either old plugin or in-tree layouts."""
    candidates: List[Path] = []

    plugin_spec = importlib.util.find_spec("vllm_tt_plugin")
    if plugin_spec is not None and plugin_spec.origin is not None:
        # Old layout: <plugin_root>/src/vllm_tt_plugin/__init__.py
        plugin_root = Path(plugin_spec.origin).resolve().parent.parent.parent
        candidates.append(plugin_root / "tests" / "tt")

    vllm_spec = importlib.util.find_spec("vllm")
    if vllm_spec is not None and vllm_spec.origin is not None:
        # Current Tenstorrent fork layout: <vllm_repo>/vllm/__init__.py
        vllm_repo = Path(vllm_spec.origin).resolve().parent.parent
        candidates.append(vllm_repo / "tests" / "tt")

    for tests_dir in candidates:
        if tests_dir.is_dir():
            return tests_dir

    checked = ", ".join(str(path) for path in candidates) or "no importable vllm/vllm_tt_plugin package"
    raise RuntimeError(f"Could not find TT vLLM pytest tests. Checked: {checked}")


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
    cmd += ["--additional-config", json.dumps({"tt": tt_config})]
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
    tokenizer = AutoTokenizer.from_pretrained(hf_model)

    def completion(prompt: str, *, temperature: float, top_p: float | None = None) -> str:
        payload: dict[str, Any] = {
            "model": hf_model,
            "prompt": prompt,
            "max_tokens": 256,
            "temperature": temperature,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        response = requests.post(
            f"{server_url.rstrip('/')}/v1/completions",
            json=payload,
            timeout=DEFAULT_VLLM_RPC_TIMEOUT_MS / 1000,
        )
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"Completion request failed: {data['error']}")
        return data["choices"][0].get("text") or ""

    results: List[dict[str, Any]] = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n  Prompt {i}/{len(prompts)}: {prompt[:60]}...")
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        greedy_text = completion(rendered_prompt, temperature=0.0)
        sampled_text = completion(rendered_prompt, temperature=0.7, top_p=0.9)

        results.append(
            {
                "prompt": prompt,
                "rendered_prompt": rendered_prompt,
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


def _vllm_cli_command() -> List[str]:
    """Return a vLLM CLI invocation in the active Python environment."""
    vllm_exe = shutil.which("vllm")
    if vllm_exe is not None:
        return [vllm_exe]
    return [sys.executable, "-m", "vllm.entrypoints.cli.main"]


def _quoted_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _json_number(data: dict[str, Any], key: str) -> Optional[float]:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _decode_tps_from_ms(ms: Optional[float]) -> Optional[float]:
    return (1000.0 / ms) if ms and ms > 0 else None


def _metric_summary(raw: dict[str, Any], metric: str) -> dict[str, Optional[float]]:
    return {
        "p50": _json_number(raw, f"median_{metric}_ms"),
        "p99": _json_number(raw, f"p99_{metric}_ms"),
        "mean": _json_number(raw, f"mean_{metric}_ms"),
    }


def _write_normalized_vllm_benchmark_summary(
    *,
    raw: dict[str, Any],
    output_file: Path,
    raw_result_file: Path,
    command: List[str],
    profile_name: str,
    comparison_scope: str,
    prompt_len: int,
    output_len: int,
    num_requests: int,
    concurrency: Optional[int],
    temperature: Optional[float],
) -> dict[str, Any]:
    total_output_tokens = _json_number(raw, "total_output_tokens")
    requested_output_tokens = output_len * num_requests
    mean_tpot_ms = _json_number(raw, "mean_tpot_ms")
    median_itl_ms = _json_number(raw, "median_itl_ms")
    mean_itl_ms = _json_number(raw, "mean_itl_ms")

    summary: dict[str, Any] = {
        "source": "vllm bench serve",
        "profile": profile_name,
        "comparison_scope": comparison_scope,
        "raw_result_file": str(raw_result_file),
        "command": command,
        "command_string": _quoted_cmd(command),
        "config": {
            "prompt_len": prompt_len,
            "output_len": output_len,
            "num_requests": num_requests,
            "concurrency": concurrency,
            "temperature": temperature,
            "generation_config_mode": "server_default" if temperature is None else "explicit",
            "dataset_name": "random",
            "ignore_eos": True,
        },
        "elapsed_s": _json_number(raw, "duration"),
        "completed_requests": raw.get("completed"),
        "total_input_tokens": _json_number(raw, "total_input_tokens"),
        "total_output_tokens": total_output_tokens,
        "requested_output_tokens": requested_output_tokens,
        "missing_output_tokens": (
            max(0, requested_output_tokens - int(total_output_tokens)) if total_output_tokens is not None else None
        ),
        "ttft_ms": _metric_summary(raw, "ttft"),
        "tpot_ms": _metric_summary(raw, "tpot"),
        "itl_ms": _metric_summary(raw, "itl"),
        "e2el_ms": _metric_summary(raw, "e2el"),
        "itl_p50_decode_tps": _decode_tps_from_ms(median_itl_ms),
        "itl_mean_decode_tps": _decode_tps_from_ms(mean_itl_ms),
        "vllm_mean_tpot_decode_tps": _decode_tps_from_ms(mean_tpot_ms),
        "mean_per_request_decode_tps": _decode_tps_from_ms(mean_tpot_ms),
        "mean_per_request_decode_tps_basis": (
            "1000 / mean_tpot_ms from vllm bench serve; this is the TPOT-derived "
            "decode value. For burst-serving profiles, scheduler admission and "
            "chunked prefill can affect TPOT, so use the single-user profile for "
            "headline decode t/s/u."
        ),
        "output_throughput_tok_per_s": _json_number(raw, "output_throughput"),
        "request_throughput_per_s": _json_number(raw, "request_throughput"),
        "total_token_throughput_tok_per_s": _json_number(raw, "total_token_throughput"),
    }
    output_file.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _print_file_tail(log_file: Path, lines: int = 40) -> None:
    if not log_file.exists():
        return
    print(f"\n=== Log tail ({log_file}, last {lines} lines) ===")
    for line in log_file.read_text(errors="replace").splitlines()[-lines:]:
        print(line)


def _run_serving_benchmark(
    *,
    server_url: str,
    hf_model: str,
    output_dir: Path,
    profile_name: str,
    comparison_scope: str,
    raw_result_filename: str,
    summary_filename: str,
    log_filename: str,
    prompt_len: int,
    output_len: int,
    num_requests: int,
    concurrency: Optional[int],
    temperature: Optional[float],
    additional_args: List[str],
) -> dict[str, Any]:
    """
    Run one vLLM serving benchmark profile.

    The primary caller writes `vllm_result.json` and `vllm_benchmark.json`;
    secondary profiles use distinct filenames.
    """
    print(f"\n=== Running vLLM serving benchmark: {profile_name} ===")
    print(
        f"  prompt_len={prompt_len}, output_len={output_len}, "
        f"num_requests={num_requests}, concurrency={concurrency}, temperature={temperature}"
    )
    print(f"  comparison_scope: {comparison_scope}")

    raw_result_file = output_dir / raw_result_filename
    summary_file = output_dir / summary_filename
    log_file = output_dir / log_filename
    for stale_file in (raw_result_file, summary_file):
        if stale_file.exists():
            stale_file.unlink()

    cmd: List[str] = [
        *_vllm_cli_command(),
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--model",
        hf_model,
        "--base-url",
        server_url.rstrip("/"),
        "--endpoint",
        "/v1/completions",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(prompt_len),
        "--random-output-len",
        str(output_len),
        "--num-prompts",
        str(num_requests),
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--save-result",
        "--result-filename",
        str(raw_result_file),
    ]
    if concurrency is not None:
        cmd.extend(["--max-concurrency", str(concurrency)])
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    cmd.extend(additional_args)

    print(f"  command: {_quoted_cmd(cmd)}")
    print(f"  log: {log_file}")
    with open(log_file, "wb") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        _print_file_tail(log_file)
        raise RuntimeError(f"`vllm bench serve` failed with exit code {result.returncode}; inspect {log_file}")
    if not raw_result_file.exists():
        _print_file_tail(log_file)
        raise RuntimeError(f"`vllm bench serve` succeeded but did not write {raw_result_file}")

    raw = json.loads(raw_result_file.read_text())
    expected_completed = raw.get("num_prompts", num_requests)
    if raw.get("completed") != expected_completed:
        _print_file_tail(log_file)
        raise RuntimeError(
            "`vllm bench serve` did not complete every prompt: "
            f"completed={raw.get('completed')} num_prompts={expected_completed}. "
            f"Raw result: {raw_result_file}"
        )

    summary = _write_normalized_vllm_benchmark_summary(
        raw=raw,
        output_file=summary_file,
        raw_result_file=raw_result_file,
        command=cmd,
        profile_name=profile_name,
        comparison_scope=comparison_scope,
        prompt_len=prompt_len,
        output_len=output_len,
        num_requests=num_requests,
        concurrency=concurrency,
        temperature=temperature,
    )

    def _fmt(v: Optional[float], unit: str) -> str:
        return f"{v:.1f}{unit}" if v is not None else "n/a"

    print(f"\n=== Serving benchmark summary: {profile_name} ===")
    print(f"  Requests : {summary['completed_requests']}/{expected_completed} completed")
    print(f"  TTFT     : P50={_fmt(summary['ttft_ms']['p50'], 'ms')}  P99={_fmt(summary['ttft_ms']['p99'], 'ms')}")
    print(f"  TPOT     : mean={_fmt(summary['tpot_ms']['mean'], 'ms')}  P99={_fmt(summary['tpot_ms']['p99'], 'ms')}")
    print(f"  ITL      : P50={_fmt(summary['itl_ms']['p50'], 'ms')}  P99={_fmt(summary['itl_ms']['p99'], 'ms')}")
    print(f"  Output   : {_fmt(summary['output_throughput_tok_per_s'], ' tok/s')} aggregate")
    print(f"  Decode   : {_fmt(summary['vllm_mean_tpot_decode_tps'], ' t/s/u')} from mean TPOT")
    print(f"  Raw      : {raw_result_file}")
    print(f"  Summary  : {summary_file}")
    return summary


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
        help=(
            "Tokens per synthetic prompt for the primary single-user benchmark "
            f"(default {DEFAULT_BENCH_PROMPT_LEN})."
        ),
    )
    parser.add_argument(
        "--benchmark-output-len",
        type=int,
        default=DEFAULT_BENCH_OUTPUT_LEN,
        help=(
            "Tokens to generate per request in the primary single-user benchmark "
            f"(default {DEFAULT_BENCH_OUTPUT_LEN})."
        ),
    )
    parser.add_argument(
        "--benchmark-num-requests",
        type=int,
        default=DEFAULT_BENCH_NUM_REQUESTS,
        help=("Total requests sent in the primary single-user benchmark " f"(default {DEFAULT_BENCH_NUM_REQUESTS})."),
    )
    parser.add_argument(
        "--benchmark-concurrency",
        type=int,
        default=DEFAULT_BENCH_CONCURRENCY,
        help=(
            "`vllm bench serve --max-concurrency` for the primary single-user benchmark "
            f"(default {DEFAULT_BENCH_CONCURRENCY})."
        ),
    )
    parser.add_argument(
        "--benchmark-temperature",
        type=float,
        default=DEFAULT_BENCH_TEMPERATURE,
        help=(
            "Temperature passed to `vllm bench serve` for the benchmark. "
            f"Default {DEFAULT_BENCH_TEMPERATURE} preserves greedy t/s/u comparability. "
            "Use --benchmark-use-server-generation-config to omit this flag and reproduce exact nightly "
            "server-generation-config behavior."
        ),
    )
    parser.add_argument(
        "--benchmark-use-server-generation-config",
        action="store_true",
        help=(
            "Do not pass `--temperature` to `vllm bench serve`; the server/model generation_config decides "
            "sampling parameters. This matches current vLLM nightly behavior for models that do not add "
            "temperature via additional benchmark args, but is not comparable to the default greedy single-user "
            "t/s/u metric."
        ),
    )
    parser.add_argument(
        "--benchmark-ci-serving",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CI_SERVING_BENCHMARK,
        help=(
            "Also run the vLLM-nightly-shaped 100/100/32 serving-burst profile as secondary evidence "
            f"(default {DEFAULT_CI_SERVING_BENCHMARK}). Use --no-benchmark-ci-serving to skip it."
        ),
    )
    parser.add_argument(
        "--ci-benchmark-prompt-len",
        type=int,
        default=DEFAULT_CI_BENCH_PROMPT_LEN,
        help=(
            "Tokens per synthetic prompt for the secondary CI serving-burst benchmark "
            f"(default {DEFAULT_CI_BENCH_PROMPT_LEN})."
        ),
    )
    parser.add_argument(
        "--ci-benchmark-output-len",
        type=int,
        default=DEFAULT_CI_BENCH_OUTPUT_LEN,
        help=(
            "Tokens to generate per request in the secondary CI serving-burst benchmark "
            f"(default {DEFAULT_CI_BENCH_OUTPUT_LEN})."
        ),
    )
    parser.add_argument(
        "--ci-benchmark-num-requests",
        type=int,
        default=DEFAULT_CI_BENCH_NUM_REQUESTS,
        help=(
            "Total requests sent in the secondary CI serving-burst benchmark "
            f"(default {DEFAULT_CI_BENCH_NUM_REQUESTS})."
        ),
    )
    parser.add_argument(
        "--ci-benchmark-concurrency",
        type=int,
        default=DEFAULT_CI_BENCH_CONCURRENCY,
        help=(
            "Optional `vllm bench serve --max-concurrency` for the secondary CI serving-burst benchmark. "
            "Default is unset, matching vLLM nightly CI."
        ),
    )
    parser.add_argument(
        "--additional-benchmark-args",
        type=str,
        default="",
        help=(
            "Catch-all for other `vllm bench serve` args not covered by the typed flags. "
            'Quoted, e.g. "--request-rate 0.25 --metric-percentiles 50,90,99".'
        ),
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

    additional_server_args = shlex.split(args.additional_server_args) if args.additional_server_args else []
    additional_benchmark_args = shlex.split(args.additional_benchmark_args) if args.additional_benchmark_args else []
    benchmark_temperature = None if args.benchmark_use_server_generation_config else args.benchmark_temperature

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
                additional_args=additional_server_args,
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
                primary_summary = _run_serving_benchmark(
                    server_url=server_url,
                    hf_model=args.hf_model,
                    output_dir=output_dir,
                    profile_name="single_user_decode",
                    comparison_scope=(
                        "Primary batch-1 serving decode profile for headline t/s/u. TTFT is reported separately "
                        "and is not blended into the TPOT-derived decode value beyond normal single-request timing."
                    ),
                    raw_result_filename="vllm_result.json",
                    summary_filename="vllm_benchmark.json",
                    log_filename="vllm_benchmark.log",
                    prompt_len=args.benchmark_prompt_len,
                    output_len=args.benchmark_output_len,
                    num_requests=args.benchmark_num_requests,
                    concurrency=args.benchmark_concurrency,
                    temperature=benchmark_temperature,
                    additional_args=additional_benchmark_args,
                )
                if args.benchmark_ci_serving:
                    ci_summary = _run_serving_benchmark(
                        server_url=server_url,
                        hf_model=args.hf_model,
                        output_dir=output_dir,
                        profile_name="ci_serving_burst",
                        comparison_scope=(
                            "vLLM-nightly-style 100/100/32 serving-burst profile for CI parity and serving "
                            "capacity. Do not use as headline decode t/s/u because burst admission and chunked "
                            "prefill can affect TPOT."
                        ),
                        raw_result_filename="vllm_ci_serving_result.json",
                        summary_filename="vllm_ci_serving_benchmark.json",
                        log_filename="vllm_ci_serving_benchmark.log",
                        prompt_len=args.ci_benchmark_prompt_len,
                        output_len=args.ci_benchmark_output_len,
                        num_requests=args.ci_benchmark_num_requests,
                        concurrency=args.ci_benchmark_concurrency,
                        temperature=benchmark_temperature,
                        additional_args=additional_benchmark_args,
                    )
                    primary_summary["secondary_benchmarks"] = {
                        "ci_serving_burst": {
                            "summary_file": str(output_dir / "vllm_ci_serving_benchmark.json"),
                            "raw_result_file": str(output_dir / "vllm_ci_serving_result.json"),
                            "log_file": str(output_dir / "vllm_ci_serving_benchmark.log"),
                            "config": ci_summary["config"],
                            "output_throughput_tok_per_s": ci_summary["output_throughput_tok_per_s"],
                            "vllm_mean_tpot_decode_tps": ci_summary["vllm_mean_tpot_decode_tps"],
                            "comparison_scope": ci_summary["comparison_scope"],
                        }
                    }
                    (output_dir / "vllm_benchmark.json").write_text(json.dumps(primary_summary, indent=2) + "\n")

        print("\n" + "=" * 60)
        print(f"vLLM checks complete. Results in {output_dir}")
        print("=" * 60)
    finally:
        if server_proc is not None:
            _shutdown(server_proc, server_log)


if __name__ == "__main__":
    _main()
