# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Device profiling harness: compare rotary-embedding op time (HF vs mllama RoPE) via Tracy + simple_text_demo.

Runs once per parametrized demo phase (``prefill`` or ``decode``): 1 layer, seqlen 1024, 2 generated
tokens, batch sizes 1 and 32 under `python -m tracy`, parses the ops CSV and captured demo logs (TTFT,
decode tok/s/user), logs a Markdown table (via loguru), and removes temp dirs on success.

Set the model with env `HF_MODEL` only. Phase is selected via :func:`pytest.mark.parametrize` on
``rope_perf_mode`` (``prefill`` or ``decode``).

Optional: set env `MESH_DEVICE` (e.g. ``N150``, ``N300``) — passed through to the Tracy subprocess via
``os.environ.copy()`` in :func:`_rope_perf_subprocess_env`, same as ``simple_text_demo`` expects.

Example:
  HF_MODEL=meta-llama/Llama-3.2-1B-Instruct pytest models/tt_transformers/tests/test_rope_performance.py -s
"""
import csv
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from loguru import logger

# Subprocess owns the mesh; parent must not open UMD / hold the device (see test_device_perf.py
# lines 23–26). Do not call ttnn.get_num_devices() here — it acquires CHIP_IN_USE and deadlocks the child.
pytestmark = pytest.mark.no_reset_default_device

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_TEST = "models/tt_transformers/demo/simple_text_demo.py::test_demo_text"
# Tracy respawns pytest using shell=True + " ".join(argv), so `-k "a and b"` cannot appear on the command line.
# Pytest parses PYTEST_ADDOPTS with shlex in the child process instead (see _pytest/config/__init__.py).
PYTEST_ADDOPTS_DEVICE_PERF = '-k "device-perf and performance"'

TIMEOUT_TEST = 1200

ROPE_OPS = frozenset(
    {
        "RotaryEmbeddingDeviceOperation",
        "RotaryEmbeddingLlamaDeviceOperation",
        "RotaryEmbeddingLlamaFusedQKDeviceOperation",
    }
)

COL_TRACE = "METAL TRACE ID"
COL_SESSION = "METAL TRACE REPLAY SESSION ID"
COL_OP = "OP CODE"
COL_DEVICE = "DEVICE ID"
COL_DUR = "DEVICE FW DURATION [ns]"


def model_display_id(hf_model: str) -> str:
    return hf_model.replace("/", "_").replace("-", "_")


def parse_ns(raw: Any) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def find_ops_perf_csv(output_root: Path, name_append: str) -> Path:
    """Mirror Tracy layout: OUTPUT/reports/<name_append>/<YYYY_MM_DD_HH_MM_SS>/ops_perf*.csv"""
    cfg_dir = output_root / "reports" / name_append
    if not cfg_dir.is_dir():
        raise FileNotFoundError(f"No reports subdir for {name_append!r} under {output_root}")
    ts_dirs = [d for d in cfg_dir.iterdir() if d.is_dir()]
    if not ts_dirs:
        raise FileNotFoundError(f"No timestamp subdirs under {cfg_dir}")
    latest = sorted(ts_dirs, key=lambda p: p.name, reverse=True)[0]
    matches = sorted(latest.glob("ops_perf*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one ops_perf*.csv in {latest}, found {len(matches)}")
    return matches[0]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def per_step_rope_ns(rows: list[dict[str, str]]) -> list[tuple[Any, Any, float, float]]:
    """
    Returns list of (trace_id, session_id, sum_max_ns, sum_mean_ns) per trace/session,
    matching diff_rope_prof.per_step_rope_ops_time + scalar grouping.
    """
    warmup_indices = [i for i, r in enumerate(rows) if "warmup" in str(r.get(COL_OP, "")).lower()]
    if len(warmup_indices) >= 1:
        # Warmup, if present, is a signpost that takes one row, so we skip it
        body = rows[warmup_indices[-1] + 1 :]
    else:
        body = rows
    rot_rows = [r for r in body if r.get(COL_OP) in ROPE_OPS]
    if not rot_rows:
        raise ValueError("No rotary embedding ops after filtering")

    # _op_index = cumcount per (trace, session, op, device)
    cum: dict[tuple[Any, Any, Any, Any], int] = defaultdict(int)
    keyed: list[tuple[tuple[Any, Any, Any, int], float | None]] = []
    for r in rot_rows:
        tid = r.get(COL_TRACE)
        sid = r.get(COL_SESSION)
        op = r.get(COL_OP)
        dev = r.get(COL_DEVICE)
        gk = (tid, sid, op, dev)
        idx = cum[gk]
        cum[gk] = idx + 1
        keyed.append(((tid, sid, op, idx), parse_ns(r.get(COL_DUR))))

    # Inner group: (trace, session, op, op_index) -> max/mean of DEVICE FW over rows in group
    inner_groups: dict[tuple[Any, Any, Any, int], list[float]] = defaultdict(list)
    for k, dur in keyed:
        if dur is not None:
            inner_groups[k].append(dur)

    inner_sums: dict[tuple[Any, Any], list[tuple[float, float]]] = defaultdict(list)
    for (tid, sid, op, op_i), vals in inner_groups.items():
        if not vals:
            continue
        mx = max(vals)
        mean = sum(vals) / len(vals)
        inner_sums[(tid, sid)].append((mx, mean))

    out: list[tuple[Any, Any, float, float]] = []
    for (tid, sid), pairs in inner_sums.items():
        smx = sum(p[0] for p in pairs)
        smn = sum(p[1] for p in pairs)
        out.append((tid, sid, smx, smn))
    return out


def scalar_rope_time_ns(rows: list[dict[str, str]]) -> tuple[float, float]:
    per = per_step_rope_ns(rows)
    if not per:
        raise ValueError("No per-step rope aggregates")
    n = len(per)
    max_mean = sum(t[2] for t in per) / n
    mean_mean = sum(t[3] for t in per) / n
    return max_mean, mean_mean


def pct_diff(hf_val: float, llama_val: float) -> float:
    if hf_val == 0:
        return float("nan")
    return 100.0 * (hf_val - llama_val) / hf_val


def ratio(hf_val: float, llama_val: float) -> float:
    if llama_val == 0:
        return float("nan")
    return hf_val / llama_val


# Logged by simple_text_demo.py after inference (see Performance metrics block).
_RE_TTFT_MS = re.compile(r"Average Time to First Token \(TTFT\):\s*([0-9.]+)\s*ms")
_RE_AVG_SPEED = re.compile(
    r"Average speed:\s*([0-9.]+)\s*ms @\s*([0-9.]+)\s*tok/s/user",
    re.IGNORECASE,
)


def parse_simple_text_demo_perf_log(log: str) -> tuple[float | None, float | None]:
    """
    Parse TTFT (ms) and decode throughput (tok/s/user) from captured pytest/demo subprocess output.

    If a pattern appears multiple times (e.g. several tests), the last match is used.
    """
    ttft_hits = _RE_TTFT_MS.findall(log)
    speed_hits = _RE_AVG_SPEED.findall(log)
    ttft_ms = float(ttft_hits[-1]) if ttft_hits else None
    tok_s_u = float(speed_hits[-1][1]) if speed_hits else None
    return ttft_ms, tok_s_u


def _fmt_pct_diff(hf_val: float | None, llama_val: float | None) -> str:
    if hf_val is None or llama_val is None:
        return "n/a"
    if hf_val == 0:
        return "n/a"
    return f"{pct_diff(hf_val, llama_val):.2f}"


def _fmt_ratio_metric(hf_val: float | None, llama_val: float | None) -> str:
    if hf_val is None or llama_val is None:
        return "n/a"
    r = ratio(hf_val, llama_val)
    if math.isnan(r):
        return "n/a"
    return f"{r:.4f}"


def _fmt_float_opt(x: float | None, *, ndigits: int) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{ndigits}f}"


def _rope_perf_demo_pytest_args(
    *, batch_size: int, use_hf_rope: bool, mode: str, timeout: int | None = None
) -> list[str]:
    """Arguments after ``python -m pytest`` for the simple_text_demo RoPE perf configuration."""
    args = [
        DEMO_TEST,
        "--mode",
        mode,
        "--num_layers",
        "1",
        "--max_seq_len",
        "1024",
        "--max_generated_tokens",
        "2",
        "--batch_size",
        str(batch_size),
    ]
    if use_hf_rope:
        args.append("--use_hf_rope")
    if timeout is not None:
        args.append("--timeout")
        args.append(str(timeout))
    return args


def rope_perf_pytest_argv(
    *, batch_size: int, use_hf_rope: bool, mode: str = "prefill", timeout: int | None = None
) -> list[str]:
    """
    Full argv to run the demo test under pytest without Tracy (same workload as the Tracy harness).

    Example: ``subprocess.run(rope_perf_pytest_argv(...), cwd=repo_root, env=env, check=True)``
    """
    return [
        sys.executable,
        "-m",
        "pytest",
        *_rope_perf_demo_pytest_args(batch_size=batch_size, use_hf_rope=use_hf_rope, mode=mode, timeout=timeout),
    ]


def rope_perf_tracy_argv(*, output_dir: Path | str, run_name: str, pytest_argv: list[str]) -> list[str]:
    """
    Wrap a full ``python -m pytest ...`` argv produced by :func:`rope_perf_pytest_argv` with Tracy.

    ``pytest_argv`` must be ``[sys.executable, '-m', 'pytest', ...]`` so everything after ``pytest``
    is forwarded to the child pytest invocation.
    """
    if len(pytest_argv) < 3 or pytest_argv[1:3] != ["-m", "pytest"]:
        raise ValueError("pytest_argv must start with [executable, '-m', 'pytest', ...]")
    demo_args = pytest_argv[3:]
    return [
        sys.executable,
        "-m",
        "tracy",
        "-v",
        "-r",
        "-p",
        "-o",
        str(output_dir),
        "-n",
        run_name,
        "--check-exit-code",
        "-m",
        "pytest",
        *demo_args,
    ]


def _run_tracy(
    *,
    repo_root: Path,
    env: dict[str, str],
    argv: list[str],
    timeout_sec: int | None = None,
) -> str:
    """
    Run Tracy + pytest; capture combined stdout/stderr for parsing demo perf logs, and log child output
    with ``logger.info`` so ``pytest -s`` still shows it (via loguru sinks).
    """
    proc = subprocess.run(
        argv,
        cwd=repo_root,
        env=env,
        check=True,
        timeout=timeout_sec if timeout_sec is not None else int(os.environ.get("ROPE_PERF_TRACY_TIMEOUT_SEC", "3600")),
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        logger.info(proc.stdout)
    if proc.stderr:
        logger.info(proc.stderr)
    return (proc.stdout or "") + "\n" + (proc.stderr or "")


def _run_tracy_demo(
    *,
    repo_root: Path,
    batch_size: int,
    use_hf_rope: bool,
    run_name: str,
    mode: str,
    timeout: int | None = None,
) -> tuple[list[dict[str, str]], str]:
    env = _rope_perf_subprocess_env()

    with TemporaryDirectory(prefix="rope_perf_") as tmp:
        pytest_cmd = rope_perf_pytest_argv(batch_size=batch_size, use_hf_rope=use_hf_rope, mode=mode, timeout=timeout)
        argv = rope_perf_tracy_argv(output_dir=tmp, run_name=run_name, pytest_argv=pytest_cmd)
        captured_log = _run_tracy(repo_root=repo_root, env=env, argv=argv)
        csv_path = find_ops_perf_csv(Path(tmp), run_name)
        return load_rows(csv_path), captured_log


def _rope_perf_subprocess_env() -> dict[str, str]:
    """Environment for the rope-perf demo subprocess (with or without Tracy).

    Starts from ``os.environ.copy()`` so ``MESH_DEVICE`` and other vars are inherited unchanged; only
    rope-perf-specific keys are overwritten below. Do not inherit parent pytest's ``PYTEST_ADDOPTS``;
    device-perf test selection is applied via ``PYTEST_ADDOPTS`` (see module comment above).
    """
    env = os.environ.copy()
    env["PYTEST_ADDOPTS"] = PYTEST_ADDOPTS_DEVICE_PERF
    return env


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "model_id",
        "mode",
        "gen_tok",
        "batch_size",
        "hf_ttft_ms",
        "mllama_ttft_ms",
        "pct_diff_ttft_%",
        "ratio_ttft",
        "hf_tok_s_u",
        "mllama_tok_s_u",
        "pct_diff_tok_s_u_%",
        "ratio_tok_s_u",
        "hf_rope_max_mean_ns",
        "mllama_rope_max_mean_ns",
        "delta_rope_max_mean_ns",
        "pct_diff_rope_max_%",
        "pct_diff_rope_mean_%",
        "ratio_rope_max",
        "ratio_rope_mean",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[h]) for h in headers) + " |")
    return "\n".join(lines)


@pytest.mark.timeout(14400)
@pytest.mark.parametrize("rope_perf_mode", ["prefill", "decode"])
def test_rope_performance_comparison_table(rope_perf_mode: str):
    """
    Compare HF vs mllama RoPE device time for ``rope_perf_mode`` (prefill or decode), bs in {1, 32},
    1 layer, seqlen 1024, 2 generated tokens. Model from ``HF_MODEL`` env only.
    Optional ``MESH_DEVICE`` is inherited by the Tracy subprocess. Logs a Markdown table (run pytest with -s).
    """
    hf_model = os.environ.get("HF_MODEL")
    assert (
        hf_model is not None
    ), "HF_MODEL is not set, it needs to be set to a HuggingFace model name, for example: export HF_MODEL=meta-llama/Llama-3.2-1B-Instruct"
    model_id = model_display_id(hf_model)
    table_rows: list[dict[str, Any]] = []
    for batch_size in (1, 32):
        base = f"ropeperf_{rope_perf_mode}_g2_sl1024_bs{batch_size}"
        llama_rows, llama_log = _run_tracy_demo(
            repo_root=REPO_ROOT,
            batch_size=batch_size,
            use_hf_rope=False,
            run_name=f"{base}_mllama",
            mode=rope_perf_mode,
            timeout=TIMEOUT_TEST,
        )
        hf_rows, hf_log = _run_tracy_demo(
            repo_root=REPO_ROOT,
            batch_size=batch_size,
            use_hf_rope=True,
            run_name=f"{base}_hf",
            mode=rope_perf_mode,
            timeout=TIMEOUT_TEST,
        )
        lm, lmean = scalar_rope_time_ns(llama_rows)
        hm, hmean = scalar_rope_time_ns(hf_rows)
        hf_ttft_ms, hf_tok_s_u = parse_simple_text_demo_perf_log(hf_log)
        ml_ttft_ms, ml_tok_s_u = parse_simple_text_demo_perf_log(llama_log)
        table_rows.append(
            {
                "model_id": model_id,
                "mode": rope_perf_mode,
                "gen_tok": 2,
                "batch_size": batch_size,
                "hf_ttft_ms": _fmt_float_opt(hf_ttft_ms, ndigits=2),
                "mllama_ttft_ms": _fmt_float_opt(ml_ttft_ms, ndigits=2),
                "pct_diff_ttft_%": _fmt_pct_diff(hf_ttft_ms, ml_ttft_ms),
                "ratio_ttft": _fmt_ratio_metric(hf_ttft_ms, ml_ttft_ms),
                "hf_tok_s_u": _fmt_float_opt(hf_tok_s_u, ndigits=2),
                "mllama_tok_s_u": _fmt_float_opt(ml_tok_s_u, ndigits=2),
                "pct_diff_tok_s_u_%": _fmt_pct_diff(hf_tok_s_u, ml_tok_s_u),
                "ratio_tok_s_u": _fmt_ratio_metric(hf_tok_s_u, ml_tok_s_u),
                "hf_rope_max_mean_ns": f"{hm:.0f}",
                "mllama_rope_max_mean_ns": f"{lm:.0f}",
                "delta_rope_max_mean_ns": f"{hm - lm:.0f}",
                "pct_diff_rope_max_%": f"{pct_diff(hm, lm):.2f}",
                "pct_diff_rope_mean_%": f"{pct_diff(hmean, lmean):.2f}",
                "ratio_rope_max": f"{ratio(hm, lm):.4f}",
                "ratio_rope_mean": f"{ratio(hmean, lmean):.4f}",
            }
        )

    md = _markdown_table(table_rows)
    mesh_device_env = os.environ.get("MESH_DEVICE")
    mesh_device_display = mesh_device_env if mesh_device_env else "(unset — demo mesh from available devices)"
    logger.info("\n### RoPE device time (HF vs mllama)\n")
    logger.info(f"`HF_MODEL`: `{hf_model}`")
    logger.info(f"`MESH_DEVICE`: `{mesh_device_display}`")
    logger.info(f"`rope_perf_mode`: `{rope_perf_mode}`")
    logger.info(
        "*Difference metrics (per table row; HF vs mllama):* "
        "`pct_diff_ttft_%` / `pct_diff_tok_s_u_%` / `pct_diff_rope_*` = 100 × (HF − mllama) / HF. "
        "For TTFT and RoPE (time), a positive % means HF uses more time than mllama. "
        "For decode `tok/s/user`, a positive % means HF is faster (higher throughput). "
        "`ratio_*` = HF / mllama. "
        "`delta_rope_max_mean_ns` = HF − mllama (aggregated RoPE max-mean ns).\n"
    )
    logger.info(md)
    logger.info("")
