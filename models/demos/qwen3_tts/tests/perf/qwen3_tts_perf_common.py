# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared machinery for the two single-window Tracy reports in this folder.

Each report is produced by **one** pytest test that spawns its own Tracy capture:
the test re-executes its own file as a plain script under ``python -m tracy``
(running that file's ``main()``), then turns the resulting
``ops_perf_results_*.csv`` into the report artifacts. So ``pytest <file>`` is the
whole command — no separate tracy invocation, no hand-run ``tt-perf-report``.

Why a subprocess at all: the device profiler writes its CSV only when the
profiled process exits, so a test can never read its own capture. The multi-window
reports that used to live here worked around that with a shell driver; these two
windows are small enough to do it in-process.

**The profiled device graph is not redefined here.** Both windows call straight
into ``qwen3_tts_perf_layers.py``, which owns the layer construction and emits the
``start`` / ``stop`` signposts, so a report here always describes that one op
sequence — including its ``QWEN3_TTS_BF8_WEIGHTS`` dtype
handling. That module's own docstring records what happens when a second copy of
the layer setup goes stale: it profiled bfloat16 gate/up at 116 us against the
92 us the model actually ran. Importing is what keeps the two honest.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

try:
    from tracy import signpost
except ModuleNotFoundError:  # plain pytest run: the outer driver never signposts

    def signpost(*_a, **_k):
        pass


REPO_ROOT = Path(__file__).resolve().parents[5]
PERF_DIR = Path(__file__).resolve().parent
REPORTS_DIR = PERF_DIR / "reports"
_OPSLIST = PERF_DIR / "qwen3_tts_perf_report_opslist.py"

# The profiler's DRAM buffer holds TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT programs
# and DROPS markers past it with no error, leaving a partial CSV. A single layer is
# tens of ops, so this is a wide margin — the AR-frame windows need 20000.
_OP_SUPPORT_COUNT = int(os.environ.get("QWEN3_TTS_PERF_OP_SUPPORT_COUNT", "2000"))

# Optional regression gate, off by default: device kernel us for the window.
# Left opt-in because the number is per-SKU (N150 has no collectives, N300 TP=2
# splits the heads) and this folder is a measurement tool, not a CI golden.
_BUDGET_ENV = "QWEN3_TTS_PERF_BUDGET_US"


# ── the profiled windows ─────────────────────────────────────────────────────
# Imported, not reimplemented — see the module docstring. ``qwen3_tts_perf_layers``
# owns every device graph these reports profile.
from models.demos.qwen3_tts.tests.perf import qwen3_tts_perf_layers as _layers


def open_perf_device():
    """Device opened the way the demo opens it (honours MESH_DEVICE)."""
    return _layers.open_device()


def close_perf_device(device, mesh_shape) -> None:
    _layers.close_device(device, mesh_shape)


def build_talker_layer(device):
    """One Talker ``DecoderLayer`` with the deployed weight dtype and random weights."""
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSTalkerConfig

    return _layers.make_talker_layer(device, Qwen3TTSTalkerConfig())


def run_prefill_single_layer_window(device, layer, seq_len: int) -> None:
    """One Talker decoder layer, prefill, at a demo TRACE bucket, between signposts."""
    _layers.run_talker_prefill(device, layer, seq_len)


def run_talker_decode_layer_window(device, layer) -> None:
    """One Talker decoder layer, one **deployed** decode step, between signposts.

    The deployed form: device ``cur_pos_tensor`` + a full ``[1, heads, 1, kv_max]``
    mask, so the layer uses ``paged_fused_update_cache`` and attends over the whole
    KV cache. The eager fallback in the same module slices the cache to one position
    and is a graph the demo never runs.
    """
    _layers.run_talker_decode_traced(device, layer)


def prefill_buckets() -> tuple[int, ...]:
    return _layers.DEMO_TALKER_PREFILL_BUCKETS


def profile_window(start: str, stop: str, warmup: bool = True):
    """Name one window's signposts (and skip its compile pass) — see the tests module."""
    return _layers.profile_window(start, stop, warmup)


def build_code_predictor(device):
    """Production ``CodePredictor`` with a single layer and random weights.

    Reuses ``qwen3_tts_perf_layers.synthetic_cp_sd`` so the weights match the graph
    the CP windows profile.
    """
    from models.demos.qwen3_tts.tt.code_predictor import CodePredictor
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig, Qwen3TTSTalkerConfig

    talker_h = Qwen3TTSTalkerConfig().hidden_size
    cfg = Qwen3TTSCodePredictorConfig(num_hidden_layers=1)
    return CodePredictor(
        device=device,
        config=cfg,
        talker_hidden_size=talker_h,
        state_dict=_layers.synthetic_cp_sd(cfg, talker_hidden=talker_h, num_layers=1),
    )


def run_cp_prefill_layer_window(device, code_predictor) -> None:
    """One CodePredictor layer at the demo's CP prefill (seq=2), between signposts."""
    _layers.run_cp_layer_prefill(device, code_predictor)


def run_cp_decode_layer_window(device, code_predictor) -> None:
    """One CodePredictor layer at the demo's CP decode (seq=1), between signposts."""
    _layers.run_cp_layer_decode(device, code_predictor)


# Demo generation defaults (server.py Qwen3TTSConfig): the sampler's cost depends on
# top_k only through the k=64 tile, but temperature scales the noise the host folds in.
_SAMPLING_TOP_K = 50
_SAMPLING_TEMPERATURE = 0.9


def run_cp_sampling_window(
    device,
    start: str = "cp_sampling_start",
    stop: str = "cp_sampling_stop",
    warmup: bool = True,
) -> None:
    """One device sampling call on a CP logit row, between signposts.

    The deployed chain from ``_DeviceSampler.append_sampling``:
    ``pad -> topk(k=64) -> add(gumbel row) -> sampling(k=1)``. It needs no weights, so
    this window stays checkpoint-free like the rest of the folder.

    ``refresh_noise`` is deliberately OUTSIDE the window: it is a host ``torch.rand``
    plus one ~4 KB H2D that the demo does once per frame, not once per call, so timing
    it here would attribute host work to sampling.

    Sampling is not part of any layer, and one call costs more device time than a whole
    CP layer — which is why a layers-only report is blind to the frame's largest
    non-layer cost.
    """
    import torch

    import ttnn
    from models.demos.qwen3_tts.tt.model_config import Qwen3TTSCodePredictorConfig
    from models.demos.qwen3_tts.tt.server import _DeviceSampler

    vocab = Qwen3TTSCodePredictorConfig().vocab_size
    sampler = _DeviceSampler(device, top_k=_SAMPLING_TOP_K, temperature=_SAMPLING_TEMPERATURE, seed=42)
    out_tok = sampler.alloc_token_buf()
    sampler.refresh_noise()
    logits = ttnn.from_torch(
        torch.randn(1, 1, 1, vocab, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Slot 0 compiles, slot 1 is measured: every call in a frame must consume a
    # different Gumbel row, and reusing one would also reuse its program. ``warmup``
    # mirrors ``profile_window``'s flag — the compile call is NOT signposted, so when
    # this window runs inside an enclosing start/stop it must be skipped or a whole
    # second sampling chain lands in the enclosing window.
    if warmup:
        sampler.append_sampling(logits, 0, out_tok)
        ttnn.synchronize_device(device)
    signpost(start)
    sampler.append_sampling(logits, 1, out_tok)
    ttnn.synchronize_device(device)
    signpost(stop)


# ── capture + report ─────────────────────────────────────────────────────────


def _free_port() -> int:
    """A port no one is listening on, for tracy's ``-t`` capture socket.

    Tracy defaults to 8088, and two captures in one pytest session then race: the
    second attaches to the first's lingering socket and comes back with
    "No profiling data could be captured" and one zone, after running the whole
    workload. Binding port 0 and releasing hands out a fresh port each time, which
    also avoids TIME_WAIT from an earlier run of the same window.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _latest_ops_csv(profiler_dir: Path) -> Path:
    reports = profiler_dir / "reports"
    if not reports.is_dir():
        raise AssertionError(f"tracy wrote no reports folder under {profiler_dir}")
    runs = sorted(p for p in reports.iterdir() if p.is_dir())
    if not runs:
        raise AssertionError(f"tracy wrote no run folder under {reports}")
    run = runs[-1]
    csv = run / f"ops_perf_results_{run.name}.csv"
    if not csv.is_file():
        raise AssertionError(f"no ops_perf_results CSV in {run}")
    return csv


def _emit_ops_list(out: Path, csv: Path, name: str, label: str, start: str, stop: str) -> dict:
    """Run the per-op report over one signpost window; returns its totals."""
    suffix = "" if name == "" else f"_{name}"
    totals_path = out / f"totals{suffix}.json"
    r = subprocess.run(
        [
            sys.executable,
            str(_OPSLIST),
            "--window",
            label,
            "--start",
            start,
            "--end",
            stop,
            "--json",
            str(totals_path),
            str(csv),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # Exits 1 with "no ops between signposts ..." when the window never opened —
    # which is the failure that matters here, so it is fatal.
    assert r.returncode == 0, f"per-op report for '{start}'->'{stop}' failed: {r.stderr.strip()}"
    (out / f"ops_list{suffix}.md").write_text(r.stdout)
    return json.loads(totals_path.read_text())


def capture_tracy_report(
    window: str,
    script: Path,
    *,
    min_ops: int,
    label: str = "",
    sub_windows: dict | None = None,
) -> dict:
    """Profile ``script`` under Tracy and write this window's report artifacts.

    Writes to ``perf/reports/<window>/``:

      ``run.log``            the tracy run, stdout + stderr
      ``ops.csv``            the raw ops_perf_results CSV, every column, every op
      ``ops_list.md``        full per-op list + rollups (the primary artifact)
      ``totals.json``        ops / device_ms / gap_ms / chips for the window
      ``tt-perf-report.txt`` the ranked view, when ``tt-perf-report`` is installed

    Returns the ``totals.json`` dict.
    """
    out = REPORTS_DIR / window
    out.mkdir(parents=True, exist_ok=True)
    profiler_dir = REPO_ROOT / "generated" / "profiler" / f"qwen3_tts_perf_{window}"

    cmd = [
        sys.executable,
        "-m",
        "tracy",
        "-p",
        "-v",
        "-r",
        "-o",
        str(profiler_dir),
        "--op-support-count",
        str(_OP_SUPPORT_COUNT),
        "-t",
        str(_free_port()),
        str(script),
    ]
    env = dict(os.environ)
    env.setdefault("TT_METAL_HOME", str(REPO_ROOT))
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if p)

    started = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    log = f"$ {' '.join(cmd)}\n\n{proc.stdout}\n{proc.stderr}"
    (out / "run.log").write_text(log)
    assert "No profiling data could be captured" not in log, (
        f"tracy captured no zones — the workload ran but nothing was recorded, usually a "
        f"capture-socket collision; see {out / 'run.log'}"
    )
    assert proc.returncode == 0, f"tracy capture failed (rc={proc.returncode}); see {out / 'run.log'}"

    # A run that overflows the profiler's DRAM buffer produces a PARTIAL csv and no
    # error. Refuse to publish that as a report.
    assert "Profiler DRAM buffers were full" not in log, (
        f"profiler dropped markers — raise {_OP_SUPPORT_COUNT=} via "
        f"QWEN3_TTS_PERF_OP_SUPPORT_COUNT; see {out / 'run.log'}"
    )

    csv = _latest_ops_csv(profiler_dir)
    # A plain (unprofiled) pytest run writes no new report dir, so "the newest CSV"
    # can silently be a stale one from an earlier day. Require it to be from this run.
    mtime = csv.stat().st_mtime
    assert mtime >= started - 1, f"{csv} predates this capture (mtime {time.ctime(mtime)}) — no CSV from this run"
    shutil.copy(csv, out / "ops.csv")

    totals = _emit_ops_list(out, out / "ops.csv", "", label or window, "start", "stop")
    # Sub-windows are sliced out of the SAME csv, so the outer report and every
    # per-block report describe one capture of one process.
    totals["sub_windows"] = {
        name: _emit_ops_list(out, out / "ops.csv", name, f"{label or window}: {name}", sp_start, sp_stop)
        for name, (sp_start, sp_stop) in (sub_windows or {}).items()
    }

    # tt-perf-report is a nice-to-have second view and is NOT allowed to fail the
    # test: it is installed separately from the repo and trails it (it currently dies
    # with "Unknown math fidelity: HiFi3" on any CodePredictor window). ops_list.md is
    # the primary artifact and reads the same CSV.
    tpr = shutil.which("tt-perf-report")
    if tpr:
        r = subprocess.run(
            [tpr, "--start-signpost", "start", "--end-signpost", "stop", "--no-stacked-report", str(out / "ops.csv")],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        (out / "tt-perf-report.txt").write_text(r.stdout + r.stderr)
    else:
        (out / "tt-perf-report.txt").write_text("tt-perf-report is not installed; see ops_list.md\n")

    assert totals["ops"] >= min_ops, (
        f"only {totals['ops']} ops in the {window} window (expected >= {min_ops}) — "
        f"the capture looks truncated; see {out / 'ops_list.md'}"
    )
    return totals


def report_summary(window: str, totals: dict, **extra) -> str:
    out = REPORTS_DIR / window
    shape = " ".join(f"{k}={v}" for k, v in extra.items())
    return (
        f"[{window}] {shape}\n"
        f"[{window}] {totals['ops']} device ops, {totals['device_ms']:.3f} ms device kernel, "
        f"{totals['gap_ms']:.3f} ms op-to-op gap"
        + (f", {totals['chips']} chips merged" if totals.get("chips", 1) > 1 else "")
        + "".join(
            f"\n[{window}]   {n}: {t['ops']} ops, {t['device_ms'] * 1e3:.1f} us"
            for n, t in (totals.get("sub_windows") or {}).items()
        )
        + "\n"
        f"[{window}] report: {out / 'ops_list.md'}\n"
        f"[{window}]         {out / 'tt-perf-report.txt'}\n"
        f"[{window}]         {out / 'ops.csv'}"
    )


def check_budget(window: str, totals: dict) -> None:
    """Enforce ``QWEN3_TTS_PERF_BUDGET_US`` on device kernel time, when it is set."""
    budget = os.environ.get(_BUDGET_ENV)
    if not budget:
        return
    measured_us = totals["device_ms"] * 1e3
    limit = float(budget)
    print(f"[{window}] device kernel {measured_us:.1f} us against {_BUDGET_ENV}={limit:.1f} us")
    assert measured_us <= limit, f"[{window}] {measured_us:.1f} us exceeds {_BUDGET_ENV}={limit:.1f} us"
