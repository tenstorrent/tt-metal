# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Kimi-50k chunked-prefill perf sweep for ring joint SDPA.

Sweeps three knobs and reports *pure-compute math utilization* (CCL cores
stripped) at the chunk whose prefix is closest to the 50k+5k (sp=8) target:

  * seq_len_per_device  : per-device chunk rows, 512..1024 step 32
  * q_chunk_size        : {32, 64, 96, 128}
  * k_chunk_size        : from 128, step 32, ascending; the k-loop breaks on the
                          first OOM / failure (or once k_chunk covers the whole
                          per-device K shard, beyond which nothing changes).

Convention (matches how the user refers to sizes): everything is quoted in the
sp=8 deployment, so chunk_size(sp8) = seq_len_per_device * 8. This QuietBox is
sp=4, so the *actual* on-device chunk is seq_len_per_device * sp_size (= half of
the quoted sp=8 value). The target prefix is round(51200 / chunk_sp8) chunks,
e.g. a 4.5k chunk reports the 49.5k+4.5k point.

Two tests live here:
  * test_chunked_single_shot  — env-driven, profiled in a subprocess by the
                                orchestrator. Runs exactly ONE SDPA call: the
                                target chunk against a K/V cache pre-grown to the
                                full prefix (no replay of earlier chunks).
  * test_kimi_chunked_perf_sweep — the orchestrator. Loops the grid, profiles
                                each config via tracy, computes math util, and
                                rewrites a live markdown table after every run.

Run the orchestrator (it holds the device lock for the whole sweep):
  scripts/run_safe_pytest.sh tests/nightly/blackhole/sdpa/test_kimi_chunked_perf_sweep.py::test_kimi_chunked_perf_sweep
"""
import os
import math
import re
import time
import traceback

import pytest
from loguru import logger

from tests.nightly.sdpa_perf_utils import ARCH_CONSTANTS
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    MESH_CONFIG,
    CHUNKED_PREFILL_MODEL_CONFIGS,
    run_ring_joint_sdpa_chunked,
)

# 50k+5k (sp=8): 5k == 5120 == 640*8, so "50k" anchors to 10*5120 == 51200.
TARGET_PREFIX_SP8 = 51200
SP8 = 8  # the deployment the user quotes sizes in

MODEL_NAME = "kimi50k"

# Sweep grid
PER_DEVICE_MIN = 512
PER_DEVICE_MAX = 1024
PER_DEVICE_STEP = 32
Q_CHUNK_SIZES = [32, 64, 96, 128]
K_CHUNK_START = 128
K_CHUNK_STEP = 32

# Round sp=8 chunk sizes (4k..8k -> per_device 512..1024 step 128); run these first
# so the headline 50k+5k (per_device=640) point and the round sizes land early.
_PRIORITY_PER_DEVICE = [640, 512, 768, 896, 1024]

RESULTS_MD = os.path.join(os.environ.get("TT_METAL_HOME", os.getcwd()), "kimi_50k_chunked_perf_sweep.md")

INNER_NODE = "tests/nightly/blackhole/sdpa/test_kimi_chunked_perf_sweep.py::test_chunked_single_shot"
SUBDIR = "ttnn_kimi_chunked_perf_sweep"


def _n_prefix_for(chunk_sp8: int) -> int:
    """Number of full prefix chunks that lands closest to the 50k(sp8) target."""
    target = int(os.environ.get("KIMI_SWEEP_TARGET_PREFIX_SP8", TARGET_PREFIX_SP8))
    return max(1, round(target / chunk_sp8))


def _fmt_k(tokens_sp8: int) -> str:
    """Format an sp=8 token count in k-units, e.g. 49152 -> '48k', 4608 -> '4.5k'."""
    v = tokens_sp8 / 1024.0
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{s}k"


# Emitted once per program build by ring_joint_sdpa_program_factory.cpp (log_info).
SUBBLOCK_RE = re.compile(
    r"RJSDPA_SUBBLOCK q_chunk=(\d+) k_chunk=(\d+) "
    r"qk_out_subblock\(h x w\)=(\d+)x(\d+) av_out_subblock\(h x w\)=(\d+)x(\d+) dst=(\d+)"
)


def _logsize(path):
    try:
        return os.path.getsize(path) if path else 0
    except OSError:
        return 0


def _capture_subblock(path, start, q_chunk, k_chunk):
    """Scan the slice of the sweep log written by the just-finished subprocess for the
    RJSDPA_SUBBLOCK line; prefer the one whose q/k match this config."""
    empty = {"qk_sb": None, "av_sb": None, "dst": None}
    if not path:
        return empty
    try:
        with open(path, "r", errors="replace") as f:
            f.seek(start)
            blob = f.read()
    except OSError:
        return empty
    matched = last = None
    for m in SUBBLOCK_RE.finditer(blob):
        last = m
        if int(m.group(1)) == q_chunk and int(m.group(2)) == k_chunk:
            matched = m
    m = matched or last
    if not m:
        return empty
    return {
        "qk_sb": f"{m.group(3)}x{m.group(4)}",
        "av_sb": f"{m.group(5)}x{m.group(6)}",
        "dst": int(m.group(7)),
    }


# ---------------------------------------------------------------------------
# Inner test: one profiled SDPA call at the target chunk. Env-driven so the
# orchestrator can vary (per_device, q, k, n_prefix) without re-parametrizing.
# ---------------------------------------------------------------------------
@pytest.mark.timeout(900)
def test_chunked_single_shot():
    if os.environ.get("KIMI_SWEEP_ACTIVE") != "1":
        pytest.skip("inner single-shot test — driven by test_kimi_chunked_perf_sweep via env vars")

    mesh_config = MESH_CONFIG
    sp = mesh_config.sp_size

    per_device = int(os.environ["KIMI_SWEEP_PER_DEVICE"])
    q_chunk = int(os.environ["KIMI_SWEEP_Q"])
    k_chunk = int(os.environ["KIMI_SWEEP_K"])
    n_prefix = int(os.environ["KIMI_SWEEP_NPREFIX"])

    chunk_size = per_device * sp  # on-device (sp=4) global chunk
    total_seq = (n_prefix + 1) * chunk_size  # prefix chunks + the target chunk

    model = CHUNKED_PREFILL_MODEL_CONFIGS[MODEL_NAME]

    run_ring_joint_sdpa_chunked(
        mesh_config,
        model,
        chunk_size=chunk_size,
        total_seq=total_seq,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        do_check=False,
        only_chunk=n_prefix,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def _per_device_order():
    override = os.environ.get("KIMI_SWEEP_PER_DEVICES")
    if override:
        return [int(x) for x in override.split(",") if x.strip()]
    allv = list(range(PER_DEVICE_MIN, PER_DEVICE_MAX + 1, PER_DEVICE_STEP))
    head = [p for p in _PRIORITY_PER_DEVICE if p in allv]
    tail = [p for p in allv if p not in head]
    return head + tail


def _q_chunk_sizes():
    override = os.environ.get("KIMI_SWEEP_QS")
    if override:
        return [int(x) for x in override.split(",") if x.strip()]
    return Q_CHUNK_SIZES


def _measure(model, mesh_config, per_device, q_chunk, k_chunk, n_prefix):
    """Profile one config; return a result dict (status SUCCESS/FAIL)."""
    from tracy.process_model_log import run_device_profiler
    from tests.nightly.sdpa_perf_utils import post_process_ops_log

    sp = mesh_config.sp_size
    chunk_size = per_device * sp
    total_seq = (n_prefix + 1) * chunk_size
    prefix_k_global = n_prefix * chunk_size  # on-device global prefix rows

    os.environ["KIMI_SWEEP_ACTIVE"] = "1"
    os.environ["KIMI_SWEEP_PER_DEVICE"] = str(per_device)
    os.environ["KIMI_SWEEP_Q"] = str(q_chunk)
    os.environ["KIMI_SWEEP_K"] = str(k_chunk)
    os.environ["KIMI_SWEEP_NPREFIX"] = str(n_prefix)

    command = f"pytest {INNER_NODE}"
    float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]", "PM FPU UTIL (%)"]
    cols = ["ATTRIBUTES"]

    run_device_profiler(command, SUBDIR, device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log(
        SUBDIR,
        float_columns=float_cols,
        columns=cols,
        op_name="RingJointSDPADeviceOperation",
        sum_vals=False,
        has_signposts=False,
    )

    durations = r["DEVICE KERNEL DURATION [ns]"].tolist()
    core_counts = r["CORE COUNT"].tolist()
    if len(durations) == 0:
        raise RuntimeError("profiler returned no RingJointSDPA ops (inner test skipped or crashed)")

    # One SDPA call -> one entry per device. Critical path = slowest device.
    dur_ns = max(durations)
    ccount = max(int(c) for c in core_counts)
    effective_cores = (ccount // mesh_config.grid_rows) * mesh_config.grid_rows

    q_per_dev = chunk_size // sp  # == per_device
    # model.nhq is PER RING and heads-per-ring == heads-per-device (heads shard across
    # tp_axis), matching test_ring_joint_attention_create_chunked_perf_table on main.
    nh_per_dev = model.nhq
    d_q, d_v = model.d_q, model.d_v

    # Rectangle (Q_chunk vs prefix, non-causal) + triangle (Q_chunk vs current chunk, causal).
    rect_flops = 2 * q_per_dev * prefix_k_global * (d_q + d_v) * nh_per_dev
    tri_flops = q_per_dev * chunk_size * (d_q + d_v) * nh_per_dev
    chunk_flops = rect_flops + tri_flops

    constants = ARCH_CONSTANTS["blackhole"]
    cycles = dur_ns * constants["clock_ghz"]
    theoretical = effective_cores * cycles * constants["mm_flops_per_cycle_per_core"]
    util = (chunk_flops / theoretical) * 100 if theoretical > 0 else 0.0

    fpu = r.get("PM FPU UTIL (%)", [])
    fpu_min = float(min(fpu)) if len(fpu) > 0 else 0.0
    fpu_max = float(max(fpu)) if len(fpu) > 0 else 0.0

    return {
        "status": "OK",
        "duration_ms": dur_ns / 1e6,
        "cores": effective_cores,
        "util": util,
        "fpu_min": fpu_min,
        "fpu_max": fpu_max,
        "chunk_flops": chunk_flops,
    }


def _write_md(rows, started_at, done, total_planned, note=""):
    sp = MESH_CONFIG.sp_size
    lines = []
    lines.append("# Kimi-50k chunked-prefill perf sweep (ring joint SDPA)")
    lines.append("")
    lines.append(
        f"Live results — updated after every config. Sizes in **sp=8** units "
        f"(chunk_sp8 = seq_len_per_device × 8). This box is **sp={sp}**, so the "
        f"on-device chunk is half the quoted value."
    )
    lines.append("")
    lines.append(
        f"Model `{MODEL_NAME}`: nhq={CHUNKED_PREFILL_MODEL_CONFIGS[MODEL_NAME].nhq}, "
        f"d_q={CHUNKED_PREFILL_MODEL_CONFIGS[MODEL_NAME].d_q}, "
        f"d_v={CHUNKED_PREFILL_MODEL_CONFIGS[MODEL_NAME].d_v}, causal, bf16 Q / bf8 KV."
    )
    elapsed = time.time() - started_at
    per = (elapsed / done) if done else 0.0
    lines.append(f"Progress: **{done}** configs measured in {elapsed/60:.1f} min " f"(~{per:.0f}s/config). {note}")
    lines.append("")
    lines.append(
        "| seq/dev | chunk(sp8) | target (sp8) | q_chunk | k_chunk | n_prefix | "
        "dur (ms) | cores | FPU% | qk subblk | av subblk | **math util** | status |"
    )
    lines.append(
        "|--------:|-----------:|:-------------|--------:|--------:|---------:|"
        "---------:|------:|-----:|:---------:|:---------:|--------------:|:-------|"
    )
    for row in rows:
        qk_sb = row.get("qk_sb") or "-"
        av_sb = row.get("av_sb") or "-"
        if row["status"] == "OK":
            fpu = f"{row['fpu_min']:.0f}-{row['fpu_max']:.0f}"
            lines.append(
                f"| {row['per_device']} | {row['chunk_sp8_lbl']} | {row['target_lbl']} | "
                f"{row['q']} | {row['k']} | {row['n_prefix']} | {row['duration_ms']:.3f} | "
                f"{row['cores']} | {fpu} | {qk_sb} | {av_sb} | **{row['util']:.1f}%** | OK |"
            )
        else:
            lines.append(
                f"| {row['per_device']} | {row['chunk_sp8_lbl']} | {row['target_lbl']} | "
                f"{row['q']} | {row['k']} | {row['n_prefix']} | - | - | - | {qk_sb} | {av_sb} | - | "
                f"{row['status']} |"
            )
    lines.append("")
    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(lines))


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Perf sweep — run locally")
@pytest.mark.timeout(0)  # may run for hours; rely on per-config inner timeouts
def test_kimi_chunked_perf_sweep():
    mesh_config = MESH_CONFIG
    sp = mesh_config.sp_size
    if sp < 2:
        pytest.skip(f"Ring joint chunked prefill requires >=2 devices, got sp={sp}")

    model = CHUNKED_PREFILL_MODEL_CONFIGS[MODEL_NAME]

    per_devices = _per_device_order()
    q_chunk_sizes = _q_chunk_sizes()
    # Cap k steps per (per_device,q) pair (smoke testing); 0 = unlimited (until OOM/shard).
    k_count_cap = int(os.environ.get("KIMI_SWEEP_KCOUNT", "0"))
    # Sweep stdout/stderr log; the inner subprocess's RJSDPA_SUBBLOCK line lands here.
    logfile = os.environ.get("KIMI_SWEEP_LOGFILE")
    started_at = time.time()
    rows = []
    done = 0
    consecutive_dry_pairs = 0  # (per_device,q) pairs that produced zero OK results

    logger.info(
        f"Kimi chunked perf sweep: per_device {per_devices}, q {Q_CHUNK_SIZES}, "
        f"k from {K_CHUNK_START} step {K_CHUNK_STEP}; results -> {RESULTS_MD}"
    )
    _write_md(rows, started_at, done, 0, note="starting…")

    for per_device in per_devices:
        chunk_size = per_device * sp
        chunk_sp8 = per_device * SP8
        n_prefix = _n_prefix_for(chunk_sp8)
        prefix_sp8 = n_prefix * chunk_sp8
        chunk_sp8_lbl = _fmt_k(chunk_sp8)
        target_lbl = f"{_fmt_k(prefix_sp8)}+{_fmt_k(chunk_sp8)}"

        # Largest useful k_chunk: once it covers the whole per-device K shard the
        # work is a single k-chunk and larger values only grow CBs (OOM, no change).
        n_local_kv = (n_prefix + 1) * per_device  # per-device K/V rows at target

        for q_chunk in q_chunk_sizes:
            pair_ok = 0
            k_chunk = K_CHUNK_START
            k_steps = 0
            while True:
                base = {
                    "per_device": per_device,
                    "chunk_sp8_lbl": chunk_sp8_lbl,
                    "target_lbl": target_lbl,
                    "q": q_chunk,
                    "k": k_chunk,
                    "n_prefix": n_prefix,
                }
                t0 = time.time()
                before = _logsize(logfile)
                try:
                    res = _measure(model, mesh_config, per_device, q_chunk, k_chunk, n_prefix)
                    base.update(res)
                    base.update(_capture_subblock(logfile, before, q_chunk, k_chunk))
                    pair_ok += 1
                    done += 1
                    logger.info(
                        f"[OK] seq/dev={per_device} ({chunk_sp8_lbl}) q={q_chunk} k={k_chunk} "
                        f"n_prefix={n_prefix} -> util={res['util']:.1f}% "
                        f"dur={res['duration_ms']:.3f}ms cores={res['cores']} "
                        f"qk_sb={base.get('qk_sb')} av_sb={base.get('av_sb')} "
                        f"({time.time()-t0:.0f}s)"
                    )
                    rows.append(base)
                    _write_md(rows, started_at, done, 0)
                except Exception as exc:  # noqa: BLE001 — OOM/hang/etc. expected at the memory wall
                    msg = f"{type(exc).__name__}: {exc}".splitlines()[0][:120]
                    base["status"] = f"FAIL ({msg})"
                    logger.warning(
                        f"[FAIL] seq/dev={per_device} q={q_chunk} k={k_chunk}: {msg} "
                        f"-> closing k-loop for this (seq/dev, q)"
                    )
                    logger.debug(traceback.format_exc())
                    rows.append(base)
                    _write_md(rows, started_at, done, 0)
                    break  # OOM / failure: stop increasing k for this (per_device, q)

                k_steps += 1
                if k_count_cap and k_steps >= k_count_cap:
                    break
                k_chunk += K_CHUNK_STEP
                if k_chunk > n_local_kv:
                    # Covered the whole shard; nothing new beyond this.
                    break

            if pair_ok == 0:
                consecutive_dry_pairs += 1
                # If many (per_device,q) pairs in a row yield nothing, the device is
                # likely wedged — bail rather than churn for hours on a dead device.
                if consecutive_dry_pairs >= 6:
                    _write_md(
                        rows,
                        started_at,
                        done,
                        0,
                        note="ABORTED: 6 consecutive dry (per_device,q) pairs — suspected device hang.",
                    )
                    pytest.fail(
                        "Aborting sweep: 6 consecutive (per_device, q) pairs produced no successful "
                        "measurement (suspected device hang). See markdown for partial results."
                    )
            else:
                consecutive_dry_pairs = 0

    _write_md(rows, started_at, done, 0, note="DONE.")
    logger.info(f"Sweep complete: {done} configs measured. Results in {RESULTS_MD}")
