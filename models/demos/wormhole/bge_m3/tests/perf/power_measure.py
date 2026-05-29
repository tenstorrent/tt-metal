# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""BGE-M3 DP power measurement — sustained forward loop + IPMI power correlation.

Why a separate file: perf.py mixes many concerns. This file does ONE thing:
run the forward pass in a tight loop for a fixed wall-clock duration, logging
each iteration's Unix start/stop, while a background IPMI poller logs system
power with Unix timestamps. We then intersect by time to get average power
DURING compute.

Key idea (sampling mismatch):
  - one forward is ~6 ms (B1/DP32) or ~70 ms (B32/DP32)
  - IPMI samples at ~0.1-0.3 Hz (reads take 3-10 s)
  => you cannot measure a single iteration. Instead run a SUSTAINED loop long
     enough (default 60 s) for IPMI to collect 15-30 samples inside the window,
     then average only the samples whose timestamp falls within [loop_start,
     loop_end]. A short warmup/cooldown guard band is trimmed so we don't
     average ramp-up/idle power.

Output (per config, under MEASURE_OUT_DIR):
  iters_<cfg>.csv   one row per forward: iter, t_start_epoch, t_stop_epoch, dur_ms
  power_<cfg>.csv   IPMI poller output (epoch, power sensors)
  summary_<cfg>.txt human-readable correlation result

Configs (parametrized): dp32_b1, dp32_b32.

Run (32-chip Galaxy):
  cd /home/tt-admin/gtobar && source local_env.sh && cd tt-metal
  python -m pytest \
    models/demos/wormhole/bge_m3/tests/perf/power_measure.py -k dp32_b1 -sv

Env knobs:
  MEASURE_SECONDS   compute-loop duration per config (default 60)
  MEASURE_GUARD_S   seconds trimmed from each end of the window (default 5)
  MEASURE_OUT_DIR   output dir (default /tmp/bge_power)
  MEASURE_IPMI_INTERVAL  poller target interval seconds (default 1.0)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest
from loguru import logger

import ttnn
from models.demos.utils.common_demo_utils import get_mesh_mappers

# Reuse the input/D2H helpers already written + tested in perf.py.
from models.demos.wormhole.bge_m3.tests.perf.perf import (
    SEQ_LEN,
    _allocate_d2h_stack,
    _d2h_step_optimized,
    allocate_dp_device_tensors,
    prepare_inputs,
    to_dp_host_tensors,
)
from models.demos.wormhole.bge_m3.tt.common import create_tt_model
from models.demos.wormhole.bge_m3.tt.model_config import determine_device_name

MEASURE_SECONDS = float(os.environ.get("MEASURE_SECONDS", "60"))
MEASURE_GUARD_S = float(os.environ.get("MEASURE_GUARD_S", "5"))
# Idle baseline: device built + synced but NOT running, captured before the
# active loop so we can isolate dynamic compute power from the static floor.
MEASURE_IDLE_S = float(os.environ.get("MEASURE_IDLE_S", "100"))
# Default output under the gtobar workspace (healthy root btrfs), NOT /tmp and
# never on the suspended ZFS pool. Override with MEASURE_OUT_DIR.
OUT_DIR = os.environ.get("MEASURE_OUT_DIR", os.path.expanduser("~/gtobar/bge_power"))
IPMI_INTERVAL = os.environ.get("MEASURE_IPMI_INTERVAL", "1.0")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ──────────────────────────────────────────────────────────────────────────────
# IPMI poller process management
# ──────────────────────────────────────────────────────────────────────────────


def _start_power_poller(power_csv: str) -> subprocess.Popen:
    """Launch ipmi_power_poller.py as a separate process writing power_csv."""
    poller = os.path.join(_THIS_DIR, "ipmi_power_poller.py")
    proc = subprocess.Popen(
        [sys.executable, poller, power_csv, IPMI_INTERVAL],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return proc


def _stop_power_poller(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ──────────────────────────────────────────────────────────────────────────────
# Correlation: average power over samples inside [window_start, window_end]
# ──────────────────────────────────────────────────────────────────────────────


def _correlate(power_csv: str, window_start: float, window_end: float):
    """Return dict {sensor: avg_W} over IPMI samples whose epoch is in-window,
    plus the count of samples used. Trims nothing here -- caller passes the
    already-guard-trimmed window."""
    import csv

    # Count samples PER SENSOR, only for valid (non-NA) values. A failed IPMI
    # read writes "NA" for every sensor that cycle; those rows must not inflate
    # the divisor (the original bug averaged active power down toward idle).
    sums = {}
    counts = {}
    rows_total = 0
    # A partial/concurrent write can leave NUL bytes in the CSV; the csv module
    # raises "line contains NUL" on those. Strip NULs and drop blank lines so a
    # few corrupt samples never lose the whole run's report.
    with open(power_csv, newline="") as f:
        clean = (line.replace("\x00", "") for line in f if line.strip("\x00\r\n "))
        reader = csv.DictReader(clean)
        for row in reader:
            rows_total += 1
            try:
                ep = float(row["epoch"])
            except (KeyError, ValueError):
                continue
            if ep < window_start or ep > window_end:
                continue
            for k, v in row.items():
                if not k.endswith("_W"):
                    continue
                try:
                    fv = float(v)
                except ValueError:
                    continue  # NA -> skip this sensor for this row, do not count
                sums[k] = sums.get(k, 0.0) + fv
                counts[k] = counts.get(k, 0) + 1
    avgs = {k: (sums[k] / counts[k]) for k in sums if counts.get(k, 0) > 0}
    n = max(counts.values()) if counts else 0
    return avgs, n, rows_total


# ──────────────────────────────────────────────────────────────────────────────
# Test
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mesh_device, per_device_batch",
    [
        ((4, 8), 1),
        ((4, 8), 32),
    ],
    indirect=["mesh_device"],
    ids=["dp32_b1", "dp32_b32"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50_000_000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.timeout(0)  # phases are bounded by MEASURE_* env knobs, not pytest's 300s
def test_power_measure_dp(mesh_device, per_device_batch):
    """Sustained forward loop with IPMI power correlation.

    Runs forward (+H2D+D2H) in a tight loop for MEASURE_SECONDS, logging each
    iteration's Unix start/stop. A background IPMI poller logs power. We then
    average power over samples inside the guard-trimmed compute window.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    dtype = ttnn.bfloat8_b
    device_name = determine_device_name(mesh_device)[0]
    num_devices = mesh_device.get_num_devices()
    global_batch = per_device_batch * num_devices
    mask_dtype = dtype if per_device_batch in (1, 32) else ttnn.bfloat16
    hidden = 1024
    cfg = f"dp{num_devices}_b{per_device_batch}"

    iters_csv = os.path.join(OUT_DIR, f"iters_{cfg}.csv")
    power_csv = os.path.join(OUT_DIR, f"power_{cfg}.csv")
    summary_txt = os.path.join(OUT_DIR, f"summary_{cfg}.txt")

    inputs_mesh_mapper, _, _ = get_mesh_mappers(mesh_device)
    assert inputs_mesh_mapper is not None, f"need multi-device mesh; got {num_devices}"

    logger.info(
        f"[{cfg}] Building model: per_device_batch={per_device_batch} "
        f"global_batch={global_batch} num_devices={num_devices} {device_name}"
    )
    model_args, model, _ = create_tt_model(
        mesh_device=mesh_device,
        max_batch_size=per_device_batch,
        max_seq_len=SEQ_LEN,
        dtype=dtype,
    )

    host_inputs = prepare_inputs(model_args.tokenizer, global_batch, SEQ_LEN, model_args.pad_token_id)
    host_tensors = to_dp_host_tensors(host_inputs, mask_dtype, inputs_mesh_mapper)
    device_tensors = allocate_dp_device_tensors(host_inputs, mesh_device, mask_dtype, inputs_mesh_mapper)

    logger.info(f"[{cfg}] Compiling + capturing trace...")
    out = model.forward(**device_tensors)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)
    trace_out = model.capture_trace(**device_tensors, mesh_device=mesh_device, cq_id=0)
    for _ in range(3):
        model.execute_trace(blocking=True)

    dram_staging, dest_torch = _allocate_d2h_stack(trace_out, mesh_device, global_batch, hidden)
    h2d_keys = ("input_ids", "attention_mask", "token_type_ids", "position_ids")

    # Warm the full pipeline a few times before measuring.
    for _ in range(3):
        for k in h2d_keys:
            ttnn.copy_host_to_device_tensor(host_tensors[k], device_tensors[k])
        ttnn.synchronize_device(mesh_device)
        model.execute_trace(blocking=True)
        _d2h_step_optimized(trace_out, mesh_device, dram_staging, dest_torch)

    # ── Start the IPMI poller ───────────────────────────────────────────────
    logger.info(f"[{cfg}] Starting IPMI poller -> {power_csv}")
    poller = _start_power_poller(power_csv)
    # Give the poller a moment to take its first sample.
    time.sleep(2.0)

    # ── Idle baseline phase: device built + synced but NOT executing ────────
    idle_start = time.time()
    if MEASURE_IDLE_S > 0:
        logger.info(f"[{cfg}] Idle baseline for {MEASURE_IDLE_S:.0f}s " f"(device built + synced, no compute)")
        ttnn.synchronize_device(mesh_device)
        time.sleep(MEASURE_IDLE_S)
    idle_end = time.time()

    # ── Active phase: sustained compute loop ────────────────────────────────
    logger.info(
        f"[{cfg}] Running sustained loop for {MEASURE_SECONDS:.0f}s " f"(guard {MEASURE_GUARD_S:.0f}s each end)"
    )
    iter_rows = []  # (iter_idx, t_start, t_stop, dur_ms)
    loop_start = time.time()
    i = 0
    while True:
        t_start = time.time()
        if t_start - loop_start >= MEASURE_SECONDS:
            break
        # Full H2D -> Forward -> D2H, all timed as one iteration.
        for k in h2d_keys:
            ttnn.copy_host_to_device_tensor(host_tensors[k], device_tensors[k])
        ttnn.synchronize_device(mesh_device)
        model.execute_trace(blocking=True)
        _d2h_step_optimized(trace_out, mesh_device, dram_staging, dest_torch)
        t_stop = time.time()
        iter_rows.append((i, t_start, t_stop, (t_stop - t_start) * 1000.0))
        i += 1
    loop_end = time.time()

    # Let the poller capture a couple more samples, then stop it.
    time.sleep(2.0)
    _stop_power_poller(poller)
    model.release_trace()

    # ── Write per-iteration CSV ─────────────────────────────────────────────
    with open(iters_csv, "w") as f:
        f.write("iter,t_start_epoch,t_stop_epoch,dur_ms\n")
        for idx, ts, te, dur in iter_rows:
            f.write(f"{idx},{ts:.6f},{te:.6f},{dur:.4f}\n")

    # ── Latency stats (from Unix start/stop) ────────────────────────────────
    durs = sorted(r[3] for r in iter_rows)
    n_iter = len(durs)
    assert n_iter > 0, "no iterations ran"
    avg_ms = sum(durs) / n_iter
    p50_ms = durs[n_iter // 2]
    min_ms = durs[0]
    max_ms = durs[-1]
    thr = global_batch / (avg_ms / 1000.0)

    # ── Power correlation over guard-trimmed windows ────────────────────────
    # Active window
    win_start = loop_start + MEASURE_GUARD_S
    win_end = loop_end - MEASURE_GUARD_S
    avgs, n_samples, rows_total = _correlate(power_csv, win_start, win_end)
    # Model-loaded window (skip a guard band each end so we don't catch build/settle)
    idle_avgs, idle_n = {}, 0
    if MEASURE_IDLE_S > 2 * MEASURE_GUARD_S:
        idle_avgs, idle_n, _ = _correlate(power_csv, idle_start + MEASURE_GUARD_S, idle_end - MEASURE_GUARD_S)

    # ── Report ──────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 78)
    lines.append(f"  BGE-M3 POWER MEASURE  |  {cfg}  S={SEQ_LEN}  {device_name}")
    lines.append(f"  global_batch={global_batch}  num_devices={num_devices}")
    lines.append("=" * 78)
    lines.append(f"  Idle baseline  : {MEASURE_IDLE_S:.0f}s (device built+synced, no compute)")
    lines.append(
        f"  Compute window : {MEASURE_SECONDS:.0f}s requested, "
        f"{loop_end - loop_start:.1f}s actual  ({n_iter} iters)"
    )
    lines.append(f"  Guard band     : {MEASURE_GUARD_S:.0f}s each end")
    lines.append("-" * 78)
    lines.append("  Per-iteration latency (Unix start/stop):")
    lines.append(f"    avg={avg_ms:.3f} ms  p50={p50_ms:.3f} ms  " f"min={min_ms:.3f} ms  max={max_ms:.3f} ms")
    lines.append(f"    throughput={thr:.1f} emb/s  ({global_batch * SEQ_LEN / (avg_ms / 1000.0):.0f} tok/s)")
    lines.append("-" * 78)
    lines.append(f"  IPMI power (avg over {n_samples} in-window samples " f"of {rows_total} total):")
    if n_samples == 0:
        lines.append("    !! NO IPMI samples landed inside the window.")
        lines.append("    !! Increase MEASURE_SECONDS (IPMI is ~0.1-0.3 Hz).")
    else:
        for k in (
            "Power_Total_W",
            "Power_UBB0_W",
            "Power_UBB1_W",
            "Power_UBB2_W",
            "Power_UBB3_W",
            "Power_CPU_W",
            "Power_Memory_W",
            "Power_FAN_W",
        ):
            if k in avgs:
                lines.append(f"    {k:<16} {avgs[k]:8.1f} W")
        ubb = sum(avgs.get(f"Power_UBB{j}_W", 0.0) for j in range(4))
        if ubb > 0:
            lines.append(f"    {'UBB_total (chips)':<16} {ubb:8.1f} W  " f"({ubb / num_devices:.1f} W/chip)")
        if "Power_Total_W" in avgs and thr > 0:
            lines.append(f"    {'efficiency':<16} {avgs['Power_Total_W'] / thr:8.4f} W per emb/s")
    # ── Idle baseline + dynamic (active - idle) delta ───────────────────────
    lines.append("-" * 78)
    lines.append(f"  IDLE baseline power (avg over {idle_n} in-window samples):")
    if idle_n == 0:
        lines.append("    !! No idle samples (set MEASURE_IDLE_S >= ~30s for IPMI).")
    else:
        for k in ("Power_Total_W", "Power_CPU_W", "Power_FAN_W"):
            if k in idle_avgs:
                lines.append(f"    {k:<16} {idle_avgs[k]:8.1f} W")
        idle_ubb = sum(idle_avgs.get(f"Power_UBB{j}_W", 0.0) for j in range(4))
        if idle_ubb > 0:
            lines.append(f"    {'UBB_total (chips)':<16} {idle_ubb:8.1f} W  " f"({idle_ubb / num_devices:.1f} W/chip)")
    if idle_n and n_samples and "Power_Total_W" in avgs and "Power_Total_W" in idle_avgs:
        d_total = avgs["Power_Total_W"] - idle_avgs["Power_Total_W"]
        a_ubb = sum(avgs.get(f"Power_UBB{j}_W", 0.0) for j in range(4))
        i_ubb = sum(idle_avgs.get(f"Power_UBB{j}_W", 0.0) for j in range(4))
        d_ubb = a_ubb - i_ubb
        lines.append("-" * 78)
        lines.append("  DYNAMIC compute power (active - idle):")
        lines.append(
            f"    System total    {d_total:8.1f} W  "
            f"(idle {idle_avgs['Power_Total_W']:.0f} -> active {avgs['Power_Total_W']:.0f})"
        )
        if i_ubb > 0:
            lines.append(f"    Chips (UBB)     {d_ubb:8.1f} W  " f"({d_ubb / num_devices:.1f} W/chip dynamic)")
        if thr > 0:
            lines.append(f"    dynamic eff     {d_total / thr:8.4f} W per emb/s")
    lines.append("=" * 78)

    report = "\n".join(lines)
    logger.info("\n" + report)
    with open(summary_txt, "w") as f:
        f.write(report + "\n")
    logger.info(f"[{cfg}] Wrote: {iters_csv}, {power_csv}, {summary_txt}")
