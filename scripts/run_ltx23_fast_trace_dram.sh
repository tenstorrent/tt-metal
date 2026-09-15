#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SAMPLE_INTERVAL_SECONDS="${SAMPLE_INTERVAL_SECONDS:-0.20}"
PYTEST_TIMEOUT="${PYTEST_TIMEOUT:-3600}"
MESH_FILTER="${MESH_FILTER:-4x8sp1tp0nl2_ring_is_fsdp0}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/generated/ltx23_dram/$RUN_ID}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

export LTX_DRAM_OUTPUT_DIR="$OUTPUT_DIR"
export LTX_DRAM_SAMPLE_INTERVAL="$SAMPLE_INTERVAL_SECONDS"
export LTX_TRACED=1
export RUN_WARMUP=1
export NO_PROMPT=1
export RUN_VBENCH=0
export RUN_CLIP=0
export TT_METAL_SHM_TRACKING_DISABLED="${TT_METAL_SHM_TRACKING_DISABLED:-1}"
export TT_METAL_INSPECTOR="${TT_METAL_INSPECTOR:-0}"
export TT_METAL_LOGS_PATH="${TT_METAL_LOGS_PATH:-/tmp/tt-logs}"
export LTX_VOC_TRACE="${LTX_VOC_TRACE:-0}"
export LTX_BWE_TRACE="${LTX_BWE_TRACE:-0}"

PLUGIN_PATH="$OUTPUT_DIR/ltx_dram_plugin.py"
cat >"$PLUGIN_PATH" <<'PY'
import csv
import datetime
import os
import pathlib
import sys
import threading
import time

import pytest
import ttnn


class DramSampler:
    def __init__(self, mesh_device):
        self.output_dir = pathlib.Path(os.environ["LTX_DRAM_OUTPUT_DIR"])
        self.interval = float(os.environ.get("LTX_DRAM_SAMPLE_INTERVAL", "0.20"))
        if self.interval <= 0:
            raise ValueError("LTX_DRAM_SAMPLE_INTERVAL must be greater than zero")

        devices = mesh_device.get_devices() if hasattr(mesh_device, "get_devices") else [mesh_device]
        self.devices = sorted(devices, key=lambda device: int(device.id()))
        if not self.devices:
            raise RuntimeError("No devices found in mesh_device")

        self.stop_event = threading.Event()
        self.started_at = time.monotonic()
        self.samples = {}
        self.baseline = {}
        self.peaks = {}
        self.dram_peaks = {}
        self.trace_peaks = {}
        self.errors = []
        self.csv_file = (self.output_dir / "dram_samples.csv").open("w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(
            [
                "timestamp_utc",
                "elapsed_seconds",
                "device_id",
                "dram_used_bytes",
                "dram_free_bytes",
                "dram_total_bytes",
                "trace_used_bytes",
                "trace_free_bytes",
                "trace_total_bytes",
                "allocator_used_bytes",
                "allocator_free_bytes",
                "allocator_total_bytes",
                "allocator_percent_used",
            ]
        )
        self.thread = threading.Thread(target=self._run, name="ltx-dram-sampler", daemon=True)

    @staticmethod
    def _view_bytes(device, buffer_type):
        view = ttnn.get_memory_view(device, buffer_type)
        banks = int(view.num_banks)
        return (
            banks * int(view.total_bytes_allocated_per_bank),
            banks * int(view.total_bytes_free_per_bank),
            banks * int(view.total_bytes_per_bank),
        )

    def sample(self):
        elapsed = time.monotonic() - self.started_at
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        for device in self.devices:
            device_id = int(device.id())
            try:
                dram_used, dram_free, dram_total = self._view_bytes(device, ttnn.BufferType.DRAM)
                trace_used, trace_free, trace_total = self._view_bytes(device, ttnn.BufferType.TRACE)
                used = dram_used + trace_used
                free = dram_free + trace_free
                total = dram_total + trace_total
                percent = (100.0 * used / total) if total else 0.0
                row = {
                    "timestamp_utc": timestamp,
                    "elapsed_seconds": elapsed,
                    "device_id": device_id,
                    "dram_used_bytes": dram_used,
                    "dram_free_bytes": dram_free,
                    "dram_total_bytes": dram_total,
                    "trace_used_bytes": trace_used,
                    "trace_free_bytes": trace_free,
                    "trace_total_bytes": trace_total,
                    "allocator_used_bytes": used,
                    "allocator_free_bytes": free,
                    "allocator_total_bytes": total,
                    "allocator_percent_used": percent,
                }
                if device_id not in self.baseline:
                    self.baseline[device_id] = row.copy()
                if device_id not in self.peaks or used > self.peaks[device_id]["allocator_used_bytes"]:
                    self.peaks[device_id] = row.copy()
                self.dram_peaks[device_id] = max(self.dram_peaks.get(device_id, 0), dram_used)
                self.trace_peaks[device_id] = max(self.trace_peaks.get(device_id, 0), trace_used)
                self.samples[device_id] = self.samples.get(device_id, 0) + 1
                self.writer.writerow(
                    [
                        timestamp,
                        f"{elapsed:.6f}",
                        device_id,
                        dram_used,
                        dram_free,
                        dram_total,
                        trace_used,
                        trace_free,
                        trace_total,
                        used,
                        free,
                        total,
                        f"{percent:.6f}",
                    ]
                )
            except Exception as error:
                message = f"device {device_id} at {elapsed:.3f}s: {type(error).__name__}: {error}"
                self.errors.append(message)
                print(f"DRAM sampler warning: {message}", file=sys.stderr, flush=True)
        self.csv_file.flush()

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def start(self):
        self.sample()
        self.thread.start()
        print(
            f"DRAM sampler started for {len(self.devices)} devices "
            f"at {self.interval:.3f}s intervals; output={self.output_dir}",
            flush=True,
        )

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        self.sample()
        self.csv_file.close()
        if not self.peaks:
            raise RuntimeError("DRAM sampler did not collect any successful samples")

        summary_path = self.output_dir / "dram_peak_summary.csv"
        with summary_path.open("w", newline="") as summary_file:
            fieldnames = [
                "device_id",
                "samples",
                "dram_total_gib",
                "peak_dram_used_gib",
                "trace_total_gib",
                "peak_trace_used_gib",
                "allocator_total_gib",
                "baseline_allocator_used_gib",
                "peak_allocator_used_gib",
                "peak_above_baseline_gib",
                "minimum_allocator_free_gib",
                "peak_percent_used",
                "peak_elapsed_seconds",
            ]
            writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
            writer.writeheader()
            gib = 1024**3
            for device_id in sorted(self.peaks):
                peak = self.peaks[device_id]
                baseline = self.baseline[device_id]
                writer.writerow(
                    {
                        "device_id": device_id,
                        "samples": self.samples[device_id],
                        "dram_total_gib": f"{peak['dram_total_bytes'] / gib:.4f}",
                        "peak_dram_used_gib": f"{self.dram_peaks[device_id] / gib:.4f}",
                        "trace_total_gib": f"{peak['trace_total_bytes'] / gib:.4f}",
                        "peak_trace_used_gib": f"{self.trace_peaks[device_id] / gib:.4f}",
                        "allocator_total_gib": f"{peak['allocator_total_bytes'] / gib:.4f}",
                        "baseline_allocator_used_gib": f"{baseline['allocator_used_bytes'] / gib:.4f}",
                        "peak_allocator_used_gib": f"{peak['allocator_used_bytes'] / gib:.4f}",
                        "peak_above_baseline_gib": (
                            f"{(peak['allocator_used_bytes'] - baseline['allocator_used_bytes']) / gib:.4f}"
                        ),
                        "minimum_allocator_free_gib": f"{peak['allocator_free_bytes'] / gib:.4f}",
                        "peak_percent_used": f"{peak['allocator_percent_used']:.2f}",
                        "peak_elapsed_seconds": f"{peak['elapsed_seconds']:.3f}",
                    }
                )

        if self.errors:
            (self.output_dir / "dram_sampler_errors.log").write_text("\n".join(self.errors) + "\n")
        print(f"DRAM peak summary written to {summary_path}", flush=True)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item):
    mesh_device = item.funcargs.get("mesh_device")
    sampler = None
    if mesh_device is not None and item.name.startswith("test_pipeline_distilled"):
        sampler = DramSampler(mesh_device)
        sampler.start()
    try:
        yield
    finally:
        if sampler is not None:
            sampler.stop()
PY

python3 -m py_compile "$PLUGIN_PATH"

LOG_PATH="$OUTPUT_DIR/ltx23_fast_trace.log"
echo "Running traced LTX-2.3 Fast E2E with per-device DRAM sampling"
echo "Results directory: $OUTPUT_DIR"

set +e
PYTHONPATH="$OUTPUT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    pytest -p ltx_dram_plugin \
    models/tt_dit/tests/models/ltx/test_pipeline_ltx_distilled.py::test_pipeline_distilled \
    -k "$MESH_FILTER" -s --timeout "$PYTEST_TIMEOUT" "$@" 2>&1 | tee "$LOG_PATH"
status=${PIPESTATUS[0]}
set -e

echo
if [[ -f "$OUTPUT_DIR/dram_peak_summary.csv" ]]; then
    echo "Per-device sampled DRAM peaks:"
    if command -v column >/dev/null 2>&1; then
        column -s, -t <"$OUTPUT_DIR/dram_peak_summary.csv"
    else
        cat "$OUTPUT_DIR/dram_peak_summary.csv"
    fi
else
    echo "No DRAM summary was produced; inspect $LOG_PATH" >&2
fi
echo "Raw samples: $OUTPUT_DIR/dram_samples.csv"
echo "Run log:     $LOG_PATH"

exit "$status"
