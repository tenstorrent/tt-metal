# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os, sys, csv, time, re, subprocess, tempfile
from pathlib import Path

BIN = "build/test/tt_metal/tt_fabric/bench_unicast"
SRC = "0:0"
DST = "0:1"
PAGE = "2048"
ITERS = "5"
WARMUP = "1"

ARTIFACTS = Path("artifacts/prof_runs")
HOST_CSV = Path("artifacts/unicast_sweep.csv")
COMBINED_CSV = Path("artifacts/unicast_sweep_with_kernel.csv")

sizes = [65536, 131072, 262144, 524288]  # bytes
recv_cores = [(0, 0), (1, 0), (2, 0)]  # (x,y)


def latest_report_dir(artifacts: Path, run_name: str) -> Path:
    base = artifacts / "reports" / run_name
    ts_dirs = [p for p in base.iterdir() if p.is_dir()] if base.exists() else []
    if not ts_dirs:
        raise FileNotFoundError(f"No Tracy reports at {base}")
    return sorted(ts_dirs)[-1]


def read_chip_freq_mhz_from_device_log(devlog: Path) -> float:
    # first line looks like: "ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000"
    with devlog.open() as f:
        line = f.readline()
    m = re.search(r"CHIP_FREQ\[MHz\]:\s*(\d+\.?\d*)", line or "")
    return float(m.group(1)) if m else 1000.0


def extract_kernel_times(report_dir: Path):
    """
    Compute:
      - kernel_longest_ms: longest single kernel interval across all devices/cores
    """
    devlog = report_dir / "profile_log_device.csv"
    if not devlog.exists():
        return {"kernel_longest_ms": None}

    mhz_default = read_chip_freq_mhz_from_device_log(devlog)

    # profile_log_device.csv: skip the first header line, then parse rows
    with devlog.open(newline="") as f:
        f.readline()  # ARCH line
        r = csv.DictReader(f)
        # normalize keys (strip spaces)
        field = lambda name: next((k for k in r.fieldnames if k and k.strip() == name), None)

        k_slot = field("PCIe slot")
        k_cx = field("core_x")
        k_cy = field("core_y")
        k_proc = field("RISC processor type")
        k_timer = field("timer_id")
        k_cycles = field("time[cycles since reset]")
        k_type = field("type")
        k_zone = field("zone name")

        if not all([k_slot, k_cx, k_cy, k_proc, k_timer, k_cycles, k_type, k_zone]):
            return {"kernel_longest_ms": None}

        # Collect START/END pairs per (device, core, proc, timer)
        starts = {}
        intervals = []  # list of (start_ns, end_ns)
        for row in r:
            if "KERNEL" not in (row[k_zone] or ""):
                continue

            dev = int(row[k_slot])
            freq_mhz = mhz_default

            # cycles -> ns using header chip frequency
            try:
                cyc = float(row[k_cycles])
            except (TypeError, ValueError):
                continue
            t_ns = (cyc * 1000.0) / float(freq_mhz)

            key = (dev, row[k_cx], row[k_cy], row[k_proc], row[k_timer])
            etype = (row[k_type] or "").strip()
            if etype == "ZONE_START":
                starts[key] = t_ns
            elif etype == "ZONE_END" and key in starts:
                intervals.append((starts.pop(key), t_ns))

        if not intervals:
            return {"kernel_longest_ms": None}

        # --- focus on the last run cluster to avoid long bring-up intervals ---
        last_end = max(e for _, e in intervals)
        window_ns = 100_000_000  # 100 ms window near the end of the trace
        recent = [(s, e) for (s, e) in intervals if s >= (last_end - window_ns)]

        if not recent:  # fallback if windowing removed everything
            recent = intervals

        dur_ns = [e - s for (s, e) in recent if e >= s]

        if not dur_ns:
            return {"kernel_longest_ms": None}

        longest_ms = max(dur_ns) / 1e6
        return {"kernel_longest_ms": round(float(longest_ms), 6)}


def append_combined_row(host_csv_row: dict, kernel: dict):
    COMBINED_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = dict(host_csv_row)
    row.update(kernel)
    write_header = not COMBINED_CSV.exists()
    with COMBINED_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_one_under_tracy(run_name: str, cmd_argv: list[str]) -> None:
    """
    Creating a tiny temp runner script that just runs the bench command.
    """
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    runner = ARTIFACTS / f"_bench_runner_{os.getpid()}_{time.time_ns()}.py"
    runner.write_text("import subprocess, sys\n" f"subprocess.run({repr(cmd_argv)}, check=True)\n")

    env = os.environ.copy()
    env["TT_METAL_DEVICE_PROFILER"] = "1"  # enable device profiler

    tracy_cmd = [
        sys.executable,
        "-m",
        "tracy",
        "-p",
        "-r",
        "-o",
        str(ARTIFACTS),
        "-n",
        run_name,
        str(runner),
    ]
    try:
        subprocess.run(tracy_cmd, check=True, env=env)
    finally:
        try:
            runner.unlink()
        except FileNotFoundError:
            pass


def read_last_host_row(csv_path: Path) -> dict:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {csv_path}")
    return rows[-1]


def main():
    # Make sure host CSV exists & has header if empty (so bench can append cleanly)
    if not HOST_CSV.exists():
        HOST_CSV.parent.mkdir(parents=True, exist_ok=True)
        with HOST_CSV.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "mesh",
                    "src_chip",
                    "dst_chip",
                    "send_x",
                    "send_y",
                    "recv_x",
                    "recv_y",
                    "sizeB",
                    "pageB",
                    "iters",
                    "warmup",
                    "p50_ms",
                    "p95_ms",
                    "mean_GB_s",
                ]
            )

    for size in sizes:
        for rx, ry in recv_cores:
            run_name = f"unicast_size{size}_rx{rx}_{ry}"
            bench_cmd = [
                BIN,
                "--src-dev",
                SRC,
                "--dst-dev",
                DST,
                "--size",
                str(size),
                "--page",
                PAGE,
                "--send-core",
                "0,0",
                "--recv-core",
                f"{rx},{ry}",
                "--iters",
                ITERS,
                "--warmup",
                WARMUP,
                "--csv",
                str(HOST_CSV),
            ]
            print(">>", " ".join(bench_cmd))
            run_one_under_tracy(run_name, bench_cmd)

            # 1) Read the row bench appended
            host_row = read_last_host_row(HOST_CSV)

            # 2) Parse Tracy’s report for this run
            rep_dir = latest_report_dir(ARTIFACTS, run_name)
            kernel = extract_kernel_times(rep_dir)

            # 3) Append combined row
            append_combined_row(host_row, kernel)

            # 4) One-line summary
            print(
                f"[ok] size={size} recv=({rx},{ry}) "
                f"host_p50_ms={host_row['p50_ms']} "
                f"kernel_longest_ms={kernel['kernel_longest_ms']}"
            )


if __name__ == "__main__":
    main()
