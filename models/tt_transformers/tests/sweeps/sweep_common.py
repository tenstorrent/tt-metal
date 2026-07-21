# SPDX-License-Identifier: Apache-2.0
"""
Shared helpers for the Llama-3.1-8B decode op sweeps (Blackhole P150).

PRIMARY metric = device kernel duration captured from the profiler
(ReadDeviceProfiler + get_latest_programs_perf_data), matching teja's
tt_llama_p150_matmul_sweep.py methodology. Host wall-clock is only a
diagnostic fallback and is flagged src=host so it is never confused with a
real device measurement.

Run every sweep with ALL THREE profiler env vars or rows fall back to src=host
(host wall-clock is dispatch-dominated and must NOT be used for ranking):
  export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) MESH_DEVICE=P150
  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1
Verified: with only TT_METAL_DEVICE_PROFILER=1, get_latest_programs_perf_data
returns nothing and everything is src=host (~70us noise). With all three, the
same create-heads op reports src=prof (~7-17us device kernel time).
"""
import csv
import time

import ttnn


def open_dev():
    dev = ttnn.open_device(device_id=0)
    dev.enable_program_cache()
    return dev


def device_kernel_ns(device):
    """Return (max_device_kernel_ns, core_count) from the profiler, or None."""
    try:
        ttnn.ReadDeviceProfiler(device)
        latest = ttnn.get_latest_programs_perf_data()
    except Exception:
        return None
    if not latest:
        return None
    dev_id = device.get_device_ids()[0] if hasattr(device, "get_device_ids") else 0
    progs = latest.get(dev_id) if latest else None
    if not progs:
        return None
    dur_ns, cores = None, 0
    for p in progs:
        r = p.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
        if r and r.duration is not None and (dur_ns is None or r.duration > dur_ns):
            dur_ns, cores = r.duration, p.core_count
    if not dur_ns:
        return None
    return dur_ns, cores


def timed_run(device, fn, iterations=30):
    """Run fn() iterations times; return dict(dur_ns, cores, src).

    fn must return the output tensor (deallocated here). Device-kernel time is
    preferred; host wall-clock/iter is only used if the profiler yields nothing.
    """
    # warmup + flush profiler so we only capture steady-state programs
    out = fn()
    ttnn.synchronize_device(device)
    try:
        if out is not None:
            out.deallocate(True)
    except Exception:
        pass
    try:
        ttnn.ReadDeviceProfiler(device)
    except Exception:
        pass

    t0 = time.perf_counter()
    for _ in range(iterations):
        out = fn()
        ttnn.synchronize_device(device)
        try:
            if out is not None:
                out.deallocate(True)
        except Exception:
            pass
    host_ns = (time.perf_counter() - t0) * 1e9 / iterations

    prof = device_kernel_ns(device)
    if prof:
        return dict(dur_ns=prof[0], cores=prof[1], src="prof")
    return dict(dur_ns=host_ns, cores=0, src="host")


class CSVLog:
    def __init__(self, path, header):
        self.path = path
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    def row(self, values):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(values)
