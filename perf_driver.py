"""Minimal perf driver: profile specific RingJointSDPA config_ids and print DEVICE KERNEL DURATION.
Usage: python perf_driver.py <config_id> [<config_id> ...]
  e.g. python perf_driver.py ltx_s1-q96-k256 ltx_s2-q192-k512 wan2_2_1xGLX-q224-k512 wan2_2_4xGLX-q224-k512
Mirrors test_ring_joint_attention_create_perf_table's inner loop (lines ~3319-3366).
"""
import os
import sys
from unittest import mock
from tracy.process_model_log import run_device_profiler
from tests.nightly.sdpa_perf_utils import post_process_ops_log

SUBDIR = "ttnn_ring_joint_sdpa_perf_driver"
float_cols = ["CORE COUNT", "DEVICE KERNEL DURATION [ns]", "PM FPU UTIL (%)"]
cols = ["ATTRIBUTES"]


def profile_config(config_id):
    command = (
        "pytest tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::"
        f"test_ring_joint_attention_sdpa_sweep_perf_impl[{config_id}]"
    )
    with mock.patch.dict(os.environ, {"CI": "false"}):
        run_device_profiler(command, SUBDIR, device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log(
        SUBDIR,
        float_columns=float_cols,
        columns=cols,
        op_name="RingJointSDPADeviceOperation",
        sum_vals=False,
        has_signposts=False,
    )
    n = len(r["DEVICE KERNEL DURATION [ns]"])
    if n == 0:
        print(f"RESULT {config_id}: NO_OPS (inner test skipped or produced no kernel)")
        return
    dur_ns = int(r["DEVICE KERNEL DURATION [ns]"].max())
    cores = int(r["CORE COUNT"][0]) if len(r["CORE COUNT"]) > 0 else 0
    fpu = r.get("PM FPU UTIL (%)", [])
    fpu_max = float(fpu.max()) if len(fpu) > 0 else 0.0
    print(f"RESULT {config_id}: {dur_ns/1e6:.4f} ms  ({dur_ns} ns)  cores={cores}  fpu_max={fpu_max:.1f}%  n_ops={n}")


if __name__ == "__main__":
    for cid in sys.argv[1:]:
        try:
            profile_config(cid)
        except Exception as e:
            print(f"RESULT {cid}: ERROR {type(e).__name__}: {e}")
