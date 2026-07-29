"""Profile the is_cross ring SDPA (V2A transport-bound case) and dump per-op device durations.
Usage: python cross_perf_driver.py <k_token>
  e.g. python cross_perf_driver.py ltx_v2a_stage1_6s-bh_4x8sp1tp0_ring
The cross accuracy test runs q_chunk_sizes=[64,128] internally -> multiple SDPA op instances.
Captures ALL ops (op_name="") so we see the fused RingJointSDPA and any separate all-gather op.
"""
import os
import sys
from unittest import mock
from tracy.process_model_log import run_device_profiler
from tests.nightly.sdpa_perf_utils import post_process_ops_log

SUBDIR = "ttnn_ring_joint_sdpa_cross_perf"


def profile(k_token):
    command = (
        "pytest tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py::"
        f"test_ring_joint_attention_sdpa_cross_accuracy -k {k_token}"
    )
    with mock.patch.dict(os.environ, {"CI": "false"}):
        run_device_profiler(command, SUBDIR, device_analysis_types=["device_kernel_duration"])
    # Per-op durations, keyed by op name.
    r = post_process_ops_log(
        SUBDIR,
        float_columns=["DEVICE KERNEL DURATION [ns]"],
        columns=["OP CODE"],
        op_name="",
        sum_vals=False,
        has_signposts=False,
    )
    durs = r["DEVICE KERNEL DURATION [ns]"]
    codes = r.get("OP CODE", [])
    print(f"===== {k_token}: {len(durs)} ops =====")
    for i in range(len(durs)):
        code = codes[i] if i < len(codes) else "?"
        print(f"  op[{i}] {code}: {int(durs[i])/1e6:.4f} ms ({int(durs[i])} ns)")
    # RingJointSDPA-only summary
    try:
        rj = post_process_ops_log(
            SUBDIR,
            float_columns=["DEVICE KERNEL DURATION [ns]"],
            columns=["OP CODE"],
            op_name="RingJointSDPADeviceOperation",
            sum_vals=False,
            has_signposts=False,
        )
        rjd = rj["DEVICE KERNEL DURATION [ns]"]
        if len(rjd):
            print(f"  RingJointSDPA: n={len(rjd)} max={int(rjd.max())/1e6:.4f} ms min={int(rjd.min())/1e6:.4f} ms")
    except Exception as e:
        print(f"  RingJointSDPA summary error: {e}")


if __name__ == "__main__":
    for tok in sys.argv[1:]:
        try:
            profile(tok)
        except Exception as e:
            print(f"ERROR {tok}: {type(e).__name__}: {e}")
