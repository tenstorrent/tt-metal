"""Unified perf driver: profile all self-attn + cross nodes, dump RingJointSDPA device duration.
Reused verbatim for ON and OFF builds. Prints RESULT lines that are easy to diff.
"""
import os
import sys
from unittest import mock
from tracy.process_model_log import run_device_profiler
from tests.nightly.sdpa_perf_utils import post_process_ops_log

F = "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py"
FP = "tests/nightly/blackhole/sdpa/test_ring_joint_sdpa_cross_padded.py"

# (label, subdir, command)
SELF = [
    ("ltx_s1-q96-k256", "pd_ltx_s1", f"pytest {F}::test_ring_joint_attention_sdpa_sweep_perf_impl[ltx_s1-q96-k256]"),
    ("ltx_s2-q192-k512", "pd_ltx_s2", f"pytest {F}::test_ring_joint_attention_sdpa_sweep_perf_impl[ltx_s2-q192-k512]"),
    ("wan2_2_1xGLX-q224-k512", "pd_wan1x", f"pytest {F}::test_ring_joint_attention_sdpa_sweep_perf_impl[wan2_2_1xGLX-q224-k512]"),
    ("wan2_2_4xGLX-q224-k512", "pd_wan4x", f"pytest {F}::test_ring_joint_attention_sdpa_sweep_perf_impl[wan2_2_4xGLX-q224-k512]"),
]
CROSS = [
    ("cross_stage1_6s", "pd_x_s1", f"pytest {FP} -k ltx_v2a_stage1_6s"),
    ("cross_stage2_1080p_6s_padded", "pd_x_s2_6s", f"pytest {FP} -k ltx_v2a_stage2_1080p_6s"),
]


def profile(label, subdir, command):
    try:
        with mock.patch.dict(os.environ, {"CI": "false"}):
            run_device_profiler(command, subdir, device_analysis_types=["device_kernel_duration"])
        allops = post_process_ops_log(
            subdir, float_columns=["DEVICE KERNEL DURATION [ns]"], columns=["OP CODE"],
            op_name="", sum_vals=False, has_signposts=False,
        )
        rj = post_process_ops_log(
            subdir, float_columns=["DEVICE KERNEL DURATION [ns]"], columns=["OP CODE"],
            op_name="RingJointSDPADeviceOperation", sum_vals=False, has_signposts=False,
        )
        durs = allops["DEVICE KERNEL DURATION [ns]"]
        codes = allops.get("OP CODE", [])
        print(f"=== {label}: {len(durs)} device ops ===")
        for i in range(len(durs)):
            code = codes[i] if i < len(codes) else "?"
            print(f"    {code}: {int(durs[i])/1e6:.4f} ms")
        rjd = rj["DEVICE KERNEL DURATION [ns]"]
        if len(rjd):
            print(f"RESULT {label}: RingJointSDPA n={len(rjd)} max={int(rjd.max())/1e6:.4f} ms min={int(rjd.min())/1e6:.4f} ms")
        else:
            print(f"RESULT {label}: NO RingJointSDPA ops (n_all={len(durs)})")
    except Exception as e:
        print(f"RESULT {label}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = []
    if which in ("all", "self"):
        jobs += SELF
    if which in ("all", "cross"):
        jobs += CROSS
    # optional filter by label substring
    if len(sys.argv) > 2:
        tok = sys.argv[2]
        jobs = [j for j in jobs if tok in j[0]]
    for label, subdir, command in jobs:
        profile(label, subdir, command)
