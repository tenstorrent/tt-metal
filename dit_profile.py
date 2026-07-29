"""Profile ONE LTX DiT transformer block forward (signpost-bracketed) and dump per-op device
kernel durations, aggregated by OP CODE. Shows where a denoise step's time goes (matmul vs SDPA
vs norm) -> the path toward 6s.  Usage: python dit_profile.py <k_token>
"""
import os
import sys
from collections import defaultdict
from unittest import mock
from tracy.process_model_log import run_device_profiler
from tests.nightly.sdpa_perf_utils import post_process_ops_log

SUBDIR = "ltx_dit_block_profile"


def profile(k_token):
    command = (
        "pytest models/tt_dit/tests/models/ltx/test_transformer_ltx.py::test_ltx_transformer_block "
        f"-k {k_token}"
    )
    with mock.patch.dict(os.environ, {"CI": "false", "LTX_SKIP_PCC": "1"}):
        run_device_profiler(command, SUBDIR, device_analysis_types=["device_kernel_duration"])
    r = post_process_ops_log(
        SUBDIR,
        float_columns=["DEVICE KERNEL DURATION [ns]"],
        columns=["OP CODE"],
        op_name="",
        sum_vals=False,
        has_signposts=True,   # slice to the signpost("start")..("stop") warm forward
    )
    durs = r["DEVICE KERNEL DURATION [ns]"]
    codes = r.get("OP CODE", [])
    n = len(durs)
    total = sum(int(durs[i]) for i in range(n))
    agg = defaultdict(lambda: [0, 0.0])  # code -> [count, total_ns]
    for i in range(n):
        code = codes[i] if i < len(codes) else "?"
        agg[code][0] += 1
        agg[code][1] += int(durs[i])
    print(f"===== {k_token}: {n} ops, block total device time = {total/1e6:.3f} ms =====")
    for code, (cnt, ns) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"  {ns/1e6:8.3f} ms  {100*ns/total:5.1f}%  x{cnt:<4d} {code}")


if __name__ == "__main__":
    for tok in sys.argv[1:]:
        try:
            profile(tok)
        except Exception as e:
            print(f"ERROR {tok}: {type(e).__name__}: {e}")
