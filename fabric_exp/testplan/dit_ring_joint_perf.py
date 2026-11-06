#!/usr/bin/env python3
"""DiT ring joint SDPA device time on a BH Galaxy (8x4), for the main-vs-new perf A/B (G-L1P-x step G1x-dit-rj).

Runs models/tt_dit/tests/unit/test_ring_joint_attention.py::test_ring_joint_sdpa under the device profiler (tracy),
one profiler run per DiT model input, with the parallel config the test's own `parallel_config_map["bh_glx"]` gives
(main's perf-table wrapper, test_ring_joint_sdpa_perf_table, is skipped: pytest inside pytest deadlocks on the
device). Per model it prints one line:
    DIT_RING_JOINT_SDPA model=<model> cfg=<rp>rpx<up>up-<q>-<k> calls=<n> ns=<median over calls of the slowest chip>
Exit 1 if a model produced no RingJointSDPA rows (skipped or failed).

    dit_ring_joint_perf.py [--q q256] [--k k512] [--models wan_14b_720p,flux,...]
"""
import argparse
import os
import statistics
import sys

MODELS = {  # model input id -> parallel config id for bh_glx (test_ring_joint_attention.parallel_config_map)
    "wan_14b_720p": "8rpx4up",
    "wan_14b_480p": "8rpx4up",
    "mochi": "8rpx4up",
    "flux": "8rpx4up",
    "sd35": "4rpx4up",
}
TEST = "models/tt_dit/tests/unit/test_ring_joint_attention.py::test_ring_joint_sdpa"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="q256")
    ap.add_argument("--k", default="k512")
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()
    home = os.environ["TT_METAL_HOME"]
    # Tried for job 172928 (tracy's capture did not finish within 120 s after the test on a 32-chip Galaxy): skipping
    # the device push to Tracy did not help (new pass 1 failed the same way), so that cause is WRONG; kept, harmless.
    os.environ["TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY"] = "1"
    sys.path.insert(0, os.path.join(home, "tools"))
    import pandas as pd
    from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

    missing = []
    for model in args.models.split(","):
        par = MODELS[model]
        k_expr = f"{model} and {par} and bh_glx and no_trace_no_check and {args.q} and {args.k}"
        subdir = f"dit_rj_{model}"
        print(f"=== {model}: pytest {TEST} -k '{k_expr}'", flush=True)
        run_device_profiler(
            f"-m 'pytest {TEST} -k \"{k_expr}\"'",
            subdir,
            check_test_return_code=False,
            device_analysis_types=["device_kernel_duration"],
            is_command_binary_exe=True,
        )
        try:
            df = pd.read_csv(get_latest_ops_log_filename(subdir))
        except (OSError, IndexError, ValueError) as e:
            print(f"DIT_RING_JOINT_SDPA model={model} no ops log: {e}", flush=True)
            missing.append(model)
            continue
        rows = df[df["OP CODE"].astype(str).str.contains("RingJointSDPA")]
        if rows.empty:
            print(f"DIT_RING_JOINT_SDPA model={model} no RingJointSDPA rows ({len(df)} ops in the log)", flush=True)
            missing.append(model)
            continue
        per_call = rows.groupby("GLOBAL CALL COUNT")["DEVICE KERNEL DURATION [ns]"].max()  # the slowest chip
        print(
            f"DIT_RING_JOINT_SDPA model={model} cfg={par}-{args.q}-{args.k} calls={len(per_call)} "
            f"ns={statistics.median(per_call.tolist()):.0f} min={per_call.min():.0f} max={per_call.max():.0f}",
            flush=True,
        )
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
