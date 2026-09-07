import os, runpy, sys

os.environ["RMS_TRIALS"] = "3"
os.environ["RMS_REPS"] = "1"
sys.argv = ["bench_ablate"]
runpy.run_path(
    "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/per_channel_reuse_mcast/bench_ablate.py",
    run_name="__main__",
)
