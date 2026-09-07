import os, runpy, sys

os.environ["RMS_TRIALS"] = "1"
os.environ["RMS_REPS"] = "1"
os.environ["RMS_CASES"] = "G4_blk7168"
os.environ["RMS_VARIANTS"] = "base,pd_off,mcast"
sys.argv = ["bench_mcast"]
runpy.run_path(
    "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/per_channel_reuse_mcast/bench_mcast.py",
    run_name="__main__",
)
