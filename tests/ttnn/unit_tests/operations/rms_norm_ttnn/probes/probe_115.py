import os, runpy, sys

os.environ["RMS_TRIALS"] = "3"
os.environ["RMS_REPS"] = "1"
os.environ["RMS_PC_TRACE"] = "1"
os.environ["RMS_CASES"] = "P2_int1024_gb,P1_int1024_g,P6_int7168_g,G4_blk7168,G3_blk8192"
os.environ["RMS_VARIANTS"] = "base,mcast,mcast_nh,ablate"
runpy.run_path(
    "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/per_channel_reuse_mcast/bench_mcast.py",
    run_name="__main__",
)
