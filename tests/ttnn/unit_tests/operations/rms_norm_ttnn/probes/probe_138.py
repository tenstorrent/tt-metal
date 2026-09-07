import os, runpy, sys

os.environ["RMS_TRIALS"] = "3"
os.environ["RMS_REPS"] = "3"
os.environ[
    "RMS_CASES"
] = "P1_int1024_g,P2_int1024_gb,P4_int2048_g,P5_int5120_gbr,P6_int7168_g,G3_blk8192,G4_blk7168,S1_stream_gbr,G2_w5120_gbr,G6_band512_rm"
os.environ["RMS_VARIANTS"] = "base,pd_off,mcast_wtb"
runpy.run_path(
    "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/per_channel_reuse_mcast/bench_mcast.py",
    run_name="__main__",
)
