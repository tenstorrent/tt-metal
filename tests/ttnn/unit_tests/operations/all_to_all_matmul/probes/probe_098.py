import os

os.environ["LC_RUN"] = "1"
os.environ[
    "VARS"
] = "base||;cand||place=landing;cand||wgap=6000;cand||wgap=5000,wwin=45000;cand|nowrite,nocompute|place=landing;base|nowrite,nocompute|;cand||place=landing;base||"
os.environ["OUTDIR"] = "ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/out/r5"
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/run_lc.py").read())
