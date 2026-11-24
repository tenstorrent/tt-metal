import os

os.environ["LC_RUN"] = "1"
os.environ[
    "VARS"
] = "base||;cand||wden=255;cand|nowrite,nocompute|wden=255;cand|nowrite,nocompute,noportread|wden=255;base|nowrite,nocompute|;cand||wden=255;base||"
os.environ["OUTDIR"] = "ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/out/r9"
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/run_lc.py").read())
