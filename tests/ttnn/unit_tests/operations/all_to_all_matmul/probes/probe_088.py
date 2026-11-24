import os

os.environ["LC_RUN"] = "1"
os.environ[
    "VARS"
] = "base|nowrite,nocompute,noweight|;base|nowrite,nocompute|;base|nowrite,nocompute,noportread|;cand|nowrite,nocompute|wden=1,wnum=1;cand|nowrite,nocompute,noportread|wden=1,wnum=1"
os.environ["OUTDIR"] = "ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/out/r2"
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/run_lc.py").read())
