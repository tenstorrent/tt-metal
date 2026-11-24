import os

os.environ["LC_RUN"] = "1"
os.environ["VARS"] = "base||;cand||wgap=4000;cand||wgap=6000;cand||wgap=8000;cand|nowrite,nocompute|wgap=6000"
os.environ["OUTDIR"] = "ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/out/r4"
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/run_lc.py").read())
