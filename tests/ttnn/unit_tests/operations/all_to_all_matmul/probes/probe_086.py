import os

os.environ["LC_RUN"] = "1"
os.environ["VARS"] = "base||;cand||wden=1,wnum=1;cand||wden=2,wnum=1;base||"
os.environ["OUTDIR"] = "ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/out/r1"
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/run_lc.py").read())
