import os

os.environ["LC_RUN"] = "1"
os.environ["CASES"] = "1,2,3"
os.environ["VARS"] = "base||;cand||wgap=6000;cand||wvc=2"
os.environ["OUTDIR"] = "ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/out/dom"
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/run_lc.py").read())
