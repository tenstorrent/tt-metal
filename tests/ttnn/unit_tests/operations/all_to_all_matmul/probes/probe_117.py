import os

os.environ["LC_RUN"] = "1"
os.environ["CASES"] = "0,3"
os.environ["B2B"] = "20"
os.environ["VARS"] = "cand||;cand||wgap=6000;cand||wvc=2"
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/run_lc.py").read())
