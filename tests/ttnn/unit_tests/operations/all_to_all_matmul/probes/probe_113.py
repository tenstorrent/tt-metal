import os

os.environ["LC_RUN"] = "1"
os.environ["VARS"] = "base||;cand||wport=r0;cand||wport=r11;cand||wvc=0;cand||wvc=2;base||"
os.environ["OUTDIR"] = "ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/out/r7"
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/run_lc.py").read())
