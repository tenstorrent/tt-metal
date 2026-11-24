import os

os.environ["LC_RUN"] = "1"
os.environ["EXP"] = "3"
os.environ[
    "VARS"
] = "base|nosend,noportread,nowrite,nocompute|;cand|nosend,noportread,nowrite,nocompute|wden=1,wnum=1;cand|nowrite,nocompute|wden=4,wnum=1"
os.environ["OUTDIR"] = "ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/out/r3"
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/run_lc.py").read())
