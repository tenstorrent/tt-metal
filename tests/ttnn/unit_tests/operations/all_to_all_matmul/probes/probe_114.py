import os

os.environ["LC_RUN"] = "1"
os.environ["EXP"] = "3"
os.environ[
    "VARS"
] = "cand|nowrite,nocompute,noportread|;cand|nowrite,nocompute,noportread|wskip=east;cand|nowrite,nocompute,noportread|wskip=west;cand|nowrite,nocompute,noportread|wskip=rows2-6;cand|nowrite,nocompute,noportread|wskip=rows7-11;cand|nowrite,nocompute,noportread,noweight|"
os.environ["OUTDIR"] = "ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/out/r8"
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/run_lc.py").read())
