import os

os.environ["LC_RUN"] = "1"
os.environ["ZONES"] = "1"
os.environ["VARS"] = "base|nowrite,nocompute,noweight|"
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/run_lc.py").read())
