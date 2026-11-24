import os

os.environ["TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES"] = "0"
os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = "0"
os.environ["LC_RUN"] = "1"
os.environ["ZONES"] = "1"
os.environ["ZCSV"] = "ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/out/zones/xw_npr.csv"
os.environ["VARS"] = "base|nowrite,nocompute,noportread|"
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/landing_contention/run_lc.py").read())
