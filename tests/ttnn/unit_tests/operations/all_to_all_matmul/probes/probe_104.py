import os

os.environ.update(VARS="cand||", CASES="0,1,2,3", B2B="20")
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/port_prefetch/run.py").read())
