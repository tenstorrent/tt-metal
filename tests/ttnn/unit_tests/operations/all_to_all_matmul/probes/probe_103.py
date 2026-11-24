import os

os.environ.update(VARS="cand||", GOLDEN=",".join(str(i) for i in range(18)), PAYLOADS="7232,0", CFG="all")
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/port_prefetch/run.py").read())
