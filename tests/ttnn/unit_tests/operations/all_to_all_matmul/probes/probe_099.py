import os

os.environ.update(
    VARS="base||;cand||",
    CASES="0,1,2,3",
    OUTDIR="ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/port_prefetch/out/dom",
)
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/port_prefetch/run.py").read())
