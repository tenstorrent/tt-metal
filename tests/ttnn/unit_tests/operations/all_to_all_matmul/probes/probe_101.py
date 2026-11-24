import os

os.environ.update(
    VARS="cand||;base||",
    CASES="0,1",
    OUTDIR="ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/port_prefetch/out/dom2",
)
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/port_prefetch/run.py").read())
