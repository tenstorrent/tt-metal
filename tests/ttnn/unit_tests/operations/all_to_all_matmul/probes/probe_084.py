import os

os.environ.update(
    VARS="base||;cand||;cand||pdepth=all;cand||pdepth=all,pinfl=2;cand||pdepth=all,pinfl=4;base||",
    OUTDIR="ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/port_prefetch/out/r1",
)
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/port_prefetch/run.py").read())
