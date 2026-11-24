import os

os.environ.update(
    VARS="base||;cand||;cand||ppkt=1,pinfl=6;cand||ppkt=1,pinfl=12;cand||ppkt=1,pinfl=15,pdepth=all;base||",
    OUTDIR="ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/port_prefetch/out/r2",
)
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/port_prefetch/run.py").read())
