import os

os.environ.update(
    VARS="cand||ppkt=1,pinfl=3;cand||ppkt=1,pinfl=4;base||;cand||ppkt=1,pinfl=6,pdepth=all;cand||ppkt=1,pinfl=8;cand||ppkt=1,pinfl=2",
    OUTDIR="ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/port_prefetch/out/r3",
)
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/port_prefetch/run.py").read())
