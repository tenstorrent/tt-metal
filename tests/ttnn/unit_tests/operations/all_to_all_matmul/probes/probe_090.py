import os

os.environ.update(VARS="cand||pdepth=all", ZONES="1", CHECK="0", TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES="0")
exec(open("ttnn/ttnn/operations/all_to_all_matmul/perf_experiments/port_prefetch/run.py").read())
