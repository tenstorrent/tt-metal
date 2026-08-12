import sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments")
from pipelined_combine.harness import main

for i in range(3):
    print("REP", i, flush=True)
    main(["focus_11x10"], ["baseline", "flag", "incr"], skews=["none", "mid", "big"])
