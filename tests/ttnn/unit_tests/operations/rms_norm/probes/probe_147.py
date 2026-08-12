import sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments")
from pipelined_combine.harness import main

V = ["baseline", "flag", "incr", "incr_sem"]
for i in range(3):
    print("REP", i, flush=True)
    main(["focus_11x10"], V, skews=["none", "mid", "big"])
print("SWEEP", flush=True)
main(["focus_11x10_b4", "wshard_8x1", "wshard_7x4", "col_1x8", "small_3x3", "bshard_8x1_r16"], V, skews=["none", "big"])
