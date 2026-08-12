import sys, os

sys.path.insert(0, os.path.join(os.environ.get("TT_METAL_HOME", "."), "tests/ttnn/unit_tests/operations/rms_norm"))
from perf_zone_harness import main

main(["decode7168"])
