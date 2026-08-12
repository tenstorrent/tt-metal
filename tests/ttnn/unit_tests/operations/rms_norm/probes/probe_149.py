import sys, os

sys.path.insert(0, os.getcwd())
from tests.ttnn.unit_tests.operations.rms_norm.perf_zone_harness import main

main(["decode7168", "decode1024", "bshard1024", "wshard7168"])
