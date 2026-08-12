import sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments")
from batched_finalize import bench

bench.main(tile_counts=(4,), blocks=(4,))
