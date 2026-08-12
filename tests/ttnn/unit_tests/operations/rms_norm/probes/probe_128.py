import sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments")
from batched_finalize import bench

bench.main(tile_counts=(1, 4, 8, 16, 32, 64), blocks=(2, 4, 8))
