import sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments")
from batched_finalize import bench

bench.main(tile_counts=(1, 32), blocks=(4,), variants=["baseline_stream", "chunk_pair", "chunk_fused"])
