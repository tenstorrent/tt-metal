import sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/combine_skew")
import bench

bench.main(
    names=["decode7168", "decode5120", "decode2304", "decode1024"],
    places=["front", "byrow", "spread", "rowend"],
    reps=3,
)
