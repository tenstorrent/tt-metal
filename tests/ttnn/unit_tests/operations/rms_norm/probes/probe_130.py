import sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/combine_skew")
import bench

# lever (a): does a G with LESS skew (or exact-equal C) beat G=110 on the focus shape?
bench.main(names=["decode7168"], places=["front", "spread"], gs=[110, 55, 22, 2], reps=1)
