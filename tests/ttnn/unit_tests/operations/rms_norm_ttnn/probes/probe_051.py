import sys, os

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm_ttnn/probes")
os.environ["RMS_REPS"] = "2"
import bench_r3

bench_r3.sweep([("base", {})])
