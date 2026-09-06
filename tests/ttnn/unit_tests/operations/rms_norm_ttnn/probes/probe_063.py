import sys, os

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm_ttnn/probes")
os.environ["RMS_REPS"] = "3"
import bench_r3

bench_r3.sweep([("r2_base", {"PASS_A_SQ_BLOCK": 0}), ("r3", {})], list(bench_r3.CASES))
