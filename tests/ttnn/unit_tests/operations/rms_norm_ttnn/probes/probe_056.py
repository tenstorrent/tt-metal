import sys, os

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm_ttnn/probes")
os.environ["RMS_REPS"] = "1"
os.environ["RMS_TRIALS"] = "1"
import bench_r3

bench_r3.sweep([("default", {})], ["P1_int1024_g", "P5_int5120_gbr", "G1_w7168_g28", "G6_band512_rm"])
