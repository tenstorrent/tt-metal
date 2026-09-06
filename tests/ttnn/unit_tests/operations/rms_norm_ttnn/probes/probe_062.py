import sys, os

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm_ttnn/probes")
os.environ["RMS_REPS"] = "3"
import bench_r3

NAMES = [
    "G3_blk8192",
    "G4_blk7168",
    "G1_w7168_g28",
    "G2_w5120_gbr",
    "G6_band512_rm",
    "P1_int1024_g",
    "P2_int1024_gb",
    "P4_int2048_g",
]
bench_r3.sweep([("sq0", {"CB_SQ_EXACT": 0}), ("sq1", {"CB_SQ_EXACT": 1})], NAMES)
