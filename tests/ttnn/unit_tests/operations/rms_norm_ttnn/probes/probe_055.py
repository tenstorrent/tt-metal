import sys, os

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm_ttnn/probes")
os.environ["RMS_REPS"] = "2"
import bench_r3

NAMES = [
    "P1_int1024_g",
    "P2_int1024_gb",
    "P4_int2048_g",
    "P5_int5120_gbr",
    "G1_w7168_g28",
    "G2_w5120_gbr",
    "G3_blk8192",
]
bench_r3.sweep(
    [
        ("t2_derived", {"PER_CHANNEL_TRIM_GAMMA": -1, "PER_CHANNEL_TRIM_BIAS": -1}),
        ("t1_half", {"PER_CHANNEL_TRIM_GAMMA": 1, "PER_CHANNEL_TRIM_BIAS": 1}),
        ("t0_whole", {"PER_CHANNEL_TRIM_GAMMA": 0, "PER_CHANNEL_TRIM_BIAS": 0}),
        ("g2_b0", {"PER_CHANNEL_TRIM_GAMMA": -1, "PER_CHANNEL_TRIM_BIAS": 0}),
    ],
    NAMES,
)
