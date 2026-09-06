import sys, os

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm_ttnn/probes")
os.environ["RMS_REPS"] = "2"
import bench_r3

NAMES = [
    "P1_int1024_g",
    "P2_int1024_gb",
    "P4_int2048_g",
    "P5_int5120_gbr",
    "S1_stream_gbr",
    "G1_w7168_g28",
    "G2_w5120_gbr",
    "G3_blk8192",
    "G6_band512_rm",
]
bench_r3.sweep(
    [
        ("base", {}),
        ("sqblk", {"PASS_A_SQ_BLOCK": 1}),
        ("resfuse", {"RES_FUSE": 1}),
        ("sq_rf", {"PASS_A_SQ_BLOCK": 1, "RES_FUSE": 1}),
        ("sq_txn2", {"PASS_A_SQ_BLOCK": 1, "DM_TXN_ROWS_MAX": 2}),
        ("sq_txnall", {"PASS_A_SQ_BLOCK": 1, "DM_TXN_ROWS_MAX": 0}),
    ],
    NAMES,
)
