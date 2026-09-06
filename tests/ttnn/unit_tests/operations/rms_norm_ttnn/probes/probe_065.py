import sys, os

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm_ttnn/probes")
import bench_r3
from bench_r3 import CASES, sweep
import ttnn

_ML = ttnn.TensorMemoryLayout
# Refinement 4 targets: prime Wt (4064 = 127 * 32), both layouts, both row counts.
CASES.update(
    {
        "R1_prime_1row_g": ((1, 1, 32, 4064), None, _ML.INTERLEAVED, "gamma", False, 0),
        "R1_prime_1row_n": ((1, 1, 32, 4064), None, _ML.INTERLEAVED, "no_gamma", False, 0),
        "R2_prime_97row_g": ((1, 1, 3104, 4064), None, _ML.INTERLEAVED, "gamma", False, 0),
        "R3_prime_gbr": ((1, 1, 3104, 4064), None, _ML.INTERLEAVED, "gamma_bias_residual", False, 0),
        "R4_prime_1row_g_rm": ((1, 1, 32, 4064), None, _ML.INTERLEAVED, "gamma", False, 0),
        "R5_prime_97row_g_rm": ((1, 1, 3104, 4064), None, _ML.INTERLEAVED, "gamma", False, 0),
    }
)

NAMES = os.environ.get("RMS_NAMES", "").split(",") if os.environ.get("RMS_NAMES") else list(CASES)
sweep([("base", {})], names=NAMES)
