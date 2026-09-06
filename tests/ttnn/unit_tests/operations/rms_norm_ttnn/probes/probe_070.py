import sys, os

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm_ttnn/probes")
from bench_r3 import CASES, sweep
import ttnn

_ML = ttnn.TensorMemoryLayout
CASES.update(
    {
        "R3_prime_gbr": ((1, 1, 3104, 4064), None, _ML.INTERLEAVED, "gamma_bias_residual", False, 0),
        "R7_prime_gbr_rm": ((1, 1, 3104, 4064), None, _ML.INTERLEAVED, "gamma_bias_residual", False, 0),
        "R8_prime_1row_gbr_rm": ((1, 1, 32, 4064), None, _ML.INTERLEAVED, "gamma_bias_residual", False, 0),
        "R9_prime_r_rm": ((1, 1, 3104, 4064), None, _ML.INTERLEAVED, "residual", False, 0),
    }
)
NAMES = os.environ.get("RMS_NAMES", "").split(",") if os.environ.get("RMS_NAMES") else list(CASES)
sweep(
    [
        ("floor1", {"ROW_RESIDENT_MIN_CHUNK_WT": 1}),
        ("floor2", {"ROW_RESIDENT_MIN_CHUNK_WT": 2}),
        ("floor4", {"ROW_RESIDENT_MIN_CHUNK_WT": 4}),
    ],
    names=NAMES,
)
