"""Refinement 4's A/B bench: D1's divisor clamp vs D33's ragged (padded) width chunk.

Reuses bench_r3.py's harness (build / measure / sweep / CASES) verbatim and only adds
the prime-Wt cases -- 4064 = 127 * 32 and 2848 = 89 * 32, both Wt PRIME, so
`_largest_divisor_at_most` returns 1 for every cap below the whole row.

    RMS_REPS=2 RMS_NAMES="R1_prime_1row_g,R2_prime_97row_g" \
        scripts/tt-probe.sh rms_norm_ttnn < .../probes/bench_r4.py

With no RMS_NAMES it sweeps every case, targets and guards together.
"""

import sys, os

sys.path.insert(0, "tests/ttnn/unit_tests/operations/rms_norm_ttnn/probes")
from bench_r3 import CASES, sweep
import ttnn

_ML = ttnn.TensorMemoryLayout
CASES.update(
    {
        "R1_prime_1row_g": ((1, 1, 32, 4064), None, _ML.INTERLEAVED, "gamma", False, 0),
        "R1_prime_1row_n": ((1, 1, 32, 4064), None, _ML.INTERLEAVED, "no_gamma", False, 0),
        "R2_prime_97row_g": ((1, 1, 3104, 4064), None, _ML.INTERLEAVED, "gamma", False, 0),
        "R3_prime_gbr": ((1, 1, 3104, 4064), None, _ML.INTERLEAVED, "gamma_bias_residual", False, 0),
        "R4_prime_1row_g_rm": ((1, 1, 32, 4064), None, _ML.INTERLEAVED, "gamma", False, 0),
        "R5_prime_97row_g_rm": ((1, 1, 3104, 4064), None, _ML.INTERLEAVED, "gamma", False, 0),
        "R6_prime_97row_n_rm": ((1, 1, 3104, 4064), None, _ML.INTERLEAVED, "no_gamma", False, 0),
        "R7_prime_gbr_rm": ((1, 1, 3104, 4064), None, _ML.INTERLEAVED, "gamma_bias_residual", False, 0),
        "R8_prime_1row_gbr_rm": ((1, 1, 32, 4064), None, _ML.INTERLEAVED, "gamma_bias_residual", False, 0),
        "R9_p2848_g_rm": ((1, 1, 3104, 2848), None, _ML.INTERLEAVED, "gamma", False, 0),
    }
)
NAMES = os.environ.get("RMS_NAMES", "").split(",") if os.environ.get("RMS_NAMES") else list(CASES)
sweep([("divisor", {"RAGGED_WIDTH_CHUNK": 0}), ("ragged", {"RAGGED_WIDTH_CHUNK": 1})], names=NAMES)
