# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Device-perf tests for the FP8 per-token compression / decompression ops.

Measures the device time of these operators on the 5K-token prefill chunks we run
in production (the 640x7168 shape one chunk casts):

  * per_token_cast_to_fp8 -> PerTokenCastToFp8DeviceOperation  (compression)
  * per_token_cast_back   -> PerTokenCastBackDeviceOperation    (decompression)
"""

import pytest

from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_per_op
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_p150
from models.demos.deepseek_v3_d_p.utils.chunk_config import ISL_TOKENS_PER_CHIP

# Functional test file whose op launches land in the Tracy CSV the harness reads back.
_WORKER = (
    "tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_deepseek_prefill_per_token_cast.py"
)

# Production prefill shape: 640 tokens x 7168 hidden. Compression always sees this exact
# shape; decompression is exercised at the same shape purely as a regression guard.
_SEQ_LEN = ISL_TOKENS_PER_CHIP
_HIDDEN = 7168

# Device-op codes emitted by each op; the harness sums the rows whose OP CODE contains the
# substring, so incidental setup ops (host->device copies, etc.) are excluded.
_COMPRESS_OP = "PerTokenCastToFp8DeviceOperation"
_DECOMPRESS_OP = "PerTokenCastBackDeviceOperation"

# Per-op device time in ns, measured on a Blackhole P150. Recalibrate on the perf CI runner.
_COMPRESS_EXPECTED_NS = 56_700
_DECOMPRESS_EXPECTED_NS = 65_450

_MARGIN = 0.05


def _k(extra: str) -> str:
    """Pin exactly the 640x7168 case of the worker."""
    return f"{_SEQ_LEN} and {extra}"


# Compression: ROW_MAJOR bf16 input -- the Blackhole production fast path (input tilized and
# bf8-packed inside the op).
_COMPRESS_WORKER = f"pytest {_WORKER}::test_cast_to_fp8_scale_values -k '{_k('bfloat16 and ROW_MAJOR')}'"
# Decompression: bf16 output, scales kept at fp32 (full-precision fp32/HiFi4 multiply).
_DECOMPRESS_WORKER = f"pytest {_WORKER}::test_cast_back_dequant -k '{_k('bfloat16 and scales_kept_at_fp32')}'"


@pytest.mark.parametrize(
    "command, expected_per_op, model_name, comments",
    [
        (
            _COMPRESS_WORKER,
            {_COMPRESS_OP: _COMPRESS_EXPECTED_NS},
            f"deepseek_v3_compression_{_SEQ_LEN}x{_HIDDEN}",
            f"per_token_cast_to_fp8 {_SEQ_LEN}x{_HIDDEN}",
        ),
        (
            _DECOMPRESS_WORKER,
            {_DECOMPRESS_OP: _DECOMPRESS_EXPECTED_NS},
            f"deepseek_v3_decompression_{_SEQ_LEN}x{_HIDDEN}",
            f"per_token_cast_back {_SEQ_LEN}x{_HIDDEN}",
        ),
    ],
    ids=["compression", "decompression"],
)
@pytest.mark.models_device_performance_bare_metal
# Gate to P150 via tt-smi board telemetry (SMBus). This also skips Wormhole and any other
# board. Do NOT use ttnn.cluster.get_cluster_type() here: it opens and locks the chip, and
# since skipif is evaluated at collection time in the parent process, the spawned Tracy worker
# then deadlocks on CHIP_IN_USE. is_p150() reads tt-smi only, so it takes no device lock.
@pytest.mark.skipif(not is_p150(), reason="perf baselines are P150-specific; skip on any other board")
def test_per_token_cast_perf(command, expected_per_op, model_name, comments):
    run_model_device_perf_test_per_op(
        command=command,
        expected_per_op=expected_per_op,
        subdir="prefill_per_token_cast",
        model_name=model_name,
        margin=_MARGIN,
        comments=comments,
    )
