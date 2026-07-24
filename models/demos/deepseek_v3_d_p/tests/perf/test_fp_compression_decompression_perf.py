# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device performance tests for FP8 compression and decompression.

Compression and decompression are measured *separately* — each has its own
worker (one op invocation on a fixed [640, 7168] input) and its own perf wrapper
with an independent baseline, so each runs as its own Tracy capture and a
regression localizes to the responsible kernel:

  * per_token_cast_to_fp8 -> PerTokenCastToFp8DeviceOperation  (compression)
  * per_token_cast_back   -> PerTokenCastBackDeviceOperation    (decompression)

"""

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.utils.perf_utils import run_model_device_perf_test_per_op

# Shape under test: 640 tokens x 7168 hidden.
# Production runs prefill in 5K-token chunks, so compression always sees this
# exact shape. Decompression is handed a differently-shaped tensor at runtime;
# we exercise it at the same shape here purely as a regression guard.
SEQ_LEN = 640
HIDDEN = 7168
BLOCK_W = 128
E4M3_MAX = 448.0

# OP CODE substrings emitted in the Tracy CSV.
COMPRESS_OP = "PerTokenCastToFp8DeviceOperation"
DECOMPRESS_OP = "PerTokenCastBackDeviceOperation"

# Measured baselines (ns) for one op invocation on a bh_loudbox (8xP150) runner
# (2026-07-24, local): compression 55,422..57,964, decompression 63,590..66,904.
# Absolute durations are small (~60us), so a few-us jitter is a large percentage;
# the wrappers use a ±5% margin, which comfortably covers the observed spread.
_COMPRESS_EXPECTED_NS = 56_700
_DECOMPRESS_EXPECTED_NS = 65_450

# Perf tolerance. 5%
_MARGIN = 0.05


# ---------------------------------------------------------------------------
# Compression: per_token_cast_to_fp8
# ---------------------------------------------------------------------------


@pytest.mark.timeout(0)
def test_run_compression(device):
    """Worker: one per_token_cast_to_fp8 invocation."""
    if not is_blackhole():
        pytest.skip("FP8_E4M3 path requires Blackhole")

    torch.manual_seed(0)

    x = (torch.randn(SEQ_LEN, HIDDEN) * 5.0).to(torch.bfloat16)
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    signpost(f"compression {SEQ_LEN}x{HIDDEN}")
    ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x_tt)

    ttnn.synchronize_device(device)
    logger.info(f"[compression] {SEQ_LEN}x{HIDDEN} done")


@pytest.mark.models_device_performance_bare_metal
def test_device_perf_compression():
    command = (
        "pytest models/demos/deepseek_v3_d_p/tests/perf/test_fp_compression_decompression_perf.py::test_run_compression"
    )
    run_model_device_perf_test_per_op(
        command=command,
        expected_per_op={COMPRESS_OP: _COMPRESS_EXPECTED_NS},
        subdir="deepseek_v3_compression",
        model_name=f"deepseek_v3_compression_{SEQ_LEN}x{HIDDEN}",
        margin=_MARGIN,
        comments=f"per_token_cast_to_fp8 {SEQ_LEN}x{HIDDEN}",
    )


# ---------------------------------------------------------------------------
# Decompression: per_token_cast_back
# ---------------------------------------------------------------------------


@pytest.mark.timeout(0)
def test_run_decompression(device):
    """Worker: one per_token_cast_back invocation."""
    if not is_blackhole():
        pytest.skip("FP8_E4M3 path requires Blackhole")

    torch.manual_seed(0)

    e4m3 = (torch.randn(SEQ_LEN, HIDDEN) * 3.0).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    scale = torch.rand(SEQ_LEN, HIDDEN // BLOCK_W) * 4.0 - 2.0

    e4m3_tt = ttnn.from_torch(
        e4m3.float(),
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scale_tt = ttnn.from_torch(
        scale,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    signpost(f"decompression {SEQ_LEN}x{HIDDEN}")
    ttnn.experimental.deepseek_prefill.per_token_cast_back(e4m3_tt, scale_tt, output_dtype=ttnn.bfloat16)

    ttnn.synchronize_device(device)
    logger.info(f"[decompression] {SEQ_LEN}x{HIDDEN} done")


@pytest.mark.models_device_performance_bare_metal
def test_device_perf_decompression():
    command = "pytest models/demos/deepseek_v3_d_p/tests/perf/test_fp_compression_decompression_perf.py::test_run_decompression"
    run_model_device_perf_test_per_op(
        command=command,
        expected_per_op={DECOMPRESS_OP: _DECOMPRESS_EXPECTED_NS},
        subdir="deepseek_v3_decompression",
        model_name=f"deepseek_v3_decompression_{SEQ_LEN}x{HIDDEN}",
        margin=_MARGIN,
        comments=f"per_token_cast_back {SEQ_LEN}x{HIDDEN}",
    )
