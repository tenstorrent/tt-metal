# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive BF16 ULP sweep for the Blackhole and Wormhole ttnn.erfinv kernels.

Runs every one of the 65,536 BF16 encodings through ttnn.erfinv in a single
tiled tensor and checks, per tenstorrent/tt-metal#49435:

  * finite-domain accuracy: max pure ULP <= 1.0 against a float64
    torch.erfinv reference (pure ULP = |FTZ(golden) - result| /
    bf16_ulp_spacing(bf16-rounded golden), the ttnn-eltwise-op-tester metric,
    with the numerator flush keyed on the rounded golden exactly as the
    post-round-FTZ hardware behaves; threshold = min(previous
    threshold, certified ULP rounded up) — the previous kernel measured
    255.2 max pure ULP on this sweep, the replacement is
    certified at 0.8324;
  * special values, matching the silicon-certified contract: x = +/-1 -> signed Inf; |x| > 1, +/-Inf and NaN -> +Inf
  * zeros and DAZ'd subnormal inputs produce exact zeros.

Hardware model per tech_reports/Handling_Special_Value/special_values.md:
DAZ on input, post-round FTZ on output, and the format conversion pipeline
maps NaN payloads onto infinities and +/-0 onto +0. Blackhole canonicalizes
NaN payloads to +Inf; Wormhole preserves the sign of negative NaN inputs.
The previous kernels also never produced NaN here.

Set TT_EXPORT_ULP_DUMP=<path.npz> to additionally dump the raw per-encoding
device outputs (used to render the accuracy figure in the PR / tech report).

Run: pytest tests/ttnn/unit_tests/operations/eltwise/test_erfinv_bf16_exhaustive.py -v
"""

import os

import numpy as np
import pytest
import torch
import ttnn
from loguru import logger
from models.common.utility_functions import is_blackhole, is_wormhole_b0


def _all_bf16_encodings() -> torch.Tensor:
    """All 65,536 BF16 bit patterns as a (256, 256) bfloat16 tensor."""
    bits = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
    return bits.view(torch.bfloat16).reshape(256, 256)


def _daz(x64: np.ndarray) -> np.ndarray:
    """Denormals-are-zero on the BF16 input, keeping the sign."""
    tiny = np.abs(x64) < 2.0**-126
    return np.where(tiny, np.copysign(0.0, x64), x64)


def _bf16_round_ftz(y64: np.ndarray) -> np.ndarray:
    """Round float64 -> BF16 grid (RNE) then flush subnormals (post-round FTZ)."""
    y_bf16 = torch.from_numpy(y64).to(torch.bfloat16).to(torch.float64).numpy()
    subnormal = (np.abs(y_bf16) < 2.0**-126) & (y_bf16 != 0.0) & np.isfinite(y_bf16)
    return np.where(subnormal, np.copysign(0.0, y_bf16), y_bf16)


def _bf16_ulp_spacing(y: np.ndarray) -> np.ndarray:
    """BF16 ULP spacing at |y| (nextafter distance on the BF16 grid)."""
    y32 = np.abs(y.astype(np.float32))
    bits = (y32.view(np.uint32) >> np.uint32(16)).astype(np.uint32)
    nxt = np.minimum(bits + np.uint32(1), np.uint32(0x7F80))
    up = (nxt << np.uint32(16)).view(np.float32)
    cur = (bits << np.uint32(16)).view(np.float32)
    return (up - cur).astype(np.float64)


@pytest.mark.skipif(
    not (is_blackhole() or is_wormhole_b0()),
    reason="BF16 erfinv kernel replacement is supported on Blackhole and Wormhole",
)
def test_erfinv_bf16_exhaustive_ulp(device):
    x_bf16 = _all_bf16_encodings()

    tt_in = ttnn.from_torch(x_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.erfinv(tt_in)
    out = ttnn.to_torch(tt_out).to(torch.float32).numpy().astype(np.float64).reshape(-1)

    x64 = _daz(x_bf16.to(torch.float64).numpy().reshape(-1))
    golden = torch.erfinv(torch.from_numpy(x64)).numpy()

    dump_path = os.environ.get("TT_EXPORT_ULP_DUMP")
    if dump_path:
        np.savez_compressed(dump_path, x=x64, out=out, golden=golden)
        logger.info(f"dumped per-encoding sweep to {dump_path}")

    nan_lanes = np.isnan(golden)
    inf_lanes = np.isinf(golden)
    finite_lanes = ~(nan_lanes | inf_lanes)

    # Out-of-domain, NaN and +/-Inf inputs become infinities. Wormhole
    # preserves the sign of negative NaN inputs; Blackhole canonicalizes all
    # of these lanes to +Inf. The poles x = +/-1 keep their sign on both.
    expected_nonfinite = np.full(nan_lanes.sum(), np.inf)
    if is_wormhole_b0():
        nonfinite_inputs = x64[nan_lanes]
        negative_nan = np.isnan(nonfinite_inputs) & np.signbit(nonfinite_inputs)
        expected_nonfinite[negative_nan] = -np.inf
    assert np.array_equal(
        out[nan_lanes], expected_nonfinite
    ), "out-of-domain / NaN inputs must follow the architecture's infinity-sign contract"
    assert np.array_equal(out[inf_lanes], golden[inf_lanes]), "pole inputs must produce signed Inf"

    zero_golden = finite_lanes & (golden == 0.0)
    assert (out[zero_golden] == 0.0).all(), "zero results must be exactly zero"

    # Pure ULP over the finite domain, measured at the BF16-rounded golden.
    # The numerator flush mirrors the tester: when the rounded golden flushed
    # to zero, the exact golden is flushed too, so a correct FTZ zero from
    # the device scores 0 ULP instead of a spurious subnormal-window error.
    rounded_golden = _bf16_round_ftz(golden[finite_lanes])
    # Flush keyed on the ROUNDED golden only: a golden in the top half-ULP
    # below MIN_NORMAL rounds UP onto MIN_NORMAL and is not flushed -- the
    # correct device answer there is MIN_NORMAL, not zero.
    golden_ftz = np.where(
        rounded_golden == 0.0,
        np.copysign(0.0, golden[finite_lanes]),
        golden[finite_lanes],
    )
    err = np.abs(golden_ftz - out[finite_lanes])
    ulp = (err.astype(np.float32) / _bf16_ulp_spacing(rounded_golden).astype(np.float32)).astype(np.float64)
    assert np.isfinite(ulp).all(), "non-finite device output for a finite golden value"

    max_ulp = float(ulp.max())
    mean_ulp = float(ulp.mean())
    logger.info(f"ttnn.erfinv exhaustive BF16: max pure ULP {max_ulp:.4f}, mean {mean_ulp:.4f}")

    assert (
        max_ulp <= 1.0
    ), f"max pure ULP {max_ulp:.4f} exceeds the 1.0 gate (certified kernel measures 0.8324 on this sweep)"
