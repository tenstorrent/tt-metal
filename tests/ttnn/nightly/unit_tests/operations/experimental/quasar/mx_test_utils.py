# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the MX (OCP microscaling) op tests on Quasar.

MX is Quasar only: the unpacker expands MX tiles to Float16_b and the packer converts Dest back to MX, so the
goldens below quantize on the host with the same encoder from_torch uses and compare on the dequantized values.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# PCC floors per format: the element grids are 2 (MXINT2), 3 (MXFP4 e2m1, MXINT4) and 4 (MXFP8 e4m3, MXFP6) to
# 7 (MXINT8) significand bits, so the result of an op on MX inputs packed back to MX is coarser the fewer bits.
_PCC_FLOOR = {
    ttnn.mxint2: 0.90,
    ttnn.mxfp4: 0.98,
    ttnn.mxint4: 0.98,
    ttnn.mxfp6_e3m2: 0.99,
}


def pcc_floor(*dtypes):
    return min(_PCC_FLOOR.get(d, 0.995) for d in dtypes)


def skip_unless_quasar(device):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip("MX formats are Quasar only")


def mx_round_trip(x, dtype):
    """Host MX quantization: the values an MX tensor built from `x` holds (float32)."""
    if dtype == ttnn.bfloat16:
        return x.to(torch.bfloat16).to(torch.float32)
    return ttnn.to_torch(ttnn.from_torch(x.to(torch.float32), dtype=dtype, layout=ttnn.TILE_LAYOUT)).to(torch.float32)


def to_device(device, x, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        x.to(torch.float32) if dtype != ttnn.bfloat16 else x.to(torch.bfloat16),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def check(expected, actual, pcc):
    actual = actual.to(torch.float32)
    expected = expected.to(torch.float32)
    assert actual.shape == expected.shape, f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    assert torch.isfinite(actual).all(), "non-finite values in the device output"
    assert_with_pcc(expected, actual, pcc)
