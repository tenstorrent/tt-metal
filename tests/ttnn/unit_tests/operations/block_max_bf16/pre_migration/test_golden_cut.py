# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden coverage cut of block_max_bf16 (golden spec: ttnn/ttnn/bringup/block_select_ttnn/tests/unit/
test_block_select.py, ``check_block_select``).

The golden suite has no TARGET / SUPPORTED table; its axes are the mode (fold / strip-major), ``factor`` (1, 4, 16),
the length as a value or a device tensor, and its INPUTS: fold (NI, factor, n) in {(8192, 4, 1000), (8192, 1, 8192),
(4096, 4, 3), (2048, 16, 128)} on 96 rows, strip-major (S, NB, n) in {(8, 16384, 16384), (8, 2176, 2176), (3, 512, 77),
(2, 264, 1), (5, 4096, 1000)}. Every axis value appears below at least once, plus the smallest and the largest input of
each mode, on the golden topology (every device as one FABRIC_2D mesh, inputs replicated). Reference and tolerance
are the golden suite's: torch amax / strip transpose, byte-exact over the first n (rounded up to 8) columns, and a
caller's output buffer (pre-filled) untouched past them. Every device's copy is checked.
"""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.block_max_bf16.pre_migration.block_max_ref import (
    check_written,
    strip_device_view,
    strip_ref,
    up,
    up_n,
    written,
)
from tests.ttnn.unit_tests.operations.block_max_bf16.pre_migration.test_acceptance import (  # noqa: F401
    block_max_bf16_op,
)

ROWS = 96  # the golden suite's rows
FILL = 7.0  # the golden suite's pre-filled output value


def _golden_fold_input(NI: int, g: torch.Generator) -> torch.Tensor:
    """The golden suite's fold input: bf16 values widened to fp32, an all-negative row, -inf runs, a -inf row."""
    x = torch.randn(1, 1, ROWS, NI, generator=g).to(torch.bfloat16).float()
    x[..., 0, :] = -x[..., 0, :].abs() - 1
    x[..., 1, ::3] = float("-inf")
    x[..., 2, :] = float("-inf")
    return x.to(torch.bfloat16).float()


FOLD_CUT = [
    pytest.param(8192, 1, 8192, False, id="largest_ni8192_f1_n8192_value"),
    pytest.param(2048, 16, 128, True, id="smallest_ni2048_f16_n128_tensor"),
    pytest.param(8192, 4, 1000, True, id="ni8192_f4_n1000_tensor"),
    pytest.param(4096, 4, 3, False, id="ni4096_f4_n3_value"),
]


@pytest.mark.parametrize(("NI", "F", "nn", "use_t"), FOLD_CUT)
def test_golden_fold(golden_mesh, block_max_bf16_op, NI, F, nn, use_t):
    g = torch.Generator().manual_seed(NI + F + nn)
    x = _golden_fold_input(NI, g)
    xt = up(x, golden_mesh, ttnn.float32)
    want = x.view(1, 1, ROWS, NI // F, F).amax(-1).to(torch.bfloat16)  # the golden reference
    prev = torch.full((1, 1, ROWS, NI // F), FILL, dtype=torch.bfloat16)
    ot = up(prev, golden_mesh, ttnn.bfloat16)
    kw = {"n_tensor": up_n(nn, golden_mesh)} if use_t else {"n": nn}
    got = block_max_bf16_op(xt, factor=F, output_tensor=ot, **kw)
    check_written(got, want, written(nn, NI // F), prev, f"fold {NI} {F} {nn} {use_t}")


STRIP_CUT = [
    pytest.param(8, 16384, 16384, False, id="largest_s8_nb16384_n16384_value"),
    pytest.param(2, 264, 1, True, id="smallest_s2_nb264_n1_tensor"),
    pytest.param(8, 2176, 2176, True, id="s8_nb2176_n2176_tensor"),
    pytest.param(3, 512, 77, False, id="s3_nb512_n77_value"),
]


@pytest.mark.parametrize(("S", "NB", "nn", "use_t"), STRIP_CUT)
def test_golden_strip(golden_mesh, block_max_bf16_op, S, NB, nn, use_t):
    g = torch.Generator().manual_seed(S * NB + nn)
    x = torch.randn(1, 1, S, NB * 32, generator=g)
    x[..., 0, ::5] = float("-inf")
    x = x.to(torch.bfloat16)
    xt = up(strip_device_view(x, S, NB), golden_mesh, ttnn.bfloat16)
    want = strip_ref(x, S, NB)  # == the golden x.view(1, 1, S, NB, 32).permute(0, 1, 2, 4, 3).reshape(...)
    prev = torch.full((1, 1, S * 32, NB), FILL, dtype=torch.bfloat16)
    ot = up(prev, golden_mesh, ttnn.bfloat16)
    kw = {"n_tensor": up_n(nn, golden_mesh)} if use_t else {"n": nn}
    got = block_max_bf16_op(xt, strip_major=True, output_tensor=ot, **kw)
    check_written(got, want, written(nn, NB), prev, f"strip {S} {NB} {nn} {use_t}")
