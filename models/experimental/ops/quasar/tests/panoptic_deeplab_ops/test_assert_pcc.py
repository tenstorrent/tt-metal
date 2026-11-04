# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-only tests for this suite's ``op_utils.assert_pcc``: it must reject an output that
correlates with the reference but has the wrong magnitude, which PCC alone cannot see
(``ttnn.multiply(x, 4)`` returning ``x`` scores PCC 1.0). No device is opened.

    pytest models/experimental/ops/quasar/tests/panoptic_deeplab_ops/test_assert_pcc.py
"""

import pytest
import torch

from models.experimental.ops.quasar.tests.panoptic_deeplab_ops import op_utils as U


@pytest.fixture
def x():
    return torch.randn(1, 32, 64, 256, generator=torch.Generator().manual_seed(0)).bfloat16()


@pytest.mark.parametrize("pcc", [0.999, 0.97, 0.90])
def test_no_op_scalar_multiply_fails(x, pcc, expect_error):
    # The captured instance-offset scaling: multiply by 4. A no-op returns the input.
    with expect_error(AssertionError, "wrong magnitude"):
        U.assert_pcc(x.float() * 4, host=x.float(), pcc=pcc)


@pytest.mark.parametrize("pcc", [0.999, 0.97, 0.90])
def test_correct_multiply_passes(x, pcc):
    U.assert_pcc(x.float() * 4, host=(x * 4).float(), pcc=pcc)


def test_block_float_noise_passes(x):
    # Noise at the bfloat8_b floor's level (PCC ~0.98) stays inside the 0.97 floor's tolerance.
    ref = x.float()
    got = ref + 0.2 * torch.randn(ref.shape, generator=torch.Generator().manual_seed(1))
    U.assert_pcc(ref, host=got, pcc=0.97)


def test_all_zero_reference(x, expect_error):
    zeros = torch.zeros_like(x, dtype=torch.float32)
    U.assert_pcc(zeros, host=zeros, pcc=0.999)
    # The shared PCC check falls back to allclose for an all-zero reference and fails first.
    with expect_error(AssertionError, "PCC below"):
        U.assert_pcc(zeros, host=x.float(), pcc=0.999)
