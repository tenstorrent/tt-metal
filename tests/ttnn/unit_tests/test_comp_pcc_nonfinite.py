# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""comp_pcc must not mask NaN/Inf positions that differ between golden and calculated.

Before the fix, the helper replaced NaN and Inf in both tensors with zero before
correlating, so an output whose non-finite values sat at different positions than the
golden could still pass with a PCC near 1.0.
"""

import torch

from tt_py_test_utils_common.utility_functions import comp_pcc


def _ramp(size=1024):
    return torch.arange(size, dtype=torch.float32)


class TestCompPccNonFinitePositions:
    def test_nan_position_mismatch_fails(self):
        golden = _ramp()
        golden[0] = float("nan")
        calculated = _ramp()
        calculated[1] = float("nan")

        passing, pcc_val = comp_pcc(golden, calculated)

        assert not passing
        assert pcc_val == 0.0

    def test_inf_position_mismatch_fails(self):
        golden = _ramp()
        golden[0] = float("inf")
        calculated = _ramp()
        calculated[1] = float("-inf")

        passing, pcc_val = comp_pcc(golden, calculated)

        assert not passing
        assert pcc_val == 0.0

    def test_inf_sign_mismatch_fails(self):
        golden = _ramp()
        golden[0] = float("inf")
        calculated = _ramp()
        calculated[0] = float("-inf")

        passing, pcc_val = comp_pcc(golden, calculated)

        assert not passing
        assert pcc_val == 0.0

    def test_matching_nan_positions_pass(self):
        golden = _ramp()
        golden[0] = float("nan")
        calculated = golden.clone()

        passing, pcc_val = comp_pcc(golden, calculated)

        assert passing
        assert pcc_val == 1.0

    def test_matching_inf_positions_pass(self):
        golden = _ramp()
        golden[0] = float("inf")
        calculated = golden.clone()

        passing, pcc_val = comp_pcc(golden, calculated)

        assert passing
        assert pcc_val == 1.0

    def test_finite_tensors_unchanged(self):
        golden = _ramp()
        calculated = golden + 0.001

        passing, pcc_val = comp_pcc(golden, calculated)

        assert passing
        assert pcc_val > 0.99

    def test_one_all_nan_tensor_still_fails(self):
        golden = torch.full((16, 16), float("nan"))
        calculated = torch.ones((16, 16))

        passing, pcc_val = comp_pcc(golden, calculated)

        assert not passing
        assert pcc_val == 0.0
