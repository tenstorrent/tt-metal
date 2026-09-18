# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Regression test for the softplus negative tail (issue #51866).

The existing softplus tests compare PCC over uniform(-100, 100). That gate cannot
see this defect: softplus(x) for x < -5 is on the order of 1e-3 down to 1e-38,
while the bulk of the distribution is order 50, so returning exactly 0.0 for every
negative-tail input still yields PCC = 0.9999999999.

This test asserts *relative* error at points chosen so that the correct answer is a
normal bfloat16 number. It fails on the unfixed kernel (returned 0.0, relative
error 1.0 at every point) and passes once the exp tail is restored.
"""

import pytest
import torch
import ttnn


# Points spanning the tail, all with a correct value that is a normal bfloat16:
#   softplus(-87) = 1.6458e-38, bfloat16 min normal is 1.1755e-38.
NEGATIVE_TAIL_INPUTS = [-5.03125, -6.0, -8.0, -10.0, -20.0, -40.0, -60.0, -80.0, -87.0]


@pytest.mark.parametrize("x", NEGATIVE_TAIL_INPUTS)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_softplus_negative_tail_is_not_zero(device, x, dtype):
    """softplus is strictly positive everywhere; it must never return exactly 0."""
    torch_input = torch.full((1, 1, 32, 32), x, dtype=torch.float32)
    expected = torch.nn.functional.softplus(torch_input)

    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.to_torch(ttnn.softplus(tt_input))

    got = tt_output[0, 0, 0, 0].item()
    ref = expected[0, 0, 0, 0].item()

    # The defect returns exactly 0.0, so check that first with a clearer message.
    assert got != 0.0, f"softplus({x}) returned exactly 0.0; correct value is {ref:.6e} (a normal bfloat16)"

    # bfloat16 carries 8 significand bits => ~2^-8 relative resolution. Allow 2 ULP
    # of headroom for the tail approximation; the defect's relative error is 1.0.
    tolerance = 2 * 2**-8 if dtype == ttnn.bfloat16 else 1e-5
    rel_err = abs(got - ref) / abs(ref)
    assert rel_err < tolerance, f"softplus({x}): got {got:.6e}, expected {ref:.6e}, relative error {rel_err:.3e}"


def test_softplus_pcc_gate_cannot_see_the_negative_tail():
    """Documents why the assertions above are relative-error rather than PCC.

    Pure host-side arithmetic; no device needed. Reproduces the masking effect so a
    future reader does not 'simplify' this file back into a PCC comparison.
    """
    torch.manual_seed(0)
    x = torch.empty(20000).uniform_(-100, 100)
    reference = torch.nn.functional.softplus(x)

    # Model the defect: exact zero below the polynomial boundary, correct elsewhere.
    defective = reference.clone()
    defective[x < -5.0] = 0.0

    pcc = torch.corrcoef(torch.stack([reference.double(), defective.double()]))[0, 1].item()
    assert pcc > 0.999, "expected the PCC gate to be blind to this defect"
    assert (defective[x < -5.0] == 0).all()
