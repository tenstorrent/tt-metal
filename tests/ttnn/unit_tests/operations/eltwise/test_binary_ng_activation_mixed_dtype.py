# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for fused binary activations on mixed operand dtypes.

`input_tensor_a_activations` / `input_tensor_b_activations` run as a separate
copy -> activate -> pack pass into an intermediate CB before the binary op. That
pass reads its source through SrcA, but srcA is programmed for the binary op's
*LHS* operand. The FPU helper asked for the switch with the conditional
`reconfig_data_format_srca(old, new)` while naming a CB srcA never holds, so the
guard compared two formats that always agreed and the reconfigure was dropped;
the SFPU helper omitted the reconfigure entirely. Either way the activation input
was unpacked through the other operand's data format, which is silently harmless
when both operands share a dtype and corrupts ~90% of the output when they do not.

Mixed float dtypes are a supported combination (`dtype_policy::supports_mixed_float_inputs`),
so the two axes have to be crossed: no test in the tree did that before this one.

The matrix deliberately keeps the matching-dtype pairs alongside the mixed ones, so
the test documents the boundary rather than just the failure.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device

SHAPE = (1, 1, 32, 32)

# ttnn.add resolves fast_and_approximate_mode to True and lands on the FPU kernel;
# ttnn.multiply leaves it False and lands on the SFPU kernel. The two helpers had
# separate instances of this bug, so both routings must be covered.
OPS = [
    pytest.param(ttnn.add, id="add_fpu"),
    pytest.param(ttnn.multiply, id="multiply_sfpu"),
]

DTYPES = [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b]
DTYPE_PAIRS = [pytest.param(a, b, id=f"{a.name.lower()}_x_{b.name.lower()}") for a in DTYPES for b in DTYPES]


def _make(device, dtype, torch_tensor):
    return ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _check(fused, two_op):
    """Compare a fused activation against the equivalent unfused pair.

    Both run on device, so the only legitimate gap is the activation's own
    precision (SiLU picks a different exp path depending on dest-accumulation
    mode); the bug instead produces a bit-field-rotated operand, i.e. PCC ~ 0.
    """
    assert not torch.isnan(fused).any(), "fused result contains NaN"
    assert not torch.isinf(fused).any(), "fused result contains inf"

    # Guard against a degenerate (near-constant) result, which PCC alone can accept.
    ref_std = two_op.float().std()
    if ref_std > 0.1:
        assert fused.float().std() > 0.1 * ref_std, (
            f"fused output is ~constant (std={fused.float().std():.4g}) while the "
            f"unfused reference varies (std={ref_std:.4g}) -- operand format mix-up?"
        )
    assert_with_pcc(two_op, fused, 0.9999)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("a_dtype, b_dtype", DTYPE_PAIRS)
def test_rhs_activation_mixed_dtype(device, op, a_dtype, b_dtype):
    """activations on input_tensor_b -- the arm that was broken on both kernels."""
    torch.manual_seed(0)
    a = torch.randn(SHAPE)
    b = torch.linspace(-100.0, 100.0, SHAPE[-1] * SHAPE[-2]).reshape(SHAPE)

    ta, tb = _make(device, a_dtype, a), _make(device, b_dtype, b)
    two_op = ttnn.to_torch(op(ta, ttnn.abs(tb))).float()
    fused = ttnn.to_torch(op(ta, tb, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.ABS)])).float()
    _check(fused, two_op)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("a_dtype, b_dtype", DTYPE_PAIRS)
def test_lhs_activation_mixed_dtype(device, op, a_dtype, b_dtype):
    """activations on input_tensor_a -- correct on the FPU kernel by coincidence, not on the SFPU one."""
    torch.manual_seed(0)
    a = torch.linspace(-100.0, 100.0, SHAPE[-1] * SHAPE[-2]).reshape(SHAPE)
    b = torch.randn(SHAPE)

    ta, tb = _make(device, a_dtype, a), _make(device, b_dtype, b)
    two_op = ttnn.to_torch(op(ttnn.abs(ta), tb)).float()
    fused = ttnn.to_torch(op(ta, tb, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.ABS)])).float()
    _check(fused, two_op)


@pytest.mark.parametrize(
    "a_dtype, b_dtype",
    [
        pytest.param(ttnn.float32, ttnn.bfloat16, id="float32_x_bfloat16"),
        pytest.param(ttnn.bfloat16, ttnn.bfloat16, id="bfloat16_x_bfloat16"),
    ],
)
def test_silu_gate_mixed_dtype(device, a_dtype, b_dtype):
    """`x * silu(z)` with fp32 x and bf16 z -- the Qwen3.6 GDN out-gate shape of the bug.

    Large-magnitude z is included because the failure was originally misread as a
    SiLU overflow; it is not, and matching dtypes are clean at any magnitude.
    """
    torch.manual_seed(0)
    x = torch.randn(SHAPE) * 3.0
    z = torch.randn(SHAPE) * 40.0

    tx, tz = _make(device, a_dtype, x), _make(device, b_dtype, z)
    two_op = ttnn.to_torch(ttnn.multiply(tx, ttnn.silu(tz))).float()
    fused = ttnn.to_torch(
        ttnn.multiply(tx, tz, input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)])
    ).float()
    _check(fused, two_op)

    # and against the host golden, to catch a shared-mode failure the pair could hide
    ref = ttnn.to_torch(tx).float() * torch.nn.functional.silu(ttnn.to_torch(tz).float())
    assert_with_pcc(ref, fused, 0.999)
