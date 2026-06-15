# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.llk_params import MathOperation
from helpers.stimuli_generator import StimuliSpec

from ._spec import float_binary_op_spec

_ELEMENTS_PER_TILE = 1024

# Crafted lanes exercising the divide special-case branches a random sweep can't
# hit: (lane, dividend, divisor). 0/0 -> NaN, x/0 -> +/-Inf, x/x -> 1.0; the golden +
# passed_test already treat NaN==NaN and matching +/-Inf as passing.
_SPECIAL_CASE_LANES = [
    (0, 0.0, 0.0),  # 0/0   -> NaN
    (1, 1.5, 0.0),  # +x/0  -> +Inf
    (2, -1.5, 0.0),  # -x/0  -> -Inf
    (3, 2.7, 2.7),  # x/x   -> 1.0
    (4, -3.3, -3.3),  # x/x   -> 1.0
]
# Lanes whose x/x result the kernel forces to a bit-exact 1.0 (see _verify_exact_ones).
_EXACT_ONE_LANES = [3, 4]


def _inject_special_cases(src, *, src_tiles):
    """Overwrite the first few lanes of the dividend and divisor tiles with crafted
    operand pairs so the special-case branches are exercised. ``src_tiles`` is the
    (dividend_tile, divisor_tile) pair for the variant's tile layout."""
    dividend_tile, divisor_tile = src_tiles
    flat = src.flatten().clone()
    for lane, dividend, divisor in _SPECIAL_CASE_LANES:
        flat[dividend_tile * _ELEMENTS_PER_TILE + lane] = dividend
        flat[divisor_tile * _ELEMENTS_PER_TILE + lane] = divisor
    return flat.reshape(src.shape)


def _verify_exact_ones(res_tensor, *, io):
    """The kernel's x/x branch forces an exact 1.0 regardless of reciprocal rounding,
    so check those result lanes bit-exact rather than relying on isclose tolerance."""
    for lane in _EXACT_ONE_LANES:
        actual = res_tensor[lane].item()
        assert (
            actual == 1.0
        ), f"x/x special case at lane {lane}: expected exact 1.0, got {actual}"


SPEC = float_binary_op_spec(
    name="div",
    math_op=MathOperation.SfpuElwdiv,
    binop="DIV",
    # Bulk operands bounded away from zero so the reciprocal + Newton-Raphson path
    # stays well-defined; the special-case lanes below cover 0/0 and x/0 explicitly.
    stimuli=StimuliSpec.uniform(low=0.25, high=4.0),
    prepare=_inject_special_cases,
    verify=_verify_exact_ones,
)
