# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
SFPU accuracy sweep for Quasar.

Sibling of test_sfpu_accuracy.py (the WH/BH sweep) sharing run_case, the
shard->merge pipeline and the output schema. It lives here rather than in
quasar/ because the merge hooks are in accuracy/conftest.py.

The parameter list is resolver-driven, not a cartesian product: on Quasar the
(input, output, dest_acc) route is decided by resolve_quasar_sfpu_variant, so
only executable routes appear. The op list is the subset of the WH/BH
transcendentals the Quasar unary SFPU kernel dispatches, the approx axis
follows Quasar's own approx-capable list, and there is no fast-mode axis
(the Quasar kernel has none). DestSync and ImpliedMathFormat are pinned in
accuracy_harness (QUASAR_DEST_SYNC / QUASAR_IMPLIED_MATH_FORMAT).

Runs only against the Quasar simulator (no silicon): from tests/python_tests
    CHIP_ARCH=quasar pytest --run-simulator -m "accuracy and quasar" \
        accuracy/test_sfpu_accuracy_quasar.py --maxfail=5000
"""

import pytest
from conftest import skip_for_coverage
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
)
from helpers.param_config import (
    generate_quasar_sfpu_format_variants,
    input_output_formats,
)

# WH/BH transcendentals that the Quasar unary SFPU test kernel dispatches
# (see tests/helpers/include/sfpu_operations_quasar.h). Missing vs WH/BH:
# Exp2, Log, Log1p, Elu, Celu, Hardsigmoid, Erfinv.
QUASAR_TRANSCENDENTAL_OPS = [
    MathOperation.Exp,
    MathOperation.Sqrt,
    MathOperation.Rsqrt,
    MathOperation.Reciprocal,
    MathOperation.Sin,
    MathOperation.Cos,
    MathOperation.Tanh,
    MathOperation.Gelu,
    MathOperation.Silu,
    MathOperation.Atanh,
    MathOperation.Asinh,
    MathOperation.Acosh,
]

# Ops with a real approx kernel on Quasar (mirrors quasar/test_eltwise_unary_sfpu_quasar.py).
# Differs from WH/BH: Sqrt, Sin and Cos have no approx path here.
QUASAR_APPROX_CAPABLE_OPS = [
    MathOperation.Exp,
    MathOperation.Gelu,
    MathOperation.Reciprocal,
    MathOperation.Rsqrt,
]

FORMATS = input_output_formats(
    [
        DataFormat.Float32,
        DataFormat.Float16,
        DataFormat.Float16_b,
    ]
)


def _get_approx_modes(mathop):
    if mathop in QUASAR_APPROX_CAPABLE_OPS:
        return [ApproximationMode.No, ApproximationMode.Yes]
    return [ApproximationMode.No]


def _quasar_accuracy_params():
    """(formats, approx, op, dest_acc) for every executable Quasar route.

    full_format_route_sweep keeps every valid (input, output, dest_acc) route
    rather than one representative per SFPU-visible state: the input format
    changes the sampled x values, so routes the functional suite treats as
    duplicates are distinct accuracy curves.
    """
    params = []
    for op in QUASAR_TRANSCENDENTAL_OPS:
        variants = generate_quasar_sfpu_format_variants(
            op, FORMATS, full_format_route_sweep=True
        )
        for variant in variants:
            for approx in _get_approx_modes(op):
                params.append((variant.formats, approx, op, variant.dest_acc))
    return params


QUASAR_ACCURACY_PARAMS = _quasar_accuracy_params()


@skip_for_coverage
@pytest.mark.quasar
@pytest.mark.accuracy
@pytest.mark.parametrize("formats,approx_mode,mathop,dest_acc", QUASAR_ACCURACY_PARAMS)
def test_sfpu_accuracy_sweep_quasar(
    formats: InputOutputFormat,
    approx_mode: ApproximationMode,
    mathop: MathOperation,
    dest_acc: DestAccumulation,
):
    from accuracy.accuracy_harness import run_case

    run_case(mathop, formats, approx_mode, FastMode.No, dest_acc)
