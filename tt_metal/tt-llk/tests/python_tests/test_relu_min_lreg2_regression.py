# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Deterministic stale-LREG2 regression for relu_min (Wormhole only).

tt-llk#1120: Wormhole's ``_relu_min_impl_`` reads its threshold from LREG2 as an
*implicit input* -- the body is raw TTI (``TTI_SFPMOV(0, LREG2, LREG1, 0)`` before the
SFPSWAP), so the register is part of the calling contract but appears nowhere in the
signature. The ``T == float`` branch of the ``_relu_min_`` wrapper assigned a local sfpi
vector instead of loading LREG2, which compiled clean and then ran relu_min against
whatever the previously executed SFPU kernel had left in that register.

Why this file exists rather than relying on the sweeps. The sweeps only fail when some
*other* op runs first and dirties LREG2, so on the broken kernel ``pytest -k ReluMin``
passes in isolation and fails only when mixed with Sqrt/Square/Log/Sign -- the failure
is a property of test ordering, not of the kernel, and a reordered or sharded run can
go green over a reintroduced bug. This test removes the ordering entirely: the kernel
writes a known poison into LREG2 as the instruction immediately preceding the
``_relu_min_`` call, inside the same SFPU block, so nothing can come between them.

    fixed kernel:  _relu_min_ loads LREG2 itself  ->  result = max(x, threshold)
    broken kernel: the poison survives            ->  result = max(x, poison)

The poison is far outside the golden's range, so the two outcomes are separated by
orders of magnitude rather than by a tolerance.

Wormhole only. Blackhole's ``_relu_min_`` is a plain sfpi predicated form
(``v_if (a < threshold)``) that genuinely consumes its parameter, so there is no
register to poison and this test would pass vacuously there; Quasar has its own
implementation. Both are skipped rather than silently covered.
"""

import struct

import torch
from conftest import skip_for_blackhole, skip_for_quasar
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    VectorMode,
    format_dict,
)
from helpers.sfpu_dispatch_constants import RELU_MIN_THRESHOLD
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    SFPU_UNARY_SCALAR,
    SFPU_UNARY_THRESHOLD,
    VECTOR_MODE,
    generate_input_dim,
)

pytestmark = [skip_for_blackhole, skip_for_quasar]

ELEMENTS_PER_TILE = 1024

# The value written into LREG2 immediately before _relu_min_. Requirements: not the
# threshold (or the bug is invisible), positive and large enough that `max(x, poison)`
# is the poison for every input below (so a broken kernel returns a flat, unmistakable
# tile), and exactly representable in every format on this path so the expected-failure
# value is not itself approximate.
LREG2_POISON = 1024.0


def _bits(value: float) -> int:
    """Raw fp32 bit pattern, the encoding both SFPU_UNARY_* parameters expect."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _straddling_inputs() -> torch.Tensor:
    """One tile straddling the threshold, so both branches of max(x, threshold) fire.

    Deterministic rather than sampled: this test is about a register, not a domain, and
    a fixed pattern makes the failure message reproducible. Half the lanes sit below the
    threshold (where relu_min must clamp) and half above (where it must pass through) --
    the pass-through half is what a threshold-returning kernel gets wrong, and the
    clamped half is what a poisoned LREG2 gets wrong.
    """
    x = torch.empty(ELEMENTS_PER_TILE, dtype=torch.float32)
    below = torch.linspace(-8.0, RELU_MIN_THRESHOLD - 1.0, ELEMENTS_PER_TILE // 2)
    above = torch.linspace(RELU_MIN_THRESHOLD + 1.0, 12.0, ELEMENTS_PER_TILE // 2)
    x[0::2] = below
    x[1::2] = above
    return x


def test_relu_min_ignores_stale_lreg2():
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    dest_acc = DestAccumulation.Yes

    src_A = _straddling_inputs()
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)

    configuration = TestConfig(
        "sources/relu_min_lreg2_regression_test.cpp",
        formats,
        templates=[
            generate_input_dim([32, 32], [32, 32]),
            APPROX_MODE(ApproximationMode.No),
            SFPU_UNARY_SCALAR(_bits(LREG2_POISON)),
            SFPU_UNARY_THRESHOLD(_bits(RELU_MIN_THRESHOLD)),
            VECTOR_MODE(VectorMode.RC),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=True,
        compile_time_formats=True,
    )

    res = torch.tensor(
        configuration.run().result[:ELEMENTS_PER_TILE],
        dtype=format_dict[formats.output_format],
    ).to(torch.float32)

    golden = torch.maximum(src_A, torch.tensor(RELU_MIN_THRESHOLD))

    # Exact, not a tolerance: relu_min is an SFPSWAP, which *selects* one of its two
    # operands rather than computing anything, and the whole path is fp32. Any deviation
    # at all is a wrong operand, not rounding.
    poisoned = torch.isclose(res, torch.tensor(LREG2_POISON))
    assert not poisoned.any(), (
        f"{poisoned.sum().item()}/{res.numel()} lanes came back as the LREG2 poison "
        f"({LREG2_POISON}), so _relu_min_ used the stale register instead of its own "
        f"threshold. This is tt-llk#1120: the T == float branch of _relu_min_ must call "
        f"_sfpu_load_imm32_(p_sfpu::LREG2, ...), because _relu_min_impl_ takes the "
        f"threshold from LREG2 and never reads a parameter."
    )
    assert torch.equal(res, golden), (
        f"relu_min(x, {RELU_MIN_THRESHOLD}) mismatched on "
        f"{(res != golden).sum().item()}/{res.numel()} lanes "
        f"(max deviation {(res - golden).abs().max().item():.6g}). LREG2 was not the "
        f"poison, so the threshold load is happening -- the defect is in the clamp itself."
    )
