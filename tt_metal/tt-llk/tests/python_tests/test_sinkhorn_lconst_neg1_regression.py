# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Sinkhorn must leave the shared -1.0 constant register intact (Blackhole only).

``_sinkhorn_program_parity_mask_`` writes its lane-parity mask into programmable
constant vector register 11 via ``TTI_SFPCONFIG``. ``p_sfpu::LCONST_neg1`` is that same
register -- a core-wide constant that other SFPU kernels read as -1.0 -- and sinkhorn is
the only site in the Blackhole tree that writes register 11 at all, so nothing ever puts
-1.0 back. The header comment acknowledges the clobber but scopes the constraint to
sinkhorn's own row-norm body; it outlives the call.

The kernel composes two ops the compute API exposes side by side and validates only the
second: sinkhorn runs on dest 0 for its side effect, floor runs on dest 1.
``ckernel_sfpu_rounding_ops.h`` applies floor's "if v > trunc(v), subtract one"
correction with ``LCONST_neg1``, so:

    register intact     ->  floor(-2.5) == -3.0
    register clobbered  ->  the correction adds the parity mask (integer 0 or 2, which
                            as a float is zero or a denormal) instead of -1.0, and floor
                            degenerates to trunc:  floor(-2.5) == -2.0

Every negative non-integer separates the two outcomes by exactly 1.0, so this asserts
exact equality rather than a tolerance. Sinkhorn's own output is never read here.
"""

import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import generate_input_dim
from helpers.tilize_untilize import tilize

pytestmark = [skip_for_wormhole, skip_for_quasar]

ELEMENTS_PER_TILE = 1024


def _inputs() -> torch.Tensor:
    """A tile of bf16-exact values, half of them negative non-integers.

    Deterministic rather than sampled: the negative half is where floor and trunc differ,
    and every value is exactly representable in Float16_b so neither the golden nor the
    broken-path value is approximate. The positive half is carried so a kernel that
    damaged floor generally, rather than only its negative correction, is still caught.
    """
    x = torch.empty(ELEMENTS_PER_TILE, dtype=torch.float32)
    # -0.5, -1.5, -2.5 ... repeating: floor and trunc differ by exactly 1.0 on each.
    negatives = -(torch.arange(ELEMENTS_PER_TILE // 2, dtype=torch.float32) % 16) - 0.5
    # 0.5, 1.5, 2.5 ... repeating: floor == trunc here, so these must be unaffected.
    positives = (torch.arange(ELEMENTS_PER_TILE // 2, dtype=torch.float32) % 16) + 0.5
    x[0::2] = negatives
    x[1::2] = positives
    return x


def test_sinkhorn_restores_lconst_neg1():
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)

    src_A = _inputs()
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch.float32)

    configuration = TestConfig(
        "sources/sinkhorn_lconst_neg1_regression_test.cpp",
        formats,
        templates=[generate_input_dim([32, 32], [32, 32])],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilize(src_A, formats.input_format).flatten(),
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=DestAccumulation.No,
        compile_time_formats=True,
    )

    res = torch.tensor(
        configuration.run().result[:ELEMENTS_PER_TILE],
        dtype=format_dict[formats.output_format],
    ).to(torch.float32)

    golden = torch.floor(
        tilize(src_A, formats.input_format).flatten().to(torch.float32)
    )
    trunc = torch.trunc(tilize(src_A, formats.input_format).flatten().to(torch.float32))

    degenerated = (res == trunc) & (golden != trunc)
    assert not degenerated.any(), (
        f"{degenerated.sum().item()}/{res.numel()} lanes came back as trunc(x) instead of "
        f"floor(x), so floor's -1.0 correction used a clobbered LCONST_neg1. Sinkhorn's "
        f"parity mask lands in programmable constant register 11, which is LCONST_neg1, "
        f"and nothing restores it."
    )
    assert torch.equal(res, golden), (
        f"floor mismatched on {(res != golden).sum().item()}/{res.numel()} lanes "
        f"(max deviation {(res - golden).abs().max().item():.6g}) without matching trunc "
        f"either, so the defect is not the LCONST_neg1 clobber alone."
    )
