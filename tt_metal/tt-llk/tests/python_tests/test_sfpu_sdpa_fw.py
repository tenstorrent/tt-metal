# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the column-vector SFPU bodies in metal's
experimental/llk_sfpu/ckernel_sfpu_sdpa_fw.h.

Those bodies have no LLK API of their own; the consumer declares its own wrapper, so
sources/sfpu_sdpa_fw_test.cpp declares one and this file drives it. The header holds two
bodies, both eltwise-unary shaped and both working on one DEST tile in place:

    Recip  sfpu_reciprocal_iter<2> on an fp32 dest, else <1> plus a round to bf16
    Exp    _ckernel_sfpu_exp_accurate_ with SCALE_EN, scale as a std::uint16_t bf16 pattern

Both are narrower than their ckernel_sfpu_sdpa.h namesakes: neither reads APPROX, recip has
no legacy_compat parameter, and exp has no polynomial branch. There is therefore no
approx_mode axis here, and dest_acc is the only config that selects code.

Both write the same footprint as the ckernel_sfpu_sdpa.h bodies: columns
{0,2,4,6,8,10,12,14} of all 32 rows, with the rest of the tile left untouched.

Only calculate_recip_first_column has a consumer today, in tt-train's sdpa_fw
(sdpa_compute_utils.hpp:422). The exp body has none: tt-train's exp paths use their own
bodies in metal/common/sdpa_compute_utils_common.hpp:63, and ttnn uses the differently
signatured calculate_exponential_first_column<bool, std::uint16_t> from ckernel_sfpu_sdpa.h.
Nothing outside this test will notice if the exp body breaks.
"""

import struct
from enum import Enum

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    TILE_DIM,
    SdpaSfpuGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    SdpaOp,
    DestAccumulation,
    DestSync,
    SdpaFwOp,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    SDPA_EXP_SCALE,
    DEST_SYNC,
    SDPA_FW_OP,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

TILE_DIMENSIONS = [TILE_DIM, TILE_DIM]

# Each _fw body computes what one ckernel_sfpu_sdpa.h body computes, so SdpaSfpuGolden
# serves both headers rather than being duplicated: _fw's recip is
# calculate_recip_first_column<false> at APPROX=false, and _fw's exp is
# calculate_exponential_first_column<true, scale>. The goldens are plain torch.reciprocal and
# torch.exp(scale*x) in any case, so the mapping only picks which branch to take.
GOLDEN_OP = {
    SdpaFwOp.Recip: SdpaOp.RecipIter,
    SdpaFwOp.Exp: SdpaOp.ExpAccurate,
}

# The exp body takes its scale as a std::uint16_t bf16 pattern, so only bf16-exact values keep the
# golden aligned with the kernel. 0xBF80 is -1.0, which ttnn's sigmoid_sub passes
# (compute_common.hpp:875); 0x3F80 is +1.0, the unscaled case. 0x3E80 is 0.25, included because at
# |scale| == 1 a squared or dropped scale is invisible, so the other two only catch a sign error.
EXP_SCALE_BF16_VALUES = (0xBF80, 0x3F80, 0x3E80)


class Precision(Enum):
    """Format and dest-accumulation pairings, as one axis so only legal ones are generated.

    Bf16Dest and Fp32Dest are what a consumer runs. Fp32E2E is here because at Float16_b
    output the packer's bf16 rounding (2^-8 relative) dominates both of these kernels' own
    error, so only with fp32 all the way through is the kernel's own accuracy visible to a
    tolerance.

    Fp32E2E sets unpack_to_dest because a Float32 operand routed through SrcA would be
    truncated to that register's narrower mantissa before the SFPU saw it.
    """

    Bf16Dest = ("bf16_dest", DataFormat.Float16_b, DestAccumulation.No, False)
    Fp32Dest = ("fp32_dest", DataFormat.Float16_b, DestAccumulation.Yes, False)
    Fp32E2E = ("fp32_e2e", DataFormat.Float32, DestAccumulation.Yes, True)

    def __init__(self, label, data_format, dest_acc, unpack_to_dest):
        self.label = label
        self.data_format = data_format
        self.dest_acc = dest_acc
        self.unpack_to_dest = unpack_to_dest

    def __str__(self):
        return self.label


def _bf16_to_float(pattern: int) -> float:
    """A bf16 pattern is the top 16 bits of the equivalent fp32, so widening is exact."""
    return struct.unpack("<f", struct.pack("<I", pattern << 16))[0]


def _ramp(lo: float, hi: float) -> torch.Tensor:
    """1024 distinct values ascending across [lo, hi).

    Distinct per element so the footprint check has something to catch: a constant stimulus
    passes even when the body walks the wrong rows.
    """
    return lo + (hi - lo) * torch.arange(ELEMENTS_PER_TILE, dtype=torch.float32) / (
        ELEMENTS_PER_TILE
    )


def _tolerance(op, precision):
    """Per-path (atol, rtol), each sized from measured error.

    torch.isclose accepts |golden - res| <= atol + rtol*|res|, so atol covers the near-zero
    floor while rtol carries magnitude. atol therefore stays small even where the measured
    maximum absolute error is large, because that maximum occurs at the largest outputs,
    where the rtol term already dominates.

    Measured maxima, worst over both scales, as absolute / relative:

              Bf16Dest             Fp32Dest             Fp32E2E
      Recip   1.9e-03 / 3.9e-03    1.9e-03 / 3.9e-03    6.0e-08 / 1.2e-07
      Exp     3.0e-03 / 5.1e-03    1.9e-03 / 3.8e-03    6.0e-08 / 1.1e-07

    Bf16Dest and Fp32Dest share one tolerance because both pack to Float16_b, and the two
    columns show why: a wider dest buys recip nothing at all and exp under a factor of two,
    because the packer's bf16 rounding (2^-8 relative) is the floor either way. Only Fp32E2E
    exposes the kernels themselves. Both reach fp32 round-off there, recip through two Newton
    iterations and exp through the 21-bit accurate kernel.

    DestSync.Half and DestSync.Full produce bit-identical results at every scale and
    precision, so dest_sync does not enter the tolerance.
    """
    fp32 = precision == Precision.Fp32E2E

    if op == SdpaFwOp.Recip:
        # Outputs stay in [0.2, 0.8], so atol is half a percent of the smallest of them and rtol
        # carries the check.
        return (1e-7, 1e-6) if fp32 else (1e-3, 1.0e-2)

    # exp reaches down to 3.4e-4, where an atol of 1e-3 would accept any answer at all. bf16
    # quantization is relative, so a small atol costs nothing: the largest absolute errors sit at
    # outputs near 1, which rtol already covers.
    return (1e-7, 1e-6) if fp32 else (1e-5, 1.2e-2)


def _footprint_violations(src_2d, res_2d):
    """Columns outside the written footprint that the body changed anyway.

    Compared bit-exactly: the untouched columns are a pass-through, so a tolerance there
    would hide a wrong dst_reg stride or iteration count.
    """
    written = SdpaSfpuGolden.TRANSFORMED_COLS
    untouched = [c for c in range(TILE_DIM) if c not in written]
    changed = res_2d.to(torch.float32) != src_2d.to(torch.float32)
    return [c for c in untouched if bool(changed[:, c].any())]


def _stimulus(op, exp_scale_bf16: int) -> torch.Tensor:
    if op == SdpaFwOp.Recip:
        # Magnitudes in [1.25, 5.0), alternating sign. Kept away from zero because the body
        # runs one or two Newton iterations off a setexp seed rather than a correctly rounded
        # divide, and 1/x near zero would let the tolerance decide the result instead of the
        # kernel. Starting above 1.0 keeps any element from being its own reciprocal, which
        # would be written yet compare equal to the input and read as a footprint gap.
        #
        # Sign alternates by row, not by flat index. Flat index i sits at column i % TILE_DIM,
        # so alternating on i makes every negative land in an odd column -- and the written
        # footprint is all even columns, so the body would only ever see positive inputs and
        # the reciprocal's sign handling would go untested.
        magnitudes = _ramp(1.25, 5.0)
        rows = torch.arange(ELEMENTS_PER_TILE) // TILE_DIM
        signs = torch.where(rows % 2 == 0, torch.tensor(1.0), torch.tensor(-1.0))
        return magnitudes * signs

    # Size the input range from the scale so that scale*x spans [-8, 0] at every scale, whichever
    # way the scale points. That is the post-max-subtraction regime a softmax feeds this body, and
    # it holds exp in [3.4e-4, 1] rather than overflowing. Deriving the range from the scale instead
    # of fixing it also means the kernel has to apply the scale to land in that regime at all: a
    # dropped or squared multiply moves the output range, not just individual elements.
    scale = _bf16_to_float(exp_scale_bf16)
    span = 8.0 / abs(scale)
    return _ramp(0.0, span) if scale < 0.0 else _ramp(-span, 0.0)


def _variants():
    """(op, exp_scale_bf16, dest_sync) triples, listed rather than crossed.

    Only the exp body varies with the scale, so recip is run once. dest_sync moves the DEST
    base the params wrapper hands the body, so each op gets one DestSync.Full variant to
    catch a base that only works for Half; crossing it over everything would double the
    suite to exercise one address.
    """
    return [
        (SdpaFwOp.Recip, EXP_SCALE_BF16_VALUES[0], DestSync.Half),
        (SdpaFwOp.Recip, EXP_SCALE_BF16_VALUES[0], DestSync.Full),
        *((SdpaFwOp.Exp, scale, DestSync.Half) for scale in EXP_SCALE_BF16_VALUES),
        (SdpaFwOp.Exp, EXP_SCALE_BF16_VALUES[-1], DestSync.Full),
    ]


# dest_acc selects code, not just a storage width: recip picks sfpu_reciprocal_iter<2> against
# <1>-plus-bf16-round, and exp picks an fp32 or bf16 load/store mode inside
# _ckernel_sfpu_exp_accurate_. It is swept through Precision, which pairs it with the operand
# format so only legal combinations are generated.
#
# dst_index is fixed at 0. Both bodies address dst_reg[0] relative to a base the params wrapper has
# already applied, so a nonzero index would exercise that wrapper rather than this header, and
# every other SFPU test covers it.
@parametrize(
    variant=_variants(),
    precision=list(Precision),
)
def test_sfpu_sdpa_fw(variant, precision):
    op, exp_scale_bf16, dest_sync = variant

    formats = InputOutputFormat(precision.data_format, precision.data_format)
    torch_format = format_dict[formats.input_format]

    src_A = _stimulus(op, exp_scale_bf16).to(torch_format)
    src_B = torch.zeros_like(src_A)

    golden_tensor = get_golden_generator(SdpaSfpuGolden)(
        src_A.view(TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]),
        GOLDEN_OP[op],
        exp_scale=_bf16_to_float(exp_scale_bf16),
    )

    src_A_tilized = tilize_block(
        src_A, TILE_DIMENSIONS, stimuli_format=formats.input_format
    ).flatten()

    configuration = TestConfig(
        "sources/sfpu_sdpa_fw_test.cpp",
        formats,
        templates=[
            SDPA_FW_OP(op),
            DEST_SYNC(dest_sync),
            SDPA_EXP_SCALE(scale_bf16=exp_scale_bf16),
        ],
        variant_stimuli=StimuliConfig(
            src_A_tilized,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
        ),
        dest_acc=precision.dest_acc,
        unpack_to_dest=precision.unpack_to_dest,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = untilize_block(res_tensor, formats.output_format, TILE_DIMENSIONS)

    unexpected = _footprint_violations(src_A.view(TILE_DIM, TILE_DIM), res_tensor)
    assert not unexpected, (
        f"sdpa_fw body wrote outside its footprint: columns {unexpected} changed "
        f"but only {list(SdpaSfpuGolden.TRANSFORMED_COLS)} should be written"
    )

    atol, rtol = _tolerance(op, precision)
    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        custom_atol=atol,
        custom_rtol=rtol,
    ), (
        f"sdpa_fw SFPU result does not match golden "
        f"(atol={atol:g}, rtol={rtol:g} for {op.name}/{precision.label})"
    )
