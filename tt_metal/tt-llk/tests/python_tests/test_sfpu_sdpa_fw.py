# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Metal's llk_sfpu/ckernel_sfpu_sdpa_fw.h.

Those kernels have no LLK API of their own. Each consumer declares its own wrapper, so
sources/sfpu_sdpa_fw_test.cpp declares one and this file drives it.
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
    ApproximationMode,
    DestAccumulation,
    DestSync,
    SdpaFwOp,
    SdpaOp,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DEST_SYNC,
    SDPA_EXP_SCALE,
    SDPA_FW_OP,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

TILE_DIMENSIONS = [TILE_DIM, TILE_DIM]

# Same golden as ckernel_sfpu_sdpa.h, this just picks which branch to take.
GOLDEN_OP = {
    SdpaFwOp.Recip: SdpaOp.RecipIter,
    SdpaFwOp.Exp: SdpaOp.ExpAccurate,
}

EXP_SCALE_BF16_VALUES = (0xBF80, 0x3F80, 0x3E80)  # (-1.0, +1.0, +0.25)


class Precision(Enum):
    """
    Format and dest accumulation pairings, so only legal ones are generated.
    Fp32E2E needs unpack_to_dest.
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
    return struct.unpack("<f", struct.pack("<I", pattern << 16))[0]


def _ramp(lo: float, hi: float) -> torch.Tensor:
    """1024 distinct values ascending across [lo, hi)."""
    return lo + (hi - lo) * torch.arange(ELEMENTS_PER_TILE, dtype=torch.float32) / (
        ELEMENTS_PER_TILE
    )


def _tolerance(op, precision):
    """
    (atol, rtol) based on measured values for each path.

    Bf16Dest and Fp32Dest share one tolerance because both pack to Float16_b, where the
    packer's bf16 rounding is the error floor either way. Only Fp32E2E exposes the kernels
    themselves, and both reach fp32 round-off there
    """
    fp32 = precision == Precision.Fp32E2E

    if op == SdpaFwOp.Recip:
        return (1e-7, 1e-6) if fp32 else (1e-3, 1.0e-2)

    # exp reaches down to 3.4e-4, where an atol of 1e-3 would accept any answer at all.
    return (1e-7, 1e-6) if fp32 else (1e-5, 1.2e-2)


def _footprint_violations(src_2d, res_2d):
    """Columns outside the written footprint that the body changed, if any."""
    written = SdpaSfpuGolden.TRANSFORMED_COLS
    untouched = [c for c in range(TILE_DIM) if c not in written]
    changed = res_2d.to(torch.float32) != src_2d.to(torch.float32)
    return [c for c in untouched if bool(changed[:, c].any())]


def _stimulus(op, exp_scale_bf16: int) -> torch.Tensor:
    if op == SdpaFwOp.Recip:
        # Magnitudes in [1.25, 5.0), alternating sign. Kept away from zero because the body
        # runs one or two Newton iterations off a setexp seed rather than a correctly rounded
        # divide. Starting above 1.0 keeps any element from being its own reciprocal, which
        # would be written yet compare equal to the input and read as a footprint gap.
        #
        # Sign alternates by row, so flat index i sits at column i % TILE_DIM.
        magnitudes = _ramp(1.25, 5.0)
        rows = torch.arange(ELEMENTS_PER_TILE) // TILE_DIM
        signs = torch.where(rows % 2 == 0, torch.tensor(1.0), torch.tensor(-1.0))
        return magnitudes * signs

    # Size the input range from the scale so that scale*x spans [-8, 0] whichever way the scale
    # points. That is the post-max-subtraction regime a softmax feeds this body, and it holds exp
    # in [3.4e-4, 1] rather than overflowing. Deriving the range from the scale also means a
    # dropped or squared multiply moves the whole output range, not just individual elements.
    scale = _bf16_to_float(exp_scale_bf16)
    span = 8.0 / abs(scale)
    return _ramp(0.0, span) if scale < 0.0 else _ramp(-span, 0.0)


def _variants():
    """
    Valid variants per op.

    Only the exp body varies with the scale, so recip is run once per approx mode. dest_sync
    only moves the Dest base, so each op gets one DestSync.Full variant rather than crossing it
    over everything.
    """
    return [
        *(
            (SdpaFwOp.Recip, EXP_SCALE_BF16_VALUES[0], DestSync.Half, approx)
            for approx in ApproximationMode
        ),
        (SdpaFwOp.Recip, EXP_SCALE_BF16_VALUES[0], DestSync.Full, ApproximationMode.No),
        *(
            (SdpaFwOp.Exp, scale, DestSync.Half, ApproximationMode.No)
            for scale in EXP_SCALE_BF16_VALUES
        ),
        (SdpaFwOp.Exp, EXP_SCALE_BF16_VALUES[-1], DestSync.Full, ApproximationMode.No),
    ]


@parametrize(
    variant=_variants(),
    precision=list(Precision),
)
def test_sfpu_sdpa_fw(variant, precision):
    op, exp_scale_bf16, dest_sync, approx_mode = variant

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
            APPROX_MODE(approx_mode),
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
