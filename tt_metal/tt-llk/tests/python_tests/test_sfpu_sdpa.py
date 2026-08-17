# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Metal's llk_sfpu/ckernel_sfpu_sdpa.h.

Those kernels have no LLK API of their own. Each consumer declares its own wrapper, so
sources/sfpu_sdpa_test.cpp declares the same one ttnn's SDPA uses and this file drives it.
"""

import struct
from enum import Enum
from typing import NamedTuple

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    ELEMENTS_PER_TILE,
    TILE_DIM,
    SdpaCorrectionGolden,
    SdpaSfpuGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    DestSync,
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
    SDPA_OP,
    SDPA_SOFTPLUS_PARAMS,
)
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

TILE_DIMENSIONS = [TILE_DIM, TILE_DIM]

RECIP_OPS = (SdpaOp.RecipLegacy, SdpaOp.RecipIter)
EXP_OPS = (SdpaOp.ExpAccurate, SdpaOp.ExpPoly)

# The exp bodies take their scale as a uint16_t bf16 pattern, so only bf16-exact values
# keep the golden aligned with the kernel. 0xBF80 is -1.0, which ttnn's sigmoid_sub passes;
# 0x3F80 is +1.0, the unscaled case; 0x3E80 is 0.25, which catches a squared or dropped
# scale that is invisible at |scale| == 1.
EXP_SCALE_BF16_VALUES = (0xBF80, 0x3F80, 0x3E80)  # (-1.0, +1.0, +0.25)

APPROX_MODES = (ApproximationMode.No, ApproximationMode.Yes)


class SoftplusParams(NamedTuple):
    beta: float
    threshold: float

    def __str__(self):
        return f"beta{self.beta:g}_thr{self.threshold:g}"


# At beta 1.0 both beta and its reciprocal are 1.0, so a dropped or swapped multiply is
# invisible; the beta 2.0 cases pin them. At threshold 20.0 the body's pass-through arm is
# unreachable, so the low-threshold cases take it. Both betas are paired with both thresholds.
SOFTPLUS_LOW_THRESHOLD = 2.0 + 4.0 / ELEMENTS_PER_TILE

SOFTPLUS_CONFIGS = (
    SoftplusParams(1.0, 20.0),
    SoftplusParams(2.0, 20.0),
    SoftplusParams(1.0, SOFTPLUS_LOW_THRESHOLD),
    SoftplusParams(2.0, SOFTPLUS_LOW_THRESHOLD),
)


class Precision(Enum):
    """Format and dest accumulation pairings, so only legal ones are generated."""

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


CORRECTION_NUM_TILES = 5
CORRECTION_TILE_NAMES = ("prev_max", "worker_max", "cur_max", "prev_sum", "worker_sum")
CORRECTION_SCALE_BF16_VALUES = (0x3F80, 0x3E80)  # (1.0, 0.25)


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _bf16_to_float(pattern: int) -> float:
    return struct.unpack("<f", struct.pack("<I", pattern << 16))[0]


def _ramp(lo: float, hi: float) -> torch.Tensor:
    """1024 distinct values ascending across [lo, hi)."""
    return lo + (hi - lo) * torch.arange(ELEMENTS_PER_TILE, dtype=torch.float32) / (
        ELEMENTS_PER_TILE
    )


# (atol, rtol) per path, derived from measurements: rtol is 2x the largest relative error on that
# path, and atol is 2x the largest absolute error but capped at 1% of the smallest output the path
# produces, so the floor doesn't decide a comparison.

# Most paths that pack into bf16 share one pair because packer rounding is the main source of error.
# The exceptions are the two paths that are not limited by the rounding precision: correction
# accumulates across five tiles, and RecipLegacy under APPROX is a 7-bit arecip with no Newton step.
_TOLERANCES = {
    # op, approx (None where the body ignores it): bf16-packed, fp32 end-to-end
    (SdpaOp.Correction, None): ((1.5e-4, 2.5e-2), (1.0e-6, 2.5e-7)),
    (SdpaOp.ExpAccurate, None): ((4.0e-6, 1.2e-2), (1.2e-7, 2.5e-7)),
    (SdpaOp.ExpPoly, None): ((4.0e-6, 1.2e-2), (4.0e-6, 8.0e-6)),
    (SdpaOp.Softplus, None): ((1.0e-4, 1.0e-2), (1.0e-4, 4.0e-3)),
    (SdpaOp.RecipIter, False): ((2.5e-3, 1.2e-2), (1.2e-7, 2.5e-7)),
    (SdpaOp.RecipIter, True): ((2.5e-3, 1.2e-2), (2.5e-3, 1.2e-2)),
    (SdpaOp.RecipLegacy, False): ((2.5e-3, 1.2e-2), (1.5e-3, 3.0e-3)),
    (SdpaOp.RecipLegacy, True): ((2.5e-3, 1.0e-1), (2.5e-3, 8.0e-2)),
}


def _tolerance(op, precision, approx_mode):
    """(atol, rtol) for one path."""
    approx = approx_mode == ApproximationMode.Yes if op in RECIP_OPS else None
    bf16_packed, fp32_e2e = _TOLERANCES[(op, approx)]
    return fp32_e2e if precision == Precision.Fp32E2E else bf16_packed


def _footprint_violations(src_2d, res_2d):
    """Columns outside the written footprint that the body changed, if any."""
    written = SdpaSfpuGolden.TRANSFORMED_COLS
    untouched = [c for c in range(TILE_DIM) if c not in written]
    changed = res_2d.to(torch.float32) != src_2d.to(torch.float32)
    return [c for c in untouched if bool(changed[:, c].any())]


class Variant(NamedTuple):
    """One build of the single-tile bodies. Each axis defaults for the bodies that ignore it."""

    op: SdpaOp
    exp_scale_bf16: int = EXP_SCALE_BF16_VALUES[1]
    approx_mode: ApproximationMode = ApproximationMode.No
    softplus: SoftplusParams = SOFTPLUS_CONFIGS[0]
    dest_sync: DestSync = DestSync.Half

    def __str__(self):
        parts = [self.op.name]
        if self.op in EXP_OPS:
            parts.append(f"scale{_bf16_to_float(self.exp_scale_bf16):g}")
        elif self.op == SdpaOp.Softplus:
            parts.append(str(self.softplus))
        elif self.op in RECIP_OPS:
            parts.append(f"apx{self.approx_mode.name}")
        if self.dest_sync != DestSync.Half:
            parts.append(self.dest_sync.name)
        return "_".join(parts)


def _stimulus(variant: Variant) -> torch.Tensor:
    op = variant.op

    if op in RECIP_OPS:
        # Magnitudes in [1.25, 5.0). Kept away from zero because these bodies run 1 to 3 Newton
        # iterations off a setexp seed rather than a correctly rounded divide, and 1/x near zero
        # would let the tolerance decide the result instead of the kernel. Starting above 1.0
        # keeps any element from being its own reciprocal, which would be written yet compare
        # equal to the input and read as a footprint gap.

        # RecipIter returns 1/x, RecipLegacy returns |1/x|. Goldens are written to match that.
        # Sign alternates by row, not by flat index, because the kernel writes every other row.
        magnitudes = _ramp(1.25, 5.0)
        rows = torch.arange(ELEMENTS_PER_TILE) // TILE_DIM
        signs = torch.where(rows % 2 == 0, torch.tensor(1.0), torch.tensor(-1.0))
        return magnitudes * signs

    if op in EXP_OPS:
        # Size the input range from the scale so scale*x spans [-8, 0] whichever way the scale
        # points. That is the post-max-subtraction regime SDPA feeds these bodies, and it keeps
        # exp in [3.4e-4, 1] rather than overflowing. Deriving the range from the scale also
        # means a dropped or squared multiply moves the whole output range, not just individual
        # elements.
        scale = _bf16_to_float(variant.exp_scale_bf16)
        span = 8.0 / abs(scale)
        return _ramp(0.0, span) if scale < 0.0 else _ramp(-span, 0.0)

    # Softplus. The bound keeps |beta*x| inside the residual polynomial's [0, 5] fit domain, so
    # the test measures the polynomial rather than the clamp beyond it. Where beta*x clears the
    # threshold the body writes nothing and the golden returns x, so a low threshold covers the
    # pass-through arm without needing its own stimulus.
    bound = 4.0 / variant.softplus.beta
    return _ramp(-bound, bound)


def _templates(variant: Variant):
    """Return all the C++ template parameters."""
    return [
        SDPA_OP(variant.op),
        APPROX_MODE(variant.approx_mode),
        DEST_SYNC(variant.dest_sync),
        SDPA_EXP_SCALE(scale_bf16=variant.exp_scale_bf16),
        SDPA_SOFTPLUS_PARAMS(
            softplus_beta_bits=_f32_bits(variant.softplus.beta),
            softplus_beta_reciprocal_bits=_f32_bits(1.0 / variant.softplus.beta),
            softplus_threshold_bits=_f32_bits(variant.softplus.threshold),
        ),
    ]


def _variants():
    """Variants for the single-tile bodies, listed per body rather than crossed.

    approx_mode is live only for the reciprocal bodies: the exp bodies read their own
    SDPA_EXP_APPROX_MODE template argument, and calculate_softplus_body takes an
    APPROXIMATION_MODE parameter it never uses. Likewise only the exp bodies vary with the
    scale and only softplus with beta/threshold, so crossing every axis over every body would
    just rebuild identical kernels. dest_sync moves the Dest base the params wrapper hands the
    body, so each body gets one DestSync.Full variant to catch a base that only works for Half.
    """
    variants = [
        Variant(op, approx_mode=approx) for op in RECIP_OPS for approx in APPROX_MODES
    ]
    variants += [
        Variant(op, exp_scale_bf16=scale)
        for op in EXP_OPS
        for scale in EXP_SCALE_BF16_VALUES
    ]
    variants += [Variant(SdpaOp.Softplus, softplus=cfg) for cfg in SOFTPLUS_CONFIGS]
    variants += [
        Variant(op, dest_sync=DestSync.Full) for op in SdpaOp if op != SdpaOp.Correction
    ]
    return variants


@parametrize(
    variant=_variants(),
    precision=list(Precision),
)
def test_sfpu_sdpa(variant, precision):
    op = variant.op

    formats = InputOutputFormat(precision.data_format, precision.data_format)
    torch_format = format_dict[formats.input_format]

    src_A = _stimulus(variant).to(torch_format)
    src_B = torch.zeros_like(src_A)

    golden_tensor = get_golden_generator(SdpaSfpuGolden)(
        src_A.view(TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]),
        op,
        exp_scale=_bf16_to_float(variant.exp_scale_bf16),
        softplus_beta=variant.softplus.beta,
        softplus_threshold=variant.softplus.threshold,
    )

    src_A_tilized = tilize_block(
        src_A, TILE_DIMENSIONS, stimuli_format=formats.input_format
    ).flatten()

    configuration = TestConfig(
        "sources/sfpu_sdpa_test.cpp",
        formats,
        templates=_templates(variant),
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
        f"column-vector body wrote outside its footprint: columns {unexpected} changed "
        f"but only {list(SdpaSfpuGolden.TRANSFORMED_COLS)} should be written"
    )

    atol, rtol = _tolerance(op, precision, variant.approx_mode)
    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        custom_atol=atol,
        custom_rtol=rtol,
    ), (
        f"column-vector SFPU result does not match golden "
        f"(atol={atol:g}, rtol={rtol:g} for {variant}/{precision.label})"
    )


def _correction_stimulus_tiles(torch_format):
    """
    Five input tiles, chosen so every branch and every in-place write is observable.

    prev_max ascends while worker_max descends, so they cross mid-tile and the device's
    `v_if(prev_max < worker_max)` takes both branches within one dispatch. Elements near
    the crossover sit close enough together to exercise a near-tie.

    The cur_max seed sits far below any achievable max. Its output is overwritten
    unconditionally, so a seed that could coincide with the answer would let a
    failure-to-write read as a pass.

    Both sums are positive and differ from each other, which keeps the two correction
    products and their sum distinguishable.
    """
    tiles = {
        "prev_max": _ramp(-2.0, 2.0),
        "worker_max": _ramp(2.0, -2.0),
        "cur_max": _ramp(-100.0, -90.0),
        "prev_sum": _ramp(0.5, 4.5),
        "worker_sum": _ramp(4.5, 0.5),
    }
    return [tiles[name].to(torch_format) for name in CORRECTION_TILE_NAMES]


def _dest_configurations():
    """Dest configurations that can hold five tiles."""
    return [
        (Precision.Bf16Dest, DestSync.Half),
        (Precision.Bf16Dest, DestSync.Full),
        (Precision.Fp32Dest, DestSync.Full),
        (Precision.Fp32E2E, DestSync.Full),
    ]


# The correction body reads no APPROX, so it takes the Variant default rather than sweeping.
@parametrize(
    dest_config=_dest_configurations(),
    scale_bf16=list(CORRECTION_SCALE_BF16_VALUES),
)
def test_sfpu_sdpa_correction(dest_config, scale_bf16):
    precision, dest_sync = dest_config

    formats = InputOutputFormat(precision.data_format, precision.data_format)
    torch_format = format_dict[formats.input_format]

    src_tiles = _correction_stimulus_tiles(torch_format)
    src_B = torch.zeros(ELEMENTS_PER_TILE, dtype=torch_format)

    golden_tiles = get_golden_generator(SdpaCorrectionGolden)(
        [t.view(TILE_DIM, TILE_DIM) for t in src_tiles],
        scale=_bf16_to_float(scale_bf16),
    )

    src_A_tilized = torch.cat(
        [
            tilize_block(
                t, TILE_DIMENSIONS, stimuli_format=formats.input_format
            ).flatten()
            for t in src_tiles
        ]
    )

    configuration = TestConfig(
        "sources/sfpu_sdpa_test.cpp",
        formats,
        templates=_templates(
            Variant(SdpaOp.Correction, exp_scale_bf16=scale_bf16, dest_sync=dest_sync)
        ),
        variant_stimuli=StimuliConfig(
            src_A_tilized,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=CORRECTION_NUM_TILES,
            tile_count_B=1,
            tile_count_res=CORRECTION_NUM_TILES,
        ),
        dest_acc=precision.dest_acc,
        unpack_to_dest=precision.unpack_to_dest,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    res_flat = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tiles = [
        untilize_block(
            res_flat[i * ELEMENTS_PER_TILE : (i + 1) * ELEMENTS_PER_TILE],
            formats.output_format,
            TILE_DIMENSIONS,
        )
        for i in range(CORRECTION_NUM_TILES)
    ]

    for name, src, res, golden in zip(
        CORRECTION_TILE_NAMES, src_tiles, res_tiles, golden_tiles
    ):
        # All five regions are addressed by the same strided walk, so a write outside the
        # footprint in any one of them is a stride bug.
        unexpected = _footprint_violations(src.view(TILE_DIM, TILE_DIM), res)
        assert not unexpected, (
            f"correction body wrote outside its footprint in the {name} region: "
            f"columns {unexpected} changed but only "
            f"{list(SdpaSfpuGolden.TRANSFORMED_COLS)} should be written"
        )

        if name == "cur_max":
            # cur_max is an eltwise max of two representable inputs, with no
            # arithmetic and nothing to round, so it is checked exactly.
            cols = torch.tensor(SdpaSfpuGolden.TRANSFORMED_COLS, dtype=torch.long)
            assert torch.equal(
                res[:, cols].to(torch.float32), golden[:, cols].to(torch.float32)
            ), "correction cur_max must be an exact elementwise max"
            continue

        atol, rtol = _tolerance(SdpaOp.Correction, precision, ApproximationMode.No)
        assert passed_test(
            golden,
            res,
            formats.output_format,
            custom_atol=atol,
            custom_rtol=rtol,
        ), (
            f"correction result does not match golden in the {name} region "
            f"(atol={atol:g}, rtol={rtol:g} for {precision.label})"
        )
