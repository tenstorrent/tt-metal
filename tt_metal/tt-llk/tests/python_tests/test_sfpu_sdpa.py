# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Metal's llk_sfpu/ckernel_sfpu_sdpa.h.

Those kernels have no LLK API of their own. Each consumer declares its own wrapper, so
sources/sfpu_sdpa_test.cpp declares the same one ttnn's SDPA uses and this file drives it.
"""

import struct
from enum import Enum

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
# keep the golden aligned with the kernel. 0xBF80 is -1.0, which ttnn's sigmoid_sub passes.
# 0x3F80 is +1.0, the unscaled case.
EXP_SCALE_BF16_VALUES = (0xBF80, 0x3F80)  # (-1.0, +1.0)

SOFTPLUS_BETA = 1.0
SOFTPLUS_THRESHOLD = 20.0


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


def _tolerance(op, precision, approx_mode):
    """(atol, rtol) based on measured values for each path."""
    approx = approx_mode == ApproximationMode.Yes
    fp32 = precision == Precision.Fp32E2E

    if op == SdpaOp.Correction:
        return (1e-6, 1e-6) if fp32 else (5e-3, 2.0e-2)
    if op == SdpaOp.Softplus:
        return (5e-4, 4.0e-3) if fp32 else (2e-3, 1.0e-2)
    if op == SdpaOp.ExpAccurate:
        return (1e-7, 1e-6) if fp32 else (1e-3, 1.2e-2)
    if op == SdpaOp.ExpPoly:
        return (1e-6, 1e-5) if fp32 else (1e-3, 1.2e-2)
    if op == SdpaOp.RecipIter:
        if approx:
            return (1e-4, 1.2e-2) if fp32 else (1e-3, 1.2e-2)
        return (1e-7, 1e-6) if fp32 else (1e-3, 1.0e-2)
    if approx:
        return (2e-3, 8.0e-2)
    return (1e-4, 4.0e-3) if fp32 else (1e-3, 1.0e-2)


def _footprint_violations(src_2d, res_2d):
    """Columns outside the written footprint that the body changed, if any."""
    written = SdpaSfpuGolden.TRANSFORMED_COLS
    untouched = [c for c in range(TILE_DIM) if c not in written]
    changed = res_2d.to(torch.float32) != src_2d.to(torch.float32)
    return [c for c in untouched if bool(changed[:, c].any())]


def _stimulus(op, exp_scale_bf16: int) -> torch.Tensor:
    if op in RECIP_OPS:
        # Magnitudes in [1.25, 5.0). Kept away from zero because these bodies run 1 to 3 Newton
        # iterations off a setexp seed rather than a correctly rounded divide, and 1/x near zero
        # would let the tolerance decide the result instead of the kernel. Starting above 1.0
        # keeps any element from being its own reciprocal, which would be written yet compare
        # equal to the input and read as a footprint gap.
        magnitudes = _ramp(1.25, 5.0)

        if op is SdpaOp.RecipLegacy:
            # Positive only. The kernel effectively returns the absolute value.
            return magnitudes

        # RecipIter handles the sign itself, so it gets both positives and negatives.
        # Sign alternates by row, so flat index i sits at column i % TILE_DIM.
        rows = torch.arange(ELEMENTS_PER_TILE) // TILE_DIM
        signs = torch.where(rows % 2 == 0, torch.tensor(1.0), torch.tensor(-1.0))
        return magnitudes * signs

    if op in EXP_OPS:
        # Take the input range from the sign of the scale so scale*x lands in [-8, 0]
        # either way. That is the post-max-subtraction regime SDPA feeds these bodies, and
        # it keeps exp in [3.4e-4, 1] rather than overflowing.
        scale = _bf16_to_float(exp_scale_bf16)
        return _ramp(0.0, 8.0) if scale < 0.0 else _ramp(-8.0, 0.0)

    # Softplus. Stay inside the degree-6 residual polynomial's [0, 5] fit domain,
    # so the test measures the polynomial rather than the clamp to zero beyond it.
    return _ramp(-4.0, 4.0)


def _templates(op, exp_scale_bf16: int, approx_mode, dest_sync=DestSync.Half):
    """Return all the C++ template parameters."""
    return [
        SDPA_OP(op),
        APPROX_MODE(approx_mode),
        DEST_SYNC(dest_sync),
        SDPA_EXP_SCALE(scale_bf16=exp_scale_bf16),
        SDPA_SOFTPLUS_PARAMS(
            beta_bits=_f32_bits(SOFTPLUS_BETA),
            beta_reciprocal_bits=_f32_bits(1.0 / SOFTPLUS_BETA),
            threshold_bits=_f32_bits(SOFTPLUS_THRESHOLD),
        ),
    ]


def _variants():
    """(op, exp_scale_bf16) pairs for the single-tile bodies."""
    variants = []
    for op in SdpaOp:
        if op == SdpaOp.Correction:
            continue
        if op in EXP_OPS:
            variants.extend((op, scale) for scale in EXP_SCALE_BF16_VALUES)
        else:
            variants.append((op, EXP_SCALE_BF16_VALUES[0]))
    return variants


@parametrize(
    op_and_scale=_variants(),
    precision=list(Precision),
    approx_mode=[ApproximationMode.No, ApproximationMode.Yes],
)
def test_sfpu_sdpa(op_and_scale, precision, approx_mode):
    op, exp_scale_bf16 = op_and_scale
    dest_acc = precision.dest_acc
    torch.manual_seed(0)

    formats = InputOutputFormat(precision.data_format, precision.data_format)
    torch_format = format_dict[formats.input_format]

    src_A = _stimulus(op, exp_scale_bf16).to(torch_format)
    src_B = torch.zeros_like(src_A)

    golden_tensor = get_golden_generator(SdpaSfpuGolden)(
        src_A.view(TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]),
        op,
        exp_scale=_bf16_to_float(exp_scale_bf16),
        softplus_beta=SOFTPLUS_BETA,
        softplus_threshold=SOFTPLUS_THRESHOLD,
    )

    src_A_tilized = tilize_block(
        src_A, TILE_DIMENSIONS, stimuli_format=formats.input_format
    ).flatten()

    configuration = TestConfig(
        "sources/sfpu_sdpa_test.cpp",
        formats,
        templates=_templates(op, exp_scale_bf16, approx_mode),
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
        dest_acc=dest_acc,
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

    atol, rtol = _tolerance(op, precision, approx_mode)
    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        custom_atol=atol,
        custom_rtol=rtol,
    ), (
        f"column-vector SFPU result does not match golden "
        f"(atol={atol:g}, rtol={rtol:g} for {op.name}/{precision.label}/"
        f"apx={approx_mode.name})"
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


@parametrize(
    dest_config=_dest_configurations(),
    scale_bf16=list(CORRECTION_SCALE_BF16_VALUES),
    approx_mode=[ApproximationMode.No],
)
def test_sfpu_sdpa_correction(dest_config, scale_bf16, approx_mode):
    precision, dest_sync = dest_config
    dest_acc = precision.dest_acc
    torch.manual_seed(0)

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
            SdpaOp.Correction, scale_bf16, approx_mode, dest_sync=dest_sync
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
        dest_acc=dest_acc,
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

        atol, rtol = _tolerance(SdpaOp.Correction, precision, approx_mode)
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
