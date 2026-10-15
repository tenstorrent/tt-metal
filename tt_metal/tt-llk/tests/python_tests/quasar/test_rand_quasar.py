# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math
import struct
from dataclasses import dataclass

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.llk_params import (
    ApproximationMode,
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathOperation,
    PerfRunType,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    QuasarSfpuVariant,
    generate_quasar_sfpu_format_variants,
    input_output_formats,
    parametrize,
    runtime,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    RAND_RANGE,
    RAND_SEED,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    FACE_C_DIM,
    MAX_NUM_FACES,
)


# rand has no element-wise golden, so these check its properties: range, uniformity, per-lane and
# per-row independence, and seed determinism.
@dataclass(frozen=True, repr=False)
class RandCase:
    name: str
    from_value: float
    scale: float
    formats: tuple  # DataFormats whose Dest can represent [from, from + scale]

    def __repr__(self) -> str:
        return self.name


_RAND_FLOAT_FORMATS = (DataFormat.Float16, DataFormat.Float32, DataFormat.Float16_b)

# Exponent <= 31 cannot absorb the 2^-31 normalization, forcing the per-row SFPMULI body.
_RAND_PER_ROW_NORMALIZE_SCALE = 2.0**-96

RAND_CASES = (
    RandCase("unit", 1.0, 2.0, _RAND_FLOAT_FORMATS),
    RandCase("signed", -4.0, 8.0, _RAND_FLOAT_FORMATS),
    # Float16 cannot represent 2^-96.
    RandCase(
        "per_row_normalize",
        0.0,
        _RAND_PER_ROW_NORMALIZE_SCALE,
        (DataFormat.Float32, DataFormat.Float16_b),
    ),
    RandCase("zero_scale", 1.5, 0.0, _RAND_FLOAT_FORMATS),
)

# rand does not depend on Dest sync, and 4 tiles fit one Dest section in every mode.
_RAND_DEST_SYNC = DestSync.Half
_RAND_DIMS = [64, 64]

# Outside every RandCase interval, so an unwritten element fails the range check.
_RAND_UNWRITTEN_SENTINEL = -100.0

_RAND_DEFAULT_SEED = 0x12345678

_RAND_MEAN_SIGMAS = 6.0
_RAND_STD_REL_TOL = 0.15
_RAND_HISTOGRAM_BINS = 8
_RAND_HISTOGRAM_MIN_FRACTION = 0.5
_RAND_HISTOGRAM_MAX_FRACTION = 1.5
_RAND_FP32_MIN_DISTINCT_FRACTION = 0.95
_RAND_NARROW_MIN_DISTINCT = 64  # 16-bit outputs
_RAND_ROW_MIN_DISTINCT_DIVISOR = 2
_RAND_COL_MIN_DISTINCT_DIVISOR = 4
# A missing finalizer makes a lane echo a nearby lane a few rows later: |r| = 0.128 on the emulator
# with it replaced by a plain copy, vs ~0.04 for the real kernel.
_RAND_CORR_MAX_ROW_OFFSET = 8
_RAND_CORR_MAX_COL_OFFSET = 4
_RAND_MAX_OFFSET_CORRELATION = 0.08


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def generate_rand_combinations():
    combinations = []
    for case in RAND_CASES:
        for variant in generate_quasar_sfpu_format_variants(
            MathOperation.Rand, input_output_formats(list(case.formats))
        ):
            combinations.append((case, variant, _RAND_DEST_SYNC, runtime(_RAND_DIMS)))
    return combinations


def _rand_output_ulp(magnitude: float, output_format: DataFormat) -> float:
    mantissa_bits = {
        DataFormat.Float16_b: 7,
        DataFormat.Float16: 10,
        DataFormat.Float32: 23,
    }[output_format]
    return (
        math.ldexp(1.0, math.frexp(magnitude)[1] - 1 - mantissa_bits)
        if magnitude
        else 0.0
    )


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / torch.sqrt((a * a).sum() * (b * b).sum()))


def _max_offset_correlation(face_rows: torch.Tensor):
    """Return (max |pearson r|, (dr, dc)) over the small row / column offsets."""
    rows, cols = face_rows.shape
    worst = (0.0, (0, 0))
    for dr in range(_RAND_CORR_MAX_ROW_OFFSET + 1):
        for dc in range(-_RAND_CORR_MAX_COL_OFFSET, _RAND_CORR_MAX_COL_OFFSET + 1):
            if dr == 0 and dc <= 0:
                continue  # (0, 0) is the element itself; (0, -dc) mirrors (0, dc)
            c0, c1 = max(0, -dc), min(cols, cols - dc)
            a = face_rows[: rows - dr, c0:c1]
            b = face_rows[dr:, c0 + dc : c1 + dc]
            r = abs(_pearson(a.flatten(), b.flatten()))
            if r > worst[0]:
                worst = (r, (dr, dc))
    return worst


def _check_rand_properties(values: torch.Tensor, case: RandCase, output_format):
    n = values.numel()
    lo = case.from_value
    hi = case.from_value + case.scale

    assert torch.isfinite(values).all(), "rand produced a non-finite value"

    out_of_range = ((values < lo) | (values > hi)).nonzero().flatten()
    assert out_of_range.numel() == 0, (
        f"{out_of_range.numel()} of {n} values fall outside [{lo}, {hi}]; first: "
        f"{values[out_of_range[:8]].tolist()} at {out_of_range[:8].tolist()}"
    )

    if case.scale == 0.0:
        assert torch.all(values == lo), (
            f"scale == 0 must give a constant {lo} tile, got "
            f"{torch.unique(values)[:8].tolist()}"
        )
        return

    # Plus one output ulp for the truncating 16-bit store.
    sigma = case.scale / math.sqrt(12.0)
    mean = float(values.mean())
    mean_tol = _RAND_MEAN_SIGMAS * sigma / math.sqrt(n) + _rand_output_ulp(
        max(abs(lo), abs(hi)), output_format
    )
    assert (
        abs(mean - (lo + hi) / 2) <= mean_tol
    ), f"mean {mean} is not within {mean_tol} of the midpoint {(lo + hi) / 2}"

    std = float(values.std())
    assert (
        abs(std - sigma) <= _RAND_STD_REL_TOL * sigma
    ), f"standard deviation {std} is not within {_RAND_STD_REL_TOL:.0%} of the uniform {sigma}"

    counts = torch.histc(values, bins=_RAND_HISTOGRAM_BINS, min=lo, max=hi)
    expected = n / _RAND_HISTOGRAM_BINS
    assert torch.all(
        (counts >= _RAND_HISTOGRAM_MIN_FRACTION * expected)
        & (counts <= _RAND_HISTOGRAM_MAX_FRACTION * expected)
    ), f"histogram over [{lo}, {hi}] is not uniform: {counts.tolist()} (expected ~{expected})"

    distinct = torch.unique(values).numel()
    min_distinct = (
        int(_RAND_FP32_MIN_DISTINCT_FRACTION * n)
        if output_format == DataFormat.Float32
        else _RAND_NARROW_MIN_DISTINCT
    )
    assert (
        distinct >= min_distinct
    ), f"only {distinct} distinct values in {n} draws (need >= {min_distinct})"

    # One PRNG step covers a row pair (32 lanes = 2 face rows x FACE_C_DIM).
    face_rows = values.reshape(-1, FACE_C_DIM)
    row_distinct = torch.tensor([torch.unique(r).numel() for r in face_rows])
    row_min = FACE_C_DIM // _RAND_ROW_MIN_DISTINCT_DIVISOR
    assert torch.all(row_distinct >= row_min), (
        f"lanes are not independent: a face row holds only {int(row_distinct.min())} "
        f"distinct values of {FACE_C_DIM}"
    )
    col_distinct = torch.tensor([torch.unique(c).numel() for c in face_rows.T])
    col_min = face_rows.shape[0] // _RAND_COL_MIN_DISTINCT_DIVISOR
    assert torch.all(col_distinct >= col_min), (
        f"the PRNG does not advance per row pair: a face column holds only "
        f"{int(col_distinct.min())} distinct values over {face_rows.shape[0]} rows"
    )
    distinct_rows = torch.unique(face_rows, dim=0).shape[0]
    assert (
        distinct_rows == face_rows.shape[0]
    ), f"{face_rows.shape[0] - distinct_rows} face rows repeat an earlier row"

    max_corr, (dr, dc) = _max_offset_correlation(face_rows)
    assert max_corr < _RAND_MAX_OFFSET_CORRELATION, (
        f"draws {dr} rows and {dc} columns apart are correlated: |r| = {max_corr:.3f} "
        f"(limit {_RAND_MAX_OFFSET_CORRELATION})"
    )


def _rand_config(
    case: RandCase,
    format_variant: QuasarSfpuVariant,
    dest_sync: DestSync,
    input_dimensions,
    seed: int = _RAND_DEFAULT_SEED,
):
    formats = format_variant.formats
    input_torch_format = format_dict[formats.input_format]
    element_count = input_dimensions[0] * input_dimensions[1]
    tile_count = element_count // (DEFAULT_TILE_R_DIM * DEFAULT_TILE_C_DIM)
    src_A = torch.full(
        (element_count,), _RAND_UNWRITTEN_SENTINEL, dtype=input_torch_format
    )
    # src_B is unused by a unary op, but StimuliConfig requires an operand-B buffer.
    src_B = torch.zeros_like(src_A)

    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=(PerfRunType.L1_TO_L1,),
        test_config_kwargs={
            "test_name": "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp",
            "formats": formats,
            "templates": [
                MATH_OP(mathop=MathOperation.Rand),
                APPROX_MODE(ApproximationMode.No),
                IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
                DATA_COPY_TYPE(DataCopyType.A2D),
                UNPACKER_ENGINE_SEL(
                    UnpackerEngine.UnpDest
                    if format_variant.unpack_to_dest
                    else UnpackerEngine.UnpA
                ),
                DEST_SYNC(dest_sync),
                TYPECAST_FORMATS(),
                RAND_RANGE(
                    rand_from_bits=_fp32_bits(case.from_value),
                    rand_scale_bits=_fp32_bits(case.scale),
                ),
                RAND_SEED(rand_seed=seed),
            ],
            "runtimes": [
                TILE_COUNT(tile_count),
                NUM_FACES(MAX_NUM_FACES),
                TEST_FACE_DIMS(),
                DEST_INDEX(0),
                LOOP_FACTOR(1),
            ],
            "variant_stimuli": StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=tile_count,
                tile_count_B=tile_count,
                tile_count_res=tile_count,
                num_faces=MAX_NUM_FACES,
            ),
            "unpack_to_dest": format_variant.unpack_to_dest,
            "dest_acc": format_variant.dest_acc,
        },
    )
    format_variant.apply_formats(configuration.formats_config)
    return configuration


def _run_rand(configuration, format_variant: QuasarSfpuVariant, element_count: int):
    res_from_L1 = configuration.run().result
    values = torch.tensor(
        res_from_L1, dtype=format_dict[format_variant.formats.output_format]
    ).to(torch.float64)
    assert (
        values.numel() == element_count
    ), f"result has {values.numel()} elements, expected {element_count}"
    return values


@pytest.mark.quasar
@parametrize(rand_case_formats_sync_dims=generate_rand_combinations())
def test_rand_quasar(rand_case_formats_sync_dims):
    case, format_variant, dest_sync, input_dimensions = rand_case_formats_sync_dims[0]
    configuration = _rand_config(case, format_variant, dest_sync, input_dimensions)
    values = _run_rand(
        configuration, format_variant, input_dimensions[0] * input_dimensions[1]
    )
    _check_rand_properties(values, case, format_variant.formats.output_format)


_RAND_OTHER_SEED = 0x9E3779B9
_RAND_LOCKUP_SEED = 0xFFFFFFFF  # XNOR LFSR lock-up state, repaired by init_rand
_RAND_LOCKUP_REPAIR_SEED = 0xFFFFFFFE
_RAND_MIN_DIFFERENT_FRACTION = 0.9


def _rand_seed_variant() -> QuasarSfpuVariant:
    """Float32 end to end, so every PRNG bit the kernel keeps reaches L1."""
    return next(
        v
        for v in generate_quasar_sfpu_format_variants(
            MathOperation.Rand, input_output_formats([DataFormat.Float32])
        )
        if v.dest_acc == DestAccumulation.Yes
    )


@pytest.mark.quasar
def test_rand_seed_quasar():
    case = RAND_CASES[0]
    format_variant = _rand_seed_variant()
    element_count = _RAND_DIMS[0] * _RAND_DIMS[1]
    seeds = (
        _RAND_DEFAULT_SEED,
        _RAND_OTHER_SEED,
        _RAND_LOCKUP_SEED,
        _RAND_LOCKUP_REPAIR_SEED,
    )
    configs = {
        seed: _rand_config(case, format_variant, _RAND_DEST_SYNC, _RAND_DIMS, seed)
        for seed in seeds
    }
    # compile-producer skips on run(), so prepare every ELF first.
    for configuration in configs.values():
        configuration.prepare()

    # Another seed runs in between, so an unseeded PRNG would not repeat run one.
    first = _run_rand(configs[_RAND_DEFAULT_SEED], format_variant, element_count)
    other = _run_rand(configs[_RAND_OTHER_SEED], format_variant, element_count)
    again = _run_rand(configs[_RAND_DEFAULT_SEED], format_variant, element_count)
    lockup = _run_rand(configs[_RAND_LOCKUP_SEED], format_variant, element_count)
    repair = _run_rand(configs[_RAND_LOCKUP_REPAIR_SEED], format_variant, element_count)

    for values in (first, other, lockup):
        _check_rand_properties(values, case, format_variant.formats.output_format)

    mismatched = (first != again).nonzero().flatten()
    assert mismatched.numel() == 0, (
        f"the same seed gave a different tile in {mismatched.numel()} of {element_count} "
        f"elements; first at {mismatched[:8].tolist()}"
    )
    different = int((first != other).sum())
    assert different >= _RAND_MIN_DIFFERENT_FRACTION * element_count, (
        f"seeds {_RAND_DEFAULT_SEED:#x} and {_RAND_OTHER_SEED:#x} agree on "
        f"{element_count - different} of {element_count} elements"
    )
    assert torch.equal(lockup, repair), (
        f"seed {_RAND_LOCKUP_SEED:#x} must be repaired to {_RAND_LOCKUP_REPAIR_SEED:#x}, "
        f"but their tiles differ in {int((lockup != repair).sum())} elements"
    )
