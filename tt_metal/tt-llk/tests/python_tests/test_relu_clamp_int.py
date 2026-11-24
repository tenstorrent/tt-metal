# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Wormhole metal ``relu_clamp_int`` / ``relu_clamp_uint`` (Int32 / UInt32 relu_min / relu_max).

The kernels store through ADDR_MOD_2 (ADDR_MOD_6 programmed to +2 by the init), except in a
build with ``MM_THROTTLE > 0``, where a throttled matmul owns ADDR_MOD_6 and the body advances
Dest itself. ``relu_clamp_int_test.cpp`` writes such a matmul ADDR_MOD_6 after the init in that
build. Every row holds different words and several tiles share Dest, so a wrong Dest advance
moves results between rows and tiles; the compare is exact.

Wormhole only: Blackhole and Quasar keep their own relu_clamp implementations.
"""

from dataclasses import dataclass

import torch
from conftest import skip_for_blackhole, skip_for_quasar
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, Transpose
from helpers.param_config import parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_variant_parameters import (
    APPROX_MODE,
    LOOP_FACTOR,
    NUM_FACES,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    TemplateParameter,
)

pytestmark = [skip_for_blackhole, skip_for_quasar]

UINT32_MASK = 0xFFFFFFFF
INT32_MAX = 2**31 - 1
INT32_MIN = -(2**31)

# RELU_CLAMP_VARIANT values in relu_clamp_int_test.cpp.
RELU_MIN_INT, RELU_MAX_INT, RELU_MIN_UINT, RELU_MAX_UINT = range(4)


@dataclass
class RELU_CLAMP_VARIANT(TemplateParameter):
    variant: int = RELU_MIN_INT

    def convert_to_cpp(self) -> str:
        return f"constexpr int RELU_CLAMP_VARIANT = {self.variant};"


@dataclass
class RELU_CLAMP_THRESHOLD(TemplateParameter):
    """The threshold as its raw 32-bit word (two's complement for Int32)."""

    threshold: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr std::uint32_t RELU_CLAMP_THRESHOLD = {self.threshold & UINT32_MASK:#x}u;"


@dataclass
class RELU_CLAMP_MM_THROTTLE(TemplateParameter):
    level: int = 0

    def convert_to_cpp(self) -> str:
        return f"#define MM_THROTTLE {self.level}"


def _stimuli(threshold: int, unsigned: bool, count: int) -> torch.Tensor:
    """Words around *threshold*, the type's extremes and random words, no two neighbours equal."""
    if unsigned:
        lo, hi = 0, UINT32_MASK
        specials = [0, 1, 2, INT32_MAX - 1, INT32_MAX, 2**31, 2**31 + 1, UINT32_MASK]
    else:
        lo, hi = INT32_MIN, INT32_MAX
        specials = [INT32_MIN, INT32_MIN + 1, -2, -1, 0, 1, 2, INT32_MAX - 1, INT32_MAX]
        specials += [s * 2**k for k in range(31) for s in (1, -1)]
    specials += [threshold + d for d in (-2, -1, 0, 1, 2)]
    specials = [v for v in specials if lo <= v <= hi]

    generator = torch.Generator().manual_seed(threshold & UINT32_MASK)
    words = torch.randint(0, 2**32, (count,), generator=generator, dtype=torch.int64)
    values = words if unsigned else torch.where(words > INT32_MAX, words - 2**32, words)
    # Specials at a stride coprime to the 16-word face row, so they land on every row and column.
    positions = torch.arange(len(specials)) * 37 % count
    values[positions] = torch.tensor(specials, dtype=torch.int64)
    return values


def _golden(values: torch.Tensor, variant: int, threshold: int) -> torch.Tensor:
    t = torch.tensor(threshold, dtype=torch.int64)
    if variant in (RELU_MIN_INT, RELU_MIN_UINT):
        return torch.maximum(values, t)
    if variant == RELU_MAX_INT:
        return torch.clamp(torch.minimum(values, t), min=0)
    return torch.minimum(values, t)


def run_relu_clamp(
    variant: int,
    threshold: int,
    mm_throttle: int = 0,
    input_dimensions=(64, 64),
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    unsigned = variant in (RELU_MIN_UINT, RELU_MAX_UINT)
    data_format = DataFormat.UInt32 if unsigned else DataFormat.Int32
    formats = InputOutputFormat(data_format, data_format)
    dims = list(input_dimensions)
    tile_cnt, _, faces = calculate_tile_and_face_counts(
        dims, dims, face_r_dim=16, num_faces=4
    )

    src_A = None
    if not is_perf:
        src_A = _stimuli(threshold, unsigned, dims[0] * dims[1])

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types or [],
        test_config_kwargs={
            "test_name": "sources/relu_clamp_int_test.cpp",
            "formats": formats,
            "templates": [
                RELU_CLAMP_VARIANT(variant),
                RELU_CLAMP_THRESHOLD(threshold),
                *([RELU_CLAMP_MM_THROTTLE(mm_throttle)] if mm_throttle else []),
                APPROX_MODE(ApproximationMode.No),
                TILE_COUNT(tile_cnt),
                LOOP_FACTOR(loop_factor),
                NUM_FACES(num_faces=faces),
                UNPACK_TRANS_FACES(Transpose.No),
                UNPACK_TRANS_WITHIN_FACE(Transpose.No),
            ],
            "runtimes": [],
            "variant_stimuli": StimuliConfig(
                src_A,
                formats.input_format,
                src_A,
                formats.input_format,
                formats.output_format,
                tile_count_A=tile_cnt,
                tile_count_B=tile_cnt,
                tile_count_res=tile_cnt,
                twos_complement=not unsigned,
            ),
            "unpack_to_dest": True,
            "dest_acc": DestAccumulation.Yes,
            "compile_time_formats": True,
        },
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res = configuration.run().result[: src_A.numel()]
    # The Int32 reader decodes the packer's sign-magnitude words, so compare values.
    got = torch.tensor(res, dtype=torch.int64)
    want = _golden(src_A, variant, threshold)
    bad = torch.nonzero(got != want).flatten()
    assert (
        bad.numel() == 0
    ), f"{bad.numel()}/{want.numel()} words wrong; first: " + ", ".join(
        f"[{i}] in {int(src_A[i]) & UINT32_MASK:#x} got {int(got[i])} want {int(want[i])}"
        for i in bad[:8].tolist()
    )


_INT_RELU_MIN_THRESHOLDS = [0, 1, 6, 1000, 2**30, INT32_MAX, -1, -6, INT32_MIN + 1]
_INT_RELU_MAX_THRESHOLDS = [0, 6, 1000, INT32_MAX, -6]
# Both halves of the uint32 path: threshold top bit clear and set.
_UINT_THRESHOLDS = [0, 6, INT32_MAX, 2**31, 2**31 + 6, UINT32_MASK]
_MM_THROTTLE = [0, 1]


@parametrize(threshold=_INT_RELU_MIN_THRESHOLDS, mm_throttle=_MM_THROTTLE)
def test_relu_min_int32(threshold, mm_throttle):
    run_relu_clamp(RELU_MIN_INT, threshold, mm_throttle)


@parametrize(threshold=_INT_RELU_MAX_THRESHOLDS, mm_throttle=_MM_THROTTLE)
def test_relu_max_int32(threshold, mm_throttle):
    run_relu_clamp(RELU_MAX_INT, threshold, mm_throttle)


@parametrize(
    variant=[RELU_MIN_UINT, RELU_MAX_UINT],
    threshold=_UINT_THRESHOLDS,
    mm_throttle=_MM_THROTTLE,
)
def test_relu_clamp_uint32(variant, threshold, mm_throttle):
    run_relu_clamp(variant, threshold, mm_throttle)
