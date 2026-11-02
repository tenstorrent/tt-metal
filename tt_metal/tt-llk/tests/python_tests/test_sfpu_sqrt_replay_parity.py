# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch
from conftest import skip_for_blackhole, skip_for_quasar
from helpers.data_format_inference import unpacks_to_dest
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    FusedSort,
    MathOperation,
    PerfRunType,
    StableSort,
    Transpose,
    format_dict,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import calculate_tile_and_face_counts
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    DEST_SYNC,
    FAST_MODE,
    FUSED_SORT,
    ITERATIONS,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    PERF_RUN_TYPE,
    STABLE_SORT,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
    DestSync,
    TemplateParameter,
)
from helpers.tile_constants import MAX_FACE_R_DIM, MAX_NUM_FACES

pytestmark = [skip_for_blackhole, skip_for_quasar]

DIMS = [256, 256]


@dataclass
class SFPI_LOOP(TemplateParameter):
    on: bool

    def convert_to_cpp(self) -> str:
        return "#define TT_METAL_DISABLE_SFPU_REPLAY_OPTIMIZATION" if self.on else ""


def _all_bf16():
    return (
        torch.arange(-32768, 32768, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
    )


def _fp32_sweep():
    g = torch.Generator().manual_seed(0)
    bits = torch.randint(
        -(2**31), 2**31 - 1, (DIMS[0] * DIMS[1],), dtype=torch.int64, generator=g
    ).to(torch.int32)
    special = torch.tensor(
        [
            0x00000000,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0x00000001,
            0x007FFFFF,
            0x3F800000,
        ],
        dtype=torch.int64,
    ).to(torch.int32)
    bits[: special.numel()] = special
    return bits.view(torch.float32)


def _config(fmt, dest_acc, approx, fast, iterations, src_A, sfpi_loop):
    tile_cnt_A, tile_cnt_B, faces = calculate_tile_and_face_counts(
        DIMS, DIMS, face_r_dim=MAX_FACE_R_DIM, num_faces=MAX_NUM_FACES
    )
    formats = InputOutputFormat(fmt, fmt)
    config = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Sqrt),
            APPROX_MODE(approx),
            ITERATIONS(iterations),
            FAST_MODE(fast),
            STABLE_SORT(StableSort.No),
            FUSED_SORT(FusedSort.No),
            CLAMP_NEGATIVE(True),
            DEST_SYNC(DestSync.Half),
            SFPI_LOOP(sfpi_loop),
            PERF_RUN_TYPE(PerfRunType.L1_TO_L1),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            LOOP_FACTOR(1),
            NUM_FACES(num_faces=faces),
            UNPACK_TRANS_FACES(Transpose.No),
            UNPACK_TRANS_WITHIN_FACE(Transpose.No),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            fmt,
            torch.zeros_like(src_A),
            fmt,
            fmt,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=unpacks_to_dest(fmt, dest_acc),
    )
    return config


@pytest.mark.parametrize("iterations", [8, 32], ids=lambda n: f"iter_{n}")
@pytest.mark.parametrize(
    "fast", [FastMode.No, FastMode.Yes], ids=lambda m: f"fast_{m.name}"
)
@pytest.mark.parametrize(
    "approx",
    [ApproximationMode.No, ApproximationMode.Yes],
    ids=lambda m: f"approx_{m.name}",
)
@pytest.mark.parametrize(
    "fmt, dest_acc",
    [
        (DataFormat.Float16_b, DestAccumulation.No),
        (DataFormat.Float16_b, DestAccumulation.Yes),
        (DataFormat.Float32, DestAccumulation.Yes),
    ],
    ids=["bf16_dest16", "bf16_dest32", "fp32_dest32"],
)
def test_sqrt_matches_sfpi_loop(fmt, dest_acc, approx, fast, iterations):
    src_A = _all_bf16() if fmt == DataFormat.Float16_b else _fp32_sweep()
    configs = [
        _config(fmt, dest_acc, approx, fast, iterations, src_A, sfpi_loop=False),
        _config(fmt, dest_acc, approx, fast, iterations, src_A, sfpi_loop=True),
    ]
    for config in configs:
        config.prepare()
    ours, ref = (
        torch.tensor(config.run().result, dtype=format_dict[fmt]) for config in configs
    )
    assert ours.numel() == ref.numel() == src_A.numel()
    itype = torch.int16 if fmt == DataFormat.Float16_b else torch.int32
    diff = (ours.view(itype) != ref.view(itype)) & ~(ours.isnan() & ref.isnan())
    assert not diff.any(), (
        f"{int(diff.sum())} of {diff.numel()} outputs differ from the SFPI loop, "
        f"first at input bits {src_A.view(itype)[diff.nonzero()[0]].item():#x}"
    )
