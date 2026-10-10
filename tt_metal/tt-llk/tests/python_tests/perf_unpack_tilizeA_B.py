# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf test of the reduce with a tilized operand A (tilizeA_B_reduce_init, unpack_tilizeA_B_block and the column reduce:
the pool2d and bilinear upsample configuration). Axes: pool MAX (negative infinity source fill) or AVG (zero fill), the
input face geometry the pool program factory programs (face_r_dim = min(window, 16); 2 faces up to a 16-row window, 4
above), the block width in tiles and the number of block calls per row (equal to the width: one tile per call), the
formats, HiFi4 as the pool. All five run types."""

from collections import OrderedDict

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    PerfRunType,
    ReducePool,
)
from helpers.param_config import parametrize
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    IN_TILE_DIMS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    MATH_OP,
    NUM_BLOCKS,
    NUM_FACES,
    TILE_COUNT,
    generate_input_dim,
)

F = DataFormat
NO, YES = DestAccumulation.No, DestAccumulation.Yes
BF16 = InputOutputFormat(F.Float16_b, F.Float16_b)
FP32 = InputOutputFormat(F.Float32, F.Float32)
POOLS = {"max": ReducePool.Max, "avg": ReducePool.Average}

VARIANTS = OrderedDict()


def add(pool, fn, fmt, acc, r, nf, ct, nb=1):
    VARIANTS[
        f"{pool}_{fn}_{'acc' if acc == YES else 'noacc'}_r{r}f{nf}_ct{ct}_nb{nb}"
    ] = dict(pool=POOLS[pool], formats=fmt, acc=acc, r=r, nf=nf, ct=ct, nb=nb)


for r, nf in ((16, 4), (16, 2), (9, 2), (4, 2), (1, 2)):
    add("max", "bf16", BF16, NO, r, nf, 8)
for r, nf in ((16, 4), (9, 2)):
    add("max", "bf16", BF16, NO, r, nf, 1)
    add("max", "bf16", BF16, NO, r, nf, 8, nb=8)
    add("avg", "bf16", BF16, NO, r, nf, 8)
    add("max", "fp32", FP32, NO, r, nf, 8)


@pytest.mark.perf
@skip_for_wormhole
@skip_for_quasar
@parametrize(variant=list(VARIANTS.keys()))
def test_perf_unpack_tilizeA_B(perf_report, variant):
    if isinstance(variant, tuple):
        variant = variant[0]
    v = VARIANTS[variant]
    formats, dest_acc, ct = v["formats"], v["acc"], v["ct"]
    rows = v["r"] * (2 if v["nf"] == 4 else 1)  # operand A tile: rows x 32

    configuration = PerfConfig(
        "sources/unpack_tilizeA_B_perf.cpp",
        formats,
        run_types=[
            PerfRunType.L1_TO_L1,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ],
        templates=[
            MATH_OP(mathop=MathOperation.ReduceColumn, pool_type=v["pool"]),
            MATH_FIDELITY(MathFidelity.HiFi4),
        ],
        runtimes=[
            generate_input_dim((32, ct * 32), (32, ct * 32)),
            TILE_COUNT(ct),
            NUM_BLOCKS(v["nb"]),
            LOOP_FACTOR(32),
            NUM_FACES(v["nf"]),
            IN_TILE_DIMS(in0_r_dim=rows, in0_c_dim=32, in1_r_dim=rows, in1_c_dim=32),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=ct,
            tile_count_B=1,
            tile_count_res=ct,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )
    configuration.run(perf_report)
