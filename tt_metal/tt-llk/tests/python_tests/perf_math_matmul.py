# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from itertools import chain, product

import pytest
from helpers.format_config import DataFormat, is_dest_acc_needed
from helpers.golden_generators import TILE_DIM
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    MathFidelity,
    PerfRunType,
    StochasticRounding,
)
from helpers.matmul_sweep import (
    MatmulConfig,
    generate_face_layout_config_sweep,
    generate_tile_dims,
    skip_matmul_combination,
    sweep_tiny_tiles_matmul,
)
from helpers.param_config import DEST_SYNC_TILE_LIMITS, input_output_formats
from helpers.perf.core import PerfConfig
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    DEST_INDEX,
    DEST_SYNC,
    IN_TILE_DIMS,
    LOOP_FACTOR,
    MATH_FIDELITY,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    PARTIAL_FACE,
    THROTTLE_LEVEL,
    TILE_COUNT,
    UNPACK_TRANS_FACES,
    UNPACK_TRANS_WITHIN_FACE,
)

MATMUL_FORMATS = input_output_formats(
    [
        DataFormat.Bfp8_b,
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
    ]
)
DEST_ACC_MODES = [DestAccumulation.No, DestAccumulation.Yes]
DEST_SYNC_MODES = [DestSync.Half, DestSync.Full]
STOCHASTIC_ROUNDING_MODES = [StochasticRounding.No]
MATH_FIDELITIES = [
    MathFidelity.LoFi,
    MathFidelity.HiFi2,
    MathFidelity.HiFi3,
    MathFidelity.HiFi4,
]

# Dest-fill RT x CT from dest capacity, plus 2×4 / 4×2 when they fit.
# KT 1 and 4 (long-K lives on perf_matmul).
PERF_KT_DIMS = (1, 4)
THROTTLE_LEVELS = (0, 5)
DEST_HANDOFF_NUM_BLOCKS = 4


def _dest_capacity(dest_sync, dest_acc) -> int:
    return DEST_SYNC_TILE_LIMITS[dest_sync] // (
        2 if dest_acc == DestAccumulation.Yes else 1
    )


# 2-D dest blocks that are neither a vector (1×N / N×1) nor a square.
RECT_DEST_BLOCKS = ((2, 4), (4, 2))


def _dest_fill_rt_ct_pairs(max_tiles):
    """Power-of-two dest-fill (rt, ct) pairs, plus 2×4 / 4×2 when they fit dest."""
    pairs = []
    rt_dim = 1
    while rt_dim <= max_tiles:
        if max_tiles % rt_dim == 0:
            pairs.append((rt_dim, max_tiles // rt_dim))
        rt_dim *= 2
    pairs.extend((rt, ct) for rt, ct in RECT_DEST_BLOCKS if rt * ct <= max_tiles)
    return list(dict.fromkeys(pairs))


def _fits_tiny_perf_tile_shape(cfg) -> bool:
    rt_dim = cfg.tile_dimensions.rt_dim
    ct_dim = cfg.tile_dimensions.ct_dim
    kt_dim = cfg.tile_dimensions.kt_dim
    max_tiles = _dest_capacity(cfg.dest_sync, cfg.dest_acc)
    return (
        cfg.dst_index == 0 and rt_dim == 1 and kt_dim == 1 and ct_dim in (1, max_tiles)
    )


def generate_perf_matmul_combinations():
    """Regular matmul: dest-filling RT x CT grids, plus 2×4 / 4×2 when they fit dest, with KT in {1, 4}."""
    combinations = []
    bfloat16_formats = {DataFormat.Float16_b, DataFormat.Float32}

    for fmt in MATMUL_FORMATS:
        is_fpu_bfloat16 = (
            fmt.input_format in bfloat16_formats
            and fmt.output_format in bfloat16_formats
        )
        for dest_acc in DEST_ACC_MODES:
            if is_dest_acc_needed(fmt) and dest_acc == DestAccumulation.No:
                continue
            if (
                dest_acc == DestAccumulation.No
                and fmt.input_format == DataFormat.Float16_b
                and fmt.output_format == DataFormat.Float16
            ):
                continue

            for dest_sync in DEST_SYNC_MODES:
                max_tiles = _dest_capacity(dest_sync, dest_acc)
                for stochastic_mode in STOCHASTIC_ROUNDING_MODES:
                    for rt_dim, ct_dim in _dest_fill_rt_ct_pairs(max_tiles):
                        for kt_dim in PERF_KT_DIMS:
                            if skip_matmul_combination(
                                stochastic_mode,
                                dest_acc,
                                is_fpu_bfloat16,
                                kt_dim,
                            ):
                                continue
                            tile_dims = generate_tile_dims(
                                (
                                    [rt_dim * TILE_DIM, kt_dim * TILE_DIM],
                                    [kt_dim * TILE_DIM, ct_dim * TILE_DIM],
                                )
                            )
                            for face_layout_config in generate_face_layout_config_sweep(
                                math_matmul=True
                            ):
                                combinations.append(
                                    MatmulConfig(
                                        tile_dimensions=tile_dims,
                                        face_layout_config=face_layout_config,
                                        formats=fmt,
                                        stochastic_rnd=stochastic_mode,
                                        dst_index=0,
                                        dest_sync=dest_sync,
                                        dest_acc=dest_acc,
                                    )
                                )
    return combinations


MATMUL_COMBINATIONS = generate_perf_matmul_combinations()

TINY_TILES_MATMUL_COMBINATIONS = [
    cfg
    for cfg in sweep_tiny_tiles_matmul(
        MATMUL_FORMATS,
        DEST_ACC_MODES,
        STOCHASTIC_ROUNDING_MODES,
        DEST_SYNC_MODES,
        math_matmul=True,
    )
    if _fits_tiny_perf_tile_shape(cfg)
]

ALL_TEST_PARAMS = list(
    chain(
        (
            (fidelity, cfg, throttle, 1)
            for fidelity, cfg, throttle in product(
                MATH_FIDELITIES, MATMUL_COMBINATIONS, THROTTLE_LEVELS
            )
        ),
        (
            (fidelity, cfg, 0, 1)
            for fidelity, cfg in product(
                MATH_FIDELITIES, TINY_TILES_MATMUL_COMBINATIONS
            )
        ),
        (
            (fidelity, cfg, 0, DEST_HANDOFF_NUM_BLOCKS)
            for fidelity, cfg in product(MATH_FIDELITIES, MATMUL_COMBINATIONS)
            if cfg.tile_dimensions.kt_dim == 1
        ),
    )
)


@pytest.mark.perf
@pytest.mark.parametrize(
    "math_fidelity,matmul_config,throttle,num_blocks", ALL_TEST_PARAMS
)
def test_perf_math_matmul(
    math_fidelity,
    matmul_config,
    throttle,
    num_blocks,
    perf_report,
):
    """
    Performance test for matmul operations.

    Regular matmul uses dest-filling RT x CT grids sized to dest capacity, plus
    2×4 / 4×2 when they fit, with KT in {1, 4} and throttle 0 or 5. Tiny tiles
    cover ct=1 and dest-fill ct. A dest-handoff slice repeats those grids
    (NUM_BLOCKS=4, KT=1, throttle=0).
    """
    formats = matmul_config.formats
    in0_dimensions = matmul_config.tile_dimensions.in0_dimensions
    in1_dimensions = matmul_config.tile_dimensions.in1_dimensions
    transpose = matmul_config.face_layout_config.unpack_transpose_faces
    num_faces_in0 = matmul_config.face_layout_config.num_faces_in0
    num_faces_in1 = matmul_config.face_layout_config.num_faces_in1
    num_faces = matmul_config.face_layout_config.num_faces

    if is_dest_acc_needed(formats) and matmul_config.dest_acc == DestAccumulation.No:
        pytest.skip("Dest accumulation must be enabled for this format")

    run_types = [
        PerfRunType.L1_TO_L1,
        PerfRunType.UNPACK_ISOLATE,
        PerfRunType.MATH_ISOLATE,
        PerfRunType.PACK_ISOLATE,
        PerfRunType.L1_CONGESTION,
    ]

    variant_tile_count = (
        matmul_config.tile_dimensions.rt_dim
        * matmul_config.tile_dimensions.ct_dim
        * matmul_config.tile_dimensions.kt_dim
    )

    configuration = PerfConfig(
        "sources/math_matmul_test.cpp",
        formats,
        run_types,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(matmul_config.dest_sync),
            THROTTLE_LEVEL(throttle),
        ],
        runtimes=[
            DEST_INDEX(matmul_config.dst_index),
            UNPACK_TRANS_FACES(transpose),
            UNPACK_TRANS_WITHIN_FACE(transpose),
            TILE_COUNT(variant_tile_count * num_blocks),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(
                matmul_config.tile_dimensions.rt_dim
                * matmul_config.tile_dimensions.ct_dim
            ),
            NUM_FACES(
                num_faces, num_faces_in0, num_faces_in1
            ),  # In0 -> Input A, In1 -> Input B
            PARTIAL_FACE(  # In0 -> Input A, In1 -> Input B
                partial_a=matmul_config.face_layout_config.partial_face_in0,
                partial_face_pack=matmul_config.face_layout_config.partial_face_pack,
                partial_b=matmul_config.face_layout_config.partial_face_in1,
                partial_face_math=matmul_config.face_layout_config.partial_face_math,
            ),
            CRK_TILE_DIMM(
                matmul_config.tile_dimensions.ct_dim,
                matmul_config.tile_dimensions.rt_dim,
                matmul_config.tile_dimensions.kt_dim,
            ),
            IN_TILE_DIMS(
                matmul_config.tile_dimensions.in0_tile_r_dim,
                matmul_config.tile_dimensions.in0_tile_c_dim,
                matmul_config.tile_dimensions.in1_tile_r_dim,
                matmul_config.tile_dimensions.in1_tile_c_dim,
            ),
            LOOP_FACTOR(1024),
        ],
        variant_stimuli=StimuliConfig(
            None,
            formats.input_format,
            None,
            formats.input_format,
            formats.output_format,
            tile_count_A=matmul_config.tile_dimensions.tile_cnt_in0,
            tile_count_B=matmul_config.tile_dimensions.tile_cnt_in1,
            tile_count_res=matmul_config.tile_dimensions.output_tile_cnt * num_blocks,
        ),
        dest_acc=matmul_config.dest_acc,
    )

    configuration.run(perf_report)
