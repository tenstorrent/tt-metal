# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.data_format_inference import unpacks_to_dest
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import (
    DataCopyGolden,
    TilizeGolden,
    get_golden_generator,
    quantize_mx_tensor_chunked,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    PerfRunType,
    UnpackerEngine,
)
from helpers.param_config import (
    DEST_SYNC_TILE_LIMITS,
    flat_axes,
    generate_perf_input_dimensions,
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
    quasar_dest_sync_modes,
    select_perf_tile_sizes,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import BootMode
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_SYNC,
    LOOP_FACTOR,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    generate_input_dim,
)
from helpers.tile_constants import (
    MX_SUPPORTED_TILE_SIZES,
    is_mx_unsupported_tile_dims,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test, result_tensor

UNPACK_TILIZE_TILE_SIZES = [
    (32, 32),
    (1, 32),
    (2, 32),
]

# Rows the SrcA/SrcB tilize takes in more than one MOP run: full_ct_dim -> block_ct_dim, a
# proper divisor (the init asserts block_ct_dim divides full_ct_dim), so each run past the
# first starts at a nonzero l1_tile_idx.
MULTI_RUN_BLOCK_CT_DIM = {2: 1, 4: 2, 8: 4, 16: 8}


def generate_unpack_tilize_combinations(
    formats_list: List[FormatConfig],
    *,
    is_perf=False,
):
    """
    Generate unpack_tilize combinations.

    Rules:
    1. 32-bit formats require DestAccumulation.Yes

    Args: List of input-output format pairs

    Returns: List of (format, dest_acc, dest_sync, unpacker_sel, input_dimensions,
             tile_dimensions, block_ct_dim) tuples; block_ct_dim is None for a run per row,
             otherwise the SrcA/SrcB run length, a divisor of the row
    """
    combinations = []

    dest_sync_modes = quasar_dest_sync_modes(is_perf=is_perf)

    for fmt in formats_list:
        in_fmt = fmt.input_format
        out_fmt = fmt.output_format

        dest_acc_modes = (
            (DestAccumulation.Yes,)
            if in_fmt.is_32_bit()
            else (
                (DestAccumulation.No,)
                if in_fmt in [DataFormat.Int16]
                else (DestAccumulation.No, DestAccumulation.Yes)
            )
        )
        # 32-bit tilize uses unpack_to_dest (UNP_DEST)
        unpacker_engines = (
            (UnpackerEngine.UnpDest,)
            if in_fmt.is_32_bit()
            else (UnpackerEngine.UnpA, UnpackerEngine.UnpB)
        )
        tile_sizes = (
            select_perf_tile_sizes(UNPACK_TILIZE_TILE_SIZES)
            if is_perf
            else UNPACK_TILIZE_TILE_SIZES
        )

        for dest_acc in dest_acc_modes:
            for dest_sync in dest_sync_modes:
                for unpacker_sel in unpacker_engines:
                    # Strided tilize supports dest accumulation only on UNP_A,
                    # so UnpB with dest accumulation is skipped for every shape
                    if (
                        dest_acc == DestAccumulation.Yes
                        and unpacker_sel == UnpackerEngine.UnpB
                    ):
                        continue
                    for tile_dims in tile_sizes:
                        if is_mx_unsupported_tile_dims(in_fmt, out_fmt, tile_dims):
                            continue
                        if (
                            unpacks_to_dest(in_fmt, dest_acc)
                            and tile_dims not in MX_SUPPORTED_TILE_SIZES
                        ):
                            continue
                        tile_shape = construct_tile_shape(tile_dims)
                        dimensions_list = (
                            generate_perf_input_dimensions(
                                dest_acc, dest_sync, tile_shape
                            )
                            if is_perf
                            else generate_unary_input_dimensions(
                                dest_acc, dest_sync=dest_sync, tile_shape=tile_shape
                            )
                        )
                        for dimensions in dimensions_list:
                            combinations.append(
                                (
                                    fmt,
                                    dest_acc,
                                    dest_sync,
                                    unpacker_sel,
                                    dimensions,
                                    tile_dims,
                                    None,
                                )
                            )
                        # Several runs per row: one row of the DEST capacity and two half rows, 32x32 only.
                        if (
                            not is_perf
                            and in_fmt
                            in (
                                DataFormat.Float16_b,
                                DataFormat.Float32,
                                DataFormat.Int32,
                            )
                            and tile_dims == (32, 32)
                        ):
                            capacity = DEST_SYNC_TILE_LIMITS[dest_sync] // (
                                2 if dest_acc == DestAccumulation.Yes else 1
                            )
                            for rows, cols in ((1, capacity), (2, capacity // 2)):
                                if cols in MULTI_RUN_BLOCK_CT_DIM:
                                    combinations.append(
                                        (
                                            fmt,
                                            dest_acc,
                                            dest_sync,
                                            unpacker_sel,
                                            [
                                                rows * tile_shape.total_row_dim(),
                                                cols * tile_shape.total_col_dim(),
                                            ],
                                            tile_dims,
                                            MULTI_RUN_BLOCK_CT_DIM[cols],
                                        )
                                    )

    return combinations


UNPACK_TILIZE_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float32,
        DataFormat.Int32,
        DataFormat.Int16,
        DataFormat.Int8,
        DataFormat.UInt8,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    ],
    same=True,  # Input format and output format are the same
)
# Column names of the tuples ``generate_unpack_tilize_combinations`` returns.
UNPACK_TILIZE_FLAT_NAMES = (
    "formats",
    "dest_acc",
    "dest_sync",
    "unpacker_sel",
    "input_dimensions",
    "tile_dimensions",
    "block_ct_dim",
)
ALL_UNPACK_TILIZE_COMBINATIONS = generate_unpack_tilize_combinations(
    UNPACK_TILIZE_FORMATS
)
PERF_UNPACK_TILIZE_COMBINATIONS = generate_unpack_tilize_combinations(
    UNPACK_TILIZE_FORMATS,
    is_perf=True,
)


@pytest.mark.quasar
@parametrize(
    **flat_axes(UNPACK_TILIZE_FLAT_NAMES, ALL_UNPACK_TILIZE_COMBINATIONS),
    run_types=[[PerfRunType.L1_TO_L1]],
    loop_factor=[1],
)
def test_unpack_tilize_quasar(
    formats,
    dest_acc,
    dest_sync,
    unpacker_sel,
    input_dimensions,
    tile_dimensions,
    block_ct_dim,
    run_types,
    loop_factor,
    boot_mode=BootMode.DEFAULT,
    *,
    is_perf=False,
    perf_report=None,
):
    tile_shape = construct_tile_shape(tile_dimensions)

    sequential_spec = StimuliSpec.sequential()
    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
        spec_A=sequential_spec,
        spec_B=sequential_spec,
    )

    num_faces = tile_shape.total_num_faces()

    generate_golden = get_golden_generator(TilizeGolden)
    golden_src = src_B if unpacker_sel == UnpackerEngine.UnpB else src_A
    if formats.input_format.is_mx_format():
        golden_src = quantize_mx_tensor_chunked(golden_src, formats.input_format)
    golden_tensor = generate_golden(
        golden_src,
        input_dimensions,
        formats.output_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
    )

    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    test_config_kwargs = {
        "test_name": "sources/quasar/unpack_tilize_quasar_test.cpp",
        "formats": formats,
        "templates": [
            generate_input_dim(
                input_dimensions,
                input_dimensions,
                block_ct_dim=block_ct_dim,
                tile_dimensions=tile_dimensions,
            ),
            UNPACKER_ENGINE_SEL(unpacker_sel),
            DATA_COPY_TYPE(
                DataCopyType.B2D
                if unpacker_sel == UnpackerEngine.UnpB
                else DataCopyType.A2D
            ),
            DEST_SYNC(dest_sync),
        ],
        "runtimes": [
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(tile_shape.face_r_dim),
            NUM_FACES(num_faces),
            NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
            NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
            face_r_dim=tile_shape.face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        "unpack_to_dest": unpacks_to_dest(formats.input_format, dest_acc),
        "dest_acc": dest_acc,
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
        boot_mode=boot_mode,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    res_from_L1 = configuration.run().result

    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
    ), "Assert against golden failed"


# (unpacker engine, input dimensions, tile dimensions, block_ct_dim): the 32x32 UNPACR_TILIZE path in several runs
# per row and over two rows, and the strided path for 1x32 and 2x32 tiles, on both unpackers.
TILIZE_THEN_UNARY_CASES = [
    (engine, input_dimensions, tile_dimensions, block_ct_dim)
    for engine in (UnpackerEngine.UnpA, UnpackerEngine.UnpB)
    for input_dimensions, tile_dimensions, block_ct_dim in (
        ([32, 128], (32, 32), 2),
        ([64, 64], (32, 32), 1),
        ([1, 128], (1, 32), None),
        ([2, 128], (2, 32), None),
    )
]


@pytest.mark.quasar
@parametrize(
    **flat_axes(
        ("unpacker_sel", "input_dimensions", "tile_dimensions", "block_ct_dim"),
        TILIZE_THEN_UNARY_CASES,
    ),
)
def test_unpack_tilize_then_unary_quasar(
    unpacker_sel, input_dimensions, tile_dimensions, block_ct_dim
):
    """Unpack tilize, then a plain unary unpack of the same input on the same unpacker with no teardown in between.

    tilize_uninit() is a no-op on Quasar, so the unary unpack must not depend on any state tilize leaves behind.
    """
    formats = input_output_formats([DataFormat.Float16_b], same=True)[0]
    dest_acc = DestAccumulation.No
    tile_shape = construct_tile_shape(tile_dimensions)
    num_faces = tile_shape.total_num_faces()

    sequential_spec = StimuliSpec.sequential()
    src_A, tile_cnt, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
        spec_A=sequential_spec,
        spec_B=sequential_spec,
    )
    golden_src = src_B if unpacker_sel == UnpackerEngine.UnpB else src_A

    # Section 0 is the tilized input, section 1 the input copied tile by tile.
    tilize_golden = get_golden_generator(TilizeGolden)(
        golden_src,
        input_dimensions,
        formats.output_format,
        num_faces=num_faces,
        tile_dimensions=tile_dimensions,
    )
    copy_golden = get_golden_generator(DataCopyGolden)(
        golden_src,
        formats.output_format,
        num_faces=num_faces,
        input_dimensions=input_dimensions,
        face_r_dim=tile_shape.face_r_dim,
        input_format=formats.input_format,
        tile_shape=tile_shape,
    )
    golden_tensor = torch.cat([tilize_golden.flatten(), copy_golden.flatten()])

    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=[PerfRunType.L1_TO_L1],
        test_config_kwargs={
            "test_name": "sources/quasar/unpack_tilize_then_unary_quasar_test.cpp",
            "formats": formats,
            "templates": [
                generate_input_dim(
                    input_dimensions,
                    input_dimensions,
                    block_ct_dim=block_ct_dim,
                    tile_dimensions=tile_dimensions,
                ),
                UNPACKER_ENGINE_SEL(unpacker_sel),
                DATA_COPY_TYPE(
                    DataCopyType.B2D
                    if unpacker_sel == UnpackerEngine.UnpB
                    else DataCopyType.A2D
                ),
                DEST_SYNC(DestSync.Half),
            ],
            "runtimes": [
                TILE_COUNT(tile_cnt),
                TEST_FACE_DIMS(tile_shape.face_r_dim),
                NUM_FACES(num_faces),
                NUM_FACES_R_DIM(tile_shape.num_faces_r_dim),
                NUM_FACES_C_DIM(tile_shape.num_faces_c_dim),
            ],
            "variant_stimuli": StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=tile_cnt,
                tile_count_B=tile_cnt,
                tile_count_res=2 * tile_cnt,
                num_faces=num_faces,
                face_r_dim=tile_shape.face_r_dim,
                tile_dimensions=tile_dimensions,
                use_dense_tile_dimensions=True,
            ),
            "unpack_to_dest": False,
            "dest_acc": dest_acc,
        },
    )
    res_from_L1 = configuration.run().result
    res_tensor = result_tensor(res_from_L1, formats.output_format, golden_tensor)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
    ), "Assert against golden failed"
