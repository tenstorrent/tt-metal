# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
from helpers.chip_architecture import ChipArchitecture, get_chip_architecture
from helpers.data_format_inference import unpacks_to_dest
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    get_golden_generator,
    quantize_input_to_unpack_format,
)
from helpers.llk_params import (
    BlocksCalculationAlgorithm,
    BroadcastType,
    DestAccumulation,
    DestSync,
    PerfRunType,
    format_dict,
)
from helpers.param_config import (
    generate_perf_input_dimensions,
    get_num_blocks_and_num_tiles_in_block,
    input_output_formats,
    parametrize,
    select_perf_tile_sizes,
)
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_variant_parameters import (
    ACC_TO_DEST,
    BROADCAST_TYPE,
    DEST_SYNC,
    LOOP_FACTOR,
    NUM_BLOCKS,
    NUM_FACES,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    NUM_TILES_IN_BLOCK,
    SEED_BROADCAST_TYPE,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.tile_constants import get_tile_params
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test

supported_formats = [
    DataFormat.Int32,
    DataFormat.UInt32,
    DataFormat.UInt16,
    DataFormat.Float32,
    DataFormat.Float16_b,
    DataFormat.Bfp8_b,
]

# Sweep tile dimensions from tiny ([1,32]..[16,32]) through full ([32,32]).
# Tiny tiles have fewer faces (num_faces=2) and variable face_r_dim;
# full 32x32 tiles have 4 faces with face_r_dim=16.
# BroadcastType.None_ is a datacopy (unpack A -> DEST -> pack to L1).
BCAST_TILE_DIMENSIONS = [[1, 32], [2, 32], [4, 32], [8, 32], [16, 32], [32, 32]]
BCAST_PERF_TILE_DIMENSIONS = [
    list(tile_dimensions)
    for tile_dimensions in select_perf_tile_sizes(BCAST_TILE_DIMENSIONS)
]
BCAST_FORMATS = input_output_formats(supported_formats, same=True)
BCAST_TYPES = [
    BroadcastType.None_,
    BroadcastType.Column,
    BroadcastType.Row,
    BroadcastType.Scalar,
]
BCAST_DEST_SYNC = DestSync.Half


def get_valid_dest_acc_bcast(formats):
    """32-bit formats require dest accumulation."""
    if formats.input_format.is_32_bit():
        return [DestAccumulation.Yes]
    return [DestAccumulation.Yes, DestAccumulation.No]


def _bfp8_b_tile_filter(formats, tile_dimensions):
    """Bfp8_b requires a minimum of 16 exponents per face (tile height >= 16)."""
    if formats.input_format == DataFormat.Bfp8_b:
        return [dims for dims in tile_dimensions if dims[0] >= 16]
    return tile_dimensions


def get_valid_tile_dimensions_bcast(formats):
    return _bfp8_b_tile_filter(formats, BCAST_TILE_DIMENSIONS)


def get_valid_perf_tile_dimensions_bcast(formats):
    return _bfp8_b_tile_filter(formats, BCAST_PERF_TILE_DIMENSIONS)


def get_valid_broadcast_types(formats, dest_acc, tile_dimensions):
    broadcast_types = []
    for broadcast_type in BCAST_TYPES:
        # Data copy mechanism for SFPU (expects in low-bits) and Packer (expects in high-bits) conflict
        if (
            broadcast_type == BroadcastType.None_
            and dest_acc == DestAccumulation.Yes
            and formats.input_format == DataFormat.UInt16
        ):
            continue
        # TODO(#47328): Column broadcast for tiny tiles needs kernel support
        if broadcast_type == BroadcastType.Column and tile_dimensions != [32, 32]:
            continue
        # TODO: pgardner - known WH issue with row broadcast + dest accumulation
        if (
            get_chip_architecture() == ChipArchitecture.WORMHOLE
            and broadcast_type == BroadcastType.Row
            and dest_acc == DestAccumulation.Yes
            and formats.input_format in (DataFormat.Float16_b, DataFormat.Bfp8_b)
        ):
            continue
        broadcast_types.append(broadcast_type)
    return broadcast_types


def get_perf_input_dimensions_bcast(dest_acc, tile_dimensions):
    """The tall dest-full matrix; the wide one runs identical linear work."""
    if tile_dimensions not in BCAST_PERF_TILE_DIMENSIONS:
        return []
    return generate_perf_input_dimensions(
        dest_acc, BCAST_DEST_SYNC, construct_tile_shape(tuple(tile_dimensions))
    )[:1]


def get_input_dimensions_bcast(dest_acc, tile_dimensions):
    """A single tile, plus every perf geometry, so the shared kernel is
    golden-checked at the shapes perf_bcast.py benchmarks."""
    return [list(tile_dimensions)] + get_perf_input_dimensions_bcast(
        dest_acc, tile_dimensions
    )


def _run_unpack_bcast_test(
    formats,
    dest_acc,
    tile_dimensions,
    broadcast_type,
    input_dimensions,
    *,
    is_perf: bool = False,
    perf_report=None,
    run_types=None,
    loop_factor: int = 1,
    acc_to_dest=None,
    seed_broadcast_type=None,
):
    """Unary broadcast of A, or with acc_to_dest set, a plain seed_broadcast_type pass of A followed by an
    acc_to_dest broadcast_type pass of B (sources/unpack_A_bcast_acc_to_dest_test.cpp).
    """
    if is_perf and perf_report is None:
        raise ValueError("perf_report must be provided when is_perf=True")

    if run_types is None:
        run_types = [PerfRunType.L1_TO_L1]

    # --- Tile geometry ---------------------------------------------------
    # get_tile_params returns (face_r_dim, num_faces_r_dim, num_faces_c_dim).
    # For tiny tiles (e.g. [4,32]): face_r_dim=4, num_faces=2.
    # For full tiles ([32,32]):     face_r_dim=16, num_faces=4.
    face_r_dim, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim

    # --- Stimuli generation ----------------------------------------------
    # generate_stimuli(..., tile_dimensions=...) produces dense data for any tile size.
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions,
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        BCAST_DEST_SYNC,
        dest_acc,
        formats,
        input_dimensions,
        tile_dimensions,
        BlocksCalculationAlgorithm.Standard,
    )

    # --- Kernel configuration --------------------------------------------
    templates = [
        BROADCAST_TYPE(broadcast_type),
        DEST_SYNC(BCAST_DEST_SYNC),
    ]
    if acc_to_dest is not None:
        templates += [
            ACC_TO_DEST(acc_to_dest),
            SEED_BROADCAST_TYPE(seed_broadcast_type),
        ]
    test_config_kwargs = {
        "test_name": (
            "sources/unpack_A_bcast_test.cpp"
            if acc_to_dest is None
            else "sources/unpack_A_bcast_acc_to_dest_test.cpp"
        ),
        "formats": formats,
        "templates": templates,
        "runtimes": [
            NUM_FACES(num_faces, num_faces, num_faces),
            NUM_FACES_R_DIM(num_faces_r_dim, num_faces_r_dim),
            NUM_FACES_C_DIM(num_faces_c_dim, num_faces_c_dim),
            TILE_COUNT(tile_cnt_A),
            TEST_FACE_DIMS(face_r_dim=face_r_dim),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
            NUM_BLOCKS(num_blocks),
            LOOP_FACTOR(loop_factor),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=tile_dimensions,
            use_dense_tile_dimensions=True,
        ),
        "dest_acc": dest_acc,
        "unpack_to_dest": unpacks_to_dest(formats.input_format, dest_acc),
    }

    configuration = create_test_or_perf_config(
        is_perf=is_perf,
        run_types=run_types,
        test_config_kwargs=test_config_kwargs,
    )
    if is_perf:
        configuration.run(perf_report)
        return

    # --- Golden model ----------------------------------------------------
    # Broadcast types use BroadcastGolden which handles all face geometries.
    # Datacopy (None_) golden is the input as unpacked from L1, cast to the output format.
    generate_broadcast_golden = get_golden_generator(BroadcastGolden)

    def broadcast_golden(bcast_type, src):
        return generate_broadcast_golden(
            bcast_type,
            src,
            formats.output_format,
            num_faces=num_faces,
            tile_cnt=tile_cnt_A,
            face_r_dim=face_r_dim,
        )

    if acc_to_dest is not None:
        golden_tensor = broadcast_golden(broadcast_type, src_B)
        if acc_to_dest:
            golden_tensor = (
                broadcast_golden(seed_broadcast_type, src_A).float()
                + golden_tensor.float()
            ).to(format_dict[formats.output_format])
    elif broadcast_type != BroadcastType.None_:
        golden_tensor = broadcast_golden(broadcast_type, src_A)
    else:
        golden_tensor = quantize_input_to_unpack_format(src_A, formats.input_format).to(
            format_dict[formats.output_format]
        )

    res_from_L1 = configuration.run().result

    # --- Assertions ------------------------------------------------------
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])

    # Pretty red/green diff output via passed_test (tolerance-based)
    tile_shape = construct_tile_shape(tile_dimensions)
    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        tile_shape=tile_shape,
    )

    # Datacopy/bcast should be bit-exact for float formats (no compute loss); an accumulated sum rounds.
    if not acc_to_dest and formats.input_format in (
        DataFormat.Float32,
        DataFormat.Float16_b,
    ):
        assert torch.equal(
            golden_tensor, res_tensor
        ), "Datacopy/bcast should be exact for float formats"


@parametrize(
    formats=BCAST_FORMATS,
    dest_acc=get_valid_dest_acc_bcast,
    tile_dimensions=get_valid_tile_dimensions_bcast,
    broadcast_type=get_valid_broadcast_types,
    input_dimensions=get_input_dimensions_bcast,
)
def test_unpack_bcast(
    formats,
    dest_acc,
    tile_dimensions,
    broadcast_type,
    input_dimensions,
):
    _run_unpack_bcast_test(
        formats, dest_acc, tile_dimensions, broadcast_type, input_dimensions
    )


# acc_to_dest broadcasts through ELWADD with dest accumulation, which needs 16-bit float inputs:
# 32-bit inputs unpack straight to Dest and UInt16 broadcasts through MOVB2D, neither of which accumulates.
BCAST_ACC_FORMATS = input_output_formats(
    [DataFormat.Float16_b, DataFormat.Bfp8_b], same=True
)
BCAST_ACC_TILE_DIMENSIONS = [32, 32]
BCAST_ACC_INPUT_DIMENSIONS = [[128, 64]]
BCAST_ACC_TYPES = [BroadcastType.Column, BroadcastType.Row, BroadcastType.Scalar]


def _is_wh_row_dest_acc(dest_acc, broadcast_type):
    """The known WH row broadcast + dest accumulation issue is in the plain (MOVB2D) row broadcast."""
    return (
        get_chip_architecture() == ChipArchitecture.WORMHOLE
        and broadcast_type == BroadcastType.Row
        and dest_acc == DestAccumulation.Yes
    )


def get_valid_acc_to_dest(dest_acc, broadcast_type):
    # The acc_to_dest=False control ends with a plain row broadcast; the accumulating ELWADD row broadcast runs.
    if _is_wh_row_dest_acc(dest_acc, broadcast_type):
        return [True]
    return [False, True]


@parametrize(
    formats=BCAST_ACC_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    broadcast_type=BCAST_ACC_TYPES,
    acc_to_dest=get_valid_acc_to_dest,
    input_dimensions=BCAST_ACC_INPUT_DIMENSIONS,
)
def test_unpack_bcast_acc_to_dest(
    formats,
    dest_acc,
    broadcast_type,
    acc_to_dest,
    input_dimensions,
):
    """Seed Dest with a plain broadcast of A, then broadcast B with acc_to_dest.

    Accumulating gives seed_bcast(A) + bcast(B); acc_to_dest=False is the matched overwrite control, bcast(B).
    The WH row case seeds with a column broadcast to keep the plain row broadcast out of the accumulating cell.
    """
    seed_broadcast_type = (
        BroadcastType.Column
        if _is_wh_row_dest_acc(dest_acc, broadcast_type)
        else broadcast_type
    )
    _run_unpack_bcast_test(
        formats,
        dest_acc,
        BCAST_ACC_TILE_DIMENSIONS,
        broadcast_type,
        input_dimensions,
        acc_to_dest=acc_to_dest,
        seed_broadcast_type=seed_broadcast_type,
    )
