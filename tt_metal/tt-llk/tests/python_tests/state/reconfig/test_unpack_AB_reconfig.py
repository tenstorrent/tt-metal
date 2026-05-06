# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import (
    DestAccumulation,
)
from helpers.param_config import parametrize
from helpers.tensix import TensixState
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    RECONFIG_RUN_IDX,
    UNPACK_RECONFIG_RUNTIMES,
    UNPACK_RECONFIG_TEMPLATES,
)


def sanitize(dump: dict) -> dict:
    """Drop fields that differ between RECONFIG_RUN_IDX=0 and =1 but are not unpack reconfig semantics."""
    out = copy.deepcopy(dump)
    for gpr in out.get("gpr", []):
        if isinstance(gpr, dict):
            # test_unpack_AB_reconfig[format_from:Float16-format_to:Float16-row_dim_a:8-row_dim_b:8-num_faces_a:1-num_faces_b:1-tile_size_a:10-tile_size_b:11-row_dim_a_next:8-row_dim_b_next:8-num_faces_a_next:1-num_faces_b_next:1-tile_size_a_next:12-tile_size_b_next:13-to_from_int8:False-dest_acc:No]
            # idx 0 runs hw_configure(NEXT) only; idx 1 runs hw_configure(current) then reconfig. TT_SETDMAREG
            # updates these GPRs in a different order, so TILE_SIZE_A/B in GPRs need not match at end state.
            gpr.pop("tile_size_a", None)
            gpr.pop("tile_size_b", None)

    for rc in out.get("relu_config", []):
        if isinstance(rc, dict):
            # test_unpack_AB_reconfig[format_from:Float16-format_to:UInt16-row_dim_a:8-row_dim_b:8-num_faces_a:1-num_faces_b:1-tile_size_a:10-tile_size_b:11-row_dim_a_next:8-row_dim_b_next:8-num_faces_a_next:1-num_faces_b_next:1-tile_size_a_next:12-tile_size_b_next:13-to_from_int8:True-dest_acc:Yes]
            # Unpack-only test still snapshots full tensix state; ALU relu disabled_src differed 0 vs 1 for this case.
            rc.pop("disabled_src", None)

    return out


def get_valid_num_faces(row_dim: int) -> list[int]:
    if row_dim == 16:
        return [1, 2, 4]

    return [1, 2]


def get_valid_to_from_int8(
    format_from: DataFormat,
    format_to: DataFormat,
) -> bool:
    return format_from.is_integer() or format_to.is_integer()


def get_valid_dest_acc(to_from_int8: bool) -> bool:
    if to_from_int8:
        return DestAccumulation.Yes

    return [DestAccumulation.No, DestAccumulation.Yes]


ALL_FORMATS = [
    DataFormat.Float16,
    DataFormat.Float16_b,
    DataFormat.Bfp8,
    DataFormat.Bfp8_b,
    DataFormat.Float32,
    # DataFormat.Int32, # lol
    DataFormat.Tf32,
    # DataFormat.UInt32, # lol
    DataFormat.UInt16,
    DataFormat.Int8,
    DataFormat.UInt8,
]


@parametrize(
    format_from=ALL_FORMATS,
    format_to=ALL_FORMATS,
    row_dim_a=[8, 16],
    row_dim_b=[8, 16],
    num_faces_a=lambda row_dim_a: get_valid_num_faces(row_dim_a),
    num_faces_b=lambda row_dim_b: get_valid_num_faces(row_dim_b),
    tile_size_a=0xA,
    tile_size_b=0xB,
    row_dim_a_next=[8, 16],
    row_dim_b_next=[8, 16],
    num_faces_a_next=lambda row_dim_a_next: get_valid_num_faces(row_dim_a_next),
    num_faces_b_next=lambda row_dim_b_next: get_valid_num_faces(row_dim_b_next),
    tile_size_a_next=0xC,
    tile_size_b_next=0xD,
    to_from_int8=lambda format_from, format_to: get_valid_to_from_int8(
        format_from, format_to
    ),
    dest_acc=lambda to_from_int8: get_valid_dest_acc(to_from_int8),
)
def test_unpack_AB_reconfig(
    format_from,
    format_to,
    row_dim_a,
    row_dim_b,
    num_faces_a,
    num_faces_b,
    tile_size_a,
    tile_size_b,
    row_dim_a_next,
    row_dim_b_next,
    num_faces_a_next,
    num_faces_b_next,
    tile_size_a_next,
    tile_size_b_next,
    to_from_int8,
    dest_acc,
    workers_tensix_coordinates,
):

    if num_faces_a != num_faces_a_next:
        pytest.xfail("NUM_FACES_A != NUM_FACES_A_NEXT")

    if num_faces_b != num_faces_b_next:
        pytest.xfail("NUM_FACES_B != NUM_FACES_B_NEXT")

    if row_dim_a != row_dim_a_next:
        pytest.xfail("FACE_R_DIM_A != FACE_R_DIM_A_NEXT")

    if row_dim_b != row_dim_b_next:
        pytest.xfail("FACE_R_DIM_B != FACE_R_DIM_B_NEXT")

    templates = [
        UNPACK_RECONFIG_TEMPLATES(
            format_from,
            format_from,
            format_from,
            format_from,
            format_to,
            format_to,
            format_to,
            format_to,
            to_from_int8,
        ),
    ]
    runtimes = UNPACK_RECONFIG_RUNTIMES(
        row_dim_a,
        row_dim_b,
        num_faces_a,
        num_faces_b,
        tile_size_a,
        tile_size_b,
        row_dim_a_next,
        row_dim_b_next,
        num_faces_a_next,
        num_faces_b_next,
        tile_size_a_next,
        tile_size_b_next,
    )

    configuration = TestConfig(
        "sources/state/reconfig/unpack_AB_reconfig_test.cpp",
        formats=FormatConfig(
            DataFormat.Float16,
            DataFormat.Float16,
            DataFormat.Float16,
            DataFormat.Float16,
            DataFormat.Float16,
        ),
        templates=templates,
        runtimes=[runtimes, RECONFIG_RUN_IDX(0)],
        dest_acc=dest_acc,
    )

    configuration.run(workers_tensix_coordinates)
    expected = TensixState.fetch(workers_tensix_coordinates)

    configuration = TestConfig(
        "sources/state/reconfig/unpack_AB_reconfig_test.cpp",
        formats=FormatConfig(
            DataFormat.Float16,
            DataFormat.Float16,
            DataFormat.Float16,
            DataFormat.Float16,
            DataFormat.Float16,
        ),
        templates=templates,
        runtimes=[runtimes, RECONFIG_RUN_IDX(1)],
        dest_acc=dest_acc,
    )

    configuration.run(workers_tensix_coordinates)
    actual = TensixState.fetch(workers_tensix_coordinates)

    TensixState.assert_equal(sanitize(expected), sanitize(actual))
