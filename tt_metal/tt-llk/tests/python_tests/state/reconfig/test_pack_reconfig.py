# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import copy

from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import DestAccumulation
from helpers.param_config import parametrize
from helpers.tensix import TensixState
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    PACK_RECONFIG_RUNTIMES,
    PACK_RECONFIG_TEMPLATES,
    RECONFIG_RUN_IDX,
)

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


SHAPES = {
    (1, 32): (1, 32, 2, 1, 0),
    (2, 32): (2, 32, 2, 1, 0),
    (4, 32): (4, 32, 2, 1, 0),
    (8, 32): (8, 32, 2, 1, 0),
    (16, 32): (16, 32, 2, 0, 0),
    (32, 32): (16, 32, 4, 0, 0),
    (32, 16): (16, 16, 2, 0, 1),
    (32, 8): (16, 8, 2, 1, 1),
    (16, 8): (16, 8, 1, 1, 1),
    (8, 8): (8, 8, 1, 1, 1),
}


def sanitize(dump: dict) -> dict:
    out = copy.deepcopy(dump)
    for gpr in out.get("gpr", []):
        if isinstance(gpr, dict):
            # scratch registers, ignore
            gpr.pop("tmp_lo", None)
            gpr.pop("tmp_hi", None)

            # unused, ignore
            gpr.pop("tile_header", None)

            # pop the exponent section size cache, reconfigure updates only the required values??
            for k in list(gpr.keys()):
                if k.startswith("exp") and "sec_size" in k:
                    gpr.pop(k, None)

    ac = out.get("address_counters")
    if isinstance(ac, dict):
        # reset by relevant init, ignore
        ac.pop("adcs2_packers_channel1_x_cr", None)
        ac.pop("adcs2_packers_channel1_x_counter", None)

    for pc in out.get("pack_config", []):
        if isinstance(pc, dict):
            # reset by execute, ignore
            pc.pop("l1_dest_addr", None)

            # ???, ignore
            pc.pop("exp_section_size", None)

    for ctr in out.get("pack_counters", []):
        if isinstance(ctr, dict):
            # ???, ignore
            ctr.pop("pack_reads_per_xy_plane", None)
    return out


# @skip_for_wormhole
# @skip_for_blackhole
@parametrize(
    format_from=ALL_FORMATS,
    format_from_next=ALL_FORMATS,
    format_to=ALL_FORMATS,
    format_to_next=ALL_FORMATS,
    shape=list(SHAPES.keys()),
    shape_next=list(SHAPES.keys()),
)
def test_pack_AB_reconfig(
    format_from,
    format_from_next,
    format_to,
    format_to_next,
    shape,
    shape_next,
    workers_tensix_coordinates,
):
    face_r_dim, tile_c_dim, num_faces, partial_face, narrow_tile = SHAPES[shape]
    (
        face_r_dim_next,
        tile_c_dim_next,
        num_faces_next,
        partial_face_next,
        narrow_tile_next,
    ) = SHAPES[shape_next]

    tile_size = 0x0A
    tile_size_next = 0xB

    templates = [
        PACK_RECONFIG_TEMPLATES(
            format_from,
            format_to,
            format_from_next,
            format_to_next,
        ),
    ]
    runtimes = PACK_RECONFIG_RUNTIMES(
        tile_size,
        face_r_dim,
        tile_c_dim,
        num_faces,
        partial_face,
        narrow_tile,
        tile_size_next,
        face_r_dim_next,
        tile_c_dim_next,
        num_faces_next,
        partial_face_next,
        narrow_tile_next,
    )

    configuration = TestConfig(
        "sources/state/reconfig/pack_reconfig_test.cpp",
        formats=FormatConfig(
            DataFormat.Float16,
            DataFormat.Float16,
            DataFormat.Float16,
            DataFormat.Float16,
            DataFormat.Float16,
        ),
        templates=templates,
        runtimes=[runtimes, RECONFIG_RUN_IDX(0)],
        dest_acc=DestAccumulation.Yes,
    )

    configuration.run(workers_tensix_coordinates)
    expected = TensixState.fetch(workers_tensix_coordinates)

    configuration = TestConfig(
        "sources/state/reconfig/pack_reconfig_test.cpp",
        formats=FormatConfig(
            DataFormat.Float16,
            DataFormat.Float16,
            DataFormat.Float16,
            DataFormat.Float16,
            DataFormat.Float16,
        ),
        templates=templates,
        runtimes=[runtimes, RECONFIG_RUN_IDX(1)],
        dest_acc=DestAccumulation.Yes,
    )

    configuration.run(workers_tensix_coordinates)
    actual = TensixState.fetch(workers_tensix_coordinates)

    TensixState.assert_equal(sanitize(expected), sanitize(actual))
