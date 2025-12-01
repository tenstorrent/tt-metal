# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity, MathOperation


def get_valid_dest_accumulation_modes(formats):
    """
    Base constraints for Dest Accumulation modes.

    - Dest accumulation must be ENABLED for the following format combinations:
        - Input format has B type exponent (bfp8_b, float16_b)
        - Output format is A type exponent (float16)
        Reason: HW limitation, Packer cannot convert expB to expA, so we convert it to Float32 first as intermediate. (Source???)
    - Dest accumulation must be ENABLED for the following format combination:
        - Input format is 32bit integer format (int32, uint32)
        Reason: HW limitation, Unpacker cannot unpack 32bit integer formats into SrcA and SrcB registers
    - Otherwise it can be ENABLED or DISABLED

    NOTE: There are more combos that fit this rule, but aren't handled in the codebase
        So I'm not sure if they should also be handled here.
    """

    if (
        formats.input in [DataFormat.Bfp8_b, DataFormat.Float16_b]
        and formats.output == DataFormat.Float16
    ):
        return [DestAccumulation.Yes]

    if formats.input in [DataFormat.Int32, DataFormat.UInt32]:
        return [DestAccumulation.Yes]

    return [DestAccumulation.No, DestAccumulation.Yes]


def get_valid_math_fidelities(format, operation, PERF_RUN: bool = False):
    """
    Base constraints for Math Fidelity modes.

    - Regular mode:
        - Math fidelity must be LoFi for ElwAdd and ElwSub operations
        - Otherwise it can be LoFi, HiFi2, HiFi3, HiFi4.

    - Performance mode:
        - Ignores Math fidelity settings that are higher than necessary for full precision
    """

    if operation in [MathOperation.Elwadd, MathOperation.Elwsub]:
        return [MathFidelity.LoFi]

    # HiFi2 will multiply BFP8 and BFP8_b in full precision, skip HiFi3 and HiFi4
    if PERF_RUN and format.input in [DataFormat.Bfp8_b, DataFormat.Bfp8]:
        return [MathFidelity.LoFi, MathFidelity.HiFi2]

    # todo: once support for any of these is added, add them here
    # LoFi will multiply FP8, BFP4, BFP4a, BFP2, BFP2a in full precision, skip HiFi2 and higher

    return [
        MathFidelity.LoFi,
        MathFidelity.HiFi2,
        MathFidelity.HiFi3,
        MathFidelity.HiFi4,
    ]


def get_valid_dest_indices(
    dest_sync: DestSync,
    dest_acc: DestAccumulation,
    tile_count: int,
    all_indices: bool = False,
):
    """
    Base constraint for valid destination register indices.

    Capacity of the destination register is 16 tiles with 16bit datums.

    - When using DestSync.Half, the capacity is halved due to software double buffering.
    - When using DestAccumulation.Yes, the capacity is halved due to using tiles with 32bit datums.

    By default the function only returns the lowest and highest possible indices.
    This is to limit the number of tests. Use all_indices=True force the function to return all possible indices.
    """

    capacity_tiles = 16

    if dest_sync == DestSync.Half:
        capacity_tiles = capacity_tiles // 2

    if dest_acc == DestAccumulation.Yes:
        capacity_tiles = capacity_tiles // 2

    if tile_count > capacity_tiles:
        raise ValueError(
            f"Tried to fit {tile_count} tiles when Dest capacity is {capacity_tiles}"
        )

    start_index = 0
    end_index = capacity_tiles - tile_count

    if all_indices:
        return list(range(start_index, end_index + 1))

    return [start_index] if start_index == end_index else [start_index, end_index]
