# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync
from helpers.param_config import input_output_formats

FAST_UNTILIZE_NUM_FACES = 4
FAST_UNTILIZE_FACE_R = 16
FAST_UNTILIZE_FACE_C = 16
FAST_UNTILIZE_TILE_R = 32
FAST_UNTILIZE_TILE_C = 32
FAST_UNTILIZE_TILE_FACE_ROWS = FAST_UNTILIZE_NUM_FACES * FAST_UNTILIZE_FACE_R
FAST_UNTILIZE_RT_DIMS = [1, 2, 4]
FAST_UNTILIZE_BASE_CT_DIMS = [2, 3, 4, 5, 6, 7, 8]
FAST_UNTILIZE_EXTENDED_CT_DIMS = [9, 12, 16]
FAST_UNTILIZE_CT_DIMS = FAST_UNTILIZE_BASE_CT_DIMS + FAST_UNTILIZE_EXTENDED_CT_DIMS
FAST_UNTILIZE_DIMS = [
    (rt_dim, ct_dim)
    for rt_dim in FAST_UNTILIZE_RT_DIMS
    for ct_dim in FAST_UNTILIZE_CT_DIMS
]
FAST_UNTILIZE_DEST_SYNC_MODES = [DestSync.Half, DestSync.Full]


def fast_untilize_formats():
    return [
        *input_output_formats([DataFormat.Float16_b, DataFormat.Float32], same=True),
        InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Float32),
        InputOutputFormat(DataFormat.Bfp4_b, DataFormat.Float16_b),
        InputOutputFormat(DataFormat.Bfp4_b, DataFormat.Float32),
    ]


def fast_untilize_dest_acc_modes(formats):
    if formats.output_format == DataFormat.Float32:
        return [DestAccumulation.Yes]
    return [DestAccumulation.No, DestAccumulation.Yes]
