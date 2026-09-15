# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.chip_architecture import ChipArchitecture
from helpers.llk_params import Transpose


def unpack_matmul_golden(call, state, node, operation, config):
    a_rows = node.src_a.tile_shape.total_row_dim()
    b_cols = node.src_b.tile_shape.total_col_dim()
    out_tiles_x = operation.max_output_dimensions[1] // b_cols
    row0, col0 = divmod(call.in0, out_tiles_x)
    rt = state.inputs.block_tiles_y
    ct = state.inputs.block_tiles_x
    tensor_a = state.inputs.view_a.row_major_region(
        slice(row0 * a_rows, (row0 + rt) * a_rows), slice(None)
    )
    tensor_b = state.inputs.view_b.row_major_region(
        slice(None), slice(col0 * b_cols, (col0 + ct) * b_cols)
    )
    if (
        config.architecture in (ChipArchitecture.WORMHOLE, ChipArchitecture.BLACKHOLE)
        and node.transpose_within_face == Transpose.Yes
    ):
        b_rows = node.src_b.tile_shape.total_row_dim()
        if (b_rows, b_cols) == (32, 16):
            raise ValueError("Matmul transpose does not support 32x16 RHS tiles")
        tiles_b = tensor_b.reshape(-1, b_rows, ct, b_cols)
        if (b_rows, b_cols) == (16, 32):
            transposed = tiles_b.new_zeros(tiles_b.shape)
            transposed[:, :, :, :16] = tiles_b[:, :, :, :16].transpose(1, 3)
        else:
            transposed = tiles_b.transpose(1, 3)
        tensor_b = transposed.reshape(tensor_b.shape)
    state.source_registers.push(tensor_a, tensor_b)
