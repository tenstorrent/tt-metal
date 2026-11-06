# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Result tile shape of a Quasar FPU reduce, and the matching golden and L1 layout.

A column reduce (REDUCE_COL) leaves one row per face column in dest, laid out as a 1xN
tile (1x32, or 1x16 for a one-face-wide input), so the kernel packs it with that 1xN
shape. Row and scalar reduce results keep the input tile shape.
"""

import torch

from .llk_params import ReduceDimension, format_tile_sizes
from .stimuli_config import StimuliConfig
from .tile_constants import FACE_C_DIM, calculate_tile_size_bytes
from .tile_shape import TileShape


def reduce_result_tile_shape(
    reduce_dim: ReduceDimension, input_tile_shape: TileShape
) -> TileShape:
    """Tile shape the reduce result is packed with: 1xN for a column reduce, else the input shape."""
    if reduce_dim != ReduceDimension.Column:
        return input_tile_shape
    return TileShape(
        face_r_dim=1,
        face_c_dim=FACE_C_DIM,
        num_faces_r_dim=1,
        num_faces_c_dim=input_tile_shape.num_faces_c_dim,
    )


def reduce_result_golden(
    golden: torch.Tensor,
    reduce_dim: ReduceDimension,
    input_tile_shape: TileShape,
    tile_count: int,
) -> torch.Tensor:
    """Cut a full-tile reduce golden down to the packed result.

    The reduce goldens return one input-shaped tile per result. For a column reduce,
    keep row 0 of each face in the first face row, which is the 1xN result tile.
    """
    if reduce_dim != ReduceDimension.Column:
        return golden
    faces = golden.reshape(
        tile_count,
        input_tile_shape.total_num_faces(),
        input_tile_shape.face_r_dim,
        FACE_C_DIM,
    )
    return faces[:, : input_tile_shape.num_faces_c_dim, 0, :].flatten()


class ReduceStimuliConfig(StimuliConfig):
    """Dense L1 layout where the inputs (A, B) use one tile shape and the result another.

    The base fields (tile_dimensions, num_faces, face_r_dim) describe the result tile;
    input_tile_shape describes A and B.
    """

    def __init__(self, *args, input_tile_shape: TileShape, **kwargs):
        self.input_tile_shape = input_tile_shape
        super().__init__(*args, **kwargs)

    def _calculate_tile_sizes(self):
        super()._calculate_tile_sizes()
        if any(True for _ in self._active_optional_operands()):
            raise ValueError("ReduceStimuliConfig supports only operands A, B and Res")
        self.tile_size_A_bytes = calculate_tile_size_bytes(
            self.stimuli_A_format,
            self.input_tile_shape.tile_dims,
            format_tile_sizes,
            use_srcs=self._operand_use_srcs("A"),
        )
        self.tile_size_B_bytes = calculate_tile_size_bytes(
            self.stimuli_B_format,
            self.input_tile_shape.tile_dims,
            format_tile_sizes,
            use_srcs=self._operand_use_srcs("B"),
        )
        self.buf_b_addr = self.buf_a_addr + self.tile_size_A_bytes * self.tile_count_A
        self.buf_res_addr = self.buf_b_addr + self.tile_size_B_bytes * self.tile_count_B

    def _write_dense_tile_dimensions(self, location: str = "0,0"):
        for operand, buffer, tile_count, data_format, address, tile_size in (
            (
                "A",
                self.buffer_A,
                self.tile_count_A,
                self.stimuli_A_format,
                self.buf_a_addr,
                self.tile_size_A_bytes,
            ),
            (
                "B",
                self.buffer_B,
                self.tile_count_B,
                self.stimuli_B_format,
                self.buf_b_addr,
                self.tile_size_B_bytes,
            ),
        ):
            pack_function = StimuliConfig.get_packer(data_format)
            if not pack_function:
                raise ValueError(
                    f"Unsupported data format for operand {operand}: {data_format.name}"
                )
            StimuliConfig.write_matrix_w_tile_dimensions(
                buffer,
                tile_count,
                pack_function,
                address,
                tile_size,
                self.input_tile_shape.total_num_faces(),
                self.input_tile_shape.face_r_dim,
                self.input_tile_shape.tile_dims,
                location,
                use_srcs=self._operand_use_srcs(operand),
                twos_complement=self.twos_complement,
            )
