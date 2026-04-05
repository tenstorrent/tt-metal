# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standardized tensor shape representation for LLK operations.

Replaces inconsistent tile size parameters (num_faces, face_r_dim,
narrow_tile, partial_face, VectorMode) with a unified class.
"""

from dataclasses import dataclass
from typing import Tuple

from .tile_constants import FACE_C_DIM, get_tile_params


@dataclass
class TileShape:
    """
    Standardized tensor shape representation for LLK operations.

    A tile is composed of faces arranged in a grid:
    - Total tile rows = face_r_dim * num_faces_r_dim
    - Total tile cols = face_c_dim * num_faces_c_dim

    Example: 32x32 tile = 4 faces of 16x16 (num_faces_r_dim=2, num_faces_c_dim=2)
    Example: 32x16 tile = 2 faces of 16x16 (num_faces_r_dim=2, num_faces_c_dim=1)
    """

    # Tile shape parameters
    face_r_dim: int  # Row dimension of each face (typically 16)
    face_c_dim: int  # Column dimension of each face (always 16 for HW)
    num_faces_r_dim: int  # Number of faces in row dimension
    num_faces_c_dim: int  # Number of faces in column dimension

    def total_row_dim(self) -> int:
        """Get total tile row dimension"""
        return self.face_r_dim * self.num_faces_r_dim

    def total_col_dim(self) -> int:
        """Get total tile column dimension"""
        return self.face_c_dim * self.num_faces_c_dim

    def total_tile_size(self) -> int:
        """Get total number of datums in the tile"""
        return self.total_row_dim() * self.total_col_dim()

    def total_num_faces(self) -> int:
        """Get total number of faces"""
        return self.num_faces_r_dim * self.num_faces_c_dim


def validate_tile_shape_tile_dependent_ops(tile_shape: TileShape) -> None:
    """
    Operations that are dependent on face positioning within a tile will have
    this function called to validate the tensor shape.

    Args:
        tile_shape: Tensor shape to validate

    Raises:
        AssertionError: If tensor shape is invalid
    """
    VALID_FACE_R_DIMS = [1, 2, 4, 8, 16]
    VALID_NUM_FACES = [1, 2, 4]

    assert (
        tile_shape.total_num_faces() in VALID_NUM_FACES
    ), f"total num_faces must be 1, 2, or 4, got {tile_shape.total_num_faces()}"
    assert (
        tile_shape.face_r_dim in VALID_FACE_R_DIMS
    ), f"face_r_dim must be 1, 2, 4, 8, or 16, got {tile_shape.face_r_dim}"
    assert (
        tile_shape.face_c_dim == FACE_C_DIM
    ), f"face_c_dim must be 16, got {tile_shape.face_c_dim}"


def construct_tile_shape(tile_dimensions: Tuple[int, int] = (32, 32)) -> TileShape:

    face_rows, num_faces_r_dim, num_faces_c_dim = get_tile_params(tile_dimensions)

    return TileShape(
        face_r_dim=face_rows,
        face_c_dim=FACE_C_DIM,
        num_faces_r_dim=num_faces_r_dim,
        num_faces_c_dim=num_faces_c_dim,
    )
