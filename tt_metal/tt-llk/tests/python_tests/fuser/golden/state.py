# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from collections import deque
from copy import copy
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from helpers.llk_params import format_dict
from helpers.tilize_untilize import tilize_block, untilize_block

if TYPE_CHECKING:
    from ..l1_operation import L1Operation
    from ..operand import Operand


def tile_dimensions(tile_shape) -> tuple:
    return (tile_shape.total_row_dim(), tile_shape.total_col_dim())


def tile_operation(operation: "L1Operation") -> "L1Operation":
    dimensions = tile_dimensions(operation.tile_shape)
    single = copy(operation)
    single.max_output_dimensions = dimensions
    single.block_size = dimensions
    single.block_tiles_x = 1
    single.block_tiles_y = 1
    return single


class OperandTiles:
    def __init__(self, operand: "Operand", tensor: torch.Tensor):
        self.operand = operand
        self.tile_dims = tile_dimensions(operand.tile_shape)
        self.num_faces = operand.tile_shape.total_num_faces()
        self._tiles = tilize_block(
            tensor,
            operand.dimensions,
            operand.data_format,
            num_faces=self.num_faces,
            tile_dimensions=self.tile_dims,
        ).view(operand.tile_count, -1)
        self._row_major = None

    def tile(self, index: int) -> torch.Tensor:
        return untilize_block(
            self._tiles[index].flatten(),
            self.operand.data_format,
            self.tile_dims,
            tile_dimensions=self.tile_dims,
            num_faces=self.num_faces,
        ).reshape(self.tile_dims)

    def tilized_tile(self, index: int) -> torch.Tensor:
        return self._tiles[index].reshape(self.tile_dims).clone()

    def tilized_region(self, rows: slice, cols: slice) -> torch.Tensor:
        return self._tiles.reshape(self.operand.dimensions)[rows, cols].clone()

    def row_major_region(self, rows: slice, cols: slice) -> torch.Tensor:
        if self._row_major is None:
            self._row_major = untilize_block(
                self._tiles.flatten(),
                self.operand.data_format,
                self.operand.dimensions,
                tile_dimensions=self.tile_dims,
                num_faces=self.num_faces,
            )
        return self._row_major[rows, cols].contiguous()


class OutputLayout(Enum):
    ROW_MAJOR = auto()  # untilize the whole tile grid (also the L1-accumulation path)
    TILED = auto()  # tile-concatenated L1 result (tilize unpack path)
    UNTILIZE = auto()  # arrange tiles spatially, untilize once (untilize packer)


def finalize_output(
    layout: OutputLayout, buffer: Dict[int, List[torch.Tensor]], operand: "Operand"
) -> torch.Tensor:
    tile_dims = tile_dimensions(operand.tile_shape)
    num_faces = operand.tile_shape.total_num_faces()
    data_format = operand.data_format
    dtype = format_dict[data_format]

    def summed(index: int) -> torch.Tensor:
        tiles = buffer.get(index)
        if not tiles:
            return torch.zeros(tile_dims, dtype=dtype)
        total = tiles[0]
        for tile in tiles[1:]:
            total = total + tile
        return total.reshape(tile_dims)

    tiles = torch.stack([summed(i) for i in range(operand.tile_count)])
    if layout == OutputLayout.TILED:
        return tiles.reshape(operand.dimensions)

    rows, cols = tile_dims
    tile_grid = (
        tiles.to(dtype)
        .reshape(operand.tile_count_y, operand.tile_count_x, rows, cols)
        .permute(0, 2, 1, 3)
        .reshape(operand.dimensions)
    )

    if layout == OutputLayout.UNTILIZE:
        return untilize_block(
            tile_grid.flatten(),
            data_format,
            operand.dimensions,
            tile_dimensions=tile_dims,
            num_faces=num_faces,
        )

    return tile_grid


class SourceRegisters:
    def __init__(self):
        self.a = deque()
        self.b = deque()

    def push(self, tile_a: Optional[torch.Tensor], tile_b: Optional[torch.Tensor]):
        if tile_a is not None:
            self.a.append(tile_a)
        if tile_b is not None:
            self.b.append(tile_b)

    def pop(self):
        return (
            self.a.popleft() if self.a else None,
            self.b.popleft() if self.b else None,
        )

    def pop_operands(self, dimensions):
        tensor_a, tensor_b = self.pop()
        if tensor_a is None:
            tensor_a = torch.zeros(dimensions)
        if tensor_b is None:
            tensor_b = torch.zeros(dimensions)
        return tensor_a, tensor_b


class DestBank:
    def __init__(
        self,
        tiles: int,
        tile_dims: tuple,
        num_faces: int,
        dtype,
        block_tiles_x: int = None,
        block_tiles_y: int = None,
    ):
        self.tile_dims = tile_dims
        self.num_faces = num_faces
        # Actual extents of the region this bank holds, so block/row-granular
        # goldens can locate a call's row/column within the block (remainder aware).
        self.block_tiles_x = block_tiles_x
        self.block_tiles_y = block_tiles_y
        self._tiles = [torch.zeros(tile_dims, dtype=dtype) for _ in range(tiles)]

    def __len__(self) -> int:
        return len(self._tiles)

    def get(self, index: int) -> torch.Tensor:
        return self._tiles[index].clone()

    def set(self, index: int, tile: torch.Tensor) -> None:
        self._tiles[index] = tile.reshape(self.tile_dims).clone()

    def tilized(self, data_format) -> torch.Tensor:
        rows, cols = self.tile_dims
        tiles = torch.stack(self._tiles).reshape(len(self) * rows, cols)
        return tilize_block(
            tiles,
            tiles.shape,
            data_format,
            self.num_faces,
            tile_dimensions=self.tile_dims,
        ).flatten()

    def update_from_tilized(self, tensor: torch.Tensor, data_format) -> None:
        rows, cols = self.tile_dims
        tiles = untilize_block(
            tensor,
            data_format,
            (len(self) * rows, cols),
            tile_dimensions=self.tile_dims,
            num_faces=self.num_faces,
        ).reshape(len(self), rows, cols)
        self._tiles = list(tiles.unbind())


class Inputs:
    def __init__(
        self,
        view_a: Optional[OperandTiles],
        view_b: Optional[OperandTiles],
        block_tiles_x: int = None,
        block_tiles_y: int = None,
    ):
        self.view_a = view_a
        self.view_b = view_b
        # Actual extents of the region, so row/block-granular unpack goldens can
        # gather all the tiles a single call covers (remainder aware).
        self.block_tiles_x = block_tiles_x
        self.block_tiles_y = block_tiles_y

    def tile_a(self, index) -> Optional[torch.Tensor]:
        if index is None or self.view_a is None:
            return None
        return self.view_a.tile(index)

    def tile_b(self, index) -> Optional[torch.Tensor]:
        if index is None or self.view_b is None:
            return None
        return self.view_b.tile(index)


class GoldenState:
    def __init__(self, dest: DestBank, relu_configs=None):
        self.dest = dest
        self.inputs: Optional[Inputs] = None
        self.source_registers = SourceRegisters()
        self.output = None
        self.relu_configs = {} if relu_configs is None else relu_configs

    def begin_fpu(self, inputs: Inputs) -> None:
        self.inputs = inputs
        self.source_registers = SourceRegisters()
