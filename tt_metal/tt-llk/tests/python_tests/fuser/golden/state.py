# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from collections import deque
from typing import TYPE_CHECKING, Dict, List, Optional

import torch
from helpers.llk_params import format_dict
from helpers.tilize_untilize import tilize_block, untilize_block

if TYPE_CHECKING:
    from ..operand import Operand


def tile_dimensions(tile_shape) -> tuple:
    return (tile_shape.total_row_dim(), tile_shape.total_col_dim())


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

    def tile(self, index: int) -> torch.Tensor:
        return untilize_block(
            self._tiles[index].flatten(),
            self.operand.data_format,
            self.tile_dims,
            tile_dimensions=self.tile_dims,
            num_faces=self.num_faces,
        ).reshape(self.tile_dims)

    def strided_tile(self, index: int) -> torch.Tensor:
        row, col = divmod(index, self.operand.tile_count_x)
        rows, cols = self.tile_dims
        return self._tiles.reshape(self.operand.dimensions)[
            row * rows : (row + 1) * rows, col * cols : (col + 1) * cols
        ].clone()

    def block(self, index: int, rows: int, cols: int) -> torch.Tensor:
        indices = [
            index + row * self.operand.tile_count_x + col
            for row in range(rows)
            for col in range(cols)
        ]
        return untilize_block(
            self._tiles[indices].flatten(),
            self.operand.data_format,
            (rows * self.tile_dims[0], cols * self.tile_dims[1]),
            tile_dimensions=self.tile_dims,
            num_faces=self.num_faces,
        )


def finalize_output(
    buffer: Dict[int, List[torch.Tensor]], operand: "Operand"
) -> torch.Tensor:
    result = torch.zeros(
        operand.dimensions[0] * operand.dimensions[1],
        dtype=format_dict[operand.data_format],
    )
    for offset, writes in buffer.items():
        total = writes[0].to(result.dtype)
        for value in writes[1:]:
            total = total + value.to(result.dtype)
        result[offset : offset + total.numel()] = total.flatten()
    return untilize_block(
        result,
        operand.data_format,
        operand.dimensions,
        tile_dimensions=tile_dimensions(operand.tile_shape),
        num_faces=operand.tile_shape.total_num_faces(),
    )


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
    ):
        self.tile_dims = tile_dims
        self.num_faces = num_faces
        self.block_tiles_x = None
        self.block_tiles_y = None
        self._tiles = [torch.zeros(tile_dims, dtype=dtype) for _ in range(tiles)]
        self.written_tiles = set()

    def __len__(self) -> int:
        return len(self._tiles)

    def get(self, index: int) -> torch.Tensor:
        return self._tiles[index].clone()

    def set(self, index: int, tile: torch.Tensor) -> None:
        self._tiles[index] = tile.reshape(self.tile_dims).clone()
        self.written_tiles.add(index)

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

    def update_from_tilized(self, tensor: torch.Tensor, data_format, indices) -> None:
        rows, cols = self.tile_dims
        tiles = untilize_block(
            tensor,
            data_format,
            (len(self) * rows, cols),
            tile_dimensions=self.tile_dims,
            num_faces=self.num_faces,
        ).reshape(len(self), rows, cols)
        for index in indices:
            self.set(index, tiles[index])


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
