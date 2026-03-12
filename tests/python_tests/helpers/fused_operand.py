# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from helpers.llk_params import DataFormat, format_dict
from helpers.stimuli_generator import generate_random_face
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from helpers.tile_shape import TileShape, construct_tile_shape
from helpers.tilize_untilize import tilize_block


@dataclass
class Operand:
    name: str
    dimensions: Optional[Tuple[int, int]] = None
    data_format: Optional[DataFormat] = None
    tile_shape: Optional[TileShape] = None
    l1_address: Optional[int] = None
    is_output: bool = False
    sfpu: bool = True
    _data: Optional[torch.Tensor] = None
    _raw_data: Optional[torch.Tensor] = None
    _master_golden: Optional[torch.Tensor] = None
    l1_golden: Optional[torch.Tensor] = None
    _tile_count: Optional[int] = None
    tile_count_x: Optional[int] = None
    tile_count_y: Optional[int] = None

    def __post_init__(self):
        if not self.is_output and (self.dimensions is None or self.data_format is None):
            raise ValueError(
                f"Input operand '{self.name}' must have dimensions and data_format"
            )

        if self.tile_shape is None:
            self.tile_shape = construct_tile_shape(
                (DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM)
            )

        if self.dimensions is not None:
            self.tile_count_x = self.dimensions[1] // self.tile_shape.total_col_dim()
            self.tile_count_y = self.dimensions[0] // self.tile_shape.total_row_dim()
            self._tile_count = self.tile_count_x * self.tile_count_y

    def is_input(self) -> bool:
        return not self.is_output

    def generate_data(self, const_value=None):
        if self._data is not None:
            return

        if self.dimensions is None or self.data_format is None:
            raise ValueError(
                f"Cannot generate data for operand '{self.name}' without dimensions and format"
            )

        height, width = self.dimensions[0], self.dimensions[1]
        tile_rows = self.tile_shape.total_row_dim()
        tile_cols = self.tile_shape.total_col_dim()
        tile_count = (height // tile_rows) * (width // tile_cols)

        faces_needed = tile_count * self.tile_shape.total_num_faces()
        faces_data = []

        for _ in range(faces_needed):
            face = generate_random_face(
                stimuli_format=self.data_format,
                const_value=const_value,
                const_face=const_value is not None,
                sfpu=self.sfpu,
                face_r_dim=self.tile_shape.face_r_dim,
                negative_values=False,
            )
            faces_data.extend(face.tolist())

        dtype = (
            format_dict[self.data_format]
            if self.data_format != DataFormat.Bfp8_b
            else torch.bfloat16
        )
        raw_data = torch.tensor(faces_data[: height * width], dtype=dtype).view(
            height, width
        )

        if self.data_format != DataFormat.Bfp8_b:
            tilized_data = tilize_block(
                raw_data, dimensions=self.dimensions, stimuli_format=self.data_format
            )
        else:
            tilized_data = raw_data

        self._raw_data = raw_data
        self._data = tilized_data
        self._tile_count = tile_count

    def set_data(self, raw_data: torch.Tensor):
        self._raw_data = raw_data

        if self.data_format != DataFormat.Bfp8_b:
            tilized_data = tilize_block(
                raw_data, dimensions=self.dimensions, stimuli_format=self.data_format
            )
        else:
            tilized_data = raw_data

        self._data = tilized_data

    @property
    def data(self) -> Optional[torch.Tensor]:
        if self._data is None and self.is_input():
            self.generate_data()
        return self._data

    @data.setter
    def data(self, value: torch.Tensor):
        self._data = value

    @property
    def raw_data(self) -> Optional[torch.Tensor]:
        if self._raw_data is None and self.is_input():
            self.generate_data()
        return self._raw_data

    @property
    def master_golden(self) -> Optional[torch.Tensor]:
        if self.is_input():
            return self.raw_data
        return self._master_golden

    @property
    def tile_count(self) -> Optional[int]:
        if self._tile_count is None:
            if self.dimensions is not None:
                tile_rows = self.tile_shape.total_row_dim()
                tile_cols = self.tile_shape.total_col_dim()
                self._tile_count = (self.dimensions[0] // tile_rows) * (
                    self.dimensions[1] // tile_cols
                )
            elif self.is_input():
                self.generate_data()
        return self._tile_count

    def __str__(self) -> str:
        return f"{self.name}, {self.dimensions}, {self.data_format}, L1 Addr: {hex(self.l1_address)}"


class OperandMapping:
    def __init__(
        self,
        src_a: str,
        src_b: str,
        output: str,
        operand_registry: "OperandRegistry" = None,
    ):
        self.src_a = src_a
        self.src_b = src_b
        self.output = output
        self.operand_registry = operand_registry

    def create_output_operand(
        self,
        operand_registry: "OperandRegistry",
        output_format: DataFormat,
        output_dims: Tuple[int, int],
    ):
        if self.output in operand_registry.operands:
            return

        max_output_dims = self.resolve_output_dimensions(operand_registry)

        if output_dims[0] > max_output_dims[0] or output_dims[1] > max_output_dims[1]:
            raise ValueError(f"Max output dimensions are {max_output_dims}")

        operand_registry.add_output(
            name=self.output,
            dimensions=output_dims,
            data_format=output_format,
        )

    def resolve_output_dimensions(
        self, operand_registry: "OperandRegistry"
    ) -> Tuple[int, int]:
        src_a_op = operand_registry.get(self.src_a)
        src_b_op = operand_registry.get(self.src_b)

        M = src_a_op.dimensions[0]
        N = src_b_op.dimensions[1]

        return (M, N)

    def get_output_tile_count(self, operand_registry: "OperandRegistry") -> int:
        dims = self.resolve_output_dimensions(operand_registry)
        output_op = operand_registry.get(self.output)
        tile_rows = output_op.tile_shape.total_row_dim()
        tile_cols = output_op.tile_shape.total_col_dim()
        return (dims[0] // tile_rows) * (dims[1] // tile_cols)


class OperandRegistry:
    def __init__(self):
        self.operands: dict[str, Operand] = {}

    def add_input(
        self,
        name: str,
        dimensions: Tuple[int, int, int, int],
        data_format: DataFormat,
        address: int = None,
        sfpu: bool = True,
    ) -> Operand:
        if name in self.operands:
            raise ValueError(f"Operand '{name}' already exists")

        operand = Operand(
            name=name,
            dimensions=dimensions,
            data_format=data_format,
            l1_address=address,
            is_output=False,
            sfpu=sfpu,
        )
        self.operands[name] = operand
        return operand

    def add_output(
        self,
        name: str,
        address: int = None,
        dimensions: Optional[Tuple[int, int]] = None,
        data_format: Optional[DataFormat] = None,
    ) -> Operand:
        if name in self.operands:
            raise ValueError(f"Output operand '{name}' already exists")

        operand = Operand(
            name=name,
            dimensions=dimensions,
            data_format=data_format,
            l1_address=address,
            is_output=True,
        )
        self.operands[name] = operand
        return operand

    def get(self, name: str) -> Operand:
        if name not in self.operands:
            raise KeyError(f"Operand '{name}' not found")
        return self.operands[name]

    def get_all_inputs(self) -> list[Operand]:
        return [op for op in self.operands.values() if op.is_input()]

    def get_all_outputs(self) -> list[Operand]:
        return [op for op in self.operands.values() if op.is_output]

    def update_data(self, name: str, data: torch.Tensor):
        if name not in self.operands:
            raise KeyError(f"Operand '{name}' not found")
        self.operands[name].data = data

    def create_mapping(
        self,
        src_a: str,
        src_b: str,
        output: str,
        src_a_dims: Tuple[int, int] = [32, 32],
        src_b_dims: Tuple[int, int] = [32, 32],
        output_dims: Tuple[int, int] = [64, 64],
        input_format: DataFormat = DataFormat.Float16_b,
        output_format: DataFormat = DataFormat.Float16_b,
        src_a_tensor: torch.Tensor = None,
        src_b_tensor: torch.Tensor = None,
        src_a_const_value: Optional[float] = None,
        src_b_const_value: Optional[float] = None,
    ) -> OperandMapping:
        if src_a not in self.operands:
            self.add_input(src_a, dimensions=src_a_dims, data_format=input_format)
        else:
            existing = self.operands[src_a]
            if list(existing.dimensions) != list(src_a_dims):
                raise ValueError(
                    f"Operand '{src_a}' already exists with dimensions {existing.dimensions}, got {src_a_dims}"
                )

        if src_b not in self.operands:
            self.add_input(src_b, dimensions=src_b_dims, data_format=input_format)
        else:
            existing = self.operands[src_b]
            if list(existing.dimensions) != list(src_b_dims):
                raise ValueError(
                    f"Operand '{src_b}' already exists with dimensions {existing.dimensions}, got {src_b_dims}"
                )

        if src_a_tensor is not None:
            self.operands[src_a].set_data(src_a_tensor)
        else:
            self.operands[src_a].generate_data(const_value=src_a_const_value)

        if src_b_tensor is not None:
            self.operands[src_b].set_data(src_b_tensor)
        else:
            self.operands[src_b].generate_data(const_value=src_b_const_value)

        mapping = OperandMapping(
            src_a=src_a,
            src_b=src_b,
            output=output,
            operand_registry=self,
        )

        mapping.create_output_operand(self, output_format, output_dims)

        return mapping
