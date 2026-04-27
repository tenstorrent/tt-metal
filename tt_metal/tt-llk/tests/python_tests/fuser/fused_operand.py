# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from helpers.llk_params import DataFormat, PartialFace, format_dict, format_tile_sizes
from helpers.stimuli_generator import generate_random_face
from helpers.tile_constants import (
    DEFAULT_TILE_C_DIM,
    DEFAULT_TILE_R_DIM,
    calculate_tile_size_bytes,
)
from helpers.tile_shape import TileShape, construct_tile_shape
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.unpack import unpack_res_tiles
from ttexalens.tt_exalens_lib import read_from_device, write_to_device


@dataclass
class Operand:
    name: str
    dimensions: Tuple[int, int]
    data_format: DataFormat
    tile_shape: TileShape = construct_tile_shape(
        (DEFAULT_TILE_R_DIM, DEFAULT_TILE_C_DIM)
    )
    l1_address: Optional[int] = None
    is_output: bool = False
    sfpu: bool = True
    _raw_data: Optional[torch.Tensor] = None
    _master_golden: Optional[torch.Tensor] = None
    const_value: Optional[float] = None
    l1_golden: Optional[torch.Tensor] = None
    tile_count: Optional[int] = None
    tile_count_x: Optional[int] = None
    tile_count_y: Optional[int] = None
    tile_size: Optional[int] = None

    def __post_init__(self):
        self.tile_count_x = self.dimensions[1] // self.tile_shape.total_col_dim()
        self.tile_count_y = self.dimensions[0] // self.tile_shape.total_row_dim()
        self.tile_count = self.tile_count_x * self.tile_count_y

        self.partial_face = (
            PartialFace.Yes
            if (
                self.tile_shape.total_row_dim() < 16
                or self.tile_shape.total_col_dim() < 16
            )
            else PartialFace.No
        )

        TILE_SIZES = {
            DataFormat.Bfp8_b: 68,
            DataFormat.Float32: 256,
        }

        self.tile_size = (
            TILE_SIZES.get(self.data_format, 128) * self.tile_shape.total_num_faces()
        ) // 4

    def is_input(self) -> bool:
        return not self.is_output

    def generate_data(self):
        height, width = self.dimensions[0], self.dimensions[1]

        faces_needed = self.tile_count * self.tile_shape.total_num_faces()
        faces_data = []

        for _ in range(faces_needed):
            face = generate_random_face(
                stimuli_format=self.data_format,
                const_value=self.const_value,
                const_face=self.const_value is not None,
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

        self._raw_data = raw_data

    def set_data(self, raw_data: torch.Tensor):
        self._raw_data = raw_data

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

    def __str__(self) -> str:
        return f"{self.name}, {self.dimensions}, {self.data_format}, L1 Addr: {hex(self.l1_address)}"

    def calculate_l1_size(self) -> int:
        """Calculate the size in L1 memory for this operand."""
        tile_elements = self.tile_shape.total_tile_size()
        return self.data_format.num_bytes_per_tile(tile_elements) * self.tile_count

    def pack_for_l1(self) -> List[Tuple[int, List]]:
        """Pack operand data for writing to L1 memory.

        Returns:
            List of (address, packed_data) tuples, one per tile.
        """
        from helpers.pack import (
            pack_bfp8_b,
            pack_bfp16,
            pack_fp8_e4m3,
            pack_fp16,
            pack_fp32,
            pack_int8,
            pack_int16,
            pack_int32,
            pack_uint8,
            pack_uint16,
            pack_uint32,
        )

        packers = {
            DataFormat.Float16: pack_fp16,
            DataFormat.Float16_b: pack_bfp16,
            DataFormat.Float32: pack_fp32,
            DataFormat.Bfp8_b: pack_bfp8_b,
            DataFormat.Int32: pack_int32,
            DataFormat.UInt32: pack_uint32,
            DataFormat.Int16: pack_int16,
            DataFormat.UInt16: pack_uint16,
            DataFormat.Fp8_e4m3: pack_fp8_e4m3,
            DataFormat.Int8: pack_int8,
            DataFormat.UInt8: pack_uint8,
        }

        pack_function = packers.get(self.data_format)
        if not pack_function:
            raise ValueError(f"Unsupported data format: {self.data_format.name}")

        tile_elements = self.tile_shape.total_tile_size()
        tile_size = self.data_format.num_bytes_per_tile(tile_elements)

        if self.data_format != DataFormat.Bfp8_b:
            buffer = tilize_block(
                self.raw_data,
                dimensions=self.dimensions,
                stimuli_format=self.data_format,
            ).flatten()
        else:
            buffer = self.raw_data.flatten()

        packed_tiles = []

        for i in range(self.tile_count):
            start_idx = tile_elements * i
            tile_data = buffer[start_idx : start_idx + tile_elements]

            if self.data_format == DataFormat.Bfp8_b:
                packed = pack_function(
                    tile_data, num_faces=self.tile_shape.total_num_faces()
                )
            else:
                packed = pack_function(tile_data)

            addr = self.l1_address + i * tile_size
            packed_tiles.append((addr, packed))

        return packed_tiles

    @property
    def cpp_name(self) -> str:
        return f"{self.name}_buffer"

    def cpp_value(self, dest_acc: bool) -> str:
        buffer_size = calculate_tile_size_bytes(
            data_format=self.data_format,
            tile_dimensions=(
                self.tile_shape.total_row_dim(),
                self.tile_shape.total_col_dim(),
            ),
            format_tile_sizes=format_tile_sizes,
            use_srcs=True,
            dest_acc=dest_acc,
        )
        return f"[[maybe_unused]] const Operand {self.cpp_name}({hex(self.l1_address)}, {buffer_size});\n"


class OperandRegistry:
    def __init__(self):
        self.operands: dict[str, Operand] = {}

    def create(
        self,
        name: str,
        dimensions: Tuple[int, int],
        data_format: DataFormat,
        const_value: Optional[float] = None,
    ) -> Operand:
        if name in self.operands:
            operand = self.operands[name]
            if (dimensions is not None and dimensions != operand.dimensions) or (
                data_format is not None and data_format != operand.data_format
            ):
                raise ValueError(
                    f"Operand '{name}' exists with different parameters: "
                    f"requested dimensions={dimensions}, data_format={data_format}; "
                    f"existing dimensions={operand.dimensions}, data_format={operand.data_format}"
                )

            return operand

        operand = Operand(
            name=name,
            dimensions=dimensions,
            data_format=data_format,
            is_output=False,
            const_value=const_value,
        )
        self.operands[name] = operand
        return operand

    def get(
        self,
        name: str,
    ) -> Operand:
        if name not in self.operands:
            raise ValueError(
                f"Operand '{name}' not found in registry. Define it in the 'operands' section with dims and format."
            )

        return self.operands[name]

    def get_all_inputs(self) -> list[Operand]:
        return [op for op in self.operands.values() if op.is_input()]

    def get_all_outputs(self) -> list[Operand]:
        return [op for op in self.operands.values() if op.is_output]

    DEFAULT_L1_START_ADDRESS = 0x00021000
    DEFAULT_L1_END_ADDRESS = 0x00169FFF

    def allocate_l1_addresses(
        self,
        start_address: int = DEFAULT_L1_START_ADDRESS,
        end_address: int = DEFAULT_L1_END_ADDRESS,
    ) -> None:
        """
        Allocate L1 addresses for all operands.

        Addresses are allocated sequentially:
        1. All input operands
        2. All output operands
        """

        if (
            start_address < self.DEFAULT_L1_START_ADDRESS
            or start_address > self.DEFAULT_L1_END_ADDRESS
        ):
            raise ValueError(
                f"Start L1 address for operands must be between "
                f"{self.DEFAULT_L1_START_ADDRESS} and {self.DEFAULT_L1_END_ADDRESS}"
            )

        current_address = start_address

        for operand in self.get_all_inputs():
            if operand.l1_address is None:
                operand.l1_address = current_address
                current_address += operand.calculate_l1_size()

        for operand in self.get_all_outputs():
            if operand.l1_address is None:
                operand.l1_address = current_address
                current_address += operand.calculate_l1_size()
                if current_address > end_address:
                    raise ValueError(
                        "There is not enough space on device for all operands"
                    )

    def write_inputs_to_l1(self, location: str = "0,0") -> None:
        """Write all input operands to L1 memory."""

        for operand in self.get_all_inputs():
            for addr, packed_data in operand.pack_for_l1():
                write_to_device(location, addr, packed_data)

    def read_outputs_from_l1(self, location: str = "0,0") -> None:
        """Read output operands from L1 memory."""

        for output in self.get_all_outputs():
            if output.l1_address is None:
                raise ValueError(
                    f"Output operand '{output.name}' does not have an L1 address."
                )

            output_dimensions = output.dimensions
            output_format = output.data_format
            tile_cnt = output.tile_count
            tile_elements = output.tile_shape.total_tile_size()

            read_bytes_cnt = output_format.num_bytes_per_tile(tile_elements) * tile_cnt
            read_data = read_from_device(
                location, output.l1_address, num_bytes=read_bytes_cnt
            )

            res_from_l1 = unpack_res_tiles(
                read_data,
                output_format,
                tile_count=tile_cnt,
                sfpu=False,
                num_faces=output.tile_shape.total_num_faces(),
                face_r_dim=output.tile_shape.face_r_dim,
            )

            torch_format = format_dict[output_format]
            tilized_tensor = torch.tensor(res_from_l1, dtype=torch_format)

            if output_format != DataFormat.Bfp8_b and output_dimensions is not None:
                raw_tensor = untilize_block(
                    tilized_tensor,
                    stimuli_format=output_format,
                    dimensions=output_dimensions,
                )
            else:
                raw_tensor = tilized_tensor

            output._raw_data = raw_tensor

    def generate_cpp(self, dest_acc: bool):
        code = "// Inputs\n"
        for operand in self.get_all_inputs():
            code += operand.cpp_value(dest_acc)

        code += "// Outputs\n"
        for operand in self.get_all_outputs():
            code += operand.cpp_value(dest_acc)

        return code
