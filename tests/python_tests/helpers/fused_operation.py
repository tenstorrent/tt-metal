# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Tuple, Type

import torch

from .chip_architecture import ChipArchitecture, get_chip_architecture
from .format_config import DataFormat
from .fused_math import Math
from .fused_operand import Operand, OperandMapping
from .fused_packer import Packer
from .fused_unpacker import Unpacker, UnpackerTilizeA
from .llk_params import (
    DataCopyType,
    DestSync,
    MathFidelity,
    StochasticRounding,
    Tilize,
    Transpose,
    format_tile_sizes,
)
from .matmul_sweep import validate_tile_dimensions


@dataclass
class FusedOperation:
    unpacker: Type[Unpacker]
    math: Math
    packer: Type[Packer]
    operand_mapping: OperandMapping
    stage_id: int = 0
    num_stages: int = 1
    math_fidelity: MathFidelity = MathFidelity.HiFi4
    unpack_to_dest: bool = False
    unpack_transpose_faces: Transpose = Transpose.No
    unpack_transpose_within_face: Transpose = Transpose.No
    math_transpose_faces: Transpose = Transpose.No
    throttle: int = 0
    stochastic_rnd: StochasticRounding = StochasticRounding.No
    data_copy_type: DataCopyType = DataCopyType.A2D
    tiny_tiles: bool = False
    partial_face_A: bool = False
    partial_face_B: bool = False
    partial_face: bool = False
    num_faces: int = 4
    num_faces_A: int = 4
    num_faces_B: int = 4
    in0_tile_r_dim: int = 32
    in0_tile_c_dim: int = 32
    in1_tile_r_dim: int = 32
    in1_tile_c_dim: int = 32
    face_r_dim: int = 16
    face_c_dim: int = 16
    dest_sync: DestSync = DestSync.Half
    dst_index: int = 0
    srca_reuse_count: int = 4
    output_pack_dims: Tuple[int, int] = None

    def __post_init__(self):
        mapping = self.operand_mapping
        registry = mapping.operand_registry

        src_a = registry.get(mapping.src_a)
        src_b = registry.get(mapping.src_b)
        output = registry.get(mapping.output)

        input_A_dimensions = src_a.dimensions if src_a.dimensions else [32, 32]
        input_B_dimensions = src_b.dimensions if src_b.dimensions else [32, 32]

        from .format_config import InputOutputFormat

        formats = InputOutputFormat(
            input_format=src_a.data_format, output_format=output.data_format
        )

        TILE_SIZES = {
            DataFormat.Bfp8_b: 68,
            DataFormat.Float32: 256,
        }

        pack_size = TILE_SIZES.get(output.data_format, 128)
        unpack_size_a = TILE_SIZES.get(src_a.data_format, 128)
        unpack_size_b = TILE_SIZES.get(src_b.data_format, 128)

        if self.tiny_tiles:
            pack_size = (pack_size // self.num_faces) * (
                self.in0_tile_r_dim // self.face_r_dim
            )
            unpack_size_a = (unpack_size_a // self.num_faces_A) * (
                self.in0_tile_r_dim // self.face_r_dim
            )

        self.tile_size_pack = pack_size
        self.tile_size_unpack_a = unpack_size_a
        self.tile_size_unpack_b = unpack_size_b

        self.tile_size = 16 * 16 * self.num_faces

        self.buffer_A_tile_size = format_tile_sizes[formats.input_format]
        self.buffer_B_tile_size = format_tile_sizes[formats.input_format]
        self.buffer_Res_tile_size = format_tile_sizes[formats.output_format]

        num_rows = 32
        num_cols = 32

        validate_tile_dimensions(input_A_dimensions[0], num_rows)
        validate_tile_dimensions(input_A_dimensions[1], num_cols)
        validate_tile_dimensions(input_B_dimensions[0], num_rows)
        validate_tile_dimensions(input_B_dimensions[1], num_cols)

        full_rt_dim = input_A_dimensions[0] // num_rows
        full_ct_dim = input_B_dimensions[1] // num_cols

        self.full_rt_dim = full_rt_dim
        self.full_ct_dim = full_ct_dim
        self.block_rt_dim = full_rt_dim
        self.block_ct_dim = full_ct_dim

        self.rt_dim = input_A_dimensions[0] // num_rows
        self.ct_dim = input_B_dimensions[1] // num_cols
        self.kt_dim = input_A_dimensions[1] // num_cols

        if self.output_pack_dims is None:
            self.output_pack_dims = self.output.dimensions
        else:
            self.output_pack_dims[0] = min(
                self.output_pack_dims[0], self.output.dimensions[0]
            )
            self.output_pack_dims[1] = min(
                self.output_pack_dims[1], self.output.dimensions[1]
            )

        self.output_tiles_h = self.output_pack_dims[0] // 32
        self.output_tiles_w = self.output_pack_dims[1] // 32

        self.dest_tiles_h = self.output.dimensions[0] // 32
        self.dest_tiles_w = self.output.dimensions[1] // 32

        self.output_pack_tile_cnt = self.output_tiles_h * self.output_tiles_w

        self.output.pack_dims = self.output_pack_dims

        if (
            get_chip_architecture() == ChipArchitecture.BLACKHOLE
            and self.unpacker is UnpackerTilizeA
            and formats.input_format != DataFormat.Bfp8_b
        ):
            self.bh_tilize = Tilize.Yes
        else:
            self.bh_tilize = Tilize.No

    @property
    def src_a(self) -> Operand:
        mapping = self.operand_mapping
        return mapping.operand_registry.get(mapping.src_a)

    @property
    def src_b(self) -> Operand:
        mapping = self.operand_mapping
        return mapping.operand_registry.get(mapping.src_b)

    @property
    def output(self) -> Operand:
        mapping = self.operand_mapping
        return mapping.operand_registry.get(mapping.output)

    def unpack(self, config) -> str:
        unpacker_instance = self.unpacker()
        return unpacker_instance.exec(self, config)

    def do_math(self, config) -> str:
        return self.math.exec(self, config)

    def pack(self, config) -> str:
        packer_instance = self.packer()
        return packer_instance.exec(self, config)

    def golden(self, config) -> torch.Tensor:
        # calculate l1 golden
        src_a_dims = self.src_a.dimensions
        src_b_dims = self.src_b.dimensions

        tensor_a = self.src_a.raw_data.view(src_a_dims)
        tensor_b = self.src_b.raw_data.view(src_b_dims)

        tensor_a, tensor_b = self.unpacker().golden(tensor_a, tensor_b, self, config)
        l1_golden_tensor = self.math.golden(tensor_a, tensor_b, self, config)
        l1_golden_tensor = self.packer().golden(l1_golden_tensor, self, config)

        self.output.l1_golden = l1_golden_tensor.flatten()

        # calculate master golden
        golden_tensor_a = self.src_a.master_golden.view(src_a_dims)
        golden_tensor_b = self.src_b.master_golden.view(src_b_dims)

        golden_tensor_a, golden_tensor_b = self.unpacker().golden(
            golden_tensor_a, golden_tensor_b, self, config
        )
        master_golden_tensor = self.math.golden(
            golden_tensor_a, golden_tensor_b, self, config
        )
        master_golden_tensor = self.packer().golden(master_golden_tensor, self, config)

        self.output._master_golden = master_golden_tensor.flatten()

        self.output.dimensions = self.output_pack_dims

        return master_golden_tensor

    def __str__(self):
        return (
            f"\n{'='*60}\n"
            f"Operation {self.stage_id}\n"
            f"{'='*60}\n"
            f"  Unpacker: {self.unpacker.__name__}\n"
            f"  Math: {self.math}\n"
            f"    Fpu Math Op: {self.math.fpu}\n"
            f"    Sfpu Math Ops: {[op.__str__() for op in self.math.sfpu]}\n"
            f"  Packer: {self.packer.__name__}\n"
            f"  Src_A: {self.src_a}\n"
            f"  Src_B: {self.src_b}\n"
            f"  Output: {self.output}\n"
            f"  Math Fidelity: {self.math_fidelity}\n"
            f"  Unpack Transpose Faces: {self.unpack_transpose_faces}\n"
            f"  Unpack Transpose Within Faces: {self.unpack_transpose_within_face}\n"
        )
