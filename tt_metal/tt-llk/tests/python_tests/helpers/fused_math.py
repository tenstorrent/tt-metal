# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List, Tuple, Union

import torch

if TYPE_CHECKING:
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig

from .block_data import BlockData
from .chip_architecture import ChipArchitecture
from .fused_fpu import Fpu, ReduceBlockMaxFpu, ReduceFpu
from .fused_packer import Packer
from .fused_sfpu import Sfpu
from .fused_unpacker import Unpacker
from .llk_params import (
    BroadcastType,
    DataCopyType,
    EltwiseBinaryReuseDestType,
    PerfRunType,
    Transpose,
)
from .tilize_untilize import tilize_block, untilize_block


class ComputeNode:
    def __init__(
        self,
        unpacker: Unpacker = None,
        fpu: Fpu = None,
        sfpu: Sfpu = None,
        unpack_transpose_faces: Transpose = Transpose.No,
        unpack_transpose_within_face: Transpose = Transpose.No,
        broadcast_type: BroadcastType = BroadcastType.None_,
        data_copy_type: DataCopyType = DataCopyType.A2D,
        reuse_dest: EltwiseBinaryReuseDestType = EltwiseBinaryReuseDestType.NONE,
    ):
        if fpu is None and sfpu is None:
            raise ValueError("Compute unit needs an fpu or sfpu unit")
        if fpu is not None and sfpu is not None:
            raise ValueError("Compute unit can be only fpu or sfpu")
        if sfpu is not None and unpacker is not None:
            raise ValueError("Sfpu unit does not support unpacker")

        self.unpacker = unpacker
        self.fpu = fpu
        self.sfpu = sfpu
        self.unpack_transpose_faces = unpack_transpose_faces
        self.unpack_transpose_within_face = unpack_transpose_within_face
        self.broadcast_type = broadcast_type
        self.reuse_dest = reuse_dest

        if (
            self.broadcast_type != BroadcastType.None_
            and data_copy_type == DataCopyType.A2D
        ):
            self.data_copy_type = DataCopyType.B2D
        elif (
            self.broadcast_type == BroadcastType.None_
            and data_copy_type == DataCopyType.B2D
        ):
            self.data_copy_type = DataCopyType.A2D
        else:
            self.data_copy_type = data_copy_type

    def unpack(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if self.unpacker is None:
            return ""

        code = ""
        skip_init = config.perf_run_type in (
            PerfRunType.PACK_ISOLATE,
            PerfRunType.MATH_ISOLATE,
        )
        if not skip_init:
            code += self.unpacker().init(operation, config, self, block)

        code += self.unpacker().loop.unpack_loop(operation, config, self, block)
        if not skip_init:
            code += self.unpacker().uninit(operation, config, self, block)

        return code

    def fpu_calculate(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if self.fpu is None:
            return ""

        code = ""
        skip_init = config.perf_run_type in (
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        )
        if not skip_init:
            code += self.fpu.init(operation, config, self, block)

        code += self.fpu.loop.math_loop(operation, config, self, block)
        if not skip_init:
            code += self.fpu.uninit(operation, config, self, block)

        return code

    def sfpu_calculate(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ):
        if self.sfpu is None:
            return ""

        if config.perf_run_type in (
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ):
            return ""

        code = self.sfpu.init(operation, config, self, block)
        code += self.sfpu.calculate(operation, config, self, block)
        code += self.sfpu.uninit(operation, config, self, block)
        return code

    def math_calculate(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        block: BlockData,
    ) -> str:
        if self.fpu is not None:
            return self.fpu_calculate(operation, config, block)
        elif self.sfpu is not None:
            return self.sfpu_calculate(operation, config, block)
        else:
            raise ValueError("fpu and sfpu are not defined")

    def golden(
        self,
        input_tensor_a,
        input_tensor_b,
        tensor_a,
        tensor_b,
        tensor_dst,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.unpacker is not None:
            unpacked_tensor_a, unpacked_tensor_b = self.unpacker().golden(
                input_tensor_a, input_tensor_b, operation, config, self
            )

            if unpacked_tensor_a is not None:
                tensor_a = unpacked_tensor_a

            if unpacked_tensor_b is not None:
                tensor_b = unpacked_tensor_b

        if self.fpu is not None:
            tensor_a, tensor_b, tensor_dst = self.fpu.golden(
                tensor_a, tensor_b, tensor_dst, operation, config, self
            )

        if self.sfpu is not None:
            tilized_dst = tilize_block(
                tensor_dst,
                operation.max_output_dimensions,
                operation.output.data_format,
            )

            tile_count_x = operation.output.tile_count_x
            tile_count_y = operation.output.tile_count_y
            block_tiles_x = operation.block_tiles_x
            block_tiles_y = operation.block_tiles_y

            full_blocks_x = tile_count_x // block_tiles_x
            full_blocks_y = tile_count_y // block_tiles_y
            remaining_tiles_x = tile_count_x % block_tiles_x
            remaining_tiles_y = tile_count_y % block_tiles_y

            full_x_limit = full_blocks_x * block_tiles_x
            full_y_limit = full_blocks_y * block_tiles_y

            tile_size = tilized_dst.shape[1]

            def process_block(block_x, block_y, block_tiles_x_eff, block_tiles_y_eff):
                block_tile_ids = []
                for tile_y in range(block_tiles_y_eff):
                    for tile_x in range(block_tiles_x_eff):
                        tile_id = tile_count_x * (block_y + tile_y) + (block_x + tile_x)
                        block_tile_ids.append(tile_id)

                block_tile_cnt = len(block_tile_ids)
                if block_tile_cnt == 0:
                    return

                block_tensor = tilized_dst[block_tile_ids, :].clone().flatten()
                block_dims = (block_tile_cnt * 32, 32)

                block_tensor = self.sfpu.golden(
                    block_tensor,
                    operation,
                    config,
                    self,
                    block_dims,
                    block_tile_cnt,
                )

                tilized_dst[block_tile_ids, :] = block_tensor.view(
                    block_tile_cnt, tile_size
                )

            if full_blocks_x > 0 and full_blocks_y > 0:
                for block_x in range(0, full_x_limit, block_tiles_x):
                    for block_y in range(0, full_y_limit, block_tiles_y):
                        process_block(block_x, block_y, block_tiles_x, block_tiles_y)

            if remaining_tiles_y > 0 and full_blocks_x > 0:
                for block_x in range(0, full_x_limit, block_tiles_x):
                    process_block(
                        block_x, full_y_limit, block_tiles_x, remaining_tiles_y
                    )

            if remaining_tiles_x > 0 and full_blocks_y > 0:
                for block_y in range(0, full_y_limit, block_tiles_y):
                    process_block(
                        full_x_limit, block_y, remaining_tiles_x, block_tiles_y
                    )

            if remaining_tiles_x > 0 and remaining_tiles_y > 0:
                process_block(
                    full_x_limit, full_y_limit, remaining_tiles_x, remaining_tiles_y
                )

            tensor_dst = untilize_block(
                tilized_dst.flatten(),
                operation.output.data_format,
                operation.max_output_dimensions,
            ).reshape(operation.max_output_dimensions)

        return (
            tensor_a.reshape(operation.src_a.dimensions),
            tensor_b.reshape(operation.src_b.dimensions),
            tensor_dst.reshape(operation.max_output_dimensions),
        )

    def __str__(self):
        if self.fpu is not None:
            return f"{self.unpacker.__name__}, {self.fpu}"
        elif self.sfpu:
            return f"{self.sfpu}"
        else:
            return ""


class ComputePipeline:
    operations: List[ComputeNode]
    packer: Packer

    def __init__(self, operations: List[ComputeNode], packer: Packer):
        self.operations = operations
        self.packer = packer

    def get_unpackers(self) -> List["Unpacker"]:
        unpackers: List["Unpacker"] = []

        for operation in self.operations:
            if operation.unpacker is not None:
                unpackers.append(operation.unpacker)

        return unpackers

    def get_math_units(self) -> List[Union["Fpu", "Sfpu"]]:
        math_units = []

        for operation in self.operations:
            if operation.fpu is not None:
                math_units.append(operation.fpu)

            if operation.sfpu is not None:
                math_units.append(operation.sfpu)

        return math_units

    def bh_unpack_tilize_check(self) -> bool:
        from .fused_unpacker import UnpackerTilizeA

        has_unpack_tilize = self.has_unpacker(UnpackerTilizeA)

        has_other_unpacker = False
        for operation in self.operations:
            if operation.unpacker is not None and operation.unpacker != UnpackerTilizeA:
                has_other_unpacker = True

        return has_unpack_tilize and has_other_unpacker

    def has_unpacker(self, unpacker) -> bool:
        for operation in self.operations:
            if (
                isinstance(operation.unpacker, unpacker)
                or operation.unpacker == unpacker
            ):
                return True

        return False

    def has_fpu(self, fpu) -> bool:
        for operation in self.operations:
            if isinstance(operation.fpu, fpu):
                return True

        return False

    def get_fused_compute_with_unpacker(self) -> List["ComputeNode"]:
        return [op for op in self.operations if op.unpacker is not None]

    def get_reduce_pack_mask(self) -> str:
        reduce_op = None
        for operation in self.operations:
            if isinstance(operation.fpu, ReduceFpu):
                reduce_op = operation.fpu.operation

        if reduce_op is None:
            return None

        return f"ReduceDim::{reduce_op.cpp_enum_value}"

    def _batch_loop(
        self, operation: "FusedOperation", config: "GlobalConfig", body_fn
    ) -> str:
        block_tiles_x = operation.block_tiles_x
        block_tiles_y = operation.block_tiles_y
        tile_count_x = operation.output.tile_count_x
        tile_count_y = operation.output.tile_count_y

        full_blocks_x = tile_count_x // block_tiles_x
        full_blocks_y = tile_count_y // block_tiles_y
        remaining_tiles_x = tile_count_x % block_tiles_x
        remaining_tiles_y = tile_count_y % block_tiles_y

        full_x_limit = full_blocks_x * block_tiles_x
        full_y_limit = full_blocks_y * block_tiles_y

        def make_block(block_x, block_y, block_tiles_x_eff, block_tiles_y_eff):
            return BlockData(
                block_x=block_x,
                block_y=block_y,
                block_tiles_x=block_tiles_x_eff,
                block_tiles_y=block_tiles_y_eff,
                tile_count_x=tile_count_x,
                tile_count_y=tile_count_y,
                full_x_limit=full_x_limit,
                full_y_limit=full_y_limit,
                tile_id_global="0",
                tile_id_block="0",
            )

        code = ""

        if full_blocks_x > 0 and full_blocks_y > 0:
            code += f"for (std::uint32_t block_x = 0; block_x < {full_x_limit}; block_x += {block_tiles_x}) {{\n"
            code += f"for (std::uint32_t block_y = 0; block_y < {full_y_limit}; block_y += {block_tiles_y}) {{\n"
            code += body_fn(
                make_block("block_x", "block_y", block_tiles_x, block_tiles_y)
            )
            code += "}\n"
            code += "}\n"

        if remaining_tiles_y > 0 and full_blocks_x > 0:
            code += f"for (std::uint32_t block_x = 0; block_x < {full_x_limit}; block_x += {block_tiles_x}) {{\n"
            code += body_fn(
                make_block("block_x", full_y_limit, block_tiles_x, remaining_tiles_y)
            )
            code += "}\n"

        if remaining_tiles_x > 0 and full_blocks_y > 0:
            code += f"for (std::uint32_t block_y = 0; block_y < {full_y_limit}; block_y += {block_tiles_y}) {{\n"
            code += body_fn(
                make_block(full_x_limit, "block_y", remaining_tiles_x, block_tiles_y)
            )
            code += "}\n"

        if remaining_tiles_x > 0 and remaining_tiles_y > 0:
            code += body_fn(
                make_block(
                    full_x_limit, full_y_limit, remaining_tiles_x, remaining_tiles_y
                )
            )

        return code

    def unpack_operand_constants(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        stage = operation.stage_id
        buffer_A_address = operation.src_a.l1_address
        buffer_B_address = operation.src_b.l1_address
        buffer_A_tile_size = operation.buffer_A_tile_size
        buffer_B_tile_size = operation.buffer_B_tile_size
        unpack_a_src = operation.unpack_a_in
        unpack_a_dst = operation.unpack_a_out
        unpack_b_src = operation.unpack_a_in
        unpack_b_dst = operation.unpack_a_out

        code = (
            f"    // Operation {stage}: Fused Unpack\n"
            f"    UNUSED const Operand buffer_A{stage}({hex(buffer_A_address)}, {buffer_A_tile_size});\n"
            f"    UNUSED const Operand buffer_B{stage}({hex(buffer_B_address)}, {buffer_B_tile_size});\n"
            f"    UNUSED const std::uint32_t unpack_a_src_format{stage} = ckernel::to_underlying(DataFormat::{unpack_a_src.name});\n"
            f"    UNUSED const std::uint32_t unpack_a_dst_format{stage} = ckernel::to_underlying(DataFormat::{unpack_a_dst.name});\n"
            f"    UNUSED const std::uint32_t unpack_b_src_format{stage} = ckernel::to_underlying(DataFormat::{unpack_b_src.name});\n"
            f"    UNUSED const std::uint32_t unpack_b_dst_format{stage} = ckernel::to_underlying(DataFormat::{unpack_b_dst.name});\n"
        )
        return code

    def unpack_hw_configure(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        stage = operation.stage_id
        unpa_tile_size = operation.tile_size_unpack_a
        unpb_tile_size = operation.tile_size_unpack_b
        dest_acc = config.dest_acc.cpp_enum_value
        unpa_face_r_dim = operation.face_r_dim
        unpb_face_r_dim = operation.face_r_dim
        unpa_num_faces = operation.num_faces_A
        unpb_num_faces = operation.num_faces_B

        if stage == 1:
            code = (
                f"_llk_unpack_hw_configure_<{dest_acc}, false>(\n"
                f"    unpack_a_src_format{stage}, unpack_b_src_format{stage}, unpack_a_dst_format{stage}, unpack_b_dst_format{stage},\n"
                f"    {unpa_face_r_dim}, {unpb_face_r_dim}, {unpa_num_faces}, {unpb_num_faces}, {unpa_tile_size}, {unpb_tile_size}\n"
                f");\n"
            )
        else:
            code = (
                f"_llk_unpack_reconfig_data_format_srca_impl_<{dest_acc}, false>(\n"
                f"    unpack_a_src_format{stage}, unpack_a_dst_format{stage}, {unpa_tile_size}\n"
                f");\n"
                f"_llk_unpack_reconfig_data_format_srcb_impl_<{dest_acc}, false>(\n"
                f"    unpack_b_src_format{stage}, unpack_b_dst_format{stage}, {unpb_tile_size}\n"
                f");\n"
            )
        return code

    def unpacker_sync_with_packer(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ) -> str:
        if operation.stage_id > 1:
            return (
                "t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);\n"
                "t6_semaphore_get<>(semaphore::PACK_DONE);\n"
            )

        return ""

    def unpack_body(self, operation: "FusedOperation", config: "GlobalConfig") -> str:
        code = self.unpack_operand_constants(operation, config)

        if config.profiler_enabled:
            code += "{\n"
            code += 'ZONE_SCOPED("INIT")\n'

        code += self.unpack_hw_configure(operation, config)

        if config.profiler_enabled:
            code += "PROFILER_SYNC();\n"
            code += "}\n"
            code += "{\n"
            code += 'ZONE_SCOPED("TILE_LOOP")\n'

        code += self.unpacker_sync_with_packer(operation, config)

        if config.profiler_enabled:
            code += f"for(int loop = 0; loop < {config.loop_factor}; loop++)\n"
            code += "{\n"

        def batch_body(block: BlockData):
            body = ""
            for compute_unit in self.operations:
                body += compute_unit.unpack(operation, config, block)
            return body

        code += self._batch_loop(operation, config, batch_body)

        if config.profiler_enabled:
            code += "}\n"
            code += "PROFILER_SYNC();\n"
            code += "}\n"
            code += "{\n"
            code += 'ZONE_SCOPED("INIT")\n'
            code += "PROFILER_SYNC();\n"
            code += "}\n"

        return code

    def math_hw_configure(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        stage = operation.stage_id
        dest_acc = config.dest_acc.cpp_enum_value
        if stage == 1:
            code = f"_llk_math_hw_configure_<{dest_acc}>(math_format{stage}, math_format{stage});\n"
        else:
            code = f"_llk_math_reconfig_data_format_<{dest_acc}, false>(math_format{stage}, math_format{stage});\n"

        code += f"_llk_math_pack_sync_init_<dest_sync{stage}, {dest_acc}>();\n"

        return code

    def _math_wait_for_dest(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        if config.perf_run_type in (
            PerfRunType.MATH_ISOLATE,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ):
            return ""

        return f"_llk_math_wait_for_dest_available_<dest_sync{operation.stage_id}>();\n"

    def _math_dest_section_done(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        if config.perf_run_type in (
            PerfRunType.MATH_ISOLATE,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ):
            return ""

        dest_acc = config.dest_acc.cpp_enum_value
        return f"_llk_math_dest_section_done_<dest_sync{operation.stage_id}, {dest_acc}>();\n"

    def _math_constants(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        stage = operation.stage_id
        math_format = operation.output.data_format
        dest_sync = operation.dest_sync.cpp_enum_value

        code = f"// Operation {stage}: Math Setup\n"
        code += f"const std::uint32_t math_format{stage} = ckernel::to_underlying(DataFormat::{math_format.name});\n"
        code += f"constexpr DstSync dest_sync{stage} = {dest_sync};\n"

        return code

    def math_body(self, operation: "FusedOperation", config: "GlobalConfig") -> str:
        code = self._math_constants(operation, config)

        if config.profiler_enabled:
            code += "{\n"
            code += 'ZONE_SCOPED("INIT")\n'

        code += self.math_hw_configure(operation, config)

        if config.profiler_enabled:
            code += "PROFILER_SYNC();\n"
            code += "}\n"
            code += "{\n"
            code += 'ZONE_SCOPED("TILE_LOOP")\n'
            code += f"for(int loop = 0; loop < {config.loop_factor}; loop++)\n"
            code += "{\n"

        def batch_body(block: BlockData):
            body = self._math_wait_for_dest(operation, config)
            for compute_unit in self.operations:
                body += compute_unit.math_calculate(operation, config, block)
            body += self._math_dest_section_done(operation, config)
            return body

        code += self._batch_loop(operation, config, batch_body)

        if config.profiler_enabled:
            code += "}\n"
            code += "PROFILER_SYNC();\n"
            code += "}\n"

        return code

    def _packer_wait_for_math(self, config: "GlobalConfig") -> str:
        if config.perf_run_type in (
            PerfRunType.MATH_ISOLATE,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ):
            return ""

        return "_llk_packer_wait_for_math_done_();\n"

    def _packer_dest_section_done(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        if config.perf_run_type in (
            PerfRunType.MATH_ISOLATE,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ):
            return ""

        dest_sync = operation.dest_sync.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        return f"_llk_pack_dest_section_done_<{dest_sync}, {dest_acc}>();\n"

    def _pack_constants(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        stage = operation.stage_id
        buffer_Res_tile_size = operation.buffer_Res_tile_size
        pack_src = operation.pack_in
        pack_dst = operation.pack_out
        result_buffer_address = operation.output.l1_address

        return (
            f"// Operation {stage}: Packer\n"
            f"const Operand buffer_Res{stage}({hex(result_buffer_address)}, {buffer_Res_tile_size});\n"
            f"const std::uint32_t pack_src_format{stage} = ckernel::to_underlying(DataFormat::{pack_src.name});\n"
            f"const std::uint32_t pack_dst_format{stage} = ckernel::to_underlying(DataFormat::{pack_dst.name});\n"
        )

    def pack_hw_configure(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        stage = operation.stage_id
        bh_tilize = operation.bh_tilize.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        pack_size = operation.tile_size_pack
        face_r_dim = operation.face_r_dim
        num_faces = operation.num_faces

        if stage == 1:
            if config.architecture == ChipArchitecture.BLACKHOLE:
                code = (
                    f"_llk_pack_hw_configure_<{dest_acc}, false, {bh_tilize}>(\n"
                    f"pack_src_format{stage}, pack_dst_format{stage}, {pack_size}, {face_r_dim}, TILE_C_DIM, {num_faces}\n"
                    f");\n"
                )
            elif config.architecture == ChipArchitecture.WORMHOLE:
                code = (
                    f"_llk_pack_hw_configure_<{dest_acc}, false>(\n"
                    f"pack_src_format{stage}, pack_dst_format{stage}, {pack_size}, {face_r_dim}, {num_faces}\n"
                    f");\n"
                )
        else:
            code = (
                f"_llk_pack_reconfig_data_format_<{dest_acc}, false>(\n"
                f"pack_src_format{stage}, pack_dst_format{stage}, {pack_size}\n"
                f");\n"
            )

        return code

    def _pack_reduce_mask_config(self) -> str:
        code = ""
        if self.has_fpu(ReduceFpu):
            reduce_dim = self.get_reduce_pack_mask()
            code += f"_llk_pack_reduce_mask_config_<false, {reduce_dim}>();\n"
        elif self.has_fpu(ReduceBlockMaxFpu):
            reduce_dim = "ckernel::ReduceDim::REDUCE_ROW"
            code += f"_llk_pack_reduce_mask_config_<false, {reduce_dim}>();\n"
        return code

    def _pack_reduce_mask_clear(self) -> str:
        code = ""

        if self.has_fpu(ReduceFpu) or self.has_fpu(ReduceBlockMaxFpu):
            code = "_llk_pack_reduce_mask_clear_();\n"

        return code

    def packer_sync_with_unpacker(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ) -> str:
        stage = operation.stage_id
        num_stages = operation.num_stages
        code = ""

        if stage < num_stages:
            code += "t6_semaphore_post<>(semaphore::PACK_DONE);\n\n"

        return code

    def pack_body(self, operation: "FusedOperation", config: "GlobalConfig") -> str:
        code = self._pack_constants(operation, config)

        if config.profiler_enabled:
            code += "{\n"
            code += 'ZONE_SCOPED("INIT")\n'

        code += self.pack_hw_configure(operation, config)
        code += self._pack_reduce_mask_config()
        code += self.packer().init(operation, config, None, None)

        if config.profiler_enabled:
            code += "PROFILER_SYNC();\n"
            code += "}\n"
            code += "{\n"
            code += 'ZONE_SCOPED("TILE_LOOP")\n'

        if config.profiler_enabled:
            code += f"for(int loop = 0; loop < {config.loop_factor}; loop++)\n"
            code += "{\n"

        def batch_body(block: BlockData):
            body = self._packer_wait_for_math(config)
            body += self.packer().loop.pack_loop(operation, config, block)
            body += self._packer_dest_section_done(operation, config)
            return body

        code += self._batch_loop(operation, config, batch_body)

        if config.profiler_enabled:
            code += "}\n"
            code += "PROFILER_SYNC();\n"
            code += "}\n"
            code += "{\n"
            code += 'ZONE_SCOPED("INIT")\n'

        code += self.packer_sync_with_unpacker(operation, config)
        code += self.packer().uninit(operation, config, None, None)
        code += self._pack_reduce_mask_clear()

        if config.profiler_enabled:
            code += "PROFILER_SYNC();\n"
            code += "}\n"

        return code

    def golden(
        self,
        input_tensor_a: torch.Tensor,
        input_tensor_b: torch.Tensor,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ) -> torch.Tensor:
        tensor_a = torch.zeros(operation.src_a.dimensions)
        tensor_b = torch.zeros(operation.src_b.dimensions)
        tensor_dst = torch.zeros(operation.max_output_dimensions)
        for op in self.operations:
            tensor_a, tensor_b, tensor_dst = op.golden(
                input_tensor_a,
                input_tensor_b,
                tensor_a,
                tensor_b,
                tensor_dst,
                operation,
                config,
            )

        dimensions = operation.output.dimensions
        return tensor_dst.reshape(operation.max_output_dimensions)[
            : dimensions[0], : dimensions[1]
        ]

    def __str__(self):
        str = "Math:"
        for op in self.operations:
            str += "\n    "
            str += op.__str__()

        return str
