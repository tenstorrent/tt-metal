# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List, Union

import torch

if TYPE_CHECKING:
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig

from helpers.llk_params import GoldenType, PerfRunType

from .block_data import BlockData
from .compute_node import ComputeNode
from .fused_fpu import Fpu
from .fused_packer import Packer
from .fused_sfpu import Sfpu
from .fused_unpacker import Unpacker


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

    def get_reduce_pack_mask(self) -> str:
        for operation in self.operations:
            if operation.reduce_dim is not None:
                return operation.reduce_dim.cpp_enum_value

        return None

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
        code = ""

        if config.profiler_enabled:
            code += "{\n"
            code += 'ZONE_SCOPED("INIT")\n'

        code += config.sentinel.hw_configure_unpack(config, operation)

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
        dest_sync = operation.dest_sync.cpp_enum_value

        code = f"// Operation {stage}: Math Setup\n"
        code += f"constexpr DstSync dest_sync{stage} = {dest_sync};\n"

        return code

    def math_body(self, operation: "FusedOperation", config: "GlobalConfig") -> str:
        code = self._math_constants(operation, config)

        if config.profiler_enabled:
            code += "{\n"
            code += 'ZONE_SCOPED("INIT")\n'

        code += config.sentinel.hw_configure_math(config, operation)

        stage = operation.stage_id
        dest_acc = config.dest_acc.cpp_enum_value
        code += f"_llk_math_pack_sync_init_<dest_sync{stage}, {dest_acc}>();\n"

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
        return f"// Operation {stage}: Packer\n"

    def _pack_reduce_mask_config(self) -> str:
        reduce_dim = self.get_reduce_pack_mask()
        if reduce_dim is not None:
            return f"_llk_pack_reduce_mask_config_<false, {reduce_dim}>();\n"
        return ""

    def _pack_reduce_mask_clear(self) -> str:
        reduce_dim = self.get_reduce_pack_mask()
        if reduce_dim is not None:
            return "_llk_pack_reduce_mask_clear_();\n"
        return ""

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

        code += config.sentinel.hw_configure_pack(config, operation)
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
        operation: "FusedOperation",
        config: "GlobalConfig",
        golden_type: GoldenType,
    ) -> torch.Tensor:
        tensor_a = torch.zeros(operation.math.operations[0].src_a.dimensions)
        tensor_b = torch.zeros(operation.math.operations[0].src_b.dimensions)
        tensor_dst = torch.zeros(operation.max_output_dimensions)
        for op in self.operations:
            if op.src_a is not None:
                input_tensor_a = (
                    op.src_a.raw_data
                    if golden_type == GoldenType.L1_GOLDEN
                    else op.src_a.master_golden
                )
            else:
                input_tensor_a = None
            if op.src_b is not None:
                input_tensor_b = (
                    op.src_b.raw_data
                    if golden_type == GoldenType.L1_GOLDEN
                    else op.src_b.master_golden
                )
            else:
                input_tensor_b = None
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
