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
from .fused_sfpu import Sfpu
from .fused_unpacker import Unpacker
from .pack_node import PackNode


class ComputePipeline:
    operations: List[ComputeNode]
    pack_nodes: List[PackNode]

    def __init__(self, operations: List[ComputeNode], pack_nodes: List[PackNode]):
        self.operations = operations
        self.pack_nodes = pack_nodes

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

    def _batch_loop(
        self, operation: "FusedOperation", config: "GlobalConfig", body_fn
    ) -> str:
        block_tiles_x = operation.block_tiles_x
        block_tiles_y = operation.block_tiles_y
        tile_count_x = (
            operation.max_output_dimensions[1] // operation.tile_shape.total_col_dim()
        )
        tile_count_y = (
            operation.max_output_dimensions[0] // operation.tile_shape.total_row_dim()
        )

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

    def _pack_dest_init(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        dest_sync = operation.dest_sync.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        return f"_llk_pack_dest_init_<{dest_sync}, {dest_acc}>();\n"

    def _pack_constants(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        stage = operation.stage_id
        return f"// Operation {stage}: Packer\n"

    def _pack_reduce_mask_config(self, operation: "FusedOperation") -> str:
        if operation.reduce_dim is not None:
            reduce_dim = operation.reduce_dim.cpp_enum_value
            return f"_llk_pack_reduce_mask_config_<{reduce_dim}>();\n"
        return ""

    def _pack_reduce_mask_clear(self, operation: "FusedOperation") -> str:
        if operation.reduce_dim is not None:
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

        code += config.sentinel.hw_configure_pack(config, operation, self.pack_nodes)
        code += self._pack_dest_init(operation, config)
        code += self._pack_reduce_mask_config(operation)

        if len(self.pack_nodes) == 1:
            code += self.pack_nodes[0].configure(operation, config, None)

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
            for pack_node in self.pack_nodes:
                if len(self.pack_nodes) > 1:
                    body += pack_node.configure(operation, config, block)
                body += pack_node.pack_loop(operation, config, block)
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
        for pack_node in self.pack_nodes:
            code += pack_node.uninit(operation, config)
        code += self._pack_reduce_mask_clear(operation)

        if config.profiler_enabled:
            code += "PROFILER_SYNC();\n"
            code += "}\n"

        return code

    def _math_golden(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        golden_type: GoldenType,
    ) -> torch.Tensor:
        first_fpu = next((op for op in self.operations if op.src_a is not None), None)
        if first_fpu is not None:
            tensor_a = torch.zeros(first_fpu.src_a.dimensions)
            tensor_b = torch.zeros(first_fpu.src_b.dimensions)
        else:
            tensor_a = torch.zeros(operation.max_output_dimensions)
            tensor_b = torch.zeros(operation.max_output_dimensions)
        tensor_dst = torch.zeros(operation.max_output_dimensions)
        for op in self.operations:
            config.sentinel.configure_golden(config, operation, op)
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
        return tensor_dst

    def golden(self, operation: "FusedOperation", config: "GlobalConfig", golden_type: GoldenType):
        math_tensor = self._math_golden(operation, config, golden_type)

        for pack_node in self.pack_nodes:
            config.sentinel.configure_golden(
                config, operation, output_format=pack_node.output.data_format
            )

            dimensions = pack_node.output.dimensions
            cropped = math_tensor.reshape(operation.max_output_dimensions)[
                : dimensions[0], : dimensions[1]
            ]
            result = pack_node.golden(cropped, operation, config)

            if golden_type == GoldenType.L1_GOLDEN:
                pack_node.output.l1_golden = result
            else:
                pack_node.output._master_golden = result

    def __str__(self):
        result = "Math:"
        for op in self.operations:
            result += "\n    "
            result += op.__str__()
        result += f"\n  Pack:"
        for pn in self.pack_nodes:
            result += f"\n    "
            result += pn.output.__str__()
        return result
