# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List, Union

import torch

if TYPE_CHECKING:
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig

from helpers.llk_params import GoldenType

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

    def _all_same_operand_formats(self, ops: List[ComputeNode]) -> bool:
        def signature(op: ComputeNode):
            return (
                op.src_a.data_format if op.src_a is not None else None,
                op.src_b.data_format if op.src_b is not None else None,
            )

        return len({signature(op) for op in ops}) <= 1

    def _batch_loop(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        body_fn,
        init_fn=None,
        uninit_fn=None,
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

        def wrap(block, body):
            code = ""
            if init_fn is not None:
                code += init_fn(block)
            code += body
            if uninit_fn is not None:
                code += uninit_fn(block)
            return code

        code = ""

        if full_blocks_x > 0 and full_blocks_y > 0:
            block = make_block("block_x", "block_y", block_tiles_x, block_tiles_y)
            code += wrap(
                block,
                (
                    f"for (std::uint32_t block_x = 0; block_x < {full_x_limit}; block_x += {block_tiles_x}) {{\n"
                    f"for (std::uint32_t block_y = 0; block_y < {full_y_limit}; block_y += {block_tiles_y}) {{\n"
                    + body_fn(block)
                    + "}\n"
                    "}\n"
                ),
            )

        if remaining_tiles_y > 0 and full_blocks_x > 0:
            block = make_block(
                "block_x", full_y_limit, block_tiles_x, remaining_tiles_y
            )
            code += wrap(
                block,
                (
                    f"for (std::uint32_t block_x = 0; block_x < {full_x_limit}; block_x += {block_tiles_x}) {{\n"
                    + body_fn(block)
                    + "}\n"
                ),
            )

        if remaining_tiles_x > 0 and full_blocks_y > 0:
            block = make_block(
                full_x_limit, "block_y", remaining_tiles_x, block_tiles_y
            )
            code += wrap(
                block,
                (
                    f"for (std::uint32_t block_y = 0; block_y < {full_y_limit}; block_y += {block_tiles_y}) {{\n"
                    + body_fn(block)
                    + "}\n"
                ),
            )

        if remaining_tiles_x > 0 and remaining_tiles_y > 0:
            block = make_block(
                full_x_limit, full_y_limit, remaining_tiles_x, remaining_tiles_y
            )
            code += wrap(block, body_fn(block))

        return code

    def _zone(self, config: "GlobalConfig", name: str, body: str) -> str:
        if not config.profiler_enabled:
            return body
        code = "{\n"
        code += f'ZONE_SCOPED("{name}")\n'
        code += body
        code += "PROFILER_SYNC();\n"
        code += "}\n"
        return code

    def _zone_loop(self, config: "GlobalConfig", name: str, body: str) -> str:
        if not config.profiler_enabled:
            return body
        code = "{\n"
        code += f'ZONE_SCOPED("{name}")\n'
        code += f"for(int loop = 0; loop < {config.loop_factor}; loop++)\n"
        code += "{\n"
        code += body
        code += "}\n"
        code += "PROFILER_SYNC();\n"
        code += "}\n"
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
        unpack_ops = [cu for cu in self.operations if cu.unpacker is not None]
        hoist = len(unpack_ops) == 1
        hoist_reconfig = hoist or self._all_same_operand_formats(unpack_ops)

        init_code = config.sentinel.hw_configure_unpack(config, operation)
        if hoist_reconfig and unpack_ops:
            init_code += unpack_ops[0].unpack_reconfig(operation, config)
        if hoist and not unpack_ops[0].unpacker.per_block_init:
            init_code += unpack_ops[0].unpack_init(operation, config, None)
        code = self._zone(config, "INIT", init_code)

        code += self.unpacker_sync_with_packer(operation, config)

        init_fn = None
        uninit_fn = None
        if hoist and unpack_ops[0].unpacker.per_block_init:
            init_fn = lambda block: unpack_ops[0].unpack_init(operation, config, block)
            uninit_fn = lambda block: unpack_ops[0].unpack_uninit(
                operation, config, block
            )

        def batch_body(block: BlockData):
            body = ""
            for cu in self.operations:
                if not hoist_reconfig and cu.unpacker is not None:
                    body += cu.unpack_reconfig(operation, config)
                if not hoist:
                    body += cu.unpack_init(operation, config, block)
                body += cu.unpack_run(operation, config, block)
                if not hoist:
                    body += cu.unpack_uninit(operation, config, block)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(operation, config, batch_body, init_fn, uninit_fn),
        )

        uninit_code = ""
        if hoist and not unpack_ops[0].unpacker.per_block_init:
            uninit_code += unpack_ops[0].unpack_uninit(operation, config, None)
        code += self._zone(config, "INIT", uninit_code)

        return code

    def _math_wait_for_dest(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        if config.skip_sync:
            return ""
        dest_sync = operation.dest_sync.cpp_enum_value
        return f"_llk_math_wait_for_dest_available_<{dest_sync}>();\n"

    def _math_dest_section_done(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        if config.skip_sync:
            return ""
        dest_sync = operation.dest_sync.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        return f"_llk_math_dest_section_done_<{dest_sync}, {dest_acc}>();\n"

    def _math_pack_sync_init(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        dest_sync = operation.dest_sync.cpp_enum_value
        dest_acc = config.dest_acc.cpp_enum_value
        return f"_llk_math_pack_sync_init_<{dest_sync}, {dest_acc}>();\n"

    def _math_constants(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        return f"// Operation {operation.stage_id}: Math Setup\n"

    def math_body(self, operation: "FusedOperation", config: "GlobalConfig") -> str:
        code = self._math_constants(operation, config)
        fpu_ops = [cu for cu in self.operations if cu.fpu is not None]
        hoist = len(fpu_ops) == 1
        hoist_reconfig = hoist or self._all_same_operand_formats(fpu_ops)

        init_code = config.sentinel.hw_configure_math(config, operation)
        init_code += self._math_pack_sync_init(operation, config)
        if hoist_reconfig and fpu_ops:
            init_code += fpu_ops[0].math_reconfig(operation, config)
        if hoist and not fpu_ops[0].fpu.per_block_init:
            init_code += fpu_ops[0].math_init(operation, config, None)
        code += self._zone(config, "INIT", init_code)

        init_fn = None
        uninit_fn = None
        if hoist and fpu_ops[0].fpu.per_block_init:
            init_fn = lambda block: fpu_ops[0].math_init(operation, config, block)
            uninit_fn = lambda block: fpu_ops[0].math_uninit(operation, config, block)

        def batch_body(block: BlockData):
            body = self._math_wait_for_dest(operation, config)
            for cu in self.operations:
                if not hoist_reconfig and cu.fpu is not None:
                    body += cu.math_reconfig(operation, config)
                if not hoist or cu.fpu is None:
                    body += cu.math_init(operation, config, block)
                body += cu.math_run(operation, config, block)
                if not hoist or cu.fpu is None:
                    body += cu.math_uninit(operation, config, block)
            body += self._math_dest_section_done(operation, config)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(operation, config, batch_body, init_fn, uninit_fn),
        )

        uninit_code = ""
        if hoist and not fpu_ops[0].fpu.per_block_init:
            uninit_code += fpu_ops[0].math_uninit(operation, config, None)
        code += self._zone(config, "INIT", uninit_code)

        return code

    def _packer_wait_for_math(self, config: "GlobalConfig") -> str:
        if config.skip_sync:
            return ""
        return "_llk_packer_wait_for_math_done_();\n"

    def _packer_dest_section_done(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        if config.skip_sync:
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

    def _all_same_pack_formats(self) -> bool:
        if len(self.pack_nodes) <= 1:
            return True
        first_fmt = self.pack_nodes[0].output.data_format
        return all(pn.output.data_format == first_fmt for pn in self.pack_nodes[1:])

    def pack_body(self, operation: "FusedOperation", config: "GlobalConfig") -> str:
        code = self._pack_constants(operation, config)
        hoist = len(self.pack_nodes) == 1
        hoist_reconfig = hoist or self._all_same_pack_formats()

        init_code = config.sentinel.hw_configure_pack(
            config, operation, self.pack_nodes
        )
        init_code += self._pack_dest_init(operation, config)
        init_code += self._pack_reduce_mask_config(operation)
        if hoist_reconfig:
            init_code += self.pack_nodes[0].reconfig(operation, config)
        if hoist:
            init_code += self.pack_nodes[0].configure(operation, config, None)
        code += self._zone(config, "INIT", init_code)

        def batch_body(block: BlockData):
            body = self._packer_wait_for_math(config)
            if not hoist_reconfig:
                config.sentinel.reset_pack_formats()
            for pack_node in self.pack_nodes:
                if not hoist_reconfig:
                    body += pack_node.reconfig(operation, config)
                if not hoist:
                    body += pack_node.configure(operation, config, block)
                body += pack_node.pack_loop(operation, config, block)
                if not hoist:
                    body += pack_node.uninit(operation, config)
            body += self._packer_dest_section_done(operation, config)
            return body

        code += self._zone_loop(
            config, "TILE_LOOP", self._batch_loop(operation, config, batch_body)
        )

        uninit_code = self.packer_sync_with_unpacker(operation, config)
        if hoist:
            uninit_code += self.pack_nodes[0].uninit(operation, config)
        uninit_code += self._pack_reduce_mask_clear(operation)
        code += self._zone(config, "INIT", uninit_code)

        return code

    def golden(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
        golden_type: GoldenType,
    ):
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

        for pack_node in self.pack_nodes:
            config.sentinel.configure_golden(
                config, operation, output_format=pack_node.output.data_format
            )

            dimensions = pack_node.output.dimensions
            cropped = tensor_dst.reshape(operation.max_output_dimensions)[
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
