# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from itertools import product
from typing import TYPE_CHECKING, List, Union

import torch
from helpers.tilize_untilize import tilize_block, untilize_block

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig

from helpers.llk_params import GoldenType, L1Accumulation, format_dict

from .arch_common import fpu_common, pack_common, unpack_common
from .base_fpu import Fpu
from .base_sfpu import Sfpu
from .base_unpacker import Unpacker
from .block_data import BlockData, InvocationGranularity, KernelInvocation
from .fpu_node import FpuNode
from .pack_node import PackNode
from .sfpu_node import SfpuNode


class ComputePipeline:
    math_nodes: List[Union[FpuNode, SfpuNode]]
    pack_nodes: List[Union[PackNode, SfpuNode]]

    def __init__(
        self,
        math_nodes: List[Union[FpuNode, SfpuNode]],
        pack_nodes: List[Union[PackNode, SfpuNode]],
    ):
        self.math_nodes = math_nodes
        self.pack_nodes = pack_nodes
        nodes = math_nodes + pack_nodes
        self.explicit_blocks = bool(nodes and nodes[0].blocks is not None)
        self.block_data = []
        self.codegen_block_data = []

    def plan(self, operation: "L1Operation"):
        self.block_data = []
        self.codegen_block_data = []
        nodes = self.math_nodes + self.pack_nodes
        tile_count_x = (
            operation.max_output_dimensions[1] // operation.tile_shape.total_col_dim()
        )
        tile_count_y = (
            operation.max_output_dimensions[0] // operation.tile_shape.total_row_dim()
        )
        if self.explicit_blocks:
            for _ in nodes[0].blocks:
                self.block_data.append(
                    BlockData(
                        block_x=0,
                        block_y=0,
                        block_tiles_x=1,
                        block_tiles_y=1,
                        tile_count_x=tile_count_x,
                        tile_count_y=tile_count_y,
                        full_x_limit=tile_count_x,
                        full_y_limit=tile_count_y,
                        tile_id_global=0,
                        tile_id_block=0,
                    )
                )
            return

        def regions(tile_count, block_tiles, symbol):
            full_limit = tile_count // block_tiles * block_tiles
            result = []
            if full_limit:
                result.append(
                    (range(0, full_limit, block_tiles), symbol, block_tiles, True)
                )
            if full_limit < tile_count:
                result.append(
                    (
                        (full_limit,),
                        full_limit,
                        tile_count - full_limit,
                        False,
                    )
                )
            return full_limit, result

        full_x_limit, x_regions = regions(
            tile_count_x, operation.block_tiles_x, "block_x"
        )
        full_y_limit, y_regions = regions(
            tile_count_y, operation.block_tiles_y, "block_y"
        )

        def make_block(block_x, block_y, tiles_x, tiles_y, **kwargs):
            return BlockData(
                block_x=block_x,
                block_y=block_y,
                block_tiles_x=tiles_x,
                block_tiles_y=tiles_y,
                tile_count_x=tile_count_x,
                tile_count_y=tile_count_y,
                full_x_limit=full_x_limit,
                full_y_limit=full_y_limit,
                tile_id_global=0,
                tile_id_block=0,
                **kwargs,
            )

        for x_region, y_region in product(x_regions, y_regions):
            x_origins, codegen_x, tiles_x, loop_x = x_region
            y_origins, codegen_y, tiles_y, loop_y = y_region
            self.block_data.extend(
                make_block(block_x, block_y, tiles_x, tiles_y)
                for block_x, block_y in product(x_origins, y_origins)
            )
            self.codegen_block_data.append(
                make_block(
                    codegen_x,
                    codegen_y,
                    tiles_x,
                    tiles_y,
                    codegen=True,
                    loop_x=loop_x,
                    loop_y=loop_y,
                )
            )

    @staticmethod
    def _positions(block: BlockData, granularity: InvocationGranularity):
        if granularity == InvocationGranularity.BLOCK:
            yield 0, 0
            return
        if granularity == InvocationGranularity.ROW:
            rows = ("tile_y",) if block.codegen else range(block.block_tiles_y)
            yield from ((0, tile_y) for tile_y in rows)
            return
        if granularity != InvocationGranularity.TILE:
            return
        columns = ("tile_x",) if block.codegen else range(block.block_tiles_x)
        rows = ("tile_y",) if block.codegen else range(block.block_tiles_y)
        yield from product(columns, rows)

    @classmethod
    def _automatic_calls(cls, block: BlockData, granularity: InvocationGranularity):
        for tile_x, tile_y in cls._positions(block, granularity):
            if block.codegen:
                global_id = f"({block.tile_count_x} * ({block.block_y} + {tile_y}) + ({block.block_x} + {tile_x}))"
                dest_id = f"({tile_y} * {block.block_tiles_x} + {tile_x})"
            else:
                global_id = (
                    block.tile_count_x * (block.block_y + tile_y)
                    + block.block_x
                    + tile_x
                )
                dest_id = tile_y * block.block_tiles_x + tile_x
            yield KernelInvocation(
                in0=global_id,
                in1=global_id,
                src0=dest_id,
                src1=dest_id,
                dest=dest_id,
                out=global_id,
            )

    def _calls(self, node, block_index, block, granularity, unpack=False):
        if node.blocks is not None:
            return node.blocks[block_index]
        return (
            node.automatic_call(call, block, unpack)
            for call in self._automatic_calls(block, granularity)
        )

    def _block_calls(self, node, block_index, block):
        return self._calls(node, block_index, block, InvocationGranularity.BLOCK)

    @staticmethod
    def _wrap_calls(code, granularity, block):
        if granularity == InvocationGranularity.TILE:
            code = f"for (std::uint32_t tile_y = 0; tile_y < {block.block_tiles_y}; tile_y++) {{\n{code}}}\n"
            return f"for (std::uint32_t tile_x = 0; tile_x < {block.block_tiles_x}; tile_x++) {{\n{code}}}\n"
        if granularity == InvocationGranularity.ROW:
            return f"for (std::uint32_t tile_y = 0; tile_y < {block.block_tiles_y}; tile_y++) {{\n{code}}}\n"
        return code

    def _emit_calls(
        self,
        node,
        block_index,
        block,
        granularity,
        run,
        operation,
        config,
        unpack=False,
    ):
        code = "".join(
            run(operation, config, block, call)
            for call in self._calls(node, block_index, block, granularity, unpack)
        )
        return self._wrap_calls(code, granularity, block) if block.codegen else code

    @staticmethod
    def _global_dest_call(call, block):
        def global_index(index):
            if index is None:
                return None
            tile_y, tile_x = divmod(index, block.block_tiles_x)
            return (
                block.tile_count_x * (block.block_y + tile_y) + block.block_x + tile_x
            )

        return replace(
            call,
            src0=global_index(call.src0),
            src1=global_index(call.src1),
            dest=global_index(call.dest),
        )

    def _sfpu_golden(self, node, tensor_dst, operation, config):
        tile_shape = operation.tile_shape
        tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())
        tiles = tilize_block(
            tensor_dst,
            operation.max_output_dimensions,
            config.sentinel.golden_math_format,
            num_faces=tile_shape.total_num_faces(),
            tile_dimensions=tile_dims,
        )
        for block_index, block in enumerate(self.block_data):
            for call in self._block_calls(node, block_index, block):
                node.golden_call(
                    self._global_dest_call(call, block), tiles, operation, config
                )
        return untilize_block(
            tiles.flatten(),
            config.sentinel.golden_math_format,
            operation.max_output_dimensions,
            tile_dimensions=tile_dims,
            num_faces=tile_shape.total_num_faces(),
        ).reshape(operation.max_output_dimensions)

    def _l1_acc_golden(self, tensor, node, operation):
        output = node.output
        tile_shape = output.tile_shape
        tile_dims = (tile_shape.total_row_dim(), tile_shape.total_col_dim())
        source_tiles = tilize_block(
            tensor,
            output.dimensions,
            output.data_format,
            num_faces=tile_shape.total_num_faces(),
            tile_dimensions=tile_dims,
        ).view(output.tile_count, tile_shape.total_tile_size())
        result_tiles = torch.zeros_like(source_tiles)
        for block_index, block in enumerate(self.block_data):
            calls = self._calls(node, block_index, block, node.packer.granularity)
            for call in calls:
                tile_y, tile_x = divmod(call.dest, block.block_tiles_x)
                source_id = (
                    (block.block_y + tile_y) * output.tile_count_x
                    + block.block_x
                    + tile_x
                )
                result_tiles[call.out] = node.packer.l1_acc_golden(
                    result_tiles[call.out],
                    source_tiles[source_id],
                    output.data_format,
                )
        return untilize_block(
            result_tiles.flatten(),
            output.data_format,
            output.dimensions,
            tile_dimensions=tile_dims,
            num_faces=tile_shape.total_num_faces(),
        )

    def _get_pack_nodes(self) -> List[PackNode]:
        return [pn for pn in self.pack_nodes if isinstance(pn, PackNode)]

    def get_unpackers(self) -> List["Unpacker"]:
        unpackers: List["Unpacker"] = []

        for operation in self.math_nodes:
            if isinstance(operation, FpuNode) and operation.unpacker is not None:
                unpackers.append(operation.unpacker)

        return unpackers

    def get_math_units(self) -> List[Union["Fpu", "Sfpu"]]:
        math_units = []

        for operation in self.math_nodes:
            if isinstance(operation, FpuNode):
                math_units.append(operation.fpu)
            elif isinstance(operation, SfpuNode):
                math_units.append(operation.sfpu)

        return math_units

    def _all_same_operand_formats(self, ops: List[FpuNode]) -> bool:
        def signature(op: FpuNode):
            return (
                op.src_a.data_format if op.src_a is not None else None,
                op.src_b.data_format if op.src_b is not None else None,
            )

        return len({signature(op) for op in ops}) <= 1

    @staticmethod
    def _block_loop(body, block):
        if block.loop_y:
            body = (
                f"for (std::uint32_t block_y = 0; block_y < {block.full_y_limit}; "
                f"block_y += {block.block_tiles_y}) {{\n{body}}}\n"
            )
        if block.loop_x:
            body = (
                f"for (std::uint32_t block_x = 0; block_x < {block.full_x_limit}; "
                f"block_x += {block.block_tiles_x}) {{\n{body}}}\n"
            )
        return body

    @staticmethod
    def _with_lifecycle(body, block, init_fn, uninit_fn):
        return (
            (init_fn(block) if init_fn is not None else "")
            + body
            + (uninit_fn(block) if uninit_fn is not None else "")
        )

    def _batch_loop(
        self,
        body_fn,
        init_fn=None,
        uninit_fn=None,
    ) -> str:
        code = ""
        if not self.explicit_blocks:
            for block_index, block in enumerate(self.codegen_block_data):
                body = body_fn(block_index, block)
                if not body:
                    continue
                code += self._with_lifecycle(
                    self._block_loop(body, block), block, init_fn, uninit_fn
                )
            return code

        bodies = [
            (block, body_fn(block_index, block))
            for block_index, block in enumerate(self.block_data)
        ]
        bodies = [(block, body) for block, body in bodies if body]
        if not bodies:
            return code
        if init_fn is not None:
            code += init_fn(bodies[0][0])
        code += "".join(body for _, body in bodies)
        if uninit_fn is not None:
            code += uninit_fn(bodies[-1][0])
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

    def unpack_body(self, operation: "L1Operation", config: "GlobalConfig") -> str:
        unpack_ops = [
            cu
            for cu in self.math_nodes
            if isinstance(cu, FpuNode) and cu.unpacker is not None
        ]
        hoist = len(unpack_ops) == 1
        hoist_reconfig = hoist or self._all_same_operand_formats(unpack_ops)

        init_code = ""
        init_code += unpack_common.dvalid_init(config=config, operation=operation)
        init_code += config.sentinel.hw_configure_unpack(config, operation)
        if hoist_reconfig and unpack_ops and not config.skip_unpack_init:
            init_code += config.sentinel.configure_unpack(
                config, operation, unpack_ops[0]
            )
        if hoist and not unpack_ops[0].unpacker.per_block_init:
            init_code += unpack_ops[0].unpack_init(operation, config, None)
        code = self._zone(config, "INIT", init_code)

        code += unpack_common.sync_with_packer(config, operation)

        init_fn = None
        uninit_fn = None
        if hoist and unpack_ops[0].unpacker.per_block_init:
            init_fn = lambda block: unpack_ops[0].unpack_init(operation, config, block)
            uninit_fn = lambda block: unpack_ops[0].unpack_uninit(
                operation, config, block
            )

        def batch_body(block_index: int, block: BlockData):
            body = ""
            for cu in self.math_nodes:
                if not isinstance(cu, FpuNode) or cu.unpacker is None:
                    continue
                if not hoist_reconfig and not config.skip_unpack_init:
                    body += config.sentinel.configure_unpack(config, operation, cu)
                if not hoist:
                    body += cu.unpack_init(operation, config, block)
                body += self._emit_calls(
                    cu,
                    block_index,
                    block,
                    cu.unpacker.granularity,
                    cu.unpack_call,
                    operation,
                    config,
                    unpack=True,
                )
                if not hoist:
                    body += cu.unpack_uninit(operation, config, block)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(batch_body, init_fn, uninit_fn),
        )

        uninit_code = ""
        if hoist and not unpack_ops[0].unpacker.per_block_init:
            uninit_code += unpack_ops[0].unpack_uninit(operation, config, None)
        code += self._zone(config, "INIT", uninit_code)

        return code

    def math_body(self, operation: "L1Operation", config: "GlobalConfig") -> str:
        code = f"// Operation {operation.stage_id}: Math Setup\n"
        fpu_ops = [cu for cu in self.math_nodes if isinstance(cu, FpuNode)]
        hoist = len(fpu_ops) == 1
        hoist_reconfig = hoist or self._all_same_operand_formats(fpu_ops)

        init_code = config.sentinel.hw_configure_math(config, operation)
        init_code += fpu_common.math_pack_sync_init(config, operation)
        init_code += fpu_common.math_dest_remap_config(
            any(pn.packer.requires_dest_remap for pn in self._get_pack_nodes())
        )
        if hoist_reconfig and fpu_ops and not config.skip_math_init:
            init_code += config.sentinel.configure_math(config, operation, fpu_ops[0])
        if hoist and not fpu_ops[0].fpu.per_block_init:
            init_code += fpu_ops[0].fpu_init(operation, config, None)
        code += self._zone(config, "INIT", init_code)

        init_fn = None
        uninit_fn = None
        if hoist and fpu_ops[0].fpu.per_block_init:
            init_fn = lambda block: fpu_ops[0].fpu_init(operation, config, block)
            uninit_fn = lambda block: fpu_ops[0].fpu_uninit(operation, config, block)

        def batch_body(block_index: int, block: BlockData):
            body = fpu_common.math_wait_for_dest(config, operation)
            for cu in self.math_nodes:
                if isinstance(cu, SfpuNode):
                    body += cu.sfpu_init(operation, config, block)
                    body += self._emit_calls(
                        cu,
                        block_index,
                        block,
                        InvocationGranularity.BLOCK,
                        cu.sfpu_call,
                        operation,
                        config,
                    )
                    body += cu.sfpu_uninit(operation, config, block)
                    continue
                if not hoist_reconfig and not config.skip_math_init:
                    body += config.sentinel.configure_math(config, operation, cu)
                if not hoist:
                    body += cu.fpu_init(operation, config, block)
                body += self._emit_calls(
                    cu,
                    block_index,
                    block,
                    cu.fpu.granularity,
                    cu.fpu_call,
                    operation,
                    config,
                )
                if not hoist:
                    body += cu.fpu_uninit(operation, config, block)
            body += fpu_common.math_dest_section_done(config, operation)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(batch_body, init_fn, uninit_fn),
        )

        uninit_code = ""
        if hoist and not fpu_ops[0].fpu.per_block_init:
            uninit_code += fpu_ops[0].fpu_uninit(operation, config, None)
        code += self._zone(config, "INIT", uninit_code)

        return code

    def _all_same_pack_formats(self) -> bool:
        pack_only = self._get_pack_nodes()
        if len(pack_only) <= 1:
            return True
        first_fmt = pack_only[0].output.data_format
        return all(pn.output.data_format == first_fmt for pn in pack_only[1:])

    def pack_body(self, operation: "L1Operation", config: "GlobalConfig") -> str:
        code = f"// Operation {operation.stage_id}: Packer\n"
        pack_only = self._get_pack_nodes()
        hoist = len(pack_only) == 1 and len(self.pack_nodes) == 1
        hoist_reconfig = hoist or self._all_same_pack_formats()

        init_code = config.sentinel.hw_configure_pack(config, operation, pack_only)
        if hoist_reconfig and pack_only:
            init_code += config.sentinel.configure_pack(config, operation, pack_only[0])
        init_code += pack_common.pack_reduce_mask_config(operation)
        init_code += pack_common.pack_dest_init(config, operation, pack_only[0])
        if hoist and not pack_only[0].packer.per_block_init:
            init_code += pack_only[0].init(operation, config, None)
        code += self._zone(config, "INIT", init_code)

        init_fn = None
        uninit_fn = None
        if hoist and pack_only[0].packer.per_block_init:
            init_fn = lambda block: pack_only[0].init(operation, config, block)
            uninit_fn = lambda block: pack_only[0].uninit(operation, config)

        def batch_body(block_index: int, block: BlockData):
            body = pack_common.packer_wait_for_math(config, operation)
            if not hoist_reconfig:
                config.sentinel.reset_pack_formats()
            prev_was_pack = False
            for pack_node in self.pack_nodes:
                if isinstance(pack_node, SfpuNode):
                    if prev_was_pack:
                        body += "TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::PACK);\n"
                    body += pack_node.sfpu_init(operation, config, block)
                    body += self._emit_calls(
                        pack_node,
                        block_index,
                        block,
                        InvocationGranularity.BLOCK,
                        pack_node.sfpu_call,
                        operation,
                        config,
                    )
                    body += pack_node.sfpu_uninit(operation, config, block)
                    prev_was_pack = False
                    continue
                if not hoist_reconfig:
                    body += config.sentinel.configure_pack(config, operation, pack_node)
                if not hoist:
                    body += pack_node.init(operation, config, block)
                body += self._emit_calls(
                    pack_node,
                    block_index,
                    block,
                    pack_node.packer.granularity,
                    pack_node.pack_call,
                    operation,
                    config,
                )
                if not hoist:
                    body += pack_node.uninit(operation, config)
                prev_was_pack = True
            body += pack_common.packer_dest_section_done(config, operation)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(batch_body, init_fn, uninit_fn),
        )

        uninit_code = pack_common.packer_sync_with_unpacker(config, operation)
        if hoist and not pack_only[0].packer.per_block_init:
            uninit_code += pack_only[0].uninit(operation, config)
        uninit_code += pack_common.pack_reduce_mask_clear(operation)
        code += self._zone(config, "INIT", uninit_code)

        return code

    def golden(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        golden_type: GoldenType,
    ):
        if self.explicit_blocks:
            return self._indexed_golden(operation, config, golden_type)

        first_fpu = next(
            (
                op
                for op in self.math_nodes
                if isinstance(op, FpuNode) and op.src_a is not None
            ),
            None,
        )
        tensor_dims = (
            first_fpu.src_a.dimensions
            if first_fpu is not None
            else operation.max_output_dimensions
        )
        tensor_b_dims = (
            first_fpu.src_b.dimensions
            if first_fpu is not None and first_fpu.src_b is not None
            else tensor_dims
        )
        tensor_a = torch.zeros(tensor_dims)
        tensor_b = torch.zeros(tensor_b_dims)
        tensor_dst = torch.zeros(operation.max_output_dimensions)
        for op in self.math_nodes:
            config.sentinel.configure_golden(config, operation, op)
            if isinstance(op, SfpuNode):
                tensor_dst = self._sfpu_golden(op, tensor_dst, operation, config)
                continue
            input_tensor_a = self._golden_source(op.src_a, golden_type)
            input_tensor_b = self._golden_source(op.src_b, golden_type)
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
            if isinstance(pack_node, SfpuNode):
                tensor_dst = self._sfpu_golden(pack_node, tensor_dst, operation, config)
                continue

            config.sentinel.configure_golden(
                config, operation, output_format=pack_node.output.data_format
            )

            dimensions = pack_node.output.dimensions
            cropped = tensor_dst.reshape(operation.max_output_dimensions)[
                : dimensions[0], : dimensions[1]
            ]
            result = pack_node.golden(cropped, operation, config)
            if pack_node.pack_l1_accumulation == L1Accumulation.Yes:
                result = self._l1_acc_golden(result, pack_node, operation)
            self._store_golden(pack_node.output, result, golden_type)

    @staticmethod
    def _golden_source(operand, golden_type):
        if operand is None:
            return None
        if golden_type == GoldenType.L1_GOLDEN:
            return operand.raw_data
        return operand.master_golden

    @staticmethod
    def _store_golden(operand, result, golden_type):
        if golden_type == GoldenType.L1_GOLDEN:
            operand.l1_golden = result
        else:
            operand._master_golden = result

    def _indexed_golden(
        self,
        operation: "L1Operation",
        config: "GlobalConfig",
        golden_type: GoldenType,
    ):
        pack_nodes = self._get_pack_nodes()
        outputs = {
            id(node): torch.zeros(
                (node.output.tile_count, node.output.tile_shape.total_tile_size()),
                dtype=format_dict[node.output.data_format],
            )
            for node in pack_nodes
        }
        config.sentinel.configure_golden(
            config, operation, output_format=pack_nodes[0].output.data_format
        )
        master = golden_type == GoldenType.MASTER_GOLDEN
        nodes = self.math_nodes + self.pack_nodes
        for block_index, block in enumerate(self.block_data):
            dest_indices = [
                index
                for node in nodes
                for call in self._block_calls(node, block_index, block)
                for index in (call.src0, call.src1, call.dest)
                if index is not None
            ]
            tensor_dst = torch.zeros(
                (
                    max(dest_indices, default=0) + 1,
                    operation.tile_shape.total_tile_size(),
                ),
                dtype=format_dict[config.sentinel.golden_math_format],
            )
            for node in self.math_nodes:
                config.sentinel.configure_golden(config, operation, node)
                for call in self._block_calls(node, block_index, block):
                    tensor_dst = node.golden_call(
                        call, tensor_dst, operation, config, master
                    )
            for node in self.pack_nodes:
                if isinstance(node, SfpuNode):
                    for call in self._block_calls(node, block_index, block):
                        tensor_dst = node.golden_call(
                            call, tensor_dst, operation, config, master
                        )
                    continue
                config.sentinel.configure_golden(
                    config, operation, output_format=node.output.data_format
                )
                for call in self._block_calls(node, block_index, block):
                    outputs[id(node)] = node.golden_call(
                        call, tensor_dst, outputs[id(node)], operation, config
                    )

        for node in pack_nodes:
            tile_shape = node.output.tile_shape
            result = untilize_block(
                outputs[id(node)].flatten(),
                node.output.data_format,
                node.output.dimensions,
                tile_dimensions=(
                    tile_shape.total_row_dim(),
                    tile_shape.total_col_dim(),
                ),
                num_faces=tile_shape.total_num_faces(),
            )
            self._store_golden(node.output, result, golden_type)

    def __str__(self):
        result = "Math:"
        for op in self.math_nodes:
            result += "\n    "
            result += op.__str__()
        result += "\n  Pack:"
        for pn in self.pack_nodes:
            result += "\n    "
            if isinstance(pn, PackNode):
                result += pn.output.__str__()
            else:
                result += str(pn)
        return result
