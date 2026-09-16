# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from .l1_operation import L1Operation
    from .fuser_config import GlobalConfig

from .arch_common import fpu_common, pack_common, unpack_common
from .base_fpu import Fpu
from .base_sfpu import Sfpu
from .base_unpacker import Unpacker
from .fpu_node import FpuNode
from .pack_node import PackNode
from .pipeline_plan import PlannedBlock, plan_pipeline
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

    @property
    def custom_op(self) -> bool:
        return any(
            node.loop_spec is not None for node in self.math_nodes + self.pack_nodes
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

    def _batch_loop(
        self,
        operation: "L1Operation",
        body_fn,
        init_fn=None,
        uninit_fn=None,
    ) -> str:
        code = ""
        for planned in plan_pipeline(operation):
            body = planned.bank.emit_banks(
                lambda constants: body_fn(planned, constants)
            )
            if not body:
                continue
            if init_fn is not None:
                code += init_fn(planned)
            code += body
            if uninit_fn is not None:
                code += uninit_fn(planned)
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
            init_fn = lambda planned: unpack_ops[0].unpack_init(
                operation, config, planned.block_for(unpack_ops[0])
            )
            uninit_fn = lambda planned: unpack_ops[0].unpack_uninit(
                operation, config, planned.block_for(unpack_ops[0])
            )

        def batch_body(planned: PlannedBlock, constants):
            body = ""
            for cu in self.math_nodes:
                if not isinstance(cu, FpuNode):
                    continue
                block = planned.block_for(cu)
                if (
                    not hoist_reconfig
                    and cu.unpacker is not None
                    and not config.skip_unpack_init
                ):
                    body += config.sentinel.configure_unpack(config, operation, cu)
                if not hoist:
                    body += cu.unpack_init(operation, config, block)
                if cu.unpacker is not None:
                    body += planned.plan(cu, "unpack").emit_calls(
                        constants,
                        partial(cu.unpack_call, operation, config, block),
                    )
                if not hoist:
                    body += cu.unpack_uninit(operation, config, block)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(operation, batch_body, init_fn, uninit_fn),
        )

        uninit_code = ""
        if hoist and not unpack_ops[0].unpacker.per_block_init:
            uninit_code += unpack_ops[0].unpack_uninit(operation, config, None)
        code += self._zone(config, "UNINIT", uninit_code)

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
            init_fn = lambda planned: fpu_ops[0].fpu_init(
                operation, config, planned.block_for(fpu_ops[0])
            )
            uninit_fn = lambda planned: fpu_ops[0].fpu_uninit(
                operation, config, planned.block_for(fpu_ops[0])
            )

        def batch_body(planned: PlannedBlock, constants):
            body = fpu_common.math_wait_for_dest(config, operation)
            for cu in self.math_nodes:
                block = planned.block_for(cu)
                if isinstance(cu, FpuNode):
                    if not hoist_reconfig and not config.skip_math_init:
                        body += config.sentinel.configure_math(config, operation, cu)
                    if not hoist:
                        body += cu.fpu_init(operation, config, block)
                    body += planned.plan(cu, "math").emit_calls(
                        constants,
                        partial(cu.fpu_call, operation, config, block),
                    )
                    if not hoist:
                        body += cu.fpu_uninit(operation, config, block)
                elif isinstance(cu, SfpuNode):
                    body += cu.sfpu_init(operation, config, block)
                    body += planned.plan(cu, "sfpu").emit_calls(
                        constants,
                        partial(cu.sfpu_call, operation, config, block),
                    )
                    body += cu.sfpu_uninit(operation, config, block)
            body += fpu_common.math_dest_section_done(config, operation)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(operation, batch_body, init_fn, uninit_fn),
        )

        uninit_code = ""
        if hoist and not fpu_ops[0].fpu.per_block_init:
            uninit_code += fpu_ops[0].fpu_uninit(operation, config, None)
        code += self._zone(config, "UNINIT", uninit_code)

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
            init_fn = lambda planned: pack_only[0].init(
                operation, config, planned.block_for(pack_only[0])
            )
            uninit_fn = lambda planned: pack_only[0].uninit(operation, config)

        def batch_body(planned: PlannedBlock, constants):
            body = pack_common.packer_wait_for_math(config, operation)
            if not hoist_reconfig:
                config.sentinel.reset_pack_formats()
            prev_was_pack = False
            for pack_node in self.pack_nodes:
                block = planned.block_for(pack_node)
                if isinstance(pack_node, SfpuNode):
                    if prev_was_pack:
                        body += "TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::PACK);\n"
                    body += pack_node.sfpu_init(operation, config, block)
                    body += planned.plan(pack_node, "sfpu").emit_calls(
                        constants,
                        partial(pack_node.sfpu_call, operation, config, block),
                    )
                    body += pack_node.sfpu_uninit(operation, config, block)
                    prev_was_pack = False
                elif isinstance(pack_node, PackNode):
                    if not hoist_reconfig:
                        body += config.sentinel.configure_pack(
                            config, operation, pack_node
                        )
                    if not hoist:
                        body += pack_node.init(operation, config, block)
                    body += planned.plan(pack_node, "pack").emit_calls(
                        constants,
                        partial(pack_node.pack_call, operation, config, block),
                    )
                    if not hoist:
                        body += pack_node.uninit(operation, config)
                    prev_was_pack = True
            body += pack_common.packer_dest_section_done(config, operation)
            return body

        code += self._zone_loop(
            config,
            "TILE_LOOP",
            self._batch_loop(operation, batch_body, init_fn, uninit_fn),
        )

        uninit_code = pack_common.packer_sync_with_unpacker(config, operation)
        if hoist and not pack_only[0].packer.per_block_init:
            uninit_code += pack_only[0].uninit(operation, config)
        uninit_code += pack_common.pack_reduce_mask_clear(operation)
        code += self._zone(config, "UNINIT", uninit_code)

        return code

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
