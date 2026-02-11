# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List, Tuple, Union

import torch

if TYPE_CHECKING:
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig

from .fused_fpu import Fpu, MatmulFpu, ReduceFpu
from .fused_sfpu import Sfpu
from .fused_unpacker import MatmulUnpacker, Unpacker, UnpackerA
from .llk_params import (
    BroadcastType,
    DataCopyType,
    DestSync,
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
        if (
            fpu is not None
            and unpacker is not None
            and unpacker not in fpu.supported_unpackers()
        ):
            raise ValueError(f"{fpu} does not support {unpacker}")

        if reuse_dest != EltwiseBinaryReuseDestType.NONE and unpacker != UnpackerA:
            raise ValueError("Reuse dest is only supported with UnpackerA")

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
        batch_start,
        batch_tile_cnt,
    ):
        if self.unpacker is None:
            return ""

        if config.perf_run_type == PerfRunType.PACK_ISOLATE:
            return ""

        if config.perf_run_type == PerfRunType.MATH_ISOLATE:
            return self.unpacker().perf_set_valid(operation, config, self)

        code = self.unpacker().init(operation, config, self)
        if isinstance(self.unpacker, MatmulUnpacker) or self.unpacker == MatmulUnpacker:
            tile_idx_expr = f"{batch_start}"
            code += self.unpacker().unpack(operation, config, self, tile_idx_expr)
        else:
            for tile_idx in range(batch_tile_cnt):
                tile_idx_expr = f"{batch_start} + {tile_idx}"
                code += self.unpacker().unpack(operation, config, self, tile_idx_expr)
        code += self.unpacker().uninit(operation, config, self)
        return code

    def fpu_calculate(
        self, operation: "FusedOperation", config: "GlobalConfig", batch_tile_cnt
    ):
        if self.fpu is None:
            return ""

        if config.perf_run_type == PerfRunType.PACK_ISOLATE:
            return ""

        if config.perf_run_type in (
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ):
            code = ""
            for tile_idx in range(batch_tile_cnt):
                code += self.unpacker().perf_clear_valid(operation, config, self)
            return code

        code = self.fpu.init(operation, config, self)
        if isinstance(self.fpu, MatmulFpu):
            code += self.fpu.calculate(operation, config, self, 0)
        else:
            for tile_idx in range(batch_tile_cnt):
                code += self.fpu.calculate(operation, config, self, tile_idx)
        code += self.fpu.uninit(operation, config, self)
        return code

    def sfpu_calculate(self, operation: "FusedOperation", config: "GlobalConfig"):
        if self.sfpu is None:
            return ""

        if config.perf_run_type in (
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ):
            return ""

        code = self.sfpu.init(operation, config, self)
        code += self.sfpu.calculate(operation, config, self)
        code += self.sfpu.uninit(operation, config, self)
        return code

    def math_calculate(
        self, operation: "FusedOperation", config: "GlobalConfig", batch_tile_cnt
    ) -> str:
        if self.fpu is not None:
            return self.fpu_calculate(operation, config, batch_tile_cnt)
        elif self.sfpu is not None:
            return self.sfpu_calculate(operation, config)
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
            ).flatten()

            batch_size = operation.batch_size
            tile_cnt = operation.output.tile_count
            tile_size = 1024

            batch_start = 0
            for batch_start in range(0, tile_cnt, batch_size):
                batch_end = min(batch_start + batch_size, tile_cnt)
                batch_tile_cnt = batch_end - batch_start

                batch_start_elem = batch_start * tile_size
                batch_end_elem = batch_end * tile_size
                batch_tensor = tilized_dst[batch_start_elem:batch_end_elem].clone()

                batch_dims = (batch_tile_cnt * 32, 32)

                batch_tensor = self.sfpu.golden(
                    batch_tensor, operation, config, self, batch_dims, batch_tile_cnt
                )

                tilized_dst[batch_start_elem:batch_end_elem] = batch_tensor.flatten()

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

    def __init__(self, operations: List[ComputeNode]):
        self.operations = operations

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
        dest_acc = config.dest_acc.value
        unpa_face_r_dim = operation.face_r_dim
        unpb_face_r_dim = operation.face_r_dim
        unpa_num_faces = operation.num_faces_A
        unpb_num_faces = operation.num_faces_B

        if stage == 0:
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
        if operation.stage_id > 0:
            return (
                "t6_semaphore_wait_on_zero<p_stall::STALL_SYNC>(semaphore::PACK_DONE);\n"
                "t6_semaphore_get<>(semaphore::PACK_DONE);\n"
            )

        return ""

    def math_hw_configure(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        stage = operation.stage_id
        dest_acc = config.dest_acc.value
        if stage == 0:
            code = f"_llk_math_hw_configure_<{dest_acc}>(math_format{stage}, math_format{stage});\n"
        else:
            code = f"_llk_math_reconfig_data_format_<{dest_acc}, false>(math_format{stage}, math_format{stage});\n"

        code += f"_llk_math_pack_sync_init_<dest_sync{stage}, {dest_acc}>();\n"

        return code

    def _wait_for_dest(
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

    def _dest_section_done(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        if config.perf_run_type in (
            PerfRunType.MATH_ISOLATE,
            PerfRunType.UNPACK_ISOLATE,
            PerfRunType.PACK_ISOLATE,
            PerfRunType.L1_CONGESTION,
        ):
            return ""

        return f"_llk_math_dest_section_done_<dest_sync{operation.stage_id}, {config.dest_acc.value}>();\n"

    def _batch_loop(
        self, operation: "FusedOperation", config: "GlobalConfig", body_fn
    ) -> str:
        batch_size = operation.batch_size
        tile_cnt = operation.output.tile_count

        num_full_batches = tile_cnt // batch_size
        remaining_tiles = tile_cnt % batch_size

        code = ""

        if num_full_batches > 0:
            code += f"for (std::uint32_t batch = 0; batch < {num_full_batches}; ++batch) {{\n"
            code += body_fn(f"batch * {batch_size}", batch_size)
            code += "}\n"

        if remaining_tiles > 0:
            code += "{\n"
            code += body_fn(f"{num_full_batches * batch_size}", remaining_tiles)
            code += "}\n"

        return code

    def _math_constants(
        self, operation: "FusedOperation", config: "GlobalConfig"
    ) -> str:
        stage = operation.stage_id
        math_format = operation.output.data_format
        dest_sync = operation.dest_sync

        dest_sync_map = {
            DestSync.Half: "SyncHalf",
            DestSync.Full: "SyncFull",
        }
        dest_sync_str = dest_sync_map.get(dest_sync, "SyncHalf")

        code = f"// Operation {stage}: Math Setup\n"
        code += f"const std::uint32_t math_format{stage} = ckernel::to_underlying(DataFormat::{math_format.name});\n"
        code += f"constexpr DstSync dest_sync{stage} = DstSync::{dest_sync_str};\n"

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

        def batch_body(batch_start, batch_tile_cnt):
            body = self._wait_for_dest(operation, config)
            for compute_unit in self.operations:
                body += compute_unit.math_calculate(operation, config, batch_tile_cnt)
            body += self._dest_section_done(operation, config)
            return body

        code += self._batch_loop(operation, config, batch_body)

        if config.profiler_enabled:
            code += "}\n"
            code += "PROFILER_SYNC();\n"
            code += "}\n"

        return code

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

        def batch_body(batch_start, batch_tile_cnt):
            body = ""
            for compute_unit in self.operations:
                body += compute_unit.unpack(
                    operation, config, batch_start, batch_tile_cnt
                )
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
        str = ""
        for op in self.operations:
            str += "\n  "
            str += op.__str__()

        return str
