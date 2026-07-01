# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from helpers.chip_architecture import ChipArchitecture
from helpers.data_format_inference import (
    infer_math_format,
    infer_pack_in,
    infer_unpack_out,
)
from helpers.format_config import DataFormat
from helpers.llk_params import EltwiseBinaryReuseDestType

if TYPE_CHECKING:
    from .fpu_node import FpuNode
    from .fused_operand import Operand
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig
    from .pack_node import PackNode


@dataclass
class FuserSentinel:
    """Tracks format state for unpack, math, and pack threads across a fused pipeline.

    Each format field is None until the first hw_configure call, then holds the
    currently configured DataFormat. Subsequent calls compare against the stored
    state and emit reconfig only when formats actually change.
    """

    _unpack_A_src: Optional[DataFormat] = field(default=None, repr=False)
    _unpack_A_dst: Optional[DataFormat] = field(default=None, repr=False)
    _unpack_B_src: Optional[DataFormat] = field(default=None, repr=False)
    _unpack_B_dst: Optional[DataFormat] = field(default=None, repr=False)

    _unpack_face_r_dim_a: Optional[int] = field(default=None, repr=False)
    _unpack_num_faces_a: Optional[int] = field(default=None, repr=False)
    _unpack_face_r_dim_b: Optional[int] = field(default=None, repr=False)
    _unpack_num_faces_b: Optional[int] = field(default=None, repr=False)

    _math_format: Optional[DataFormat] = field(default=None, repr=False)

    _pack_src: Optional[DataFormat] = field(default=None, repr=False)
    _pack_dst: Optional[DataFormat] = field(default=None, repr=False)

    golden_math_format: Optional[DataFormat] = field(default=None, repr=False)
    golden_pack_src: Optional[DataFormat] = field(default=None, repr=False)

    _next_unpack_buf_desc_id: int = field(default=0, repr=False)
    _next_pack_buf_desc_id: int = field(default=8, repr=False)

    def _alloc_unpack_buf_desc_id(self) -> int:
        bid = self._next_unpack_buf_desc_id
        self._next_unpack_buf_desc_id += 1
        return bid

    def _alloc_pack_buf_desc_id(self) -> int:
        bid = self._next_pack_buf_desc_id
        self._next_pack_buf_desc_id += 1
        return bid

    def ensure_unpack_buf_desc_ids(self, compute_node: "FpuNode") -> None:
        if compute_node.src_a is not None and compute_node.src_a.buf_desc_id is None:
            compute_node.src_a.buf_desc_id = self._alloc_unpack_buf_desc_id()
        if compute_node.src_b is not None and compute_node.src_b.buf_desc_id is None:
            compute_node.src_b.buf_desc_id = self._alloc_unpack_buf_desc_id()

    def ensure_pack_buf_desc_id(self, pack_node: "PackNode") -> None:
        if pack_node.output is not None and pack_node.output.buf_desc_id is None:
            pack_node.output.buf_desc_id = self._alloc_pack_buf_desc_id()

    def reset_unpack_formats(self):
        self._unpack_A_src = None
        self._unpack_A_dst = None
        self._unpack_B_src = None
        self._unpack_B_dst = None
        self._unpack_face_r_dim_a = None
        self._unpack_num_faces_a = None
        self._unpack_face_r_dim_b = None
        self._unpack_num_faces_b = None

    def reset_math_format(self):
        self._math_format = None

    def reset_pack_formats(self):
        self._pack_src = None
        self._pack_dst = None

    @staticmethod
    def _find_format_node(
        operation: "FusedOperation",
    ) -> Optional["FpuNode"]:
        """Find the first FpuNode for format inference.

        In a pipeline with mixed FPU and SFPU nodes, only FPU nodes have src_a/src_b
        operands. SFPU nodes operate on data already in the dest register and don't
        drive format configuration.

        Returns:
            The first FpuNode, or None if all nodes are SFPU only.
        """
        from .fpu_node import FpuNode

        for node in operation.math.math_nodes:
            if isinstance(node, FpuNode):
                return node
        return None

    @staticmethod
    def _get_src_formats(
        compute_node: "FpuNode",
    ) -> Tuple[DataFormat, Optional[DataFormat]]:
        """Extract src_a and src_b data formats, handling DEST_TO_SRCA routing."""
        src_a_fmt = compute_node.src_a.data_format
        if compute_node.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
            return src_a_fmt, src_a_fmt
        if compute_node.src_b is not None:
            return src_a_fmt, compute_node.src_b.data_format
        return src_a_fmt, None

    def _infer_node_formats(
        self,
        config: "GlobalConfig",
        compute_node: "FpuNode",
        output_format: DataFormat,
    ) -> Tuple[DataFormat, DataFormat, DataFormat, DataFormat, DataFormat, DataFormat]:
        """Infer all pipeline formats from a compute node's operands.

        Returns:
            (unpack_A_src, unpack_A_dst, unpack_B_src, unpack_B_dst, math_fmt, pack_src)
        """
        src_a_fmt, src_b_fmt = self._get_src_formats(compute_node)
        unpack_to_dest = compute_node.unpack_to_dest.value
        dest_acc = config.dest_acc

        unpack_A_dst = infer_unpack_out(
            src_a_fmt, output_format, dest_acc, unpack_to_dest
        )
        if src_b_fmt is not None:
            unpack_B_src = src_b_fmt
            unpack_B_dst = infer_unpack_out(
                src_b_fmt, output_format, dest_acc, unpack_to_dest
            )
        else:
            unpack_B_src = src_a_fmt
            unpack_B_dst = unpack_A_dst

        math_fmt = infer_math_format(unpack_A_dst, unpack_B_dst)
        if math_fmt == DataFormat.Fp8_e4m3:
            math_fmt = DataFormat.Float16

        pack_src = infer_pack_in(
            src_a_fmt,
            output_format,
            math_fmt,
            dest_acc,
            unpack_to_dest,
            config.architecture,
        )
        if output_format == DataFormat.Fp8_e4m3:
            pack_src = (
                DataFormat.Float16_b if math_fmt.is_exponent_B() else DataFormat.Float16
            )

        return src_a_fmt, unpack_A_dst, unpack_B_src, unpack_B_dst, math_fmt, pack_src

    def _infer_output_formats(
        self,
        config: "GlobalConfig",
        output_format: DataFormat,
    ) -> Tuple[DataFormat, DataFormat, DataFormat, DataFormat, DataFormat, DataFormat]:
        """Infer formats for SFPU only operations that have no input operands.

        Uses the output format as both input and output with unpacking_to_dest=True
        since SFPU operates directly on the dest register.

        Returns:
            (unpack_A_src, unpack_A_dst, unpack_B_src, unpack_B_dst, math_fmt, pack_src)
        """
        dest_acc = config.dest_acc

        unpack_dst = infer_unpack_out(
            output_format, output_format, dest_acc, unpacking_to_dest=True
        )

        math_fmt = infer_math_format(unpack_dst)
        if math_fmt == DataFormat.Fp8_e4m3:
            math_fmt = DataFormat.Float16

        pack_src = infer_pack_in(
            output_format,
            output_format,
            math_fmt,
            dest_acc,
            unpacking_to_dest=True,
            chip_arch=config.architecture,
        )
        if output_format == DataFormat.Fp8_e4m3:
            pack_src = (
                DataFormat.Float16_b if math_fmt.is_exponent_B() else DataFormat.Float16
            )

        return output_format, unpack_dst, output_format, unpack_dst, math_fmt, pack_src

    def _resolve_pack_formats(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
        pack_node: "PackNode",
    ) -> Tuple[DataFormat, DataFormat]:
        """Infer pack_src and pack_dst formats for a given pack node."""
        output_format = pack_node.output.data_format

        compute_node = self._find_format_node(operation)
        if compute_node is not None:
            _, _, _, _, _, pack_src = self._infer_node_formats(
                config, compute_node, output_format
            )
        else:
            _, _, _, _, _, pack_src = self._infer_output_formats(config, output_format)

        return pack_src, output_format

    @staticmethod
    def _fmt(data_format: DataFormat) -> str:
        return f"ckernel::to_underlying(DataFormat::{data_format.name})"

    @staticmethod
    def _dfmt(data_format: DataFormat) -> str:
        return f"DataFormat::{data_format.name}"

    # Properties for reading the current format state
    @property
    def unpack_a_src_format(self) -> str:
        return self._fmt(self._unpack_A_src)

    @property
    def unpack_a_dst_format(self) -> str:
        return self._fmt(self._unpack_A_dst)

    @property
    def unpack_b_src_format(self) -> str:
        return self._fmt(self._unpack_B_src)

    @property
    def unpack_b_dst_format(self) -> str:
        return self._fmt(self._unpack_B_dst)

    @property
    def math_format(self) -> str:
        return self._fmt(self._math_format)

    @property
    def pack_src_format(self) -> str:
        return self._fmt(self._pack_src)

    @property
    def pack_dst_format(self) -> str:
        return self._fmt(self._pack_dst)

    def _emit_quasar_buf_desc(
        self,
        var_prefix: str,
        operand: "Operand",
        src_fmt: DataFormat,
        dst_fmt: DataFormat,
    ) -> str:
        """Emit buffer_descriptor_u + tdma_descriptor_t setup for a Quasar operand."""
        face_r = operand.tile_shape.face_r_dim
        face_c = operand.tile_shape.face_c_dim
        num_faces = operand.tile_shape.total_num_faces()

        code = ""
        code += f"buffer_descriptor_u bd_{var_prefix} {{}};\n"
        code += f"bd_{var_prefix}.f.l1_addr_16B = {operand.cpp_name}[0] / 16;\n"
        code += f"bd_{var_prefix}.f.format = static_cast<std::uint8_t>({self._fmt(src_fmt)});\n"
        code += f"bd_{var_prefix}.f.x_dim = {face_c};\n"
        code += f"bd_{var_prefix}.f.y_dim = {face_r};\n"
        code += f"bd_{var_prefix}.f.z_dim = {num_faces};\n"
        code += f"tdma_descriptor_t td_{var_prefix};\n"
        code += f"td_{var_prefix}.buf_desc = bd_{var_prefix};\n"
        code += f"td_{var_prefix}.buf_desc_id = {operand.buf_desc_id};\n"
        code += f"td_{var_prefix}.reg_data_format = static_cast<std::uint8_t>({self._fmt(dst_fmt)});\n"
        code += f"_configure_buf_desc_table_(td_{var_prefix}.buf_desc_id, td_{var_prefix}.buf_desc);\n"
        return code

    @staticmethod
    def _is_datacopy_node(compute_node: "FpuNode") -> bool:
        from .quasar.fpu.datacopy import DatacopyFpu

        return isinstance(compute_node.fpu, DatacopyFpu)

    def _hw_configure_unpack_quasar(
        self,
        config: "GlobalConfig",
        compute_node: "FpuNode",
        unpack_A_src: DataFormat,
        unpack_A_dst: DataFormat,
        unpack_B_src: DataFormat,
        unpack_B_dst: DataFormat,
    ) -> str:
        """Emit Quasar-specific buffer descriptor setup and configure call."""
        from .quasar.unpacker.tilize_a import UnpackerTilizeA as QsrUnpackerTilizeA
        from .quasar.unpacker.unpack_a import UnpackerA as QsrUnpackerA

        src_a = compute_node.src_a
        is_unary = isinstance(compute_node.unpacker, (QsrUnpackerA, QsrUnpackerTilizeA))
        is_datacopy_dest_acc = (
            self._is_datacopy_node(compute_node) and config.dest_acc.value
        )

        if src_a.buf_desc_id is None:
            src_a.buf_desc_id = self._alloc_unpack_buf_desc_id()

        code = self._emit_quasar_buf_desc("val_A", src_a, unpack_A_src, unpack_A_dst)

        if is_unary and is_datacopy_dest_acc:
            code += f"_llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_A);\n"
        elif is_unary:
            code += f"_llk_unpack_configure_unary_<p_unpacr::UNP_A>(td_val_A);\n"
        else:
            src_b = compute_node.src_b
            if src_b.buf_desc_id is None:
                src_b.buf_desc_id = self._alloc_unpack_buf_desc_id()
            code += self._emit_quasar_buf_desc(
                "val_B", src_b, unpack_B_src, unpack_B_dst
            )
            code += f"_llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);\n"

        return code

    def hw_configure_unpack(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
    ) -> str:
        """Emit _llk_unpack_hw_configure_ once for the first operation in the pipeline.

        Called at the top of each operation's unpack_body(). On the first call
        (when _unpack_A_src is None), emits the full hw_configure with tile shape
        parameters from the first node that has operand inputs.
        """
        compute_node = self._find_format_node(operation)
        if compute_node is None:
            return ""

        if self._unpack_A_src is not None:
            return ""

        output_format = operation.math._get_pack_nodes()[0].output.data_format
        unpack_A_src, unpack_A_dst, unpack_B_src, unpack_B_dst, _, _ = (
            self._infer_node_formats(config, compute_node, output_format)
        )

        self._unpack_A_src = unpack_A_src
        self._unpack_A_dst = unpack_A_dst
        self._unpack_B_src = unpack_B_src
        self._unpack_B_dst = unpack_B_dst

        face_r_dim_a = compute_node.src_a.tile_shape.face_r_dim
        num_faces_a = compute_node.src_a.tile_shape.total_num_faces()
        tile_size_a = compute_node.src_a.tile_size

        if compute_node.src_b is not None:
            face_r_dim_b = compute_node.src_b.tile_shape.face_r_dim
            num_faces_b = compute_node.src_b.tile_shape.total_num_faces()
            tile_size_b = compute_node.src_b.tile_size
        else:
            face_r_dim_b = face_r_dim_a
            num_faces_b = num_faces_a
            tile_size_b = tile_size_a

        self._unpack_face_r_dim_a = face_r_dim_a
        self._unpack_num_faces_a = num_faces_a
        self._unpack_face_r_dim_b = face_r_dim_b
        self._unpack_num_faces_b = num_faces_b

        if config.architecture == ChipArchitecture.QUASAR:
            return self._hw_configure_unpack_quasar(
                config,
                compute_node,
                unpack_A_src,
                unpack_A_dst,
                unpack_B_src,
                unpack_B_dst,
            )

        dest_acc = config.dest_acc.cpp_enum_value
        return (
            f"_llk_unpack_hw_configure_<{dest_acc}>(\n"
            f"    {self._fmt(unpack_A_src)}, {self._fmt(unpack_B_src)},\n"
            f"    {self._fmt(unpack_A_dst)}, {self._fmt(unpack_B_dst)},\n"
            f"    {face_r_dim_a}, {face_r_dim_b}, {num_faces_a}, {num_faces_b},\n"
            f"    {tile_size_a}, {tile_size_b}\n"
            f");\n"
        )

    def configure_unpack(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
        compute_node: "FpuNode",
    ) -> str:
        """Emit unpack reconfig calls when formats or tile shapes change between compute nodes.

        Called per node from FpuNode.unpack_configure() inside the tile loop. Compares
        the node's inferred formats and tile shape against the currently configured state
        and emits _llk_unpack_reconfig_data_format_src{a,b}_impl_ for channels that changed.
        When tile shapes differ, uses FACE_ROW_MAJOR to reprogram dim/stride registers.
        """
        output_format = operation.math._get_pack_nodes()[0].output.data_format
        new_A_src, new_A_dst, new_B_src, new_B_dst, _, _ = self._infer_node_formats(
            config, compute_node, output_format
        )

        new_face_r_dim_a = compute_node.src_a.tile_shape.face_r_dim
        new_num_faces_a = compute_node.src_a.tile_shape.total_num_faces()

        if compute_node.src_b is not None:
            new_face_r_dim_b = compute_node.src_b.tile_shape.face_r_dim
            new_num_faces_b = compute_node.src_b.tile_shape.total_num_faces()
        else:
            new_face_r_dim_b = new_face_r_dim_a
            new_num_faces_b = new_num_faces_a

        srca_fmt_changed = (
            self._unpack_A_src != new_A_src or self._unpack_A_dst != new_A_dst
        )
        srcb_fmt_changed = (
            self._unpack_B_src != new_B_src or self._unpack_B_dst != new_B_dst
        )
        srca_tile_changed = (
            self._unpack_face_r_dim_a != new_face_r_dim_a
            or self._unpack_num_faces_a != new_num_faces_a
        )
        srcb_tile_changed = (
            self._unpack_face_r_dim_b != new_face_r_dim_b
            or self._unpack_num_faces_b != new_num_faces_b
        )

        srca_changed = srca_fmt_changed or srca_tile_changed
        srcb_changed = srcb_fmt_changed or srcb_tile_changed

        if config.architecture == ChipArchitecture.QUASAR:
            from .quasar.unpacker.tilize_a import UnpackerTilizeA as QsrUnpackerTilizeA
            from .quasar.unpacker.unpack_a import UnpackerA as QsrUnpackerA

            is_unary = isinstance(
                compute_node.unpacker, (QsrUnpackerA, QsrUnpackerTilizeA)
            )

            needs_buf_desc = compute_node.src_a.buf_desc_id is None
            if not is_unary:
                needs_buf_desc = needs_buf_desc or (
                    compute_node.src_b is not None
                    and compute_node.src_b.buf_desc_id is None
                )
            if not (srca_changed or srcb_changed or needs_buf_desc):
                return ""
            if compute_node.src_a.buf_desc_id is None:
                compute_node.src_a.buf_desc_id = self._alloc_unpack_buf_desc_id()
            code = "{\n"
            code += self._emit_quasar_buf_desc(
                "val_A", compute_node.src_a, new_A_src, new_A_dst
            )
            if (
                is_unary
                and self._is_datacopy_node(compute_node)
                and config.dest_acc.value
            ):
                code += "_llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_A);\n"
            elif is_unary:
                code += "_llk_unpack_configure_unary_<p_unpacr::UNP_A>(td_val_A);\n"
            else:
                if (
                    compute_node.src_b is not None
                    and compute_node.src_b.buf_desc_id is None
                ):
                    compute_node.src_b.buf_desc_id = self._alloc_unpack_buf_desc_id()
                code += self._emit_quasar_buf_desc(
                    "val_B", compute_node.src_b, new_B_src, new_B_dst
                )
                code += "_llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val_A, td_val_B);\n"
            code += "}\n"
        elif not (srca_changed or srcb_changed):
            return ""
        else:
            dest_acc = config.dest_acc.cpp_enum_value

            if srca_changed:
                to_from_int8 = (
                    "true"
                    if self._unpack_A_src.needs_int8_math_config()
                    or new_A_src.needs_int8_math_config()
                    else "false"
                )
                dim_stride = (
                    "p_dim_stride_target::FACE_ROW_MAJOR"
                    if srca_tile_changed
                    else "p_dim_stride_target::IGNORE"
                )
                code += (
                    f"_llk_unpack_reconfig_data_format_srca_impl_<{dest_acc}, {dim_stride}, {to_from_int8}>(\n"
                    f"    {self._fmt(new_A_src)}, {self._fmt(new_A_dst)}, {compute_node.src_a.tile_size}"
                )
                if srca_tile_changed:
                    code += f", {new_face_r_dim_a}, {new_num_faces_a}"
                code += "\n);\n"

            if srcb_changed:
                srcb_tile_size = (
                    compute_node.src_a.tile_size
                    if compute_node.reuse_dest
                    is EltwiseBinaryReuseDestType.DEST_TO_SRCA
                    else (compute_node.src_b.tile_size if compute_node.src_b else None)
                )
                if srcb_tile_size is not None:
                    to_from_int8 = (
                        "true"
                        if self._unpack_B_src.needs_int8_math_config()
                        or new_B_src.needs_int8_math_config()
                        else "false"
                    )
                    dim_stride = (
                        "p_dim_stride_target::FACE_ROW_MAJOR"
                        if srcb_tile_changed
                        else "p_dim_stride_target::IGNORE"
                    )
                    code += (
                        f"_llk_unpack_reconfig_data_format_srcb_impl_<{dest_acc}, {dim_stride}, {to_from_int8}>(\n"
                        f"    {self._fmt(new_B_src)}, {self._fmt(new_B_dst)}, {srcb_tile_size}"
                    )
                    if srcb_tile_changed:
                        code += f", {new_face_r_dim_b}, {new_num_faces_b}"
                    code += "\n);\n"

        self._unpack_A_src = new_A_src
        self._unpack_A_dst = new_A_dst
        self._unpack_B_src = new_B_src
        self._unpack_B_dst = new_B_dst
        self._unpack_face_r_dim_a = new_face_r_dim_a
        self._unpack_num_faces_a = new_num_faces_a
        self._unpack_face_r_dim_b = new_face_r_dim_b
        self._unpack_num_faces_b = new_num_faces_b
        return code

    def hw_configure_math(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
    ) -> str:
        """Emit _llk_math_hw_configure_ once for the first operation in the pipeline.

        For SFPU only operations (no node with src_a), infers a format compatible
        with the output so the math hardware is always configured.
        """
        if self._math_format is not None:
            return ""

        output_format = operation.math._get_pack_nodes()[0].output.data_format
        compute_node = self._find_format_node(operation)
        if compute_node is not None:
            _, _, _, _, math_fmt, _ = self._infer_node_formats(
                config, compute_node, output_format
            )
        else:
            _, _, _, _, math_fmt, _ = self._infer_output_formats(config, output_format)

        self._math_format = math_fmt

        if config.architecture == ChipArchitecture.QUASAR:
            is_fp32 = "true" if config.dest_acc.value else "false"
            return (
                f"_llk_math_srcAB_hw_configure_<true, {is_fp32}, false>(\n"
                f"    {self._dfmt(math_fmt)}, {self._dfmt(math_fmt)}\n"
                f");\n"
            )

        dest_acc = config.dest_acc.cpp_enum_value
        return (
            f"_llk_math_hw_configure_<{dest_acc}>(\n"
            f"    {self._fmt(math_fmt)}, {self._fmt(math_fmt)}\n"
            f");\n"
        )

    def configure_math(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
        compute_node: "FpuNode",
    ) -> str:
        """Emit math reconfig when the math format changes between compute nodes.

        Called per node from FpuNode.math_configure() inside the tile loop.
        """
        if compute_node.src_a is None:
            return ""

        output_format = operation.math._get_pack_nodes()[0].output.data_format
        _, _, _, _, new_math, _ = self._infer_node_formats(
            config, compute_node, output_format
        )

        if self._math_format == new_math:
            return ""

        if config.architecture == ChipArchitecture.QUASAR:
            is_fp32 = "true" if config.dest_acc.value else "false"
            code = (
                f"_llk_math_srcAB_hw_configure_<false, {is_fp32}, false>(\n"
                f"    {self._dfmt(new_math)}, {self._dfmt(new_math)}\n"
                f");\n"
            )
        else:
            to_from_int8 = (
                "true"
                if self._math_format.needs_int8_math_config()
                or new_math.needs_int8_math_config()
                else "false"
            )
            dest_acc = config.dest_acc.cpp_enum_value
            code = (
                f"_llk_math_reconfig_data_format_<{dest_acc}, {to_from_int8}>(\n"
                f"    {self._fmt(new_math)}, {self._fmt(new_math)}\n"
                f");\n"
            )
        self._math_format = new_math
        return code

    def hw_configure_pack(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
        pack_nodes: List["PackNode"],
    ) -> str:
        """Emit _llk_pack_hw_configure_ once for the first operation in the pipeline.

        Called at the top of each operation's pack_body(). On the first call
        (when _pack_src is None), emits the full hw_configure for the first
        pack_node's format and sets sentinel state. Subsequent operations
        rely on configure_pack() to emit reconfig as needed.
        """
        if self._pack_src is not None:
            return ""

        first = pack_nodes[0]
        pack_src, pack_dst = self._resolve_pack_formats(config, operation, first)

        if config.architecture == ChipArchitecture.QUASAR:
            output = first.output
            if output.buf_desc_id is None:
                output.buf_desc_id = self._alloc_pack_buf_desc_id()

            face_r_dim = output.tile_shape.face_r_dim
            face_c_dim = output.tile_shape.face_c_dim
            num_faces_r = output.tile_shape.total_row_dim() // face_r_dim
            num_faces_c = output.tile_shape.total_col_dim() // face_c_dim

            code = ""
            code += f"buffer_descriptor_u bd_pack {{}};\n"
            code += f"bd_pack.f.l1_addr_16B = {output.cpp_name}[0] / 16;\n"
            code += f"bd_pack.f.format = static_cast<std::uint8_t>({self._fmt(pack_dst)});\n"
            code += f"bd_pack.f.x_dim = {face_c_dim};\n"
            code += f"bd_pack.f.y_dim = {face_r_dim};\n"
            code += f"bd_pack.f.z_dim = {num_faces_r * num_faces_c};\n"
            code += f"tdma_descriptor_t td_pack;\n"
            code += f"td_pack.buf_desc = bd_pack;\n"
            code += f"td_pack.buf_desc_id = {output.buf_desc_id};\n"
            code += f"td_pack.reg_data_format = static_cast<std::uint8_t>({self._fmt(pack_src)});\n"
            code += (
                f"_configure_buf_desc_table_(td_pack.buf_desc_id, td_pack.buf_desc);\n"
            )
            code += f"_llk_pack_hw_configure_<p_pacr::PACK0>(td_pack);\n"
        else:
            dest_acc = config.dest_acc.cpp_enum_value
            pack_size = first.output.tile_size
            face_r_dim = first.output.tile_shape.face_r_dim
            tile_c_dim = first.output.tile_shape.total_col_dim()
            num_faces = first.output.tile_shape.total_num_faces()

            if config.architecture == ChipArchitecture.BLACKHOLE:
                bh_pack_mode = operation.bh_tilize.pack_mode_value
                code = (
                    f"_llk_pack_hw_configure_<{dest_acc}, {bh_pack_mode}>(\n"
                    f"    {self._fmt(pack_src)}, {self._fmt(pack_dst)}, {pack_size}, {face_r_dim}, {tile_c_dim}, {num_faces}\n"
                    f");\n"
                )
            else:
                code = (
                    f"_llk_pack_hw_configure_<{dest_acc}, PackMode::Default>(\n"
                    f"    {self._fmt(pack_src)}, {self._fmt(pack_dst)}, {pack_size}, {face_r_dim}, {num_faces}\n"
                    f");\n"
                )

        self._pack_src = pack_src
        self._pack_dst = pack_dst

        return code

    def configure_pack(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
        pack_node: "PackNode",
    ) -> str:
        """Update pack format state for a specific pack node and emit reconfig if needed.

        Called from PackNode.configure() before packer.init() and _relu_config()
        so that each pack node reads its own correct formats from the sentinel.
        """
        pack_src, pack_dst = self._resolve_pack_formats(config, operation, pack_node)

        if self._pack_src == pack_src and self._pack_dst == pack_dst:
            return ""

        if config.architecture == ChipArchitecture.QUASAR:
            output = pack_node.output
            if output.buf_desc_id is None:
                output.buf_desc_id = self._alloc_pack_buf_desc_id()
            face_r_dim = output.tile_shape.face_r_dim
            face_c_dim = output.tile_shape.face_c_dim
            num_faces = output.tile_shape.total_num_faces()
            code = "{\n"
            code += f"buffer_descriptor_u bd_pack {{}};\n"
            code += f"bd_pack.f.l1_addr_16B = {output.cpp_name}[0] / 16;\n"
            code += f"bd_pack.f.format = static_cast<std::uint8_t>({self._fmt(pack_dst)});\n"
            code += f"bd_pack.f.x_dim = {face_c_dim};\n"
            code += f"bd_pack.f.y_dim = {face_r_dim};\n"
            code += f"bd_pack.f.z_dim = {num_faces};\n"
            code += f"tdma_descriptor_t td_pack;\n"
            code += f"td_pack.buf_desc = bd_pack;\n"
            code += f"td_pack.buf_desc_id = {output.buf_desc_id};\n"
            code += f"td_pack.reg_data_format = static_cast<std::uint8_t>({self._fmt(pack_src)});\n"
            code += (
                f"_configure_buf_desc_table_(td_pack.buf_desc_id, td_pack.buf_desc);\n"
            )
            code += f"_llk_pack_hw_configure_<p_pacr::PACK0>(td_pack);\n"
            code += "}\n"
        else:
            dest_acc = config.dest_acc.cpp_enum_value
            pack_size = pack_node.output.tile_size

            code = (
                f"_llk_pack_reconfig_data_format_<{dest_acc}>(\n"
                f"    {self._fmt(pack_src)}, {self._fmt(pack_dst)}, {pack_size}\n"
                f");\n"
            )
        self._pack_src = pack_src
        self._pack_dst = pack_dst
        return code

    def configure_golden(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
        compute_node=None,
        output_format: DataFormat = DataFormat.Float16_b,
    ):
        """Compute and store format values for golden generation.

        Called per compute node during golden computation. When called without
        a compute_node (at operation start), initializes from the first format
        node or from the output format. When called with a compute_node,
        recomputes only if the node is an FpuNode.
        """
        from .fpu_node import FpuNode

        if compute_node is None:
            fmt_node = self._find_format_node(operation)
            if fmt_node is not None:
                _, _, _, _, math_fmt, pack_src = self._infer_node_formats(
                    config, fmt_node, output_format
                )
            else:
                _, _, _, _, math_fmt, pack_src = self._infer_output_formats(
                    config, output_format
                )
            self.golden_math_format = math_fmt
            self.golden_pack_src = pack_src
            return

        if not isinstance(compute_node, FpuNode):
            return

        _, _, _, _, math_fmt, pack_src = self._infer_node_formats(
            config, compute_node, output_format
        )
        self.golden_math_format = math_fmt
        self.golden_pack_src = pack_src
