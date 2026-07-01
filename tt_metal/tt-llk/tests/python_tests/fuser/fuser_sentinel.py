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
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig
    from .pack_node import PackNode

from .arch_common import fpu_common, pack_common, unpack_common


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

    # Properties for reading the current format state
    @property
    def unpack_a_src_format(self) -> str:
        return self._unpack_A_src.cpp_underlying_value

    @property
    def unpack_a_dst_format(self) -> str:
        return self._unpack_A_dst.cpp_underlying_value

    @property
    def unpack_b_src_format(self) -> str:
        return self._unpack_B_src.cpp_underlying_value

    @property
    def unpack_b_dst_format(self) -> str:
        return self._unpack_B_dst.cpp_underlying_value

    @property
    def math_format(self) -> str:
        return self._math_format.cpp_underlying_value

    @property
    def pack_src_format(self) -> str:
        return self._pack_src.cpp_underlying_value

    @property
    def pack_dst_format(self) -> str:
        return self._pack_dst.cpp_underlying_value

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

        self._unpack_face_r_dim_a = compute_node.src_a.tile_shape.face_r_dim
        self._unpack_num_faces_a = compute_node.src_a.tile_shape.total_num_faces()

        if compute_node.src_b is not None:
            self._unpack_face_r_dim_b = compute_node.src_b.tile_shape.face_r_dim
            self._unpack_num_faces_b = compute_node.src_b.tile_shape.total_num_faces()
        else:
            self._unpack_face_r_dim_b = self._unpack_face_r_dim_a
            self._unpack_num_faces_b = self._unpack_num_faces_a

        self.ensure_unpack_buf_desc_ids(compute_node)

        return unpack_common.hw_configure_unpack(
            compute_node,
            config.dest_acc.cpp_enum_value,
            unpack_A_src,
            unpack_A_dst,
            unpack_B_src,
            unpack_B_dst,
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
            is_unary = unpack_common.is_unary_unpacker(compute_node)
            needs_buf_desc = compute_node.src_a.buf_desc_id is None
            if not is_unary:
                needs_buf_desc = needs_buf_desc or (
                    compute_node.src_b is not None
                    and compute_node.src_b.buf_desc_id is None
                )
            if not (srca_changed or srcb_changed or needs_buf_desc):
                return ""
            self.ensure_unpack_buf_desc_ids(compute_node)
        elif not (srca_changed or srcb_changed):
            return ""

        code = unpack_common.configure_unpack(
            compute_node,
            config.dest_acc.cpp_enum_value,
            self._unpack_A_src,
            new_A_src,
            new_A_dst,
            self._unpack_B_src,
            new_B_src,
            new_B_dst,
            srca_changed,
            srcb_changed,
            srca_tile_changed,
            srcb_tile_changed,
        )

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

        return fpu_common.hw_configure_math(config.dest_acc.cpp_enum_value, math_fmt)

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

        code = fpu_common.configure_math(
            config.dest_acc.cpp_enum_value, self._math_format, new_math
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

        self.ensure_pack_buf_desc_id(first)

        code = pack_common.hw_configure_pack(
            first.output,
            config.dest_acc.cpp_enum_value,
            pack_src,
            pack_dst,
            pack_mode=operation.bh_tilize.pack_mode_value,
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

        self.ensure_pack_buf_desc_id(pack_node)

        code = pack_common.configure_pack(
            pack_node.output,
            config.dest_acc.cpp_enum_value,
            pack_src,
            pack_dst,
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
