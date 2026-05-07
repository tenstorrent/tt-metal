# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from helpers.chip_architecture import ChipArchitecture
from helpers.data_format_inference import infer_data_formats
from helpers.format_config import DataFormat, FormatConfig
from helpers.llk_params import EltwiseBinaryReuseDestType

if TYPE_CHECKING:
    from .compute_node import ComputeNode
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig


@dataclass
class FuserSentinel:
    """Tracks format state for unpack, math, and pack threads across a fused pipeline.

    Each _format field is None until the first hw_configure call, then holds the
    currently configured FormatConfig. Subsequent calls compare against the stored
    state and emit reconfig only when formats actually change.
    """

    _unpack_format: Optional[FormatConfig] = field(default=None, repr=False)
    _math_format: Optional[FormatConfig] = field(default=None, repr=False)
    _pack_format: Optional[FormatConfig] = field(default=None, repr=False)

    @staticmethod
    def _find_format_node(
        operation: "FusedOperation",
    ) -> Optional["ComputeNode"]:
        """Find the first compute node with operand inputs for format inference.

        In a pipeline with mixed FPU and SFPU nodes, only FPU nodes have src_a/src_b
        operands. SFPU nodes operate on data already in the dest register and don't
        drive format configuration.

        Returns:
            The first ComputeNode with src_a or src_b, or None if all nodes are SFPU only.
        """
        for node in operation.math.operations:
            if node.src_a is not None or node.src_b is not None:
                return node
        return None

    @staticmethod
    def _compute_format_config(
        config: "GlobalConfig",
        operation: "FusedOperation",
        compute_node: "ComputeNode",
    ) -> FormatConfig:
        """Infer all pipeline formats from a compute node's operands and the operation output.

        Args:
            config: Global pipeline configuration (dest_acc, architecture)
            operation: Current fused operation (provides output format)
            compute_node: Node whose src_a/src_b drive format inference

        Returns:
            FormatConfig with inferred unpack, math, and pack formats
        """

        src_a_format = compute_node.src_a.data_format
        src_b_format = compute_node.src_b.data_format if compute_node.src_b else None

        if compute_node.reuse_dest == EltwiseBinaryReuseDestType.DEST_TO_SRCA:
            # DEST_TO_SRCA routes src_a L1 data to srcB, so srcB format
            # must match src_a's format, not src_b's.
            src_b_format = src_a_format

        return infer_data_formats(
            input_format=src_a_format,
            input_format_B=src_b_format,
            output_format=operation.output.data_format,
            is_fp32_dest_acc_en=config.dest_acc,
            unpacking_to_dest=compute_node.unpack_to_dest.value,
            chip_arch=config.architecture,
        )

    @staticmethod
    def _compute_format_config_from_output(
        config: "GlobalConfig",
        operation: "FusedOperation",
    ) -> FormatConfig:
        """Infer formats for SFPU only operations that have no input operands.

        When all compute nodes are SFPU (no src_a/src_b), there are no L1 inputs
        to drive format inference. We use the output format as both input and output,
        with unpacking_to_dest=True since SFPU operates directly on the dest register.

        Args:
            config: Global pipeline configuration
            operation: Current fused operation (provides output format)

        Returns:
            FormatConfig compatible with the operation's output format
        """
        return infer_data_formats(
            input_format=operation.output.data_format,
            output_format=operation.output.data_format,
            is_fp32_dest_acc_en=config.dest_acc,
            unpacking_to_dest=True,
            chip_arch=config.architecture,
        )

    @staticmethod
    def _fmt(data_format: DataFormat) -> str:
        return f"ckernel::to_underlying(DataFormat::{data_format.name})"

    # Properties for reading the current format state
    @property
    def unpack_a_src_format(self) -> str:
        return self._fmt(self._unpack_format.unpack_A_src)

    @property
    def unpack_a_dst_format(self) -> str:
        return self._fmt(self._unpack_format.unpack_A_dst)

    @property
    def unpack_b_src_format(self) -> str:
        return self._fmt(self._unpack_format.unpack_B_src)

    @property
    def unpack_b_dst_format(self) -> str:
        return self._fmt(self._unpack_format.unpack_B_dst)

    @property
    def math_format(self) -> str:
        return self._fmt(self._math_format.math)

    @property
    def pack_src_format(self) -> str:
        return self._fmt(self._pack_format.pack_src)

    @property
    def pack_dst_format(self) -> str:
        return self._fmt(self._pack_format.pack_dst)

    def hw_configure_unpack(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
    ) -> str:
        """Emit _llk_unpack_hw_configure_ once for the first operation in the pipeline.

        Called at the top of each operation's unpack_body(). On the first call
        (when _unpack_format is None), emits the full hw_configure with tile shape
        parameters from the first node that has operand inputs.

        Returns "" if no compute node in the operation has an unpacker.

        Args:
            config: Global pipeline configuration
            operation: Current fused operation

        Returns:
            C++ hw_configure call string, or "" if already configured or no inputs exist
        """
        compute_node = self._find_format_node(operation)
        if compute_node is None:
            return ""

        if self._unpack_format is not None:
            return ""

        fmt = self._compute_format_config(config, operation, compute_node)

        self._unpack_format = fmt

        dest_acc = config.dest_acc.cpp_enum_value

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

        return (
            f"_llk_unpack_hw_configure_<{dest_acc}, false>(\n"
            f"    {self._fmt(fmt.unpack_A_src)}, {self._fmt(fmt.unpack_B_src)},\n"
            f"    {self._fmt(fmt.unpack_A_dst)}, {self._fmt(fmt.unpack_B_dst)},\n"
            f"    {face_r_dim_a}, {face_r_dim_b}, {num_faces_a}, {num_faces_b},\n"
            f"    {tile_size_a}, {tile_size_b}\n"
            f");\n"
        )

    def configure_unpack(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
        compute_node: "ComputeNode",
    ) -> str:
        """Emit unpack reconfig calls when formats change between compute nodes.

        Called per node from ComputeNode.unpack() inside the tile loop. Compares
        the node's inferred formats against the currently configured state and emits
        _llk_unpack_reconfig_data_format_src{a,b}_impl_ only for channels that changed.

        Args:
            config: Global pipeline configuration
            operation: Current fused operation
            compute_node: The compute node requesting format configuration

        Returns:
            C++ reconfig call(s), or "" if formats match current state
        """
        new_fmt = self._compute_format_config(config, operation, compute_node)
        old = self._unpack_format

        srca_changed = (
            old.unpack_A_src != new_fmt.unpack_A_src
            or old.unpack_A_dst != new_fmt.unpack_A_dst
        )
        srcb_changed = (
            old.unpack_B_src != new_fmt.unpack_B_src
            or old.unpack_B_dst != new_fmt.unpack_B_dst
        )

        if not (srca_changed or srcb_changed):
            return ""

        dest_acc = config.dest_acc.cpp_enum_value
        code = ""

        if srca_changed:
            code += (
                f"_llk_unpack_reconfig_data_format_srca_impl_<{dest_acc}, p_dim_stride_target::IGNORE, false>(\n"
                f"    {self._fmt(new_fmt.unpack_A_src)}, {self._fmt(new_fmt.unpack_A_dst)}, {compute_node.src_a.tile_size}\n"
                f");\n"
            )

        if srcb_changed:
            srcb_tile_size = (
                compute_node.src_a.tile_size
                if compute_node.reuse_dest is EltwiseBinaryReuseDestType.DEST_TO_SRCA
                else (compute_node.src_b.tile_size if compute_node.src_b else None)
            )
            if srcb_tile_size is not None:
                code += (
                    f"_llk_unpack_reconfig_data_format_srcb_impl_<{dest_acc}, p_dim_stride_target::IGNORE, false>(\n"
                    f"    {self._fmt(new_fmt.unpack_B_src)}, {self._fmt(new_fmt.unpack_B_dst)}, {srcb_tile_size}\n"
                    f");\n"
                )

        self._unpack_format = new_fmt
        return code

    def hw_configure_math(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
    ) -> str:
        """Emit _llk_math_hw_configure_ once for the first operation in the pipeline.

        For SFPU only operations (no node with src_a), infers a format compatible
        with the output so the math hardware is always configured.

        Returns "" for subsequent operations since configure_math() handles reconfig.

        Args:
            config: Global pipeline configuration
            operation: Current fused operation

        Returns:
            C++ hw_configure call string, or "" if already configured
        """
        compute_node = self._find_format_node(operation)

        if self._math_format is not None:
            return ""

        if compute_node is not None:
            fmt = self._compute_format_config(config, operation, compute_node)
        else:
            fmt = self._compute_format_config_from_output(config, operation)

        self._math_format = fmt

        dest_acc = config.dest_acc.cpp_enum_value
        return (
            f"_llk_math_hw_configure_<{dest_acc}>(\n"
            f"    {self._fmt(fmt.math)}, {self._fmt(fmt.math)}\n"
            f");\n"
        )

    def configure_math(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
        compute_node: "ComputeNode",
    ) -> str:
        """Emit math reconfig when the math format changes between compute nodes.

        Called per node from ComputeNode.fpu_calculate() inside the tile loop.

        Args:
            config: Global pipeline configuration
            operation: Current fused operation
            compute_node: The compute node requesting format configuration

        Returns:
            C++ reconfig call, or "" if math format matches current state
        """
        if compute_node.src_a is None:
            return ""

        new_fmt = self._compute_format_config(config, operation, compute_node)

        if self._math_format.math == new_fmt.math:
            return ""

        dest_acc = config.dest_acc.cpp_enum_value
        code = (
            f"_llk_math_reconfig_data_format_<{dest_acc}, false>(\n"
            f"    {self._fmt(new_fmt.math)}, {self._fmt(new_fmt.math)}\n"
            f");\n"
        )
        self._math_format = new_fmt
        return code

    def hw_configure_pack(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
    ) -> str:
        """Emit pack hw_configure (first operation) or reconfig (subsequent operations).

        Unlike unpack and math, the pack thread has no per node configure_pack() calls
        in the tile loop. This method is the only place pack format changes are emitted.
        Therefore it handles both initial hw_configure AND reconfig for later operations.

        Architecture differences:
        - Blackhole: _llk_pack_hw_configure_ takes extra bh_tilize and TILE_C_DIM params
        - Wormhole: simpler signature without tilize parameters
        - Reconfig call is the same for both architectures

        Args:
            config: Global pipeline configuration
            operation: Current fused operation

        Returns:
            C++ hw_configure or reconfig call, or "" if formats unchanged
        """
        compute_node = self._find_format_node(operation)

        if compute_node is not None:
            fmt = self._compute_format_config(config, operation, compute_node)
        else:
            fmt = self._compute_format_config_from_output(config, operation)

        # Reconfig path: emit _llk_pack_reconfig_data_format_ only if pack formats changed
        if self._pack_format is not None:
            if (
                self._pack_format.pack_src == fmt.pack_src
                and self._pack_format.pack_dst == fmt.pack_dst
            ):
                return ""

            dest_acc = config.dest_acc.cpp_enum_value
            pack_size = operation.output.tile_size

            code = (
                f"_llk_pack_reconfig_data_format_<{dest_acc}, false>(\n"
                f"    {self._fmt(fmt.pack_src)}, {self._fmt(fmt.pack_dst)}, {pack_size}\n"
                f");\n"
            )
            self._pack_format = fmt
            return code

        # First time path: emit full _llk_pack_hw_configure_
        self._pack_format = fmt

        dest_acc = config.dest_acc.cpp_enum_value
        bh_tilize = operation.bh_tilize.cpp_enum_value
        pack_size = operation.output.tile_size
        face_r_dim = operation.output.tile_shape.face_r_dim
        num_faces = operation.output.tile_shape.total_num_faces()

        if config.architecture == ChipArchitecture.BLACKHOLE:
            return (
                f"_llk_pack_hw_configure_<{dest_acc}, false, {bh_tilize}>(\n"
                f"    {self._fmt(fmt.pack_src)}, {self._fmt(fmt.pack_dst)}, {pack_size}, {face_r_dim}, TILE_C_DIM, {num_faces}\n"
                f");\n"
            )

        return (
            f"_llk_pack_hw_configure_<{dest_acc}, false>(\n"
            f"    {self._fmt(fmt.pack_src)}, {self._fmt(fmt.pack_dst)}, {pack_size}, {face_r_dim}, {num_faces}\n"
            f");\n"
        )
