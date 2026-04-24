from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from helpers.chip_architecture import ChipArchitecture
from helpers.data_format_inference import infer_data_formats
from helpers.format_config import DataFormat, FormatConfig

if TYPE_CHECKING:
    from .compute_node import ComputeNode
    from .fused_operation import FusedOperation
    from .fuser_config import GlobalConfig


@dataclass
class FuserSentinel:
    _unpack_format: Optional[FormatConfig] = field(default=None, repr=False)
    _math_format: Optional[FormatConfig] = field(default=None, repr=False)
    _pack_format: Optional[FormatConfig] = field(default=None, repr=False)

    def _find_format_node(
        self,
        operation: "FusedOperation",
    ) -> "ComputeNode":
        for node in operation.math.operations:
            if node.src_a is not None or node.src_b is not None:
                return node
        return None

    def _compute_format_config(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
        compute_node: "ComputeNode",
    ) -> FormatConfig:
        return infer_data_formats(
            input_format=compute_node.src_a.data_format,
            output_format=operation.output.data_format,
            is_fp32_dest_acc_en=config.dest_acc,
            unpacking_to_dest=compute_node.unpack_to_dest.value,
            chip_arch=config.architecture,
            input_format_B=(
                compute_node.src_b.data_format if compute_node.src_b else None
            ),
        )

    def _compute_format_config_from_output(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
    ) -> FormatConfig:
        return infer_data_formats(
            input_format=operation.output.data_format,
            output_format=operation.output.data_format,
            is_fp32_dest_acc_en=config.dest_acc,
            unpacking_to_dest=False,
            chip_arch=config.architecture,
        )

    @staticmethod
    def _fmt(data_format: DataFormat) -> str:
        return f"ckernel::to_underlying(DataFormat::{data_format.name})"

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

    def _unpack_changed(self, new: FormatConfig) -> bool:
        old = self._unpack_format
        return (
            old.unpack_A_src != new.unpack_A_src
            or old.unpack_A_dst != new.unpack_A_dst
            or old.unpack_B_src != new.unpack_B_src
            or old.unpack_B_dst != new.unpack_B_dst
        )

    def hw_configure_unpack(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
    ) -> str:
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
        if compute_node.src_a is None:
            return ""

        new_fmt = self._compute_format_config(config, operation, compute_node)

        if not self._unpack_changed(new_fmt):
            return ""

        dest_acc = config.dest_acc.cpp_enum_value
        old = self._unpack_format
        code = ""

        if (
            old.unpack_A_src != new_fmt.unpack_A_src
            or old.unpack_A_dst != new_fmt.unpack_A_dst
        ):
            code += (
                f"_llk_unpack_reconfig_data_format_srca_impl_<{dest_acc}, false>(\n"
                f"    {self._fmt(new_fmt.unpack_A_src)}, {self._fmt(new_fmt.unpack_A_dst)}, {compute_node.src_a.tile_size}\n"
                f");\n"
            )

        if compute_node.src_b is not None and (
            old.unpack_B_src != new_fmt.unpack_B_src
            or old.unpack_B_dst != new_fmt.unpack_B_dst
        ):
            code += (
                f"_llk_unpack_reconfig_data_format_srcb_impl_<{dest_acc}, false>(\n"
                f"    {self._fmt(new_fmt.unpack_B_src)}, {self._fmt(new_fmt.unpack_B_dst)}, {compute_node.src_b.tile_size}\n"
                f");\n"
            )

        self._unpack_format = new_fmt
        return code

    def hw_configure_math(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
    ) -> str:
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
        compute_node = self._find_format_node(operation)

        if compute_node is not None:
            fmt = self._compute_format_config(config, operation, compute_node)
        else:
            fmt = self._compute_format_config_from_output(config, operation)

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
        else:
            return (
                f"_llk_pack_hw_configure_<{dest_acc}, false>(\n"
                f"    {self._fmt(fmt.pack_src)}, {self._fmt(fmt.pack_dst)}, {pack_size}, {face_r_dim}, {num_faces}\n"
                f");\n"
            )

    def configure_pack(
        self,
        config: "GlobalConfig",
        operation: "FusedOperation",
        compute_node: "ComputeNode",
    ) -> str:
        if compute_node.src_a is None:
            return ""

        new_fmt = self._compute_format_config(config, operation, compute_node)

        if (
            self._pack_format.pack_src == new_fmt.pack_src
            and self._pack_format.pack_dst == new_fmt.pack_dst
        ):
            return ""

        dest_acc = config.dest_acc.cpp_enum_value
        pack_size = operation.output.tile_size

        code = (
            f"_llk_pack_reconfig_data_format_<{dest_acc}, false>(\n"
            f"    {self._fmt(new_fmt.pack_src)}, {self._fmt(new_fmt.pack_dst)}, {pack_size}\n"
            f");\n"
        )
        self._pack_format = new_fmt
        return code
