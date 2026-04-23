from dataclasses import dataclass

from helpers.data_format_inference import infer_data_formats
from helpers.format_config import FormatConfig

from .compute_node import ComputeNode
from .fused_operand import Operand
from .fused_operation import FusedOperation
from .fuser_config import GlobalConfig


@dataclass
class FuserSentinel:
    src_a: Operand
    src_b: Operand
    output: Operand
    format_config: FormatConfig

    def should_reconfigure(self, src_a, src_b, output) -> bool:
        return True

    def reconfigure_self(
        self, config: GlobalConfig, operation: FusedOperation, compute_node: ComputeNode
    ):
        self.src_a = compute_node.src_a
        self.src_b = compute_node.src_b
        self.output = operation.output

        self.format_config = infer_data_formats(
            input_format=self.src_a.data_format,
            input_format_B=self.src_b.data_format,
            output_format=self.output.data_format,
            unpacking_to_dest=compute_node.unpack_to_dest,
            chip_arch=config.architecture,
        )

    @property
    def unpack_a_src_format(self):
        if self.src_a:
            return f"ckernel::to_underlying(DataFormat::{self.format_config.unpack_A_src.name})"
        return ""

    @property
    def unpack_b_src_format(self):
        if self.src_b:
            return f"ckernel::to_underlying(DataFormat::{self.format_config.unpack_B_src.name})"
        return ""

    @property
    def unpack_a_dst_format(self):
        if self._unpack_a_out_format:
            return f"ckernel::to_underlying(DataFormat::{self.format_config.unpack_A_dst.name})"
        return ""

    @property
    def unpack_b_dst_format(self):
        if self._unpack_b_out_format:
            return f"ckernel::to_underlying(DataFormat::{self.format_config.unpack_B_dst.name})"
        return ""

    @property
    def math_format(self):
        if self._math_format:
            return f"ckernel::to_underlying(DataFormat::{self.format_config.math.name})"
        return ""

    def reconfigure_unpack_formats(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ):
        return (
            f"_llk_unpack_reconfig_data_format_srca_impl_<{config.dest_acc.cpp_enum_value}, false>(\n"
            f"    {self.unpack_a_src_format}, {self.unpack_b_dst_format}, {self.src_a.tile_size}\n"
            f");\n"
            f"_llk_unpack_reconfig_data_format_srcb_impl_<{config.dest_acc.cpp_enum_value}, false>(\n"
            f"    {self.unpack_b_src_format}, {self.unpack_b_dst_format}, {self.src_a.tile_size}\n"
            f");\n"
        )

    def reconfigure_math_format(
        self,
        operation: "FusedOperation",
        config: "GlobalConfig",
    ):
        return (
            f"_llk_math_reconfig_data_format_<{config.dest_acc.cpp_enum_value}, false>(\n"
            f"    {self.math_format}, {self.math_format}\n"
            f");\n"
        )
