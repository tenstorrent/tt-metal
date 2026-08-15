# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from fuser.fpu_node import FpuNode
from fuser.sfpu_node import SfpuNode
from helpers.format_config import DataFormat

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation


def unpack_writes_dest(operation: "L1Operation") -> bool:
    return any(
        node.unpack_to_dest.value
        for node in operation.math.math_nodes
        if isinstance(node, FpuNode)
    )


def math_writes_dest(operation: "L1Operation") -> bool:
    for node in operation.math.math_nodes:
        if isinstance(node, SfpuNode):
            return True
        if isinstance(node, FpuNode) and not node.unpack_to_dest.value:
            return True
    return False


def hw_configure_math(dest_acc: str, math_fmt: DataFormat) -> str:
    return (
        f"_llk_math_srcAB_hw_configure_<true, {dest_acc}, false>(\n"
        f"    {math_fmt.cpp_enum_value}, {math_fmt.cpp_enum_value}\n"
        f");\n"
    )


def configure_math(
    dest_acc: str,
    old_math: DataFormat,
    new_math: DataFormat,
) -> str:
    return (
        f"_llk_math_srcAB_hw_configure_<false, {dest_acc}, false>(\n"
        f"    {new_math.cpp_enum_value}, {new_math.cpp_enum_value}\n"
        f");\n"
    )


def math_pack_sync_init(config: "GlobalConfig", operation: "L1Operation") -> str:
    dest_sync = operation.dest_sync.cpp_enum_value
    if config.quasar_use_dvalid:
        code = ""
        if operation.stage_id == 1:
            code += (
                "_reset_dest_register_offset_();\n"
                "_set_dest_section_base_<ckernel::TRISC_ID>(_get_dest_buffer_base_());\n"
            )
        if not math_writes_dest(operation):
            return code + "_llk_dest_dvalid_disable_<dest_dvalid::client::FPU>();\n"
        first = "" if unpack_writes_dest(operation) else ", true"
        return code + (
            "_llk_dest_dvalid_configure_<dest_dvalid::client::FPU, "
            f"dest_dvalid::client::PACK{first}>();\n"
        )
    if operation.stage_id != 1:
        return ""
    return f"_llk_math_pack_sync_init_<{dest_sync}>();\n"


def math_wait_for_dest(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync or config.quasar_use_dvalid:
        return ""
    return "_llk_math_wait_for_dest_available_();\n"


def math_dest_section_done(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync:
        return ""
    dest_sync = operation.dest_sync.cpp_enum_value
    dest_acc = config.dest_acc.cpp_enum_value
    if config.quasar_use_dvalid:
        if not math_writes_dest(operation):
            return ""
        return (
            "_llk_dest_dvalid_signal_<dest_dvalid::client::FPU, "
            f"{dest_sync}, {dest_acc}>();\n"
        )
    return f"_llk_math_dest_section_done_<{dest_sync}, {dest_acc}>();\n"
