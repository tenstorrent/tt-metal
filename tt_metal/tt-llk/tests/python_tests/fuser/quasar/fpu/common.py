# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from fuser.quasar import dest_dvalid
from helpers.format_config import DataFormat

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation


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
        # Dest addressing is bank-relative under dvalid: pin the software offset and section base to 0,
        # a previous semaphore-mode kernel may have left them on the other half.
        return (
            "_reset_dest_register_offset_();\n"
            "_set_dest_section_base_<ckernel::TRISC_ID>(_get_dest_buffer_base_());\n"
            + dest_dvalid.enable(config, operation, dest_dvalid.DestClient.FPU)
            + dest_dvalid.math_release_sfpu(config, operation)
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
    if config.quasar_use_dvalid:
        return dest_dvalid.signal(config, operation, dest_dvalid.DestClient.FPU)
    dest_sync = operation.dest_sync.cpp_enum_value
    dest_acc = config.dest_acc.cpp_enum_value
    return f"_llk_math_dest_section_done_<{dest_sync}, {dest_acc}>();\n"


def math_dest_remap_config(required: bool) -> str:
    return ""
