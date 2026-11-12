# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation


def hw_configure_math(dest_acc: str, alu_formats) -> str:
    src_a, src_b = alu_formats
    return (
        f"_llk_math_hw_configure_<{dest_acc}>(\n"
        f"    {src_a.cpp_underlying_value}, {src_b.cpp_underlying_value}\n"
        f");\n"
    )


def configure_math(
    dest_acc: str,
    old_formats,
    new_formats,
) -> str:
    to_from_int8 = str(
        any(fmt.needs_int8_math_config() for fmt in (*old_formats, *new_formats))
    ).lower()
    src_a, src_b = new_formats
    return (
        f"_llk_math_reconfig_data_format_<{dest_acc}, {to_from_int8}>(\n"
        f"    {src_a.cpp_underlying_value}, {src_b.cpp_underlying_value}\n"
        f");\n"
    )


def math_pack_sync_init(config: "GlobalConfig", operation: "L1Operation") -> str:
    if not config.quasar_use_dvalid and operation.stage_id != 1:
        return ""
    dest_sync = operation.dest_sync.cpp_enum_value
    dest_acc = config.dest_acc.cpp_enum_value
    return f"_llk_math_pack_sync_init_<{dest_sync}, {dest_acc}>();\n"


def math_wait_for_dest(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync:
        return ""
    dest_sync = operation.dest_sync.cpp_enum_value
    return f"_llk_math_wait_for_dest_available_<{dest_sync}>();\n"


def math_dest_section_done(config: "GlobalConfig", operation: "L1Operation") -> str:
    if config.skip_sync:
        return ""
    dest_sync = operation.dest_sync.cpp_enum_value
    dest_acc = config.dest_acc.cpp_enum_value
    return f"_llk_math_dest_section_done_<{dest_sync}, {dest_acc}>();\n"


def math_dest_remap_config(required: bool) -> str:
    return ""
