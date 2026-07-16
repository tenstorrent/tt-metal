# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat


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


def math_pack_sync_init(
    dest_sync: str,
    dest_acc: str,
    sfpu_on_dest: bool = False,
) -> str:
    chain = "DestChain::FPU_SFPU_PACK" if sfpu_on_dest else "DestChain::FPU_PACK"
    return f"setup_dest_dvalid<{chain}>();\n"


def math_wait_for_dest(dest_sync: str) -> str:
    return ""


def math_dest_section_done(dest_sync: str, dest_acc: str) -> str:
    return f"signal_fpu_done<{dest_sync}>();\n"
