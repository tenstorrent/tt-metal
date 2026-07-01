# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from helpers.format_config import DataFormat


def _dfmt(data_format: DataFormat) -> str:
    return f"DataFormat::{data_format.name}"


def hw_configure_math(dest_acc: str, math_fmt: DataFormat) -> str:
    return (
        f"_llk_math_srcAB_hw_configure_<true, {dest_acc}, false>(\n"
        f"    {_dfmt(math_fmt)}, {_dfmt(math_fmt)}\n"
        f");\n"
    )


def configure_math(
    dest_acc: str,
    old_math: DataFormat,
    new_math: DataFormat,
) -> str:
    return (
        f"_llk_math_srcAB_hw_configure_<false, {dest_acc}, false>(\n"
        f"    {_dfmt(new_math)}, {_dfmt(new_math)}\n"
        f");\n"
    )


def math_pack_sync_init(dest_sync: str, dest_acc: str) -> str:
    return "set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});\n"


def math_wait_for_dest(dest_sync: str) -> str:
    return ""


def math_dest_section_done(dest_sync: str, dest_acc: str) -> str:
    return f"_llk_math_set_dvalid_<p_cleardvalid::FPU, {dest_sync}>();\n"
