# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from helpers.format_config import DataFormat
from helpers.llk_params import PerfRunType

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation


def _dest_format_config(dest_acc: str, math_fmt: DataFormat) -> tuple[str, str]:
    int32_dest = dest_acc == "true" and math_fmt.needs_int8_math_config()
    fp32_dest = dest_acc == "true" and not int32_dest
    return str(fp32_dest).lower(), str(int32_dest).lower()


def hw_configure_math(dest_acc: str, math_fmt: DataFormat) -> str:
    fp32_dest, int32_dest = _dest_format_config(dest_acc, math_fmt)
    implied_math_format = "false" if int32_dest == "true" else "true"
    return (
        f"_llk_math_srcAB_hw_configure_<{implied_math_format}, {fp32_dest}, {int32_dest}>(\n"
        f"    {math_fmt.cpp_enum_value}, {math_fmt.cpp_enum_value}\n"
        f");\n"
    )


def configure_math(
    dest_acc: str,
    old_math: DataFormat,
    new_math: DataFormat,
) -> str:
    fp32_dest, int32_dest = _dest_format_config(dest_acc, new_math)
    return (
        f"_llk_math_srcAB_hw_configure_<false, {fp32_dest}, {int32_dest}>(\n"
        f"    {new_math.cpp_enum_value}, {new_math.cpp_enum_value}\n"
        f");\n"
    )


def math_pack_sync_init(config: "GlobalConfig", operation: "L1Operation") -> str:
    dest_sync = operation.dest_sync.cpp_enum_value
    if config.quasar_use_dvalid:
        if config.perf_run_type in (None, PerfRunType.L1_TO_L1):
            return "set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::PACK});\n"
        return "set_up_zero_dest_dvalid_handshake_for_math();\n"
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
        return f"_llk_math_set_dvalid_<p_cleardvalid::FPU, {dest_sync}>();\n"
    return f"_llk_math_dest_section_done_<{dest_sync}, {dest_acc}>();\n"


def math_dest_remap_config(required: bool) -> str:
    return ""
