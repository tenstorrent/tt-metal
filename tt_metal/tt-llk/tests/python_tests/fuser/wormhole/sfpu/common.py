# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation


def sfpu_on_isolated_trisc(config: "GlobalConfig") -> bool:
    return False


def math_handoff_to_sfpu(config: "GlobalConfig", operation: "L1Operation") -> str:
    return ""


def sfpu_wait_for_math(config: "GlobalConfig", operation: "L1Operation") -> str:
    return ""


def sfpu_signal_math(config: "GlobalConfig", operation: "L1Operation") -> str:
    return ""


def sfpu_sync_init(config: "GlobalConfig", operation: "L1Operation") -> str:
    return ""


def sfpu_dest_section_done(config: "GlobalConfig", operation: "L1Operation") -> str:
    return ""
