# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Isolate-SFPU stubs for wormhole.

The TRISC3 isolate-SFPU thread (and the FPU_SFPU/SFPU_FPU MATH<->SFPU dest
handshake) is quasar-only. These stubs keep arch_common.py's module import
uniform across architectures; they are never invoked on non-quasar targets.
"""

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation
    from fuser.operand import Operand


def emit_operand_init(operands: List["Operand"]) -> str:
    return ""


def sfpu_math_sync_init(config: "GlobalConfig", operation: "L1Operation") -> str:
    return ""


def math_signal_sfpu(config: "GlobalConfig", operation: "L1Operation") -> str:
    return ""


def math_wait_for_sfpu(config: "GlobalConfig", operation: "L1Operation") -> str:
    return ""


def sfpu_sync_init(config: "GlobalConfig", operation: "L1Operation") -> str:
    return ""


def sfpu_wait_for_math(config: "GlobalConfig", operation: "L1Operation") -> str:
    return ""


def sfpu_signal_math(config: "GlobalConfig", operation: "L1Operation") -> str:
    return ""


def sfpu_dest_section_done(config: "GlobalConfig", operation: "L1Operation") -> str:
    return ""
