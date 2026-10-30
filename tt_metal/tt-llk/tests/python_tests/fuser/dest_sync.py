# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from helpers.chip_architecture import ChipArchitecture

from .fpu_node import FpuNode
from .pack_node import PackNode
from .sfpu_node import SfpuNode

if TYPE_CHECKING:
    from .fuser_config import GlobalConfig
    from .l1_operation import L1Operation


def active(config: "GlobalConfig") -> bool:
    return config.architecture == ChipArchitecture.QUASAR


def client_of(node) -> str:
    if isinstance(node, SfpuNode):
        return "SFPU"
    if isinstance(node, PackNode):
        return "PACK"
    assert isinstance(node, FpuNode)
    return "UNPACK" if node.unpack_to_dest.value else "FPU"


def init(config: "GlobalConfig", operation: "L1Operation") -> str:
    if not active(config):
        return ""
    if operation.stage_id != 1:
        return ""
    return f"_llk_dest_sync_init_<{operation.dest_sync.cpp_enum_value}, {config.dest_acc.cpp_enum_value}>();\n"


def take(config: "GlobalConfig", operation: "L1Operation", node) -> str:
    if not active(config) or config.skip_sync:
        return ""
    return (
        f"_llk_dest_sync_take_<{operation.dest_sync.cpp_enum_value}, DestClient::{client_of(node)}, "
        f"{config.dest_acc.cpp_enum_value}>();\n"
    )


def takes(config: "GlobalConfig", operation: "L1Operation", nodes) -> str:
    return "".join(take(config, operation, node) for node in nodes)


def unpack_take(config: "GlobalConfig", operation: "L1Operation", node) -> str:
    return take(config, operation, node) if config.quasar_unpack_to_dest else ""


def unpack_takes(config: "GlobalConfig", operation: "L1Operation", nodes) -> str:
    return "".join(unpack_take(config, operation, node) for node in nodes)
