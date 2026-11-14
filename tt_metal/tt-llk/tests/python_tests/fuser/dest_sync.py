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
    mode = f"ckernel::{operation.dest_sync.cpp_enum_value}"
    return f"_llk_dest_sync_configure_<{mode}, {config.dest_acc.cpp_enum_value}>();\n"


def take(config: "GlobalConfig", node) -> str:
    if not active(config) or config.skip_sync:
        return ""
    return f"_llk_dest_sync_take_<DestClient::{client_of(node)}>();\n"


def takes(config: "GlobalConfig", nodes) -> str:
    return "".join(take(config, node) for node in nodes)
