# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

from helpers.chip_architecture import ChipArchitecture

from .fpu_node import FpuNode
from .pack_node import PackNode
from .sfpu_node import SfpuNode

if TYPE_CHECKING:
    from .fuser_config import GlobalConfig
    from .l1_operation import L1Operation

UNPACK = "dest_dvalid_client::UNPACK"
FPU = "dest_dvalid_client::FPU"
SFPU = "dest_dvalid_client::SFPU"
PACK = "dest_dvalid_client::PACK"
CHAIN_ORDER = (UNPACK, FPU, SFPU, PACK)


def active(config: "GlobalConfig") -> bool:
    return (
        config.architecture == ChipArchitecture.QUASAR
        and config.quasar_use_dvalid
        and not config.skip_sync
    )


def unit_of(node) -> str:
    if isinstance(node, FpuNode):
        return UNPACK if node.unpack_to_dest.value else FPU
    if isinstance(node, SfpuNode):
        return SFPU
    if isinstance(node, PackNode):
        return PACK
    raise TypeError(f"{node} does not use dest")


def units(operation: "L1Operation") -> List[str]:
    for node in operation.pack_nodes:
        if isinstance(node, SfpuNode):
            raise NotImplementedError(
                "dest chain: SFPU on the pack thread is not supported"
            )
    used = {unit_of(node) for node in operation.math_nodes + operation.pack_nodes}
    return [unit for unit in CHAIN_ORDER if unit in used]


def kernel_init(config: "GlobalConfig") -> str:
    if not active(config):
        return ""
    return "".join(f"_llk_dest_chain_join_<{unit}>();\n" for unit in (FPU, PACK))


def init(config: "GlobalConfig", operation: "L1Operation") -> str:
    if not active(config):
        return ""
    members = units(operation)
    return "".join(
        f"_llk_dest_chain_{'join' if unit in members else 'leave'}_<{unit}>();\n"
        for unit in (UNPACK, SFPU)
    )


class Section:

    def __init__(self, config: "GlobalConfig", operation: "L1Operation"):
        self.on = active(config)
        self.args = (
            f"{operation.dest_sync.cpp_enum_value}, {config.dest_acc.cpp_enum_value}"
        )

    def begin(self) -> str:
        return "_llk_dest_chain_section_begin_();\n" if self.on else ""

    def take(self, node) -> str:
        if not self.on:
            return ""
        return f"_llk_dest_chain_take_<{unit_of(node)}, {self.args}>();\n"

    def takes(self, nodes) -> str:
        return "".join(self.take(node) for node in nodes)

    def end(self) -> str:
        return f"_llk_dest_chain_section_end_<{self.args}>();\n" if self.on else ""
