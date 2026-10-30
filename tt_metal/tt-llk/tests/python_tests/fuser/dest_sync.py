# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Calls into llk_dest_sync.h. Every thread emits every call, so every thread keeps the same dest sync state."""

from typing import TYPE_CHECKING

from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat

from .fpu_node import FpuNode
from .pack_node import PackNode
from .sfpu_node import SfpuNode

if TYPE_CHECKING:
    from .fuser_config import GlobalConfig
    from .l1_operation import L1Operation

UNPACK = "DestClient::UNPACK"
FPU = "DestClient::FPU"
SFPU = "DestClient::SFPU"
PACK = "DestClient::PACK"


def active(config: "GlobalConfig") -> bool:
    return (
        config.architecture == ChipArchitecture.QUASAR
        and config.quasar_use_dvalid
        and not config.skip_sync
    )


def client_of(node) -> str:
    if isinstance(node, FpuNode):
        return UNPACK if node.unpack_to_dest.value else FPU
    if isinstance(node, SfpuNode):
        return SFPU
    if isinstance(node, PackNode):
        return PACK
    raise TypeError(f"{node} does not use dest")


def _start(config: "GlobalConfig", operation: "L1Operation") -> str:
    return (
        f"_llk_dest_sync_init_<{operation.dest_sync.cpp_enum_value}, "
        f"{config.dest_acc.cpp_enum_value}>();\n"
    )


def init(config: "GlobalConfig", operation: "L1Operation") -> str:
    """Op init: start dest sync on the first operation or a new dest sync mode, then join or leave UNPACK and SFPU."""
    if not active(config):
        return ""
    for node in operation.pack_nodes:
        if isinstance(node, SfpuNode):
            raise NotImplementedError("dest sync: SFPU on the pack thread")
    previous = operation.previous_operation
    code = ""
    if previous is None:
        code += _start(config, operation)
    elif previous.dest_sync != operation.dest_sync:
        code += "_llk_dest_sync_finish_();\n" + _start(config, operation)
    used = {client_of(node) for node in operation.math_nodes}
    for client in (UNPACK, SFPU):
        call = "join" if client in used else "leave"
        code += f"_llk_dest_sync_{call}_<{client}>();\n"
    return code


def kernel_finish(config: "GlobalConfig") -> str:
    return "_llk_dest_sync_finish_();\n" if active(config) else ""


def unpack_writes_32bit(config: "GlobalConfig", operation: "L1Operation", node) -> bool:
    """The unpacker addresses dest by the width of the format it writes, not by the dest mode."""
    output_format = operation._get_pack_nodes()[0].output.data_format
    _, unpack_dst, _, _, _, _ = config.sentinel._infer_node_formats(
        config, node, output_format, operation
    )
    return unpack_dst.is_32_bit() or unpack_dst == DataFormat.Tf32


class Section:

    def __init__(self, config: "GlobalConfig", operation: "L1Operation"):
        self.on = active(config)
        self.config = config
        self.operation = operation

    def take(self, node) -> str:
        if not self.on:
            return ""
        client = client_of(node)
        if client == UNPACK:
            wide = unpack_writes_32bit(self.config, self.operation, node)
        else:
            wide = self.config.dest_acc.value
        return f"_llk_dest_sync_take_<{client}, {str(bool(wide)).lower()}>();\n"

    def takes(self, nodes) -> str:
        return "".join(self.take(node) for node in nodes)
