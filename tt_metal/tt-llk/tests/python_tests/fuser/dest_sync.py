# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from helpers.chip_architecture import ChipArchitecture
from helpers.llk_params import DestSync

from .fpu_node import FpuNode
from .pack_node import PackNode
from .sfpu_node import SfpuNode

if TYPE_CHECKING:
    from .fuser_config import FuserConfig, GlobalConfig
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


def definitions(fuser: "FuserConfig") -> str:
    config = fuser.global_config
    if not active(config):
        return ""
    count = 2 if fuser.pipeline[0].dest_sync == DestSync.Half else 1
    repetitions = config.loop_factor if config.profiler_enabled else 1
    sections, runs = 0, []
    for operation, blocks in zip(fuser.pipeline, fuser.pipeline_plans):
        sections += repetitions * sum(
            len(block.bank.bank_assignments()) for block in blocks
        )
        writer = client_of(operation.math_nodes[0])
        if runs and runs[-1][0] == writer:
            runs[-1] = (writer, sections)
        else:
            runs.append((writer, sections))
    code = (
        f"constexpr unsigned dest_tensor_count = {count};\n"
        "constexpr unsigned dest_tensor_rows = "
        f"(ckernel::trisc::DEST_REGISTER_FULL_SIZE / {count}) >> {config.dest_acc.cpp_enum_value};\n"
        "constexpr ckernel::DestTensor dest_tensor0{0, dest_tensor_rows};\n"
        "constexpr ckernel::DestTensor dest_tensor1{dest_tensor_rows * (dest_tensor_count - 1), dest_tensor_rows};\n"
        "unsigned dest_section = 0;\n"
        "template <const ckernel::DestTensor& TENSOR>\n"
        "inline void dest_finish_tensor() {\n"
    )
    # Section k returns its tensor to the first writer of section k + count.
    for writer, end in runs:
        code += (
            f"if (dest_section + dest_tensor_count < {end}) {{\n"
            f"ckernel::_llk_dest_sync_release_<ckernel::DestClient::{writer}, TENSOR>();\n"
            "return;\n}\n"
        )
    return code + "ckernel::_llk_dest_sync_clear_<TENSOR>();\n}\n"


def init(config: "GlobalConfig", operation: "L1Operation") -> str:
    if not active(config):
        return ""
    if operation.stage_id != 1:
        return ""
    mode = f"ckernel::{operation.dest_sync.cpp_enum_value}"
    return (
        f"_llk_dest_sync_configure_<{mode}, {config.dest_acc.cpp_enum_value}>();\n"
        "dest_section = 0;\n"
    )


def take(config: "GlobalConfig", node) -> str:
    if not active(config) or config.skip_sync:
        return ""
    client = client_of(node)
    return (
        "if (dest_section % dest_tensor_count == 0) {\n"
        f"_llk_dest_sync_take_<DestClient::{client}, dest_tensor0>();\n"
        "} else {\n"
        f"_llk_dest_sync_take_<DestClient::{client}, dest_tensor1>();\n"
        "}\n"
    )


def takes(config: "GlobalConfig", nodes) -> str:
    return "".join(take(config, node) for node in nodes)


def finish(config: "GlobalConfig") -> str:
    if not active(config) or config.skip_sync:
        return ""
    return (
        "if (dest_section % dest_tensor_count == 0) {\n"
        "dest_finish_tensor<dest_tensor0>();\n"
        "} else {\n"
        "dest_finish_tensor<dest_tensor1>();\n"
        "}\n"
        "++dest_section;\n"
    )
