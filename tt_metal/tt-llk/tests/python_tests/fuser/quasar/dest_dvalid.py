# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, List

from fuser.sfpu_node import SfpuNode

if TYPE_CHECKING:
    from fuser.compute_pipeline import ComputePipeline
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation

UNPACK = "UNPACK"
FPU = "FPU"
SFPU = "SFPU"
PACK = "PACK"

UNPACK_THREAD = "unpack"
MATH_THREAD = "math"
PACK_THREAD = "pack"
SFPU_THREAD = "isolate_sfpu"


def _sfpu_thread(pipeline: "ComputePipeline") -> str:
    if any(isinstance(node, SfpuNode) for node in pipeline.pack_nodes):
        return PACK_THREAD
    return MATH_THREAD


def chain(pipeline: "ComputePipeline") -> List[str]:
    clients = []

    for node in pipeline.math_nodes:
        if isinstance(node, SfpuNode):
            clients.append(SFPU)
        elif node.unpack_to_dest.value:
            clients.append(UNPACK)
        else:
            clients.append(FPU)

    for node in pipeline.pack_nodes:
        clients.append(SFPU if isinstance(node, SfpuNode) else PACK)

    ordered = []
    for client in clients:
        if client not in ordered:
            ordered.append(client)

    return ordered if len(ordered) > 1 else []


def clients_of(pipeline: "ComputePipeline", thread: str) -> List[str]:
    sfpu = [SFPU] if _sfpu_thread(pipeline) == thread else []

    if thread == UNPACK_THREAD:
        return [UNPACK]
    if thread == MATH_THREAD:
        return [FPU] + sfpu
    if thread == PACK_THREAD:
        return [PACK] + sfpu
    return sfpu


def enable(config: "GlobalConfig", operation: "L1Operation", thread: str) -> str:
    if not config.quasar_use_dvalid:
        return ""

    order = chain(operation.math)
    clients = clients_of(operation.math, thread)

    code = "".join(
        f"_llk_dest_dvalid_disable_<dest_dvalid_client::{client}>();\n"
        for client in clients
        if client not in order
    )

    for client in clients:
        if client not in order:
            continue

        index = order.index(client)
        code += (
            f"_llk_dest_dvalid_enable_<dest_dvalid_client::{client}, "
            f"{'true' if index == 0 else 'false'}>();\n"
        )

    return code


def disable(config: "GlobalConfig", operation: "L1Operation", thread: str) -> str:
    if not config.quasar_use_dvalid or not operation.is_last_stage:
        return ""

    order = chain(operation.math)
    return "".join(
        f"_llk_dest_dvalid_disable_<dest_dvalid_client::{client}>();\n"
        for client in clients_of(operation.math, thread)
        if client in order
    )


def signal(
    config: "GlobalConfig", operation: "L1Operation", thread: str, client: str
) -> str:
    if config.skip_sync or not config.quasar_use_dvalid:
        return ""
    if client not in clients_of(operation.math, thread):
        return ""

    order = chain(operation.math)
    if client not in order:
        return ""

    next_client = order[(order.index(client) + 1) % len(order)]

    params = (
        f"dest_dvalid_client::{client}, dest_dvalid_client::{next_client}, "
        f"{operation.dest_sync.cpp_enum_value}"
    )
    if client == PACK:
        params += f", {config.dest_acc.cpp_enum_value}"

    return f"_llk_dest_dvalid_signal_<{params}>();\n"
