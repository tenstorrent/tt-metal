# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import TYPE_CHECKING, List

from fuser.fpu_node import FpuNode
from fuser.sfpu_node import SfpuNode

if TYPE_CHECKING:
    from fuser.fuser_config import GlobalConfig
    from fuser.l1_operation import L1Operation


class DestClient(Enum):
    UNPACK = "dest_dvalid_client::UNPACK"
    FPU = "dest_dvalid_client::FPU"
    SFPU = "dest_dvalid_client::SFPU"
    PACK = "dest_dvalid_client::PACK"

    @property
    def dvalid_bit(self) -> str:
        name = "UNPACK_TO_DEST" if self == DestClient.UNPACK else self.name
        return f"p_cleardvalid::{name}"


def _fpu_node_clients(node: FpuNode) -> List[DestClient]:
    from fuser.quasar.fpu.datacopy import DatacopyFpu

    if not node.unpack_to_dest.value:
        return [DestClient.FPU]
    if isinstance(node.fpu, DatacopyFpu):
        return [DestClient.UNPACK]
    return [DestClient.UNPACK, DestClient.FPU]


def dest_chain(operation: "L1Operation") -> List[DestClient]:
    chain: List[DestClient] = []
    for node in operation.math.math_nodes:
        if isinstance(node, FpuNode):
            clients = _fpu_node_clients(node)
        else:
            clients = [DestClient.SFPU]
        for client in clients:
            if chain and chain[-1] == client:
                continue
            if client in chain:
                raise ValueError(
                    f"Operation {operation.stage_id}: {client.name} touches dest twice "
                    "with another client in between; dvalid needs one section per client"
                )
            chain.append(client)

    if any(isinstance(node, SfpuNode) for node in operation.math.pack_nodes):
        raise ValueError(
            f"Operation {operation.stage_id}: SFPU nodes in the pack list are not "
            "supported with quasar_use_dvalid"
        )

    chain.append(DestClient.PACK)
    order = list(DestClient)
    if [order.index(client) for client in chain] != sorted(
        order.index(client) for client in chain
    ):
        raise ValueError(
            f"Operation {operation.stage_id}: dest chain {[c.name for c in chain]} must follow "
            "UNPACK, FPU, SFPU, PACK order; the hardware handshake hands dest to the next higher client"
        )
    return chain


def in_chain(operation: "L1Operation", client: DestClient) -> bool:
    return client in dest_chain(operation)


def chain_comment(operation: "L1Operation") -> str:
    names = " -> ".join(client.name for client in dest_chain(operation))
    return f"// Operation {operation.stage_id}: dest chain {names}\n"


def enable(config: "GlobalConfig", operation: "L1Operation", client: DestClient) -> str:
    chain = dest_chain(operation)
    if config.skip_sync or client not in chain:
        was_enabled = config.dvalid_enabled_clients.get(client, False)
        config.dvalid_enabled_clients[client] = False
        if not was_enabled:
            return ""
        return f"_llk_dest_dvalid_disable_<{client.value}>();\n"
    config.dvalid_enabled_clients[client] = True
    chain_mask = " | ".join(member.dvalid_bit for member in chain)
    return (
        chain_comment(operation)
        + f"_llk_dest_dvalid_enable_<{client.value}, {chain_mask}>();\n"
    )


def _sfpu_joins(config: "GlobalConfig", operation: "L1Operation") -> bool:
    return not config.skip_sync and in_chain(operation, DestClient.SFPU)


def math_release_sfpu(config: "GlobalConfig", operation: "L1Operation") -> str:
    """Math tells the isolated SFPU thread that this operation's chain has started."""
    if not _sfpu_joins(config, operation):
        return ""
    return "_llk_sync_post_<>(semaphore::FPU_SFPU);\n"


def sfpu_wait_for_release(config: "GlobalConfig", operation: "L1Operation") -> str:
    """The SFPU thread runs ahead of math, so it must not enable its handshake before math opens the chain."""
    if not _sfpu_joins(config, operation):
        return ""
    return (
        "_llk_sync_wait_<p_stall::STALL_THREAD, p_stall::STALL_ON_ZERO>(semaphore::FPU_SFPU);\n"
        "_llk_sync_get_<>(semaphore::FPU_SFPU);\n"
    )


def signal(config: "GlobalConfig", operation: "L1Operation", client: DestClient) -> str:
    chain = dest_chain(operation)
    if config.skip_sync or client not in chain:
        return ""
    dest_sync = operation.dest_sync.cpp_enum_value
    dest_acc = config.dest_acc.cpp_enum_value
    return f"_llk_dest_dvalid_signal_<{client.value}, {dest_sync}, {dest_acc}>();\n"
