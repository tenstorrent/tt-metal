// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/routing/flow_control/flow_control_receiver.hpp"
#include "tt_metal/impl/routing/flow_control/queue_iterator.hpp"

namespace tt_metal {
namespace routing {
namespace flow_control {

template <size_t q_capacity = 0>
struct PacketizedHostFlowControlReceiver
    : public FlowControlReceiver<PacketizedHostFlowControlReceiver<q_capacity>, q_capacity> {
    PacketizedHostFlowControlReceiver(
        size_t *remote_rdptr_ack_address,
        size_t *remote_rdptr_complete_address,
        RemoteQueuePtrManager<q_capacity> const &q_ptrs) :
        FlowControlReceiver<PacketizedHostFlowControlReceiver<q_capacity>, q_capacity>(q_ptrs),
        remote_rdptr_ack_address(remote_rdptr_ack_address),
        remote_rdptr_complete_address(remote_rdptr_complete_address) {}

    PacketizedHostFlowControlReceiver(
        size_t *remote_rdptr_ack_address,
        size_t *remote_rdptr_complete_address,
        size_t q_size,
        RemoteQueuePtrManager<q_capacity> const &q_ptrs) :
        FlowControlReceiver<PacketizedHostFlowControlReceiver<q_capacity>, q_capacity>(q_ptrs),
        remote_rdptr_ack_address(remote_rdptr_ack_address),
        remote_rdptr_complete_address(remote_rdptr_complete_address) {}

    // implements the credit sending mechanics
    void send_ack_credits_impl(size_t rdptr_ack) {
        reinterpret_cast<volatile size_t *>(remote_rdptr_ack_address)[0] = rdptr_ack;
    }

    void send_completion_credits_impl(size_t rdptr_complete) {
        reinterpret_cast<volatile size_t *>(remote_rdptr_complete_address)[0] = rdptr_complete;
    }

    size_t *remote_rdptr_ack_address;
    size_t *remote_rdptr_complete_address;
};

}  // namespace flow_control
}  // namespace routing
}  // namespace tt_metal
