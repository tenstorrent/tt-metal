// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/routing/flow_control/flow_control_sender.hpp"
#include "tt_metal/impl/routing/flow_control/queue_iterator.hpp"

namespace tt_metal {
namespace routing {
namespace flow_control {

template <size_t q_capacity = 0>
struct PacketizedHostFlowControlSender
    : public FlowControlSender<PacketizedHostFlowControlSender<q_capacity>, q_capacity> {
    PacketizedHostFlowControlSender(size_t *remote_wrptr_address, RemoteQueuePtrManager<q_capacity> const &q_ptrs) :
        FlowControlSender<PacketizedHostFlowControlSender<q_capacity>, q_capacity>(q_ptrs),
        remote_wrptr_address(remote_wrptr_address) {}

    PacketizedHostFlowControlSender(
        size_t *remote_wrptr_address, size_t q_size, RemoteQueuePtrManager<q_capacity> const &q_ptrs) :
        FlowControlSender<PacketizedHostFlowControlSender<q_capacity>, q_capacity>(q_ptrs),
        remote_wrptr_address(remote_wrptr_address) {}

    // implements the credit sending mechanics
    void send_credits_impl(size_t wrptr) { reinterpret_cast<volatile size_t *>(remote_wrptr_address)[0] = wrptr; }

    size_t *remote_wrptr_address;
};

}  // namespace flow_control
}  // namespace routing
}  // namespace tt_metal
