// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/routing/flow_control/queue_iterator.hpp"

#include <cstddef>

namespace tt_metal {
namespace routing {
namespace flow_control {

/*
 * Can this be shared between sender and receiver????
 */
template <class FlowControlSenderImpl, size_t q_capacity = 0>
struct FlowControlSender : public RemoteQueuePtrManagerFriend {
    FlowControlSender(RemoteQueuePtrManager<q_capacity> const& q_ptrs) : RemoteQueuePtrManagerFriend(), q_ptrs(q_ptrs) {}

    /*
     * returns the number of credits available (i.e. the amount of space (indirectly) available to send to
     * the downstream queue)
     * Depending on the flow control mode (paged, packeted), this number represents different sizes
     * 1 credit = 1 page in paged mode, 1 word (currently 16B) in packet mode
     */
    size_t get_num_free_local_credits() const {
        return q_ptrs.get_local_space_available();
    }

    /*
     * returns the number of credits available on the remote receiver. This will be the queue size minus the distance
     * ahead that wrptr is of the rdptr completions
     */
    size_t get_num_free_remote_credits() const {
        return q_ptrs.get_remote_space_available();
    }

    /*
     * returns the number of credits that we are waiting for in order to consider all past
     * transfers as complete to the first level ack
     */
    size_t get_num_outstanding_ack_credits() const {
        return q_ptrs.get_num_unacked_credits();
    }

    /*
     * Does the local queue have n credits worth of available buffering?
     */
    bool local_has_free_credits(size_t n) {
        return  q_ptrs.get_local_space_available() >= n;
    }
    /*
     * Does the remote queue have n credits worth of available buffering?
     */
    bool remote_has_free_credits(size_t n) {
        return q_ptrs.get_remote_space_available() >= n;
    }

    /*
     * For sender side, this indicates we have sent some payload to the consumer and we want to notify them
     * of the number of credits consumed by that payload
     * Locally this will be a decrement but remotely it will be an increment1
     * DOES NOT SEND CREDITS
     */
    void advance_write_credits(size_t n) {
        q_ptrs.advance_write_credits(n);
    }

    /*
     * implements the credit sending mechanics to the remote chip
     */
    void send_credits() {
        static_cast<FlowControlSenderImpl*>(this)->send_credits_impl(*this->get_wrptr_raw(q_ptrs));
    }

    /*
     * After receiving (ack) credits from remote receiver, increment our local (ack) credits
     * This impl is if there is no remote updater - perhaps we can spin this out as another variant
     * of the FCS to be local and remote updater
     */
    void advance_ack_credits(size_t n) {
        q_ptrs.advance_read_ack_credits(n);
    }

    /*
     * After receiving (completion) credits from remote receiver, increment our local (completion) credits
     */
    void advance_completion_credits(size_t n) {
        q_ptrs.advance_read_completion_credits(n);
    }

    private:
    RemoteQueuePtrManager<q_capacity> q_ptrs;
};


} // namespace flow_control
}  // namespace routing
} // namespace tt_metal
