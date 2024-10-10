// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/impl/routing/flow_control/queue_iterator.hpp"

#include <cstddef>

template <class FlowControlReceiverImpl, size_t q_capacity = 0>
struct FlowControlReceiver : public RemoteQueuePtrManagerFriend {
    FlowControlReceiver(RemoteQueuePtrManager<q_capacity> const& q_ptrs) : RemoteQueuePtrManagerFriend(), q_ptrs(q_ptrs) {}

    /*
     * returns the number of credits available (i.e. the amount of space (indirectly) available to send to
     * the downstream queue)
     * Depending on the flow control mode (paged, packeted), this number represents different sizes
     * 1 credit = 1 page in paged mode, 1 word (currently 16B) in packet mode
     */
    size_t get_num_unacked_credits() const { return q_ptrs.get_num_unacked_credits(); }

    /*
     * return the number of credit slots ahead is the wrptr of the rdptr completions pointer
     */
    size_t get_num_incompleted_credits() const { return q_ptrs.get_num_credits_incomplete(); }

    /*
     * returns true if any received credits were not acknowledged
     */
    bool local_has_unacknowledged_credits() const { return get_num_unacked_credits() > 0; }

    /*
     * returns true if any received credits were not completed
     */
    bool local_has_incomplete_credits() const { return get_num_incompleted_credits() > 0; }

    /*
     * returns true if there is any uncleared data locally
     */
    bool local_has_data() const { return get_num_incompleted_credits() > 0; }

    /*
     * returns true if this receiver has n credits worth of space available
     */
    bool local_has_free_credits(size_t n) { return q_ptrs.get_local_space_available() >= n; }


    /*
     * Sends the rdptr ack credits to the remote sender
     */
    void send_ack_credits() { static_cast<FlowControlReceiverImpl *>(this)->send_ack_credits_impl(*this->get_rdptr_acks_raw(q_ptrs)); }

    /*
     * Sends the rdptr completion credits to the remote sender
     */
    void send_completion_credits() { static_cast<FlowControlReceiverImpl *>(this)->send_completion_credits_impl(*this->get_rdptr_completions_raw(q_ptrs)); }

    /*
     * Advances the local rdptr_ack by n credits - call to acknowledge (to sender) the receipt of n credits (from sender)
     * Does NOT send the credits
     */
    void advance_ack_credits(size_t n) { q_ptrs.advance_read_ack_credits(n); }

    /*
     * Advances the local rdptr_completions by n credits - call to notify (to sender) the clearing of n credits in our local buffer,
     * indicating that sender can safely overwrite those n credits worth of buffering at any time.
     * Does NOT send the credits
     */
    void advance_completion_credits(size_t n) { q_ptrs.advance_read_completion_credits(n); }

   private:
    RemoteQueuePtrManager<q_capacity> q_ptrs;
};
