// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>


// Forward declaration;
struct RemoteQueuePtrManagerFriend;

template <size_t const_queue_size = 0>
struct RemoteQueuePtrManager {
    friend struct RemoteQueuePtrManagerFriend;

    constexpr RemoteQueuePtrManager(
        size_t* wrptr_ptr, size_t* rdptr_ack_ptr, size_t* rdptr_completions_ptr, size_t queue_size) :
        size(queue_size), wrptr(wrptr_ptr), rdptr_acks(rdptr_ack_ptr), rdptr_completions(rdptr_completions_ptr) {
        *wrptr = 0;
        *rdptr_acks = 0;
        *rdptr_completions = 0;
    }

    constexpr RemoteQueuePtrManager(size_t* wrptr_ptr, size_t* rdptr_ack_ptr, size_t* rdptr_completions_ptr) :
        size(const_queue_size), wrptr(wrptr_ptr), rdptr_acks(rdptr_ack_ptr), rdptr_completions(rdptr_completions_ptr) {
        *wrptr = 0;
        *rdptr_acks = 0;
        *rdptr_completions = 0;
    }

    /*
     * Updates the local copy of the write pointer by advancing by n credits. Does NOT send any credits to the remote endpoint
     * Only makes sense to call as a sender endpoint
     */
    void advance_write_credits(size_t n) { ptr_advance_impl(this->wrptr, n, this->size); }

    /*
     * Updates the local copy of the read ack pointer by advancing by n credits. Does NOT send any credits to the remote endpoint
     * Only makes sense to call as a receiver endpoint
     */
    void advance_read_ack_credits(size_t n) { ptr_advance_impl(this->rdptr_acks, n, this->size); }

    /*
     * Updates the local copy of the read completions pointer by advancing by n credits. Does NOT send any credits to the remote endpoint
     * Only makes sense to call as a receiver endpoint
     */
    void advance_read_completion_credits(size_t n) { ptr_advance_impl(this->rdptr_completions, n, this->size); }

    /*
     * Returns the number of credits that the rdptr_completions is behind the wrptr
     */
    size_t get_num_credits_incomplete() const {
        auto wrptr_cached = get(this->wrptr);
        auto rdptr_completions_cached = get(this->rdptr_completions);
        if constexpr (is_power_of_2(const_queue_size)) {
            return (wrptr_cached - rdptr_completions_cached) & pow2_full_q_mask;
        } else {
            // default impl - we can probably do better than this but this isn't our use case today
            bool acks_ahead = wrptr_cached >= rdptr_completions_cached;
            return acks_ahead ? wrptr_cached - rdptr_completions_cached
                              : wrptr_cached + ((this->size * 2) - rdptr_completions_cached);
        }
    }

    /*
     * Returns the number of credits that the rdptr_ack is behind the wrptr
     */
    size_t get_num_unacked_credits() const {
        auto wrptr_cached = get(this->wrptr);
        auto rdptr_ack_cached = get(this->rdptr_acks);
        if constexpr (is_power_of_2(const_queue_size)) {
            return (wrptr_cached - rdptr_ack_cached) & pow2_full_q_mask;
        } else {
            // default impl - we can probably do better than this but this isn't our use case today
            bool wrptr_ahead = wrptr_cached >= rdptr_ack_cached;
            return wrptr_ahead ? (wrptr_cached - rdptr_ack_cached) : wrptr_cached + ((this->size * 2) - rdptr_ack_cached);
        }
    }

    /*
     * Only call as sender.
     *
     * Returns the number of buffer credits available on the sender side.
     * Queue size - distance ahead wrptr is of rdptr_acks
     */
    size_t get_local_space_available() const { return this->size - get_num_unacked_credits(); }

    /*
     * Only call as sender.
     *
     * Returns the number of buffer credits available on the receiver side. The sender can safely
     * write this many credits worth of data to the destination.
     */
    size_t get_remote_space_available() const { return this->size - get_num_credits_incomplete(); }

    /*
     * Get the local wrptr.
     */
    size_t get_wrptr() const { return get_ptr_impl(get(this->wrptr), this->size); }

    /*
     * Get the local rdptr_acks.
     */
    size_t get_rdptr_acks() const { return get_ptr_impl(get(this->rdptr_acks), this->size); }

    /*
     * Get the local rdptr_completions.
     */
    size_t get_rdptr_completions() const { return get_ptr_impl(get(this->rdptr_completions), this->size); }

    /*
     * Get the size of the queue, in credits.
     */
    size_t get_queue_size() const { return this->size; }

    // The size of the queue, in credits
    const size_t size;

   protected:
    // The local write pointer, in credits, into the queue. Wraps at 2 * q size
    // NOTE: this can not directly be used for the index into the actual buffer
    //       instead we must call get_wrptr()
    volatile size_t *wrptr;


    // The local read pointer (ack), in credits, into the queue. Wraps at 2 * q size
    // NOTE: this can not directly be used for the index into the actual buffer
    //       instead we must call get_rdptr_acks()
    volatile size_t *rdptr_acks;

    // The local read pointer (completions), in credits, into the queue. Wraps at 2 * q size
    // NOTE: this can not directly be used for the index into the actual buffer
    //       instead we must call get_rdptr_completions()
    volatile size_t *rdptr_completions;

   private:
    // Internal helper functions
    static size_t get(volatile size_t *ptr) { return *ptr; }

    static size_t get_ptr_impl(size_t ptr, size_t q_size) {
        if constexpr (is_power_of_2(const_queue_size)) {
            return ptr & pow2_qptr_mask;
        } else {
            // default impl
            auto result = ptr;
            bool past_last_ptr_offset = q_size <= result;
            result = result - ((q_size)*past_last_ptr_offset);
            return result;
        }
    }

    static void ptr_advance_impl(volatile size_t *ptr, size_t n, size_t q_size) {
        if constexpr (is_power_of_2(const_queue_size)) {
            *ptr = (*ptr + n) & pow2_full_q_mask;
        } else {
            // default impl
            *ptr += n;
            bool past_last_internal_ptr_val = (2 * q_size) <= *ptr;
            *ptr = *ptr - (2 * q_size * past_last_internal_ptr_val);
        }
    }

    static constexpr size_t pow2_full_q_mask = (const_queue_size << 1) - 1;
    static constexpr size_t pow2_qptr_mask = const_queue_size - 1;
    static constexpr bool is_power_of_2(size_t n) { return n && !(n & (n - 1)); }
};

struct RemoteQueuePtrManagerFriend {
    // Why did I do this? Because it's way too easy to accidentally grab the raw wrptr/rdptr_acks/rdptr_completions pointers
    // and use them directly in an offset calculation which would be incorrect. For this reason I wanted to make the user
    // have to work to get access to the internals (e.g. for sending the credit pointer to remote) so they don't accidentally
    // grab the wrong credit pointer. Most users will want
    template <size_t const_queue_size>
    volatile size_t *get_wrptr_raw(RemoteQueuePtrManager<const_queue_size> const& qptrs) const { return qptrs.wrptr; }
    template <size_t const_queue_size>
    volatile size_t *get_rdptr_acks_raw(RemoteQueuePtrManager<const_queue_size> const& qptrs) const { return qptrs.rdptr_acks; }
    template <size_t const_queue_size>
    volatile size_t *get_rdptr_completions_raw(RemoteQueuePtrManager<const_queue_size> const& qptrs) const { return qptrs.rdptr_completions; }
};
