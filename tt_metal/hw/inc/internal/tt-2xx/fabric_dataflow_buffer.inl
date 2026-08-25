// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This RISC's hart id, which selects its counter bank.
FORCE_INLINE uint32_t read_hartid() {
    uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    return static_cast<uint32_t>(hartid);
}

inline FabricDataflowBuffer::FabricDataflowBuffer(DFBAccessor payload_accessor) :
    dfb_(payload_accessor),
    transaction_counters_(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        MEM_FABRIC_TXN_COUNTERS_BASE(read_hartid()) + MEM_L1_UNCACHED_BASE)),
    max_transaction_ids_(MEM_FABRIC_MAX_TRANSACTION_IDS) {
    ASSERT(dfb_.get_num_tcs_to_rr() > 0);
    ASSERT(get_num_threads() == 1);

    // Do NOT zero the bank. It is reserved L1 that persists across ops, and
    // finish() already drained every counter to zero on the way out. A nonzero
    // value here means a prior op did not drain -- clobbering it would destroy
    // the evidence and let that op's late decrement underflow ours.
    for (uint32_t transaction_id = 0; transaction_id < max_transaction_ids_; ++transaction_id) {
        ASSERT(transaction_counters_[transaction_id] == 0);
    }
}

FORCE_INLINE void FabricDataflowBuffer::compiler_fence() {
    asm volatile("" ::: "memory");
}

inline uint32_t FabricDataflowBuffer::load_counter(
    TransactionId transaction_id) const {
    ASSERT(transaction_id < max_transaction_ids_);
    // volatile through the uncached alias: the DE's decrement is visible
    // without a fence.
    return transaction_counters_[transaction_id];
}

inline void FabricDataflowBuffer::wait_for_transaction_id() {
    if (outstanding_transaction_count_ < max_transaction_ids_) {
        return;
    }

    while (!try_complete_front_transaction()) {
        invalidate_l1_cache();
    }
}

inline void FabricDataflowBuffer::wait_for_next_issue() {
    const uint32_t num_tcs = dfb_.get_num_tcs_to_rr();
    ASSERT(num_tcs > 0);

    uint32_t required_occupancy =
        outstanding_transaction_count_ / num_tcs + 1;
    while (required_occupancy >
           dfb_.get_local_num_entries()) {
        wait_complete_front_transaction();
        required_occupancy =
            outstanding_transaction_count_ / num_tcs + 1;
    }
    ASSERT(required_occupancy <=
           static_cast<uint32_t>(
               std::numeric_limits<uint16_t>::max()));
    dfb_.wait_front(static_cast<uint16_t>(required_occupancy));
}

inline uint32_t FabricDataflowBuffer::get_next_issue_read_ptr() const {
    return dfb_.get_read_ptr();
}


inline FabricDataflowBuffer::TransactionId
FabricDataflowBuffer::prepare_transaction(
    uint32_t source_read_completion_count) {
    ASSERT(source_read_completion_count > 0);
    ASSERT(outstanding_transaction_count_ < max_transaction_ids_);

    const TransactionId transaction_id =
        next_issue_transaction_id_;
    ASSERT(load_counter(transaction_id) == 0);

    transaction_counters_[transaction_id] =
        source_read_completion_count;
    compiler_fence();

    next_issue_transaction_id_++;
    if (next_issue_transaction_id_ == max_transaction_ids_) {
        next_issue_transaction_id_ = 0;
    }
    return transaction_id;
}

inline void FabricDataflowBuffer::commit_transaction() {
    ASSERT(outstanding_transaction_count_ < max_transaction_ids_);
    dfb_.advance_read_ptr();
    outstanding_transaction_count_++;
}

inline bool FabricDataflowBuffer::try_complete_front_transaction() {
    if (outstanding_transaction_count_ == 0) {
        return false;
    }

    if (load_counter(next_pop_transaction_id_) != 0) {
        return false;
    }

    dfb_.acknowledge_front();
    next_pop_transaction_id_++;
    if (next_pop_transaction_id_ == max_transaction_ids_) {
        next_pop_transaction_id_ = 0;
    }
    outstanding_transaction_count_--;
    return true;
}

inline void FabricDataflowBuffer::wait_complete_front_transaction() {
    ASSERT(outstanding_transaction_count_ > 0);
    while (!try_complete_front_transaction()) {
        invalidate_l1_cache();
    }
}

inline void FabricDataflowBuffer::finish() {
    if (finished_) {
        return;   // the destructor also calls this; an explicit early call wins
    }
    finished_ = true;
    while (outstanding_transaction_count_ > 0) {
        wait_complete_front_transaction();
    }
    dfb_.finish();
}
