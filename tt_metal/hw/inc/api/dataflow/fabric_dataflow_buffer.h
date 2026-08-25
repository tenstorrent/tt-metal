// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)

#include <cstdint>
#include <limits>

#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/assert.h"
#include "api/kernel_thread_globals.h"
#include "dev_mem_map.h"

class Fabric;

// Sender-only adapter over an ordinary DataflowBuffer. The producer continues
// to use DataflowBuffer directly. Each issued page gets one bounded software
// transaction ID whose worker-L1 counter reaches zero after the local Fabric
// DE has stopped reading that page.
class FabricDataflowBuffer {
public:
    using TransactionId = uint32_t;

    // The counter bank is reserved L1, not an allocation: a fixed region the DE
    // decrements over NoC, indexed by this RISC's hart id so two sender DMs on
    // a core do not share one. Nothing to declare and nothing to bind.
    explicit FabricDataflowBuffer(DFBAccessor payload_accessor);
    FabricDataflowBuffer(const FabricDataflowBuffer&) = delete;
    FabricDataflowBuffer& operator=(const FabricDataflowBuffer&) = delete;
    FabricDataflowBuffer(FabricDataflowBuffer&&) = delete;
    FabricDataflowBuffer& operator=(FabricDataflowBuffer&&) = delete;

    // Teardown is RAII: no payload page may still be referenced by an SWQ when
    // the kernel exits, and forgetting an explicit call is a silent corruption.
    // finish() spins, so it runs at scope exit -- place the buffer's scope where
    // you want that drain to happen.
    ~FabricDataflowBuffer() { finish(); }

    uint16_t get_id() const { return dfb_.get_id(); }
    uint32_t get_entry_size() const { return dfb_.get_entry_size(); }
    uint32_t get_total_num_entries() const { return dfb_.get_total_num_entries(); }
    uint32_t get_max_transaction_ids() const { return max_transaction_ids_; }
    uint32_t get_outstanding_transaction_count() const { return outstanding_transaction_count_; }

    // Teardown-only drain, called by the destructor. Ordinary sends reclaim
    // completed FIFO-front pages opportunistically and whenever transaction-ID
    // capacity is exhausted. Idempotent, so an explicit early call is allowed.
    void finish();

private:
    friend class Fabric;

    bool finished_ = false;  // makes finish() idempotent for the destructor

    static FORCE_INLINE void compiler_fence();

    void wait_for_transaction_id();
    void wait_for_next_issue();
    uint32_t get_next_issue_read_ptr() const;

    TransactionId prepare_transaction(uint32_t source_read_completion_count);
    void commit_transaction();
    bool try_complete_front_transaction();
    void wait_complete_front_transaction();

    uint32_t load_counter(TransactionId transaction_id) const;

    DataflowBuffer dfb_;
    // Two views of one region: the uncached alias this core loads and stores
    // through, so it sees the DE's decrements without invalidating, and the
    // plain L1 address the DE's NoC write has to target.
    volatile tt_l1_ptr uint32_t* transaction_counters_;

    uint32_t max_transaction_ids_;
    TransactionId next_issue_transaction_id_ = 0;
    TransactionId next_pop_transaction_id_ = 0;
    uint32_t outstanding_transaction_count_ = 0;
};

#include "internal/tt-2xx/fabric_dataflow_buffer.inl"

#endif  // ARCH_QUASAR && !COMPILE_FOR_TRISC
