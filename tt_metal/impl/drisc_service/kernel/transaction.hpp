// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "api/alignment.h"
#include "kernel_utils.hpp"

// One transaction = one arena extent + one in-flight read or write leg.
struct Transaction {
    abi::NocEndpoint src;
    abi::NocEndpoint dst;
    uint32_t l1_addr;
    uint32_t payload_bytes;  // what the read and write legs move, at most CHUNK_SIZE
    uint32_t alloc_bytes;    // what the arena charged, released on retire
    bool last_of_cmd;
    uint8_t client_slot;
    bool dma_read_pending;  // a DMA read leg that has not landed
    bool dma_write;         // GDDR DMA write leg, addressed as a bank offset; else a TRID-tagged NOC write
};

constexpr uint32_t MAX_INFLIGHT_TRANSACTIONS = 8;
static_assert(is_power_of_2(MAX_INFLIGHT_TRANSACTIONS));
// Each transaction has at most one DMA request in flight, so a DMA stream
// never holds more than this many and issue never waits for room.
static_assert(MAX_INFLIGHT_TRANSACTIONS <= experimental::kMaxOutstandingWrites);

// Cursors over the in-flight transactions, as three free-running counts:
// retired <= writes_issued <= accepted. [writes_issued, accepted) are reading,
// [retired, writes_issued) are writing; each phase advances oldest-first.
class TransactionRing {
public:
    bool full() const { return accepted_ - retired_ == MAX_INFLIGHT_TRANSACTIONS; }
    bool has_reading() const { return writes_issued_ != accepted_; }
    bool has_writing() const { return retired_ != writes_issued_; }

    // Claims the next slot and returns its index, the TRID for its NOC traffic.
    // Call only when !full().
    uint32_t claim_trid() { return accepted_++ % MAX_INFLIGHT_TRANSACTIONS; }

    uint32_t oldest_reading_trid() const { return writes_issued_ % MAX_INFLIGHT_TRANSACTIONS; }
    uint32_t oldest_writing_trid() const { return retired_ % MAX_INFLIGHT_TRANSACTIONS; }

    // Bytes of the oldest reading transaction's write already handed to the
    // NIU; nonzero while that write is partly issued.
    uint32_t write_issued_bytes() const { return issued_bytes_; }
    void add_write_issued_bytes(uint32_t bytes) { issued_bytes_ += bytes; }
    void mark_write_issued() {
        issued_bytes_ = 0;
        ++writes_issued_;
    }
    void retire() { ++retired_; }

private:
    uint32_t accepted_ = 0;
    uint32_t writes_issued_ = 0;
    uint32_t retired_ = 0;
    uint32_t issued_bytes_ = 0;
};
