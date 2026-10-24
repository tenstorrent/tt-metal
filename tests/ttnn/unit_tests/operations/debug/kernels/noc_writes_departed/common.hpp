// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// Shared L1 layout. The scratch tensor is sharded so every core sees it at the same base address.
//   [0, RING * 16)              ring of 16 B slots; the sender writes one 4 B tag into slot i per iteration
//   RESULTS = base + RING * 16  receiver results (RESULT_WORDS words)
//   CELL    = RESULTS + 256     sender: the L1 word used as the NoC write source
//   ACK     = CELL + 16         sender: receivers increment this once per pass
//   DATA    = RESULTS + 1024    payload source (sender) / destination (receivers)
namespace noc_writes_departed {

constexpr uint32_t SLOT_BYTES = 16;
constexpr uint32_t RESULT_WORDS = 32;

// Flush modes selected by the sender's compile-time arg.
constexpr uint32_t FLUSH_WRITES_FLUSHED = 0;   // noc_async_writes_flushed()
constexpr uint32_t FLUSH_WRITES_DEPARTED = 1;  // noc_async_writes_departed()
constexpr uint32_t FLUSH_NONE = 2;             // no wait at all: detector control

// Receiver result words.
constexpr uint32_t RES_OK = 0;        // slot held the value written before the flush
constexpr uint32_t RES_STALE = 1;     // slot held the value the sender wrote AFTER the flush
constexpr uint32_t RES_PREVIOUS = 2;  // slot held the previous iteration's post-flush value
constexpr uint32_t RES_OTHER = 3;     // anything else
constexpr uint32_t RES_PASSES = 4;
constexpr uint32_t RES_SAMPLES = 8;  // up to 8 (pass << 16 | i, value) pairs
constexpr uint32_t NUM_SAMPLES = 8;

FORCE_INLINE uint32_t sent_tag(uint32_t pass, uint32_t i) { return 0xA0000000u | ((pass & 0x3FFu) << 16) | i; }
FORCE_INLINE uint32_t overwrite_tag(uint32_t pass, uint32_t i) { return 0xB0000000u | ((pass & 0x3FFu) << 16) | i; }

FORCE_INLINE volatile tt_l1_ptr uint32_t* l1_word(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}

struct Layout {
    uint32_t base, results, cell, ack, data;
    FORCE_INLINE Layout(uint32_t base_addr, uint32_t ring) :
        base(base_addr),
        results(base_addr + ring * SLOT_BYTES),
        cell(results + 256),
        ack(results + 272),
        data(results + 1024) {}
};

}  // namespace noc_writes_departed
