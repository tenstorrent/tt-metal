// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "dev_mem_map.h"
#include "debug/assert.h"
#include "debug/dprint.h"
#include "noc_nonblocking_api.h"
#include "core_config.h"

static_assert(
    proc_type == static_cast<uint8_t>(TensixProcessorTypes::DM0) ||
        proc_type == static_cast<uint8_t>(TensixProcessorTypes::DM1),
    "PacketHeaderPool is only supported for TensixProcessorTypes::DM0 or TensixProcessorTypes::DM1");

// Simple packet header pool manager for fabric networking
// This class manages allocation of packet headers from a fixed pool
// located at MEM_PACKET_HEADER_POOL_BASE in L1 memory.
// Thread (multiple risc access) safe allocation
class PacketHeaderPool {
private:
    static uint32_t current_offset_;
    static constexpr uint32_t POOL_BASE = MEM_PACKET_HEADER_POOL_BASE;
    static constexpr uint32_t POOL_SIZE = MEM_PACKET_HEADER_POOL_SIZE;
    static constexpr uint32_t HEADER_SIZE = PACKET_HEADER_MAX_SIZE;
    static constexpr uint32_t POOL_SIZE_PER_RISC = POOL_SIZE / MaxDMProcessorsPerCoreType;
    static const uint32_t risc_pool_start = proc_type * POOL_SIZE_PER_RISC;
    static const uint32_t risc_pool_end = risc_pool_start + POOL_SIZE_PER_RISC;

public:
    FORCE_INLINE static volatile tt_l1_ptr PACKET_HEADER_TYPE* allocate_header() {
        ASSERT(current_offset_ + HEADER_SIZE <= risc_pool_end);
        if (current_offset_ + HEADER_SIZE > risc_pool_end) {
            DPRINT << "=== PACKET HEADER POOL EXHAUSTION ERROR ==="
                   << "CRITICAL: Insufficient space in packet header pool for RISC " << (uint32_t)proc_type
                   << "  - Headers Allocated: " << ((current_offset_ - risc_pool_start) / HEADER_SIZE)
                   << "  - Max Headers Capacity per RISC: " << (POOL_SIZE_PER_RISC / HEADER_SIZE)
                   << "Action: Entering infinite loop to prevent undefined behavior"
                   << "Solution: Increase MEM_PACKET_HEADER_POOL_SIZE or reduce header usage"
                   << "=================================================\n";
            while (1) {
            }  // hang intentionally
        }

        uint32_t allocated_addr = POOL_BASE + current_offset_;
        current_offset_ += HEADER_SIZE;
        return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(allocated_addr);
    }
};

uint32_t PacketHeaderPool::current_offset_ = PacketHeaderPool::risc_pool_start;
