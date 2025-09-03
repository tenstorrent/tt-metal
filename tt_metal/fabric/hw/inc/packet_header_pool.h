// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "fabric/fabric_edm_packet_header.hpp"
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
    static uint8_t route_id_;
    static const uint32_t HEADER_GROUP_SIZE_PER_RISC = NUM_PACKET_HEADERS / MaxDMProcessorsPerCoreType;

public:
    // {route_id: [header_ptr, num_headers]}
    static std::pair<volatile tt_l1_ptr PACKET_HEADER_TYPE*, uint8_t> header_table[HEADER_GROUP_SIZE_PER_RISC];

    FORCE_INLINE static volatile tt_l1_ptr PACKET_HEADER_TYPE* allocate_header(uint8_t num_headers = 1) {
        ASSERT(current_offset_ + HEADER_SIZE * num_headers <= risc_pool_end);
        if (current_offset_ + HEADER_SIZE * num_headers > risc_pool_end) {
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
        current_offset_ += HEADER_SIZE * num_headers;
        header_table[route_id_++] = {
            reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(allocated_addr), num_headers};
        return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(allocated_addr);
    }

    FORCE_INLINE static uint8_t allocate_header_n(uint8_t num_headers) {
        ASSERT(route_id_ < HEADER_GROUP_SIZE_PER_RISC);
        if (route_id_ >= HEADER_GROUP_SIZE_PER_RISC) {
            DPRINT << "=== ROUTE ID EXHAUSTION ERROR ==="
                   << "CRITICAL: Insufficient route IDs for RISC " << (uint32_t)proc_type
                   << "  - Route IDs Allocated: " << (uint32_t)route_id_
                   << "  - Max Route IDs Capacity per RISC: " << (uint32_t)HEADER_GROUP_SIZE_PER_RISC
                   << "Action: Entering infinite loop to prevent undefined behavior"
                   << "Solution: Increase HEADER_GROUP_SIZE_PER_RISC or reduce header usage"
                   << "=================================================\n";
            while (1) {
            }  // hang intentionally
        }
        allocate_header(num_headers);
        return route_id_ - 1;  // return the route_id of the allocated header
    }

    template <typename Func>
    FORCE_INLINE static void for_each_header(uint8_t route_id, Func&& func) {
        ASSERT(route_id < route_id_);
        if (route_id >= route_id_) {
            DPRINT << "=== ROUTE ID NOT FOUND ERROR ==="
                   << "CRITICAL: Route ID " << (uint32_t)route_id << " not found in header table for RISC "
                   << (uint32_t)proc_type << "  - Route IDs Allocated: " << (uint32_t)route_id_
                   << "  - Max Route IDs Capacity per RISC: " << (uint32_t)HEADER_GROUP_SIZE_PER_RISC
                   << "Action: Entering infinite loop to prevent undefined behavior"
                   << "Solution: Ensure route_id is valid before calling for_each_header"
                   << "=================================================\n";
            while (1) {
            }  // hang intentionally
        }
        auto [packet_headers, num_headers] = header_table[route_id];
        for (uint8_t i = 0; i < num_headers; i++) {
            func(packet_headers, i);
            packet_headers++;
        }
    }

    FORCE_INLINE static uint8_t get_num_headers(uint8_t route_id) {
        ASSERT(route_id < route_id_);
        if (route_id >= route_id_) {
            return 0;
        }
        return header_table[route_id].second;
    }
};

uint32_t PacketHeaderPool::current_offset_ = PacketHeaderPool::risc_pool_start;
uint8_t PacketHeaderPool::route_id_ = 0;
std::pair<volatile tt_l1_ptr PACKET_HEADER_TYPE*, uint8_t>
    PacketHeaderPool::header_table[PacketHeaderPool::HEADER_GROUP_SIZE_PER_RISC] = {};
