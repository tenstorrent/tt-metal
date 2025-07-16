// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "dev_mem_map.h"
#include "debug/assert.h"

// Simple packet header pool manager for fabric networking
// This class manages allocation and deallocation of packet headers from a fixed pool
// located at MEM_PACKET_HEADER_POOL_BASE in L1 memory.
class PacketHeaderPool {
private:
    static uint32_t current_offset_;
    static constexpr uint32_t POOL_BASE = MEM_PACKET_HEADER_POOL_BASE;
    static constexpr uint32_t POOL_SIZE = MEM_PACKET_HEADER_POOL_SIZE;
    static constexpr uint32_t HEADER_SIZE = PACKET_HEADER_MAX_SIZE;

    FORCE_INLINE static void add_and_wrap(const uint32_t size) {
        // Reset offset to start of pool if we exceed size
        current_offset_ += size;
        if (current_offset_ + HEADER_SIZE > POOL_SIZE) {
            current_offset_ = 0;
        }
    }

public:
    FORCE_INLINE static void init() { current_offset_ = 0; }

    FORCE_INLINE static volatile tt_l1_ptr PACKET_HEADER_TYPE* allocate_header() {
        ASSERT(current_offset_ + HEADER_SIZE <= POOL_SIZE, "Insufficient space in packet header pool");

        uint32_t allocated_addr = POOL_BASE + current_offset_;
        add_and_wrap(HEADER_SIZE);

        return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(allocated_addr);
    }

    FORCE_INLINE static volatile tt_l1_ptr PACKET_HEADER_TYPE* allocate_headers(uint32_t count) {
        ASSERT(count > 0, "Cannot allocate zero headers");

        uint32_t total_size = count * HEADER_SIZE;
        ASSERT(
            current_offset_ + total_size <= POOL_SIZE,
            "Insufficient space in packet header pool for {} headers",
            count);

        uint32_t allocated_addr = POOL_BASE + current_offset_;
        add_and_wrap(total_size);

        return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(allocated_addr);
    }
};

uint32_t PacketHeaderPool::current_offset_ = 0;

#define PACKET_HEADER_POOL_ALLOC() PacketHeaderPool::allocate_header()
#define PACKET_HEADER_POOL_ALLOC_N(count) PacketHeaderPool::allocate_headers(count)
#define PACKET_HEADER_POOL_RESET() PacketHeaderPool::init()
