// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include "dev_mem_map.h"
#include "debug/assert.h"
#include "debug/dprint.h"
#include "eth_chan_noc_mapping.h"

// Simple packet header pool manager for fabric networking
// This class manages allocation of packet headers from a fixed pool
// located at MEM_PACKET_HEADER_POOL_BASE in L1 memory.
class PacketHeaderPool {
private:
    static uint32_t current_offset_;
    static constexpr uint32_t POOL_BASE = MEM_PACKET_HEADER_POOL_BASE;
    static constexpr uint32_t POOL_SIZE = MEM_PACKET_HEADER_POOL_SIZE;
    static constexpr uint32_t HEADER_SIZE = PACKET_HEADER_MAX_SIZE;

public:
    FORCE_INLINE static void init() { current_offset_ = 0; }

    FORCE_INLINE static volatile tt_l1_ptr PACKET_HEADER_TYPE* allocate_header() {
        ASSERT(
            current_offset_ + HEADER_SIZE <= POOL_SIZE,
            "=== PACKET HEADER POOL EXHAUSTION ERROR === "
            "CRITICAL: Insufficient space in packet header pool. "
            " - Headers Allocated: ",
            (current_offset_ / HEADER_SIZE),
            " - Max Headers Capacity: ",
            (POOL_SIZE / HEADER_SIZE));
        if (current_offset_ + HEADER_SIZE > POOL_SIZE) {
            DPRINT << "=== PACKET HEADER POOL EXHAUSTION ERROR ===" << ENDL();
            DPRINT << "CRITICAL: Insufficient space in packet header pool." << ENDL();
            DPRINT << "  - Headers Allocated: " << (current_offset_ / HEADER_SIZE) << ENDL();
            DPRINT << "  - Max Headers Capacity: " << (POOL_SIZE / HEADER_SIZE) << ENDL();
            DPRINT << "Action: Entering infinite loop to prevent undefined behavior" << ENDL();
            DPRINT << "Solution: Increase MEM_PACKET_HEADER_POOL_SIZE or reduce header usage" << ENDL();
            DPRINT << "=================================================" << ENDL();
            while (1) {
            }  // hang intentionally
        }

        uint32_t allocated_addr = POOL_BASE + current_offset_;
        current_offset_ += HEADER_SIZE;
        return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(allocated_addr);
    }
    FORCE_INLINE static volatile tt_l1_ptr PACKET_HEADER_TYPE* allocate_header_n(uint8_t N) {
        ASSERT(
            current_offset_ + (N * HEADER_SIZE) <= POOL_SIZE,
            "=== PACKET HEADER POOL EXHAUSTION ERROR === "
            "CRITICAL: Insufficient space in packet header pool for N headers. "
            " - Headers Allocated: ",
            (current_offset_ / HEADER_SIZE),
            " - Max Headers Capacity: ",
            (POOL_SIZE / HEADER_SIZE));
        if (current_offset_ + (N * HEADER_SIZE) > POOL_SIZE) {
            DPRINT << "=== PACKET HEADER POOL EXHAUSTION ERROR ===" << ENDL();
            DPRINT << "CRITICAL: Insufficient space in packet header pool for N headers." << ENDL();
            DPRINT << "  - Headers Allocated: " << (current_offset_ / HEADER_SIZE) << ENDL();
            DPRINT << "  - Max Headers Capacity: " << (POOL_SIZE / HEADER_SIZE) << ENDL();
            DPRINT << "Action: Entering infinite loop to prevent undefined behavior" << ENDL();
            DPRINT << "Solution: Increase MEM_PACKET_HEADER_POOL_SIZE or reduce header usage" << ENDL();
            DPRINT << "=================================================" << ENDL();
            while (1) {
            }  // hang intentionally
        }

        uint32_t allocated_addr = POOL_BASE + current_offset_;
        current_offset_ += (N * HEADER_SIZE);
        return reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(allocated_addr);
    }
};

uint32_t PacketHeaderPool::current_offset_ = 0;

using route_id_t = uint8_t;
// Route ID Manager for fabric operations
// Route IDs provide an abstraction layer over packet headers,
// allowing consistent APIs across different fabric topologies (1D/2D)
class RouteIdManager {
private:
    static route_id_t next_route_id_;
    static constexpr uint8_t NUM_DIRECTIONS_1D = 2;  // Fwd, Bwd
    static constexpr uint8_t NUM_DIRECTIONS_2D = 4;  // North, East, South, West
    static constexpr uint8_t NUM_DIRECTIONS_3D = 6;  // North, East, South, West, Up, Down
    static constexpr uint8_t MAX_ROUTE_ID = PACKET_HEADER_MAX_DIRECTIONS;

    struct RouteIdMapper {
        volatile tt_l1_ptr PACKET_HEADER_TYPE* headers;  // beginning of the route ID's packet headers
        uint8_t dimension;                               // 1D, 2D, or 3D
        bool in_use;                                     // Track if this route_id is active
    };
    static RouteIdMapper route_id_mapper_[MAX_ROUTE_ID];

public:
    FORCE_INLINE static void init() { next_route_id_ = 0; }

    FORCE_INLINE static route_id_t allocate_route_id() { return next_route_id_++; }

    // Allocate packet headers for a specific route and direction
    FORCE_INLINE static volatile tt_l1_ptr PACKET_HEADER_TYPE* allocate_header_for_route(
        route_id_t route_id, uint8_t dim) {
        ASSERT(
            route_id_mapper_[route_id].in_use == false,
            "Route ID already in use: ",
            route_id,
            ". Cannot allocate new headers.");
        ASSERT(dim >= 1 && dim <= 3, "Invalid dimension: ", dim, ". Must be 1, 2, or 3.");
        if (dim == 1) {
            route_id_mapper_[route_id].headers = PacketHeaderPool::allocate_header_n(NUM_DIRECTIONS_1D);
        } else if (dim == 2) {
            route_id_mapper_[route_id].headers = PacketHeaderPool::allocate_header_n(NUM_DIRECTIONS_2D);
        } else if (dim == 3) {
            route_id_mapper_[route_id].headers = PacketHeaderPool::allocate_header_n(NUM_DIRECTIONS_3D);
        } else {
            return nullptr;
        }
        route_id_mapper_[route_id].dimension = dim;
        route_id_mapper_[route_id].in_use = true;
        return route_id_mapper_[route_id].headers;
    }
};

uint8_t RouteIdManager::next_route_id_ = 0;

RouteIdManager::RouteIdMapper RouteIdManager::route_id_mapper_[RouteIdManager::MAX_ROUTE_ID] = {};

#define PACKET_HEADER_POOL_ALLOC() PacketHeaderPool::allocate_header()
#define PACKET_HEADER_POOL_RESET() PacketHeaderPool::init()
#define ROUTE_ID_ALLOC() RouteIdManager::allocate_route_id()
#define ROUTE_ID_RESET() RouteIdManager::init()
#define ROUTE_ID_HEADER_ALLOC(route_id, dir) RouteIdManager::allocate_header_for_route(route_id, dim)
