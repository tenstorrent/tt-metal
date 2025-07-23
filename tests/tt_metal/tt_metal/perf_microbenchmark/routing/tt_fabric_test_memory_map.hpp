// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdint>
#include "assert.hpp"

namespace tt::tt_fabric::fabric_tests {

struct BaseMemoryRegion {
    uint32_t start;
    uint32_t size;

    BaseMemoryRegion() : start(0), size(0) {}
    BaseMemoryRegion(uint32_t start, uint32_t size) : start(start), size(size) {}

    uint32_t end() const { return start + size; }
    bool is_valid() const { return size > 0; }
    bool contains(uint32_t address) const { return address >= start && address < end(); }
};

/**
 * Common memory layout shared by both senders and receivers
 */
struct CommonMemoryMap {
    // Constants for memory region sizes
    static constexpr uint32_t RESULT_BUFFER_SIZE = 0x1000;  // 4KB

    // Memory regions
    BaseMemoryRegion result_buffer;

    // Default constructor
    CommonMemoryMap() : result_buffer(0, 0) {}

    // Host-side construction
    CommonMemoryMap(uint32_t result_buffer_base, uint32_t result_buffer_size) :
        result_buffer(result_buffer_base, result_buffer_size) {}

    bool is_valid() const { return result_buffer.is_valid(); }

    // Returns common kernel arguments: [result_buffer_base, result_buffer_size]
    std::vector<uint32_t> get_kernel_args() const { return {result_buffer.start, result_buffer.size}; }

    // Convenience methods for reading results
    uint32_t get_result_buffer_address() const { return result_buffer.start; }
    uint32_t get_result_buffer_size() const { return result_buffer.size; }
};

/**
 * Sender-specific memory layout (encapsulates common memory map)
 */
struct SenderMemoryMap {
    // Constants for memory region sizes
    static constexpr uint32_t PACKET_HEADER_BUFFER_SIZE = 0x1000;    // 4KB
    static constexpr uint32_t MAX_PAYLOAD_SIZE_PER_CONFIG = 0x2800;  // 10KB per config

    // Encapsulated common memory map
    CommonMemoryMap common;

    // Sender-specific memory regions
    BaseMemoryRegion packet_headers;
    BaseMemoryRegion payload_buffers;

    // sync addresses
    BaseMemoryRegion global_sync_region;
    BaseMemoryRegion local_sync_region;

    // Calculated values needed for kernel arguments
    uint32_t highest_usable_address;

    // Default constructor
    SenderMemoryMap() : common(), packet_headers(0, 0), payload_buffers(0, 0), highest_usable_address(0) {}

    SenderMemoryMap(
        uint32_t l1_unreserved_base, uint32_t l1_unreserved_size, uint32_t l1_alignment, uint32_t num_configs) :
        common(0, 0),           // Will be set below
        packet_headers(0, 0),   // Will be set below
        payload_buffers(0, 0),  // Will be set below
        highest_usable_address(l1_unreserved_base + l1_unreserved_size) {
        // Layout: [result_buffer][packet_headers][payload_buffers]
        uint32_t current_addr = l1_unreserved_base;

        // Result buffer
        uint32_t result_buffer_base = current_addr;
        current_addr += CommonMemoryMap::RESULT_BUFFER_SIZE;
        common = CommonMemoryMap(result_buffer_base, CommonMemoryMap::RESULT_BUFFER_SIZE);

        // Packet headers
        uint32_t packet_header_base = current_addr;
        current_addr += PACKET_HEADER_BUFFER_SIZE;
        packet_headers = BaseMemoryRegion(packet_header_base, PACKET_HEADER_BUFFER_SIZE);

        // Payload buffers - use small fixed size per config since sender buffer is virtual
        uint32_t payload_buffer_base = current_addr;
        uint32_t payload_buffer_size = MAX_PAYLOAD_SIZE_PER_CONFIG * num_configs;
        current_addr += payload_buffer_size;
        payload_buffers = BaseMemoryRegion(payload_buffer_base, payload_buffer_size);

        // global sync region
        uint32_t global_sync_region_base = current_addr;
        uint32_t global_sync_region_size = l1_alignment;
        current_addr += global_sync_region_size;
        global_sync_region = BaseMemoryRegion(global_sync_region_base, global_sync_region_size);

        // local sync region
        uint32_t local_sync_region_base = current_addr;
        uint32_t local_sync_region_size = l1_alignment;
        current_addr += local_sync_region_size;
        local_sync_region = BaseMemoryRegion(local_sync_region_base, local_sync_region_size);

        TT_FATAL(
            current_addr <= highest_usable_address,
            "Sender memory layout overflow: need {} bytes but only have {} bytes available",
            current_addr - l1_unreserved_base,
            l1_unreserved_size);
    }

    bool is_valid() const { return common.is_valid() && packet_headers.is_valid() && payload_buffers.is_valid(); }

    std::vector<uint32_t> get_memory_map_args() const {
        std::vector<uint32_t> args = common.get_kernel_args();

        args.push_back(packet_headers.start);
        args.push_back(payload_buffers.start);
        args.push_back(highest_usable_address);

        return args;
    }

    uint32_t get_global_sync_address() const { return global_sync_region.start; }
    uint32_t get_local_sync_address() const { return local_sync_region.start; }

    uint32_t get_global_sync_region_size() const { return global_sync_region.size; }
    uint32_t get_local_sync_region_size() const { return local_sync_region.size; }

    // Convenience methods for reading results
    uint32_t get_result_buffer_address() const { return common.get_result_buffer_address(); }
    uint32_t get_result_buffer_size() const { return common.get_result_buffer_size(); }
};

/**
 * Receiver-specific memory layout (encapsulates common memory map)
 */
struct ReceiverMemoryMap {
    // Constants for memory region sizes
    static constexpr uint32_t ATOMIC_COUNTER_BUFFER_SIZE = 0x1000;  // 4KB

    // Encapsulated common memory map
    CommonMemoryMap common;

    // Receiver-specific memory regions
    BaseMemoryRegion payload_chunks;
    BaseMemoryRegion atomic_counters;

    // Store the payload chunk size for later access
    uint32_t payload_chunk_size_;

    // Default constructor
    ReceiverMemoryMap() : common(), payload_chunks(0, 0), atomic_counters(0, 0), payload_chunk_size_(0) {}

    ReceiverMemoryMap(
        uint32_t l1_unreserved_base,
        uint32_t l1_unreserved_size,
        uint32_t l1_alignment,
        uint32_t payload_chunk_size,
        uint32_t num_configs) :
        common(0, 0),           // Will be set below
        payload_chunks(0, 0),   // Will be set below
        atomic_counters(0, 0),  // Will be set below
        payload_chunk_size_(payload_chunk_size) {
        // Layout: [result_buffer][atomic_counters][payload_chunks]
        uint32_t current_addr = l1_unreserved_base;

        // Result buffer
        uint32_t result_buffer_base = current_addr;
        current_addr += CommonMemoryMap::RESULT_BUFFER_SIZE;
        common = CommonMemoryMap(result_buffer_base, CommonMemoryMap::RESULT_BUFFER_SIZE);

        // Atomic counters
        uint32_t atomic_counter_base = current_addr;
        current_addr += ATOMIC_COUNTER_BUFFER_SIZE;
        atomic_counters = BaseMemoryRegion(atomic_counter_base, ATOMIC_COUNTER_BUFFER_SIZE);

        // Payload chunks - receivers need configurable chunk size based on number of configs
        uint32_t payload_chunk_base = current_addr;
        uint32_t payload_chunk_size_total = payload_chunk_size * num_configs;
        current_addr += payload_chunk_size_total;
        payload_chunks = BaseMemoryRegion(payload_chunk_base, payload_chunk_size_total);

        TT_FATAL(
            current_addr <= l1_unreserved_base + l1_unreserved_size,
            "Receiver memory layout overflow: need {} bytes but only have {} bytes available",
            current_addr - l1_unreserved_base,
            l1_unreserved_size);
    }

    bool is_valid() const { return common.is_valid() && payload_chunks.is_valid() && atomic_counters.is_valid(); }

    std::vector<uint32_t> get_memory_map_args() const { return common.get_kernel_args(); }

    // Convenience methods for reading results
    uint32_t get_result_buffer_address() const { return common.get_result_buffer_address(); }
    uint32_t get_result_buffer_size() const { return common.get_result_buffer_size(); }

    // Getter for payload chunk size used in this memory map
    uint32_t get_payload_chunk_size() const { return payload_chunk_size_; }
};

}  // namespace tt::tt_fabric::fabric_tests
