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
    static constexpr uint32_t LOCAL_ARGS_BUFFER_SIZE = 0x1000;  // 4KB
    static constexpr uint32_t KERNEL_CONFIG_BUFFER_SIZE = 0x1000;  // 4KB
    static constexpr uint32_t MUX_LOCAL_ADDRESSES_SIZE = 0x400;    // 1KB for mux local addresses
    static constexpr uint32_t CREDIT_ADDRESSES_SIZE =
        0x400;  // 1KB for credit addresses (64 traffic configs * 16 bytes)

    // Memory regions - still needed for individual access
    BaseMemoryRegion result_buffer;
    BaseMemoryRegion local_args_buffer;
    BaseMemoryRegion kernel_config_buffer;
    BaseMemoryRegion mux_local_addresses;
    BaseMemoryRegion credit_addresses;

    // Explicit end address tracking (no assumptions about region order)
    uint32_t end_address;

    // Credit address allocation state
    mutable uint32_t current_credit_address;

    // Default constructor
    CommonMemoryMap() :
        result_buffer(0, 0),
        local_args_buffer(0, 0),
        kernel_config_buffer(0, 0),
        mux_local_addresses(0, 0),
        credit_addresses(0, 0),
        end_address(0),
        current_credit_address(0) {}

    // Boundary-validated constructor - takes base address and highest usable address
    CommonMemoryMap(uint32_t base_address, uint32_t highest_usable_address) {
        // Layout: [local_args_buffer][result_buffer][kernel_config_buffer][mux_local_addresses][credit_addresses]
        uint32_t current_addr = base_address;

        // Local args buffer (at top of memory)
        uint32_t local_args_base = current_addr;
        current_addr += LOCAL_ARGS_BUFFER_SIZE;
        local_args_buffer = BaseMemoryRegion(local_args_base, LOCAL_ARGS_BUFFER_SIZE);

        // Result buffer
        uint32_t result_buffer_base = current_addr;
        current_addr += RESULT_BUFFER_SIZE;
        result_buffer = BaseMemoryRegion(result_buffer_base, RESULT_BUFFER_SIZE);

        // Kernel config buffer
        uint32_t kernel_config_base = current_addr;
        current_addr += KERNEL_CONFIG_BUFFER_SIZE;
        kernel_config_buffer = BaseMemoryRegion(kernel_config_base, KERNEL_CONFIG_BUFFER_SIZE);

        // Mux local addresses
        uint32_t mux_local_addresses_base = current_addr;
        current_addr += MUX_LOCAL_ADDRESSES_SIZE;
        mux_local_addresses = BaseMemoryRegion(mux_local_addresses_base, MUX_LOCAL_ADDRESSES_SIZE);

        // Credit addresses
        uint32_t credit_addresses_base = current_addr;
        current_addr += CREDIT_ADDRESSES_SIZE;
        credit_addresses = BaseMemoryRegion(credit_addresses_base, CREDIT_ADDRESSES_SIZE);

        // Initialize credit address allocation state
        current_credit_address = credit_addresses_base;

        // Explicitly track end address (no assumptions about region order)
        end_address = current_addr;

        // Verify we don't exceed the available address space
        TT_FATAL(
            end_address <= highest_usable_address,
            "CommonMemoryMap allocation exceeds available address space: allocated to {} but limit is {}",
            end_address,
            highest_usable_address);
    }

    bool is_valid() const {
        return result_buffer.is_valid() && local_args_buffer.is_valid() && kernel_config_buffer.is_valid() &&
               mux_local_addresses.is_valid() && credit_addresses.is_valid();
    }

    // Get the end address of the CommonMemoryMap allocation
    uint32_t get_end_address() const { return end_address; }

    // Returns common kernel arguments: [local_args_base, local_args_size, result_buffer_base, result_buffer_size,
    // kernel_config_base, kernel_config_size, mux_local_addresses_base, mux_local_addresses_size,
    // credit_addresses_base, credit_addresses_size]
    std::vector<uint32_t> get_kernel_args() const {
        return {
            local_args_buffer.start,
            local_args_buffer.size,
            result_buffer.start,
            result_buffer.size,
            kernel_config_buffer.start,
            kernel_config_buffer.size,
            mux_local_addresses.start,
            mux_local_addresses.size,
            credit_addresses.start,
            credit_addresses.size};
    }

    // Removed get_total_size() to eliminate circular dependency - use get_end_address() instead

    // Convenience methods for reading results
    uint32_t get_result_buffer_address() const { return result_buffer.start; }
    uint32_t get_result_buffer_size() const { return result_buffer.size; }
    uint32_t get_local_args_address() const { return local_args_buffer.start; }
    uint32_t get_local_args_size() const { return local_args_buffer.size; }
    uint32_t get_kernel_config_address() const { return kernel_config_buffer.start; }
    uint32_t get_kernel_config_size() const { return kernel_config_buffer.size; }

    // Mux local addresses - host just provides base + size, kernel handles individual calculations
    uint32_t get_mux_local_addresses_base() const { return mux_local_addresses.start; }
    uint32_t get_mux_local_addresses_size() const { return mux_local_addresses.size; }

    // Credit address allocation - allocates unique 16-byte aligned addresses for traffic configs
    uint32_t allocate_credit_address() const {
        TT_FATAL(
            current_credit_address + 16 <= credit_addresses.end(),
            "Credit address allocation exceeds available credit address space: trying to allocate at {} but limit is "
            "{}",
            current_credit_address + 16,
            credit_addresses.end());

        uint32_t addr = current_credit_address;
        current_credit_address += 16;  // 16-byte alignment for NOC efficiency

        // Initialize the credit address to 0
        *reinterpret_cast<volatile uint32_t*>(addr) = 0;

        return addr;
    }

    // Reset credit address allocation for next test run
    void reset_credit_allocation() const { current_credit_address = credit_addresses.start; }
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
    SenderMemoryMap() :
        common(),
        packet_headers(0, 0),
        payload_buffers(0, 0),
        global_sync_region(0, 0),
        local_sync_region(0, 0),
        highest_usable_address(0) {}

    SenderMemoryMap(
        uint32_t l1_unreserved_base, uint32_t l1_unreserved_size, uint32_t l1_alignment, uint32_t num_configs) :
        common(
            l1_unreserved_base,
            l1_unreserved_base + l1_unreserved_size),  // Pass boundary for validation, not size
        packet_headers(0, 0),                          // Will be set below
        payload_buffers(0, 0),                         // Will be set below
        highest_usable_address(l1_unreserved_base + l1_unreserved_size) {
        // Layout: [CommonMemoryMap regions][packet_headers][payload_buffers][sync_regions]
        uint32_t current_addr = common.get_end_address();  // Continue from where CommonMemoryMap ended

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
    uint32_t get_local_args_address() const { return common.get_local_args_address(); }
    uint32_t get_local_args_size() const { return common.get_local_args_size(); }
    uint32_t get_kernel_config_address() const { return common.get_kernel_config_address(); }
    uint32_t get_kernel_config_size() const { return common.get_kernel_config_size(); }

    // Credit address allocation methods
    uint32_t allocate_credit_address() const { return common.allocate_credit_address(); }

    // Reset method for test cleanup
    void reset() const { common.reset_credit_allocation(); }
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
        common(
            l1_unreserved_base,
            l1_unreserved_base + l1_unreserved_size),  // Pass boundary for validation, not size
        payload_chunks(0, 0),                          // Will be set below
        atomic_counters(0, 0),                         // Will be set below
        payload_chunk_size_(payload_chunk_size) {
        // Layout: [CommonMemoryMap regions][atomic_counters][payload_chunks]
        uint32_t current_addr = common.get_end_address();  // Continue from where CommonMemoryMap ended

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
    uint32_t get_local_args_address() const { return common.get_local_args_address(); }
    uint32_t get_local_args_size() const { return common.get_local_args_size(); }
    uint32_t get_kernel_config_address() const { return common.get_kernel_config_address(); }
    uint32_t get_kernel_config_size() const { return common.get_kernel_config_size(); }

    // Getter for payload chunk size used in this memory map
    uint32_t get_payload_chunk_size() const { return payload_chunk_size_; }
};

}  // namespace tt::tt_fabric::fabric_tests
