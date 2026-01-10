// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdint>
#include <tt_stl/assert.hpp>

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
 * Dynamic memory region for mutable per-device allocations (e.g., credits)
 * Manages allocation state and provides chunk allocation with bounds checking
 */
struct DynamicMemoryRegion {
    uint32_t start;
    uint32_t size;
    uint32_t stride;
    mutable uint32_t current;

    DynamicMemoryRegion(uint32_t start_, uint32_t size_, uint32_t stride_) :
        start(start_), size(size_), stride(stride_), current(start_) {}

    uint32_t end() const { return start + size; }

    // Allocate a chunk for num_elements (e.g., num_receivers)
    // Returns the base address of the allocated chunk
    uint32_t allocate_chunk(uint32_t num_elements) const {
        uint32_t chunk_size = num_elements * stride;
        TT_FATAL(
            current + chunk_size <= end(),
            "Chunk allocation exceeds region bounds: "
            "need {} bytes at address {} but region ends at {}",
            chunk_size,
            current,
            end());
        uint32_t chunk_base = current;
        current += chunk_size;
        return chunk_base;
    }

    // Get individual element address from chunk base
    uint32_t get_element_address(uint32_t chunk_base, uint32_t element_idx) const {
        return chunk_base + (element_idx * stride);
    }

    void reset() const { current = start; }
};

/**
 * Common memory layout shared by both senders and receivers
 */
struct CommonMemoryMap {
    // Constants for memory region sizes
    static constexpr uint32_t RESULT_BUFFER_SIZE = 0x1000;  // 4KB
    static constexpr uint32_t LOCAL_ARGS_BUFFER_SIZE = 0x4000;      // 16KB
    static constexpr uint32_t KERNEL_CONFIG_BUFFER_SIZE = 0x10000;  // 64KB (to accommodate big meshes)
    static constexpr uint32_t MUX_LOCAL_ADDRESSES_SIZE = 0x400;    // 1KB
    static constexpr uint32_t MUX_TERMINATION_SYNC_SIZE = 64;  // Single semaphore with padding

    // Memory regions
    BaseMemoryRegion result_buffer;
    BaseMemoryRegion local_args_buffer;
    BaseMemoryRegion kernel_config_buffer;
    BaseMemoryRegion mux_local_addresses;
    BaseMemoryRegion mux_termination_sync;

    // Explicit end address tracking (no assumptions about region order)
    uint32_t end_address;

    // Default constructor
    CommonMemoryMap() :
        result_buffer(0, 0),
        local_args_buffer(0, 0),
        kernel_config_buffer(0, 0),
        mux_local_addresses(0, 0),
        mux_termination_sync(0, 0),
        end_address(0) {}

    // Boundary-validated constructor - takes base address and highest usable address
    CommonMemoryMap(uint32_t base_address, uint32_t highest_usable_address) {
        // Layout:
        // [local_args_buffer]
        // [result_buffer]
        // [kernel_config_buffer]
        // [mux_local_addresses]
        // [mux_termination_sync]
        uint32_t current_addr = base_address;

        // Local args buffer
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

        // Mux termination sync
        uint32_t mux_termination_sync_base = current_addr;
        current_addr += MUX_TERMINATION_SYNC_SIZE;
        mux_termination_sync = BaseMemoryRegion(mux_termination_sync_base, MUX_TERMINATION_SYNC_SIZE);

        // Explicitly track end address
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
               mux_local_addresses.is_valid() && mux_termination_sync.is_valid();
    }

    // Get the end address of the CommonMemoryMap allocation
    uint32_t get_end_address() const { return end_address; }

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
            mux_termination_sync.start};
    }

    // Convenience methods for reading results
    uint32_t get_result_buffer_address() const { return result_buffer.start; }
    uint32_t get_result_buffer_size() const { return result_buffer.size; }
    uint32_t get_local_args_address() const { return local_args_buffer.start; }
    uint32_t get_local_args_size() const { return local_args_buffer.size; }
    uint32_t get_kernel_config_address() const { return kernel_config_buffer.start; }
    uint32_t get_kernel_config_size() const { return kernel_config_buffer.size; }

    // Mux local addresses
    uint32_t get_mux_local_addresses_base() const { return mux_local_addresses.start; }
    uint32_t get_mux_local_addresses_size() const { return mux_local_addresses.size; }

    // Mux termination sync address
    uint32_t get_mux_termination_sync_address() const { return mux_termination_sync.start; }
    uint32_t get_mux_termination_sync_size() const { return mux_termination_sync.size; }
};

/**
 * Sender-specific memory layout (encapsulates common memory map)
 */
struct SenderMemoryMap {
    // Constants for memory region sizes
    static constexpr uint32_t CREDIT_ADDRESSES_SIZE = 0x1000;        // 4KB
    static constexpr uint32_t PACKET_HEADER_BUFFER_SIZE = 0x8000;    // 32KB (sized for 256 mesh size)
    static constexpr uint32_t MAX_PAYLOAD_SIZE_PER_CONFIG = 0x2800;  // 10KB per config

    // Helper to compute individual receiver credit address from chunk
    static uint32_t get_receiver_credit_address(uint32_t chunk_base, uint32_t receiver_idx) {
        return chunk_base + (receiver_idx * CREDIT_ADDRESS_STRIDE);
    }

    static constexpr uint32_t CREDIT_ADDRESS_STRIDE = 16;  // 16-byte alignment per receiver

    // Encapsulated common memory map
    CommonMemoryMap common;

    // Credit address region
    BaseMemoryRegion credit_addresses;

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

        credit_addresses(0, 0),
        packet_headers(0, 0),
        payload_buffers(0, 0),
        global_sync_region(0, 0),
        local_sync_region(0, 0),
        highest_usable_address(0) {}

    SenderMemoryMap(uint32_t l1_unreserved_base, uint32_t l1_unreserved_size, uint32_t l1_alignment) :
        common(
            l1_unreserved_base,
            l1_unreserved_base + l1_unreserved_size),  // Pass boundary for validation, not size
        credit_addresses(0, 0),                        // Will be set below
        packet_headers(0, 0),                          // Will be set below
        payload_buffers(0, 0),                         // Will be set below
        highest_usable_address(l1_unreserved_base + l1_unreserved_size) {
        // Layout:
        // [CommonMemoryMap regions]
        // [credit_addresses]
        // [packet_headers]
        // [payload_buffers]
        // [sync_regions]
        uint32_t current_addr = common.get_end_address();  // Continue from where CommonMemoryMap ended

        // Credit addresses
        uint32_t credit_addresses_base = current_addr;
        current_addr += CREDIT_ADDRESSES_SIZE;
        credit_addresses = BaseMemoryRegion(credit_addresses_base, CREDIT_ADDRESSES_SIZE);

        // Packet headers
        uint32_t packet_header_base = current_addr;
        current_addr += PACKET_HEADER_BUFFER_SIZE;
        packet_headers = BaseMemoryRegion(packet_header_base, PACKET_HEADER_BUFFER_SIZE);

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

        // Payload buffers - use small fixed size per config since sender buffer is virtual
        // needs to be after the sync regions to avoid overflow
        uint32_t payload_buffer_base = current_addr;
        uint32_t payload_buffer_size = highest_usable_address - current_addr;
        current_addr += payload_buffer_size;
        payload_buffers = BaseMemoryRegion(payload_buffer_base, payload_buffer_size);

        TT_FATAL(
            current_addr <= highest_usable_address,
            "Sender memory layout overflow: need {} bytes but only have {} bytes available",
            current_addr - l1_unreserved_base,
            l1_unreserved_size);
    }

    bool is_valid() const {
        return common.is_valid() && credit_addresses.is_valid() && packet_headers.is_valid() &&
               payload_buffers.is_valid();
    }

    std::vector<uint32_t> get_memory_map_args() const {
        std::vector<uint32_t> args = common.get_kernel_args();

        args.push_back(packet_headers.start);
        args.push_back(payload_buffers.start);
        args.push_back(highest_usable_address);

        return args;
    }

    uint32_t get_credit_addresses_base() const { return credit_addresses.start; }
    uint32_t get_credit_addresses_size() const { return credit_addresses.size; }

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

    uint32_t get_mux_termination_sync_address() const { return common.get_mux_termination_sync_address(); }
    uint32_t get_mux_termination_sync_size() const { return common.get_mux_termination_sync_size(); }
};

/**
 * Receiver-specific memory layout (encapsulates common memory map)
 */
struct ReceiverMemoryMap {
    // Constants for memory region sizes
    static constexpr uint32_t ATOMIC_COUNTER_BUFFER_SIZE = 0x1000;  // 4KB
    static constexpr uint32_t CREDIT_HEADER_BUFFER_SIZE = 0x400;    // 1KB reserved for credit headers

    // Encapsulated common memory map
    CommonMemoryMap common;

    // Receiver-specific memory regions
    BaseMemoryRegion payload_chunks;
    BaseMemoryRegion atomic_counters;
    BaseMemoryRegion credit_headers;

    // Store the payload chunk size for later access
    uint32_t payload_chunk_size_;

    // Default constructor
    ReceiverMemoryMap() : payload_chunks(0, 0), atomic_counters(0, 0), credit_headers(0, 0), payload_chunk_size_(0) {}

    ReceiverMemoryMap(
        uint32_t l1_unreserved_base,
        uint32_t l1_unreserved_size,
        uint32_t /*l1_alignment*/,
        uint32_t payload_chunk_size,
        uint32_t num_configs) :
        common(
            l1_unreserved_base,
            l1_unreserved_base + l1_unreserved_size),  // Pass boundary for validation, not size
        payload_chunks(0, 0),                          // Will be set below
        atomic_counters(0, 0),                         // Will be set below
        credit_headers(0, 0),                          // Will be set below
        payload_chunk_size_(payload_chunk_size) {
        // Layout:
        // [CommonMemoryMap regions]
        // [atomic_counters]
        // [credit_headers]
        // [payload_chunks]
        uint32_t current_addr = common.get_end_address();  // Continue from where CommonMemoryMap ended

        // Atomic counters
        uint32_t atomic_counter_base = current_addr;
        current_addr += ATOMIC_COUNTER_BUFFER_SIZE;
        atomic_counters = BaseMemoryRegion(atomic_counter_base, ATOMIC_COUNTER_BUFFER_SIZE);

        // Credit headers - reserved space for credit return packet headers
        uint32_t credit_header_base = current_addr;
        current_addr += CREDIT_HEADER_BUFFER_SIZE;
        credit_headers = BaseMemoryRegion(credit_header_base, CREDIT_HEADER_BUFFER_SIZE);

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

    bool is_valid() const {
        return common.is_valid() && payload_chunks.is_valid() && atomic_counters.is_valid() &&
               credit_headers.is_valid();
    }

    std::vector<uint32_t> get_memory_map_args() const {
        auto args = common.get_kernel_args();
        args.push_back(credit_headers.start);
        args.push_back(credit_headers.start + credit_headers.size);
        return args;
    }

    // Convenience methods for reading results
    uint32_t get_result_buffer_address() const { return common.get_result_buffer_address(); }
    uint32_t get_result_buffer_size() const { return common.get_result_buffer_size(); }
    uint32_t get_local_args_address() const { return common.get_local_args_address(); }
    uint32_t get_local_args_size() const { return common.get_local_args_size(); }
    uint32_t get_kernel_config_address() const { return common.get_kernel_config_address(); }
    uint32_t get_kernel_config_size() const { return common.get_kernel_config_size(); }

    uint32_t get_mux_termination_sync_address() const { return common.get_mux_termination_sync_address(); }
    uint32_t get_mux_termination_sync_size() const { return common.get_mux_termination_sync_size(); }

    // Getter for payload chunk size used in this memory map
    uint32_t get_payload_chunk_size() const { return payload_chunk_size_; }
};

}  // namespace tt::tt_fabric::fabric_tests
