// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"

namespace tt::tt_fabric {
namespace fabric_tests {

// Maximum number of fabric connections supported per kernel (1 per direction: N, S, E, W)
// This is used to size FabricConnectionArray storage without template proliferation
static constexpr uint8_t MAX_NUM_FABRIC_CONNECTIONS = 4;

struct LocalArgsBuffer {
    uint32_t base_address = 0;
    uint32_t buffer_size = 0;
    uint32_t end_address = 0;

    void init(uint32_t base_addr, uint32_t buf_size) {
        base_address = base_addr;
        buffer_size = buf_size;
        end_address = base_address + buffer_size;
    }

    template <typename T>
    FORCE_INLINE T get_arg_val(size_t arg_idx) {
        static_assert("Error: only 4B args are supported" && sizeof(T) == 4);

        uint32_t current_offset = arg_idx * sizeof(T);
        ASSERT(current_offset + sizeof(T) <= end_address);  // Check bounds

        tt_l1_ptr T* local_args_ptr = reinterpret_cast<tt_l1_ptr T*>(base_address);
        return local_args_ptr[arg_idx];
    }
};

// Global instance of the local args buffer manager
static LocalArgsBuffer local_args_buffer;

inline void init_local_args(uint32_t base_address, uint32_t buffer_size) {
    local_args_buffer.init(base_address, buffer_size);
}

template <typename T>
FORCE_INLINE T get_local_arg_val(size_t arg_idx) {
    return local_args_buffer.get_arg_val<T>(arg_idx);
}

inline uint32_t prng_next(uint32_t n) {
    uint32_t x = n;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// Helper functions for writing test results
inline void write_test_status(uint32_t result_buffer_base, uint32_t status) {
    auto* result_buffer = reinterpret_cast<tt_l1_ptr uint32_t*>(result_buffer_base);
    result_buffer[TT_FABRIC_STATUS_INDEX] = status;
}

inline void write_test_cycles(uint32_t result_buffer_base, uint64_t cycles) {
    auto* result_buffer = reinterpret_cast<tt_l1_ptr uint32_t*>(result_buffer_base);
    result_buffer[TT_FABRIC_CYCLES_INDEX] = static_cast<uint32_t>(cycles);
    result_buffer[TT_FABRIC_CYCLES_INDEX + 1] = static_cast<uint32_t>(cycles >> 32);
}

inline void write_test_packets(uint32_t result_buffer_base, uint64_t packets) {
    auto* result_buffer = reinterpret_cast<tt_l1_ptr uint32_t*>(result_buffer_base);
    result_buffer[TT_FABRIC_WORD_CNT_INDEX] = static_cast<uint32_t>(packets);
    result_buffer[TT_FABRIC_WORD_CNT_INDEX + 1] = static_cast<uint32_t>(packets >> 32);
}

inline void clear_test_results(uint32_t result_buffer_base, uint32_t result_buffer_size) {
    auto* result_buffer = reinterpret_cast<tt_l1_ptr uint32_t*>(result_buffer_base);
    uint32_t num_words = result_buffer_size / sizeof(uint32_t);
    for (uint32_t i = 0; i < num_words; i++) {
        result_buffer[i] = 0;
    }
}

struct SequentialDataPattern {
    static constexpr uint32_t WORD_SIZE = sizeof(uint32_t);

    static void fill(uint32_t buffer_address, uint32_t payload_size, uint32_t start_value) {
        auto* buffer_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(buffer_address);
        uint32_t num_words = payload_size / WORD_SIZE;
        for (uint32_t i = 0; i < num_words; i++) {
            buffer_ptr[i] = start_value + i;
        }
    }

    static bool poll(uint32_t buffer_address, uint32_t payload_size, uint32_t start_value) {
        auto* buffer_ptr =
            reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(buffer_address + payload_size - WORD_SIZE);
        uint32_t expected_value = start_value + payload_size / WORD_SIZE - 1;
        return *buffer_ptr == expected_value;
    }

    static bool validate(uint32_t buffer_address, uint32_t payload_size, uint32_t start_value) {
        auto* buffer_ptr = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(buffer_address);
        uint32_t num_words = payload_size / WORD_SIZE;
        for (uint32_t i = 0; i < num_words; i++) {
            if (buffer_ptr[i] != (start_value + i)) {
                return false;
            }
        }
        return true;
    }
};

class StreamingBuffer {
public:
    StreamingBuffer(uint32_t base_address, uint32_t total_size, uint32_t payload_size) :
        base_address_(base_address), total_size_(total_size), payload_size_(payload_size) {
        ASSERT(total_size > 0);
        ASSERT(payload_size > 0);
        ASSERT(payload_size <= total_size);
        reset();
    }

    uint32_t get_current_offset() const { return current_offset_; }
    constexpr bool has_wrapped() const { return has_wrapped_; }

    void advance() {
        current_offset_ += payload_size_;
        // need to check if we have enough space in the buffer for another payload without wrapping
        if (current_offset_ + payload_size_ > total_size_) {
            current_offset_ = 0;
            has_wrapped_ = true;
        }
    }

    void reset() {
        current_offset_ = 0;
        has_wrapped_ = false;
    }

protected:
    uint32_t base_address_;
    uint32_t total_size_;
    uint32_t payload_size_;
    uint32_t current_offset_;
    bool has_wrapped_ = false;
};

class SenderPayloadBuffer : public StreamingBuffer {
public:
    SenderPayloadBuffer(uint32_t physical_base_address, uint32_t virtual_total_size, uint32_t payload_size) :
        StreamingBuffer(physical_base_address, virtual_total_size, payload_size) {}

    uint32_t get_physical_address() const { return base_address_; }

    void fill_data(uint32_t start_value) {
        SequentialDataPattern::fill(get_physical_address(), this->payload_size_, start_value);
    }
};

class ReceiverPayloadBuffer : public StreamingBuffer {
public:
    ReceiverPayloadBuffer(uint32_t base_address, uint32_t total_size, uint32_t payload_size) :
        StreamingBuffer(base_address, total_size, payload_size) {}

    uint32_t get_physical_address() const { return base_address_ + this->get_current_offset(); }

    bool poll_for_data(uint32_t start_value) {
        return SequentialDataPattern::poll(get_physical_address(), this->payload_size_, start_value);
    }

    bool validate_data(uint32_t start_value) const {
        return SequentialDataPattern::validate(get_physical_address(), this->payload_size_, start_value);
    }
};

struct SenderTrafficConfigMetadata {
    static SenderTrafficConfigMetadata build_from_args(size_t& arg_idx) { return SenderTrafficConfigMetadata(arg_idx); }

    SenderTrafficConfigMetadata(const SenderTrafficConfigMetadata& other) :
        num_packets(other.num_packets), seed(other.seed), payload_buffer_size(other.payload_buffer_size) {}

    uint32_t num_packets = 0;
    uint32_t seed = 0;
    uint32_t payload_buffer_size = 0;

private:
    SenderTrafficConfigMetadata(size_t& arg_idx) {
        this->num_packets = get_local_arg_val<uint32_t>(arg_idx++);
        this->seed = get_local_arg_val<uint32_t>(arg_idx++);
        this->payload_buffer_size = get_local_arg_val<uint32_t>(arg_idx++);
    }
};

struct ChipUnicastFields1D {
    static ChipUnicastFields1D build_from_args(size_t& arg_idx) {
        uint32_t num_hops = get_local_arg_val<uint32_t>(arg_idx++);
        return ChipUnicastFields1D(num_hops);
    }

    ChipUnicastFields1D(uint32_t num_hops) : num_hops(num_hops) {}

    uint32_t num_hops;
};

struct ChipUnicastFields2D {
    static ChipUnicastFields2D build_from_args(size_t& arg_idx) {
        uint16_t src_device_id = get_local_arg_val<uint32_t>(arg_idx++);
        uint16_t dst_device_id = get_local_arg_val<uint32_t>(arg_idx++);
        uint16_t dst_mesh_id = get_local_arg_val<uint32_t>(arg_idx++);
        uint16_t ew_dim = get_local_arg_val<uint32_t>(arg_idx++);
        return ChipUnicastFields2D(src_device_id, dst_device_id, dst_mesh_id, ew_dim);
    }

    ChipUnicastFields2D(uint16_t src_device_id, uint16_t dst_device_id, uint16_t dst_mesh_id, uint16_t ew_dim) :
        src_device_id(src_device_id), dst_device_id(dst_device_id), dst_mesh_id(dst_mesh_id), ew_dim(ew_dim) {}

    uint16_t src_device_id;
    uint16_t dst_device_id;
    uint16_t dst_mesh_id;
    uint16_t ew_dim;
};

struct ChipMulticastFields1D {
    static ChipMulticastFields1D build_from_args(size_t& arg_idx) {
        uint32_t mcast_start_hops = get_local_arg_val<uint32_t>(arg_idx++);
        uint32_t num_hops = get_local_arg_val<uint32_t>(arg_idx++);
        return ChipMulticastFields1D(mcast_start_hops, num_hops);
    }

    ChipMulticastFields1D(uint32_t mcast_start_hops, uint32_t num_hops) :
        mcast_start_hops(mcast_start_hops), num_hops(num_hops) {}

    uint32_t mcast_start_hops;
    uint32_t num_hops;
};

struct ChipMulticastFields2D {
    static ChipMulticastFields2D build_from_args(size_t& arg_idx) {
        uint16_t dst_device_id = get_local_arg_val<uint32_t>(arg_idx++);
        uint16_t dst_mesh_id = get_local_arg_val<uint32_t>(arg_idx++);
        uint16_t num_hops_n = get_local_arg_val<uint32_t>(arg_idx++);
        uint16_t num_hops_s = get_local_arg_val<uint32_t>(arg_idx++);
        uint16_t num_hops_e = get_local_arg_val<uint32_t>(arg_idx++);
        uint16_t num_hops_w = get_local_arg_val<uint32_t>(arg_idx++);
        return ChipMulticastFields2D(dst_device_id, dst_mesh_id, num_hops_n, num_hops_s, num_hops_e, num_hops_w);
    }

    ChipMulticastFields2D(
        uint16_t dst_device_id,
        uint16_t dst_mesh_id,
        uint16_t num_hops_n,
        uint16_t num_hops_s,
        uint16_t num_hops_e,
        uint16_t num_hops_w) :
        dst_device_id(dst_device_id),
        dst_mesh_id(dst_mesh_id),
        num_hops_n(num_hops_n),
        num_hops_s(num_hops_s),
        num_hops_e(num_hops_e),
        num_hops_w(num_hops_w) {}

    uint16_t dst_device_id;
    uint16_t dst_mesh_id;
    uint16_t num_hops_n;
    uint16_t num_hops_s;
    uint16_t num_hops_e;
    uint16_t num_hops_w;
};

struct NocUnicastWriteFields {
    template <bool IS_SOURCE>
    static NocUnicastWriteFields build_from_args(size_t& arg_idx) {
        uint32_t payload_size_bytes = get_local_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_address = get_local_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_noc_encoding = 0;
        if constexpr (IS_SOURCE) {
            dst_noc_encoding = get_local_arg_val<uint32_t>(arg_idx++);
        }
        return NocUnicastWriteFields(payload_size_bytes, dst_address, dst_noc_encoding);
    }

    NocUnicastWriteFields(uint32_t payload_size_bytes, uint32_t dst_address, uint32_t dst_noc_encoding) :
        payload_size_bytes(payload_size_bytes), dst_address(dst_address), dst_noc_encoding(dst_noc_encoding) {}

    uint32_t payload_size_bytes;
    uint32_t dst_address;
    uint32_t dst_noc_encoding;
};

struct NocUnicastAtomicIncFields {
    template <bool IS_SOURCE>
    static NocUnicastAtomicIncFields build_from_args(size_t& arg_idx) {
        uint32_t atomic_inc_val = get_local_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_address = get_local_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_noc_encoding = 0;
        if constexpr (IS_SOURCE) {
            dst_noc_encoding = get_local_arg_val<uint32_t>(arg_idx++);
        }
        return NocUnicastAtomicIncFields(atomic_inc_val, dst_address, dst_noc_encoding);
    }

    NocUnicastAtomicIncFields(uint32_t atomic_inc_val, uint32_t dst_address, uint32_t dst_noc_encoding) :
        atomic_inc_val(atomic_inc_val), dst_address(dst_address), dst_noc_encoding(dst_noc_encoding) {}

    uint32_t atomic_inc_val;
    uint32_t dst_address;
    uint32_t dst_noc_encoding;
};

struct NocUnicastWriteAtomicIncFields {
    template <bool IS_SOURCE>
    static NocUnicastWriteAtomicIncFields build_from_args(size_t& arg_idx) {
        const auto write_fields = NocUnicastWriteFields::build_from_args<IS_SOURCE>(arg_idx);
        const auto atomic_inc_fields = NocUnicastAtomicIncFields::build_from_args<IS_SOURCE>(arg_idx);
        return NocUnicastWriteAtomicIncFields(write_fields, atomic_inc_fields);
    }

    NocUnicastWriteAtomicIncFields(NocUnicastWriteFields write_fields, NocUnicastAtomicIncFields atomic_inc_fields) :
        write_fields(write_fields), atomic_inc_fields(atomic_inc_fields) {}

    NocUnicastWriteFields write_fields;
    NocUnicastAtomicIncFields atomic_inc_fields;
};

struct NocUnicastScatterWriteFields {
    static constexpr uint32_t MAX_CHUNKS = 2;

    template <bool IS_SOURCE>
    static NocUnicastScatterWriteFields build_from_args(size_t& arg_idx) {
        uint32_t payload_size_bytes = get_local_arg_val<uint32_t>(arg_idx++);
        uint32_t chunk_count = get_local_arg_val<uint32_t>(arg_idx++);
        ASSERT(chunk_count == MAX_CHUNKS);

        std::array<uint32_t, MAX_CHUNKS> dst_addresses{};
        for (uint32_t i = 0; i < chunk_count; i++) {
            dst_addresses[i] = get_local_arg_val<uint32_t>(arg_idx++);
        }

        uint32_t dst_noc_encoding = 0;
        if constexpr (IS_SOURCE) {
            dst_noc_encoding = get_local_arg_val<uint32_t>(arg_idx++);
        }

        std::array<uint16_t, MAX_CHUNKS - 1> chunk_sizes{};
        for (uint32_t i = 0; i < (chunk_count - 1); i++) {
            chunk_sizes[i] = static_cast<uint16_t>(get_local_arg_val<uint32_t>(arg_idx++));
        }

        return NocUnicastScatterWriteFields(
            payload_size_bytes, static_cast<uint8_t>(chunk_count), dst_addresses, chunk_sizes, dst_noc_encoding);
    }

    NocUnicastScatterWriteFields(
        uint32_t payload_size_bytes,
        uint8_t chunk_count,
        const std::array<uint32_t, MAX_CHUNKS>& dst_addresses,
        const std::array<uint16_t, MAX_CHUNKS - 1>& chunk_sizes,
        uint32_t dst_noc_encoding) :
        payload_size_bytes(payload_size_bytes),
        chunk_count(chunk_count),
        dst_addresses(dst_addresses),
        chunk_sizes(chunk_sizes),
        dst_noc_encoding(dst_noc_encoding) {}

    uint32_t payload_size_bytes;
    uint8_t chunk_count;
    std::array<uint32_t, MAX_CHUNKS> dst_addresses;
    std::array<uint16_t, MAX_CHUNKS - 1> chunk_sizes;
    uint32_t dst_noc_encoding;
};

template <typename T>
void setup_2d_unicast_route(uint32_t packet_header_address, const ChipUnicastFields2D& unicast_fields) {
    // Template constraint: T must be MeshPacketHeader or HybridMeshPacketHeader
    fabric_set_unicast_route(
        (T*)packet_header_address,
        unicast_fields.src_device_id,
        unicast_fields.dst_device_id,
        unicast_fields.dst_mesh_id,
        unicast_fields.ew_dim);
}

template <typename T>
void setup_2d_mcast_route(uint32_t packet_header_address, const ChipMulticastFields2D& mcast_fields) {
    // Template constraint: T must be MeshPacketHeader or HybridMeshPacketHeader
    fabric_set_mcast_route(
        (T*)packet_header_address,
        mcast_fields.dst_device_id,
        mcast_fields.dst_mesh_id,
        mcast_fields.num_hops_e,
        mcast_fields.num_hops_w,
        mcast_fields.num_hops_n,
        mcast_fields.num_hops_s);
}

/**
 * Template-based dispatch system for chip send type handling.
 * Specialized for:
 * - 1D vs 2D fabric routing
 * - Unicast vs multicast transmission
 */
template <ChipSendType chip_type, bool IS_2D_FABRIC>
struct ChipSendTypeHandler {
    static void parse_and_setup(
        size_t& arg_idx, uint32_t packet_header_address, volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header);
};

// 1D Unicast specialization
template <>
struct ChipSendTypeHandler<ChipSendType::CHIP_UNICAST, false> {
    static void parse_and_setup(
        size_t& arg_idx, uint32_t packet_header_address, volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header) {
        const auto unicast_fields = ChipUnicastFields1D::build_from_args(arg_idx);
        fabric_set_unicast_route<false>((LowLatencyPacketHeader*)packet_header, unicast_fields.num_hops);
    }
};

// 2D Unicast specialization
template <>
struct ChipSendTypeHandler<ChipSendType::CHIP_UNICAST, true> {
    static void parse_and_setup(
        size_t& arg_idx, uint32_t packet_header_address, volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header) {
        const auto unicast_fields = ChipUnicastFields2D::build_from_args(arg_idx);
        fabric_set_unicast_route(
            (HybridMeshPacketHeader*)packet_header_address, unicast_fields.dst_device_id, unicast_fields.dst_mesh_id);
    }
};

// 1D Multicast specialization
template <>
struct ChipSendTypeHandler<ChipSendType::CHIP_MULTICAST, false> {
    static void parse_and_setup(
        size_t& arg_idx, uint32_t packet_header_address, volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header) {
        const auto mcast_fields = ChipMulticastFields1D::build_from_args(arg_idx);
        packet_header->to_chip_multicast(MulticastRoutingCommandHeader{
            static_cast<uint8_t>(mcast_fields.mcast_start_hops), static_cast<uint8_t>(mcast_fields.num_hops)});
    }
};

// 2D Multicast specialization
template <>
struct ChipSendTypeHandler<ChipSendType::CHIP_MULTICAST, true> {
    static void parse_and_setup(
        size_t& arg_idx, uint32_t packet_header_address, volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header) {
        const auto mcast_fields = ChipMulticastFields2D::build_from_args(arg_idx);
        setup_2d_mcast_route<HybridMeshPacketHeader>(packet_header_address, mcast_fields);
    }
};

// Forward declaration for NOC operation function pointer types
struct SenderKernelTrafficConfig;

namespace NocOperationTypes {
using ParseSetupFunc = void (*)(SenderKernelTrafficConfig*, size_t&);
using UpdateHeaderFunc = void (*)(SenderKernelTrafficConfig*);

struct Operations {
    ParseSetupFunc parse_and_setup;
    UpdateHeaderFunc update_header;
};
}  // namespace NocOperationTypes

// NOC Operation Class Declarations (implementations after SenderKernelTrafficConfig)
struct NocWriteSenderOperations {
    static void parse_and_setup_impl(SenderKernelTrafficConfig* config, size_t& arg_idx);
    static void update_header_impl(SenderKernelTrafficConfig* config);
};

struct NocAtomicSenderOperations {
    static void parse_and_setup_impl(SenderKernelTrafficConfig* config, size_t& arg_idx);
    static void update_header_impl(SenderKernelTrafficConfig* config);
};

struct NocFusedSenderOperations {
    static void parse_and_setup_impl(SenderKernelTrafficConfig* config, size_t& arg_idx);
    static void update_header_impl(SenderKernelTrafficConfig* config);
};

struct NocScatterWriteSenderOperations {
    static void parse_and_setup_impl(SenderKernelTrafficConfig* config, size_t& arg_idx);
    static void update_header_impl(SenderKernelTrafficConfig* config);
};

/* ****************************************************************************
 * MuxCachedInfo
 * *****************************************************************************/
struct MuxCachedInfo {
    uint8_t mux_x = 0;
    uint8_t mux_y = 0;
    size_t mux_status_address = 0;
    size_t local_mux_status_address = 0;
};

/* ****************************************************************************
 * FabricConnectionArray: Unified connection management for kernel
 *
 * Provides type-erased storage for both WorkerToFabricEdmSender and
 * WorkerToFabricMuxSender connections with runtime dispatch.
 * *****************************************************************************/
struct FabricConnectionArray {
    // TODO: get the num buffers more systematically
    static constexpr uint8_t NUM_BUFFERS = 8;

    using MuxConnectionType = tt::tt_fabric::WorkerToFabricMuxSender<NUM_BUFFERS>;
    static constexpr size_t MAX_CONNECTION_SIZE = std::max(sizeof(WorkerToFabricEdmSender), sizeof(MuxConnectionType));

    // Type-erased storage for connections (sized for maximum)
    alignas(std::max(alignof(WorkerToFabricEdmSender), alignof(MuxConnectionType)))
        std::array<char, MAX_NUM_FABRIC_CONNECTIONS * MAX_CONNECTION_SIZE> storage;
    std::array<bool, MAX_NUM_FABRIC_CONNECTIONS> is_mux;

    // Cached mux info for wait_for_fabric_endpoint_ready
    std::array<MuxCachedInfo, MAX_NUM_FABRIC_CONNECTIONS> mux_cached_info;

    // Actual number of connections in use (set at initialization, bounds-checked in kernel)
    uint8_t num_connections = 0;

    // Accessors with proper type casting
    FORCE_INLINE WorkerToFabricEdmSender& get_fabric_connection(uint8_t idx) {
        return *reinterpret_cast<WorkerToFabricEdmSender*>(storage.data() + idx * MAX_CONNECTION_SIZE);
    }

    FORCE_INLINE MuxConnectionType& get_mux_connection(uint8_t idx) {
        return *reinterpret_cast<MuxConnectionType*>(storage.data() + idx * MAX_CONNECTION_SIZE);
    }

    // Parse connections from runtime args
    // Memory map is required for allocating local semaphore addresses for mux connections
    template <ProgrammableCoreType core_type = ProgrammableCoreType::TENSIX, typename MemoryMapType>
    void parse_from_args(size_t& rt_args_idx, MemoryMapType& memory_map) {
        for (uint8_t i = 0; i < num_connections; i++) {
            // Parse connection type flag
            is_mux[i] = get_arg_val<uint32_t>(rt_args_idx++) != 0;

            if (is_mux[i]) {
                // Initialize mux connection using placement new
                mux_cached_info[i].mux_x = get_arg_val<uint32_t>(rt_args_idx++);
                mux_cached_info[i].mux_y = get_arg_val<uint32_t>(rt_args_idx++);
                uint8_t worker_stream_id = get_arg_val<uint32_t>(rt_args_idx++);
                uint8_t mux_num_buffers_per_channel = get_arg_val<uint32_t>(rt_args_idx++);
                size_t mux_channel_buffer_size_bytes = get_arg_val<uint32_t>(rt_args_idx++);
                size_t mux_channel_base_address = get_arg_val<uint32_t>(rt_args_idx++);
                size_t mux_connection_info_address = get_arg_val<uint32_t>(rt_args_idx++);
                size_t mux_connection_handshake_address = get_arg_val<uint32_t>(rt_args_idx++);
                size_t mux_flow_control_address = get_arg_val<uint32_t>(rt_args_idx++);
                size_t mux_buffer_index_address = get_arg_val<uint32_t>(rt_args_idx++);
                mux_cached_info[i].mux_status_address = get_arg_val<uint32_t>(rt_args_idx++);

                // Allocate local semaphore addresses for this mux connection (cursor-based)
                const auto mux_local_addrs = memory_map.get_mux_local_addresses_for_connection();
                mux_cached_info[i].local_mux_status_address = mux_local_addrs.status_buffer_address;

                auto conn = build_connection_to_fabric_endpoint<NUM_BUFFERS>(
                    mux_cached_info[i].mux_x,
                    mux_cached_info[i].mux_y,
                    worker_stream_id,
                    mux_num_buffers_per_channel,
                    mux_channel_buffer_size_bytes,
                    mux_channel_base_address,
                    mux_connection_info_address,
                    mux_connection_handshake_address,
                    mux_flow_control_address,
                    mux_buffer_index_address,
                    mux_local_addrs.flow_control_address,
                    mux_local_addrs.teardown_address,
                    mux_local_addrs.buffer_index_address);
                new (&get_mux_connection(i)) MuxConnectionType(conn);
            } else {
                // Initialize fabric connection using placement new
                auto conn = WorkerToFabricEdmSender::build_from_args<core_type>(rt_args_idx);
                new (&get_fabric_connection(i)) WorkerToFabricEdmSender(conn);
            }
        }
    }

    // Lifecycle management
    FORCE_INLINE void open_all() {
        for (uint8_t i = 0; i < num_connections; i++) {
            if (is_mux[i]) {
                // Wait for mux to be ready before connecting
                const auto& info = mux_cached_info[i];
                tt::tt_fabric::wait_for_fabric_endpoint_ready(
                    info.mux_x, info.mux_y, info.mux_status_address, info.local_mux_status_address);
                get_mux_connection(i).open();
            } else {
                get_fabric_connection(i).open();
            }
        }
    }

    FORCE_INLINE void close_all() {
        for (uint8_t i = 0; i < num_connections; i++) {
            if (is_mux[i]) {
                get_mux_connection(i).close();
            } else {
                get_fabric_connection(i).close();
            }
        }
    }

    // Unified send operations (dispatch hidden from callers)

    // Wait for connection to have space
    template <bool BENCHMARK_MODE = false>
    FORCE_INLINE void wait_for_empty_write_slot(void* conn_ptr, uint8_t idx) {
        if constexpr (BENCHMARK_MODE) {
            // Fast path: no runtime check, direct cast
            static_cast<WorkerToFabricEdmSender*>(conn_ptr)->wait_for_empty_write_slot();
        } else {
            // Normal path: runtime dispatch using cached is_mux array
            if (is_mux[idx]) {
                static_cast<MuxConnectionType*>(conn_ptr)->wait_for_empty_write_slot();
            } else {
                static_cast<WorkerToFabricEdmSender*>(conn_ptr)->wait_for_empty_write_slot();
            }
        }
    }

    // Send header only (used for credit returns)
    template <bool BENCHMARK_MODE = false>
    FORCE_INLINE void send_header_non_blocking(void* conn_ptr, uint8_t idx, uint32_t header_addr) {
        if constexpr (BENCHMARK_MODE) {
            static_cast<WorkerToFabricEdmSender*>(conn_ptr)->send_payload_flush_non_blocking_from_address(
                header_addr, sizeof(PACKET_HEADER_TYPE));
        } else {
            if (is_mux[idx]) {
                static_cast<MuxConnectionType*>(conn_ptr)->send_payload_flush_non_blocking_from_address(
                    header_addr, sizeof(PACKET_HEADER_TYPE));
            } else {
                static_cast<WorkerToFabricEdmSender*>(conn_ptr)->send_payload_flush_non_blocking_from_address(
                    header_addr, sizeof(PACKET_HEADER_TYPE));
            }
        }
    }

    // Send payload without header (used for multi-part sends)
    template <bool BENCHMARK_MODE = false>
    FORCE_INLINE void send_payload_without_header(void* conn_ptr, uint8_t idx, uint32_t payload_addr, size_t size) {
        if constexpr (BENCHMARK_MODE) {
            static_cast<WorkerToFabricEdmSender*>(conn_ptr)->send_payload_without_header_non_blocking_from_address(
                payload_addr, size);
        } else {
            if (is_mux[idx]) {
                static_cast<MuxConnectionType*>(conn_ptr)->send_payload_without_header_non_blocking_from_address(
                    payload_addr, size);
            } else {
                static_cast<WorkerToFabricEdmSender*>(conn_ptr)->send_payload_without_header_non_blocking_from_address(
                    payload_addr, size);
            }
        }
    }

    // Send header with flush (used for completing multi-part sends)
    template <bool BENCHMARK_MODE = false>
    FORCE_INLINE void send_header_flush_blocking(void* conn_ptr, uint8_t idx, uint32_t header_addr) {
        if constexpr (BENCHMARK_MODE) {
            static_cast<WorkerToFabricEdmSender*>(conn_ptr)->send_payload_flush_blocking_from_address(
                header_addr, sizeof(PACKET_HEADER_TYPE));
        } else {
            if (is_mux[idx]) {
                static_cast<MuxConnectionType*>(conn_ptr)->send_payload_flush_blocking_from_address(
                    header_addr, sizeof(PACKET_HEADER_TYPE));
            } else {
                static_cast<WorkerToFabricEdmSender*>(conn_ptr)->send_payload_flush_blocking_from_address(
                    header_addr, sizeof(PACKET_HEADER_TYPE));
            }
        }
    }

    // Combined: send payload + header
    template <bool BENCHMARK_MODE = false>
    FORCE_INLINE void send_payload_with_header(
        void* conn_ptr, uint8_t idx, uint32_t payload_addr, size_t payload_size, uint32_t header_addr) {
        if constexpr (BENCHMARK_MODE) {
            auto* conn = static_cast<WorkerToFabricEdmSender*>(conn_ptr);
            if (payload_size > 0) {
                conn->send_payload_without_header_non_blocking_from_address(payload_addr, payload_size);
            }
            conn->send_payload_flush_non_blocking_from_address(header_addr, sizeof(PACKET_HEADER_TYPE));
        } else {
            if (is_mux[idx]) {
                auto* conn = static_cast<MuxConnectionType*>(conn_ptr);
                if (payload_size > 0) {
                    conn->send_payload_without_header_non_blocking_from_address(payload_addr, payload_size);
                }
                conn->send_payload_flush_non_blocking_from_address(header_addr, sizeof(PACKET_HEADER_TYPE));
            } else {
                auto* conn = static_cast<WorkerToFabricEdmSender*>(conn_ptr);
                if (payload_size > 0) {
                    conn->send_payload_without_header_non_blocking_from_address(payload_addr, payload_size);
                }
                conn->send_payload_flush_non_blocking_from_address(header_addr, sizeof(PACKET_HEADER_TYPE));
            }
        }
    }
};

// Line sync for each fabric connection (used by SyncKernelConfig)
struct LineSyncConfig {
    LineSyncConfig(
        FabricConnectionArray* connection_array,
        uint8_t connection_idx,
        const uint32_t packet_header_address,
        const uint32_t line_sync_val) :
        connection_manager_(connection_array), connection_idx_(connection_idx), line_sync_val(line_sync_val) {
        packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_address);

        // Cache connection pointer during initialization
        if (connection_manager_->is_mux[connection_idx_]) {
            connection_ptr_ = &connection_manager_->get_mux_connection(connection_idx_);
        } else {
            connection_ptr_ = &connection_manager_->get_fabric_connection(connection_idx_);
        }
    }

    template <bool IS_2D_FABRIC, ChipSendType CHIP_SEND_TYPE>
    void setup_packet_header(size_t& arg_idx, uint32_t packet_header_address) {
        // setup header fields. 2 rt args for 1D
        ChipSendTypeHandler<CHIP_SEND_TYPE, IS_2D_FABRIC>::parse_and_setup(
            arg_idx, packet_header_address, packet_header);

        // set up noc fields, 4 rt args
        auto fields = NocUnicastAtomicIncFields::build_from_args<true>(arg_idx);
        line_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fields.dst_address);

        uint64_t noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_address);
        packet_header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{noc_addr, fields.atomic_inc_val});
    }

    void global_sync_start() {
        connection_manager_->wait_for_empty_write_slot<false>(connection_ptr_, connection_idx_);
        connection_manager_->send_header_non_blocking<false>(connection_ptr_, connection_idx_, (uint32_t)packet_header);
    }

    void global_sync_finish(uint8_t sync_iter) {
        // sync wait
        noc_semaphore_wait_min(line_sync_ptr, line_sync_val * (sync_iter + 1));
    }

private:
    FabricConnectionArray* connection_manager_;
    void* connection_ptr_;    // Cached connection pointer
    uint8_t connection_idx_;  // Index into the connection array
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header;
    volatile tt_l1_ptr uint32_t* line_sync_ptr;
    uint32_t line_sync_val;
};

template <bool IS_MASTER_CORE, uint8_t NUM_LOCAL_CORES>
struct LocalSyncConfig {
    LocalSyncConfig(const uint32_t sync_address, const uint32_t sync_val) :
        sync_address(sync_address), sync_val(sync_val) {
        sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_address);
    }

    void setup_core_coordinates(size_t& arg_idx) {
        // Get core coordinates from runtime args
        for (uint8_t i = 0; i < NUM_LOCAL_CORES; i++) {
            sync_core_xy_encoding_[i] = get_local_arg_val<uint32_t>(arg_idx++);
        }
    }

    void local_sync(uint8_t sync_iter) {
        if constexpr (IS_MASTER_CORE) {
            // Master core: signal all local cores
            for (uint8_t i = 0; i < NUM_LOCAL_CORES; i++) {
                auto dest_noc_addr = get_noc_addr_helper(sync_core_xy_encoding_[i], sync_address);
                noc_semaphore_inc(dest_noc_addr, 1);
            }
            // Wait for all local cores to acknowledge
            uint32_t expected_val = NUM_LOCAL_CORES * (sync_iter + 1);
            noc_semaphore_wait(sync_ptr, expected_val);
        } else {
            uint32_t expected_val = sync_iter + 1;
            noc_semaphore_wait(sync_ptr, expected_val);
            // send ack back to master sender
            auto master_sender_noc_addr = get_noc_addr_helper(sync_core_xy_encoding_[0], sync_address);
            noc_semaphore_inc(master_sender_noc_addr, 1);
        }
    }

private:
    std::array<uint32_t, NUM_LOCAL_CORES> sync_core_xy_encoding_;
    uint32_t sync_address;
    volatile tt_l1_ptr uint32_t* sync_ptr;
    uint32_t sync_val;
};

struct SenderCreditInfo {
    SenderCreditInfo() = default;

    static SenderCreditInfo build_from_args(size_t& arg_idx) { return SenderCreditInfo(arg_idx); }

    uint32_t expected_receiver_count = 0;
    uint32_t credit_reception_address_base = 0;  // Base address of credit chunk (for mcast)
    uint32_t initial_credits = 0;

private:
    SenderCreditInfo(size_t& arg_idx) {
        this->expected_receiver_count = get_local_arg_val<uint32_t>(arg_idx++);
        this->credit_reception_address_base = get_local_arg_val<uint32_t>(arg_idx++);
        this->initial_credits = get_local_arg_val<uint32_t>(arg_idx++);
    }
};

// Helper class to manage sender-side credit consumption
// Encapsulates all credit checking and consumption logic in one place
struct SenderCreditManager {
    SenderCreditManager() = default;

    // Initialize from args
    void init(size_t& arg_idx, uint32_t total_credits) {
        enabled_ = get_local_arg_val<uint32_t>(arg_idx++) != 0;
        if (!enabled_) {
            return;
        }

        sender_credit_info_ = SenderCreditInfo::build_from_args(arg_idx);
        credit_semaphores_base_ptr_ =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sender_credit_info_.credit_reception_address_base);
        credit_semaphores_base_noc_addr_ = get_noc_addr(0) + sender_credit_info_.credit_reception_address_base;

        num_receivers_ = sender_credit_info_.expected_receiver_count;
        initial_credits_ = sender_credit_info_.initial_credits;
        estimated_available_credits_ = initial_credits_;
        prev_processed_credits_ = 0;
        total_credits_ = total_credits;

        ASSERT(num_receivers_ > 0);
        ASSERT(credit_semaphores_base_ptr_ != nullptr);
    }

    // Initialize credit semaphores
    void initialize() {
        if (!enabled_) {
            return;
        }

        for (uint32_t i = 0; i < num_receivers_; i++) {
            credit_semaphores_base_ptr_[i * CREDIT_STRIDE_WORDS] = 0;
        }
    }

    // Check if credits available (non-blocking, called before send)
    FORCE_INLINE bool has_credits_available(uint32_t num_packets_processed) const {
        if (!enabled_) {
            return true;  // Always available when disabled
        }

        // Fast path: if we think we have credits, return true
        if (estimated_available_credits_ > 0) {
            return true;
        }

        return const_cast<SenderCreditManager*>(this)->update_available_credits();
    }

    // Update available credits by checking all receivers (called when blocked)
    bool update_available_credits() {
        if (!enabled_) {
            return true;
        }

        invalidate_l1_cache();

        // Find minimum credits across all receivers (slowest receiver determines limit)
        uint32_t min_credits = credit_semaphores_base_ptr_[0];

        for (uint32_t i = 1; i < num_receivers_; i++) {
            uint32_t recv_credits = credit_semaphores_base_ptr_[i * CREDIT_STRIDE_WORDS];
            if (recv_credits < min_credits) {
                min_credits = recv_credits;
            }
        }

        int32_t new_credits = min_credits - prev_processed_credits_;
        if (new_credits <= 0) {
            return false;  // No new credits available
        }

        estimated_available_credits_ = new_credits;
        prev_processed_credits_ += new_credits;
        return true;
    }

    // Consume one credit (called after successful send - decrements ALL receivers for mcast)
    FORCE_INLINE void consume_credit() {
        if (!enabled_) {
            return;
        }

        ASSERT(estimated_available_credits_ > 0);
        estimated_available_credits_--;
    }

    // Wait for all credits back (called at connection close)
    bool got_all_credits_back() {
        if (!enabled_) {
            return true;
        }

        if (!got_all_credits_back_) {
            invalidate_l1_cache();
            got_all_credits_back_ = true;

            for (uint32_t i = 0; i < num_receivers_; i++) {
                if (credit_semaphores_base_ptr_[i * CREDIT_STRIDE_WORDS] < total_credits_) {
                    got_all_credits_back_ = false;
                    break;
                }
            }
        }

        return got_all_credits_back_;
    }

    bool is_enabled() const { return enabled_; }

private:
    bool enabled_ = false;
    SenderCreditInfo sender_credit_info_;

    // Per-receiver credit tracking
    volatile tt_l1_ptr uint32_t* credit_semaphores_base_ptr_ = nullptr;
    uint64_t credit_semaphores_base_noc_addr_ = 0;
    uint32_t num_receivers_ = 0;
    uint32_t initial_credits_ = 0;
    uint32_t total_credits_ = 0;
    uint32_t estimated_available_credits_ = 0;
    uint32_t prev_processed_credits_ = 0;
    bool got_all_credits_back_ = false;

    static constexpr uint32_t CREDIT_ADDRESS_STRIDE = 16;
    static constexpr uint32_t CREDIT_STRIDE_WORDS = CREDIT_ADDRESS_STRIDE / sizeof(uint32_t);
};

struct SenderKernelTrafficConfig {
    SenderKernelTrafficConfig(
        FabricConnectionArray* connection_array,
        uint8_t connection_idx,
        const SenderTrafficConfigMetadata& metadata,
        const uint32_t packet_header_address) :
        connection_manager_(connection_array),
        connection_idx_(connection_idx),
        metadata(metadata),
        noc_send_type_(static_cast<NocSendType>(0)),
        payload_buffer_(nullptr) {
        packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_address);

        // Cache connection pointer during initialization
        if (connection_manager_->is_mux[connection_idx_]) {
            connection_ptr_ = &connection_manager_->get_mux_connection(connection_idx_);
        } else {
            connection_ptr_ = &connection_manager_->get_fabric_connection(connection_idx_);
        }

        // Initialize function pointers to null (will be set in parse_and_setup_noc_send_type)
        noc_ops_.parse_and_setup = nullptr;
        noc_ops_.update_header = nullptr;
    }

    template <bool IS_2D_FABRIC>
    void parse_and_setup_chip_send_type(size_t& arg_idx, uint32_t packet_header_address) {
        ChipSendType chip_send_type = static_cast<ChipSendType>(get_local_arg_val<uint32_t>(arg_idx++));

        if (chip_send_type == ChipSendType::CHIP_UNICAST) {
            ChipSendTypeHandler<ChipSendType::CHIP_UNICAST, IS_2D_FABRIC>::parse_and_setup(
                arg_idx, packet_header_address, packet_header);
        } else if (chip_send_type == ChipSendType::CHIP_MULTICAST) {
            ChipSendTypeHandler<ChipSendType::CHIP_MULTICAST, IS_2D_FABRIC>::parse_and_setup(
                arg_idx, packet_header_address, packet_header);
        } else {
            ASSERT(false);
        }
    }

    void parse_and_setup_noc_send_type(size_t& arg_idx) {
        uint32_t noc_type_raw = get_local_arg_val<uint32_t>(arg_idx++);
        noc_send_type_ = static_cast<NocSendType>(noc_type_raw);

        // Validate NOC send type and set up operations
        switch (noc_send_type_) {
            case NocSendType::NOC_UNICAST_WRITE:
                noc_ops_.parse_and_setup = NocWriteSenderOperations::parse_and_setup_impl;
                noc_ops_.update_header = NocWriteSenderOperations::update_header_impl;
                break;
            case NocSendType::NOC_UNICAST_ATOMIC_INC:
                noc_ops_.parse_and_setup = NocAtomicSenderOperations::parse_and_setup_impl;
                noc_ops_.update_header = NocAtomicSenderOperations::update_header_impl;
                break;
            case NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC:
                noc_ops_.parse_and_setup = NocFusedSenderOperations::parse_and_setup_impl;
                noc_ops_.update_header = NocFusedSenderOperations::update_header_impl;
                break;
            case NocSendType::NOC_UNICAST_SCATTER_WRITE:
                noc_ops_.parse_and_setup = NocScatterWriteSenderOperations::parse_and_setup_impl;
                noc_ops_.update_header = NocScatterWriteSenderOperations::update_header_impl;
                break;
            default: ASSERT(false); break;
        }

        ASSERT(noc_ops_.parse_and_setup != nullptr);
        ASSERT(noc_ops_.update_header != nullptr);

        noc_ops_.parse_and_setup(this, arg_idx);
    }

    void setup_payload_buffer(uint32_t payload_buffer_address, uint32_t payload_buffer_size) {
        if (payload_size_bytes > 0) {
            payload_buffer_ = new (payload_buffer_storage.data())
                SenderPayloadBuffer(payload_buffer_address, payload_buffer_size, this->payload_size_bytes);
        } else {
            payload_buffer_ = nullptr;
        }
    }

    bool has_packets_to_send() const { return num_packets_processed < metadata.num_packets; }

    // Send exactly one packet per call (round-robin scheduling)
    // Returns: true if packet was sent, false if blocked (no credits)
    template <bool BENCHMARK_MODE>
    bool send_one_packet() {
        // STEP 1: Check credits BEFORE sending (non-benchmark mode only)
        if constexpr (!BENCHMARK_MODE) {
            if (!credit_manager_.has_credits_available(num_packets_processed)) {
                return false;  // No credits available - blocked
            }
        }

        // STEP 2: Wait for space
        connection_manager_->wait_for_empty_write_slot<BENCHMARK_MODE>(connection_ptr_, connection_idx_);

        // STEP 3: Send packet
        if constexpr (!BENCHMARK_MODE) {
            if (payload_size_bytes > 0 && payload_buffer_) {
                payload_buffer_->fill_data(metadata.seed);

                // Send payload without header
                connection_manager_->send_payload_without_header<BENCHMARK_MODE>(
                    connection_ptr_, connection_idx_, payload_buffer_->get_physical_address(), payload_size_bytes);
            }
        }

        // Send header
        connection_manager_->send_header_non_blocking<BENCHMARK_MODE>(
            connection_ptr_, connection_idx_, (uint32_t)packet_header);

        // STEP 4: Update state (after successful send)
        if constexpr (!BENCHMARK_MODE) {
            // avoid race condition where we update the ptrs but fabric write is not done yet.
            noc_async_writes_flushed();

            if (payload_size_bytes > 0 && payload_buffer_) {
                payload_buffer_->advance();
                update_header_for_next_packet();
                metadata.seed = prng_next(metadata.seed);
            }

            // STEP 5: Consume credit AFTER successful send
            credit_manager_.consume_credit();
        }

        num_packets_processed += 1;  // Always increment by 1

        return true;  // Packet sent successfully
    }

    void advance_dst_address() {
        if (payload_buffer_) {
            payload_buffer_->advance();
            update_header_for_next_packet();
        }
    }

    void reset_dst_address() {
        if (payload_buffer_) {
            payload_buffer_->reset();
            update_header_for_next_packet();
        }
    }

    bool has_wrapped() const { return payload_buffer_ ? payload_buffer_->has_wrapped() : false; }

    // Friend classes for operation implementations
    friend struct NocWriteSenderOperations;
    friend struct NocAtomicSenderOperations;
    friend struct NocFusedSenderOperations;
    friend struct NocScatterWriteSenderOperations;

private:
    void update_header_for_next_packet() {
        if (payload_buffer_) {
            noc_ops_.update_header(this);
        }
    }

public:
    FabricConnectionArray* connection_manager_;
    void* connection_ptr_;    // Cached connection pointer
    uint8_t connection_idx_;  // Index into the connection array

    SenderTrafficConfigMetadata metadata;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header;
    uint32_t payload_size_bytes = 0;
    uint32_t num_packets_processed = 0;
    uint64_t elapsed_cycles = 0;

    SenderCreditManager credit_manager_;

private:
    NocSendType noc_send_type_;
    NocOperationTypes::Operations noc_ops_;

    union NocFields {
        NocUnicastWriteFields write_fields;
        NocUnicastAtomicIncFields atomic_inc_fields;
        NocUnicastWriteAtomicIncFields write_atomic_inc_fields;
        NocUnicastScatterWriteFields scatter_write_fields;

        // Constructor needed because member types have user-defined constructors
        NocFields() {}  // Will be properly initialized later based on NOC type
    } noc_fields_;

    alignas(SenderPayloadBuffer) std::array<char, sizeof(SenderPayloadBuffer)> payload_buffer_storage;
    SenderPayloadBuffer* payload_buffer_;
};

// NOC Operation Implementations (now that SenderKernelTrafficConfig is fully defined)
inline void NocWriteSenderOperations::parse_and_setup_impl(SenderKernelTrafficConfig* config, size_t& arg_idx) {
    auto fields = NocUnicastWriteFields::build_from_args<true>(arg_idx);

    uint64_t noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_address);
    config->packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_addr}, fields.payload_size_bytes);

    config->noc_fields_.write_fields = fields;
    config->payload_size_bytes = fields.payload_size_bytes;
}

inline void NocWriteSenderOperations::update_header_impl(SenderKernelTrafficConfig* config) {
    const auto& fields = config->noc_fields_.write_fields;
    uint32_t buffer_offset = config->payload_buffer_->get_current_offset();
    uint32_t dest_address = fields.dst_address + buffer_offset;
    uint64_t noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, dest_address);
    config->packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_addr}, fields.payload_size_bytes);
}

inline void NocAtomicSenderOperations::parse_and_setup_impl(SenderKernelTrafficConfig* config, size_t& arg_idx) {
    auto fields = NocUnicastAtomicIncFields::build_from_args<true>(arg_idx);

    uint64_t noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_address);
    config->packet_header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{noc_addr, fields.atomic_inc_val});

    config->noc_fields_.atomic_inc_fields = fields;
    config->payload_size_bytes = 0;
}

inline void NocAtomicSenderOperations::update_header_impl(SenderKernelTrafficConfig* config) {
    // No-op - atomic operations use fixed addresses
}

inline void NocFusedSenderOperations::parse_and_setup_impl(SenderKernelTrafficConfig* config, size_t& arg_idx) {
    auto fields = NocUnicastWriteAtomicIncFields::build_from_args<true>(arg_idx);

    uint64_t write_noc_addr =
        get_noc_addr_helper(fields.write_fields.dst_noc_encoding, fields.write_fields.dst_address);
    uint64_t atomic_noc_addr =
        get_noc_addr_helper(fields.atomic_inc_fields.dst_noc_encoding, fields.atomic_inc_fields.dst_address);

    config->packet_header->to_noc_fused_unicast_write_atomic_inc(
        NocUnicastAtomicIncFusedCommandHeader{write_noc_addr, atomic_noc_addr, fields.atomic_inc_fields.atomic_inc_val},
        fields.write_fields.payload_size_bytes);

    config->noc_fields_.write_atomic_inc_fields = fields;
    config->payload_size_bytes = fields.write_fields.payload_size_bytes;
}

inline void NocFusedSenderOperations::update_header_impl(SenderKernelTrafficConfig* config) {
    const auto& fields = config->noc_fields_.write_atomic_inc_fields;
    uint32_t buffer_offset = config->payload_buffer_->get_current_offset();
    uint32_t write_dest_address = fields.write_fields.dst_address + buffer_offset;
    uint64_t write_noc_addr = get_noc_addr_helper(fields.write_fields.dst_noc_encoding, write_dest_address);
    uint64_t atomic_noc_addr =
        get_noc_addr_helper(fields.atomic_inc_fields.dst_noc_encoding, fields.atomic_inc_fields.dst_address);

    config->packet_header->to_noc_fused_unicast_write_atomic_inc(
        NocUnicastAtomicIncFusedCommandHeader{write_noc_addr, atomic_noc_addr, fields.atomic_inc_fields.atomic_inc_val},
        fields.write_fields.payload_size_bytes);
}

inline void NocScatterWriteSenderOperations::parse_and_setup_impl(SenderKernelTrafficConfig* config, size_t& arg_idx) {
    auto fields = NocUnicastScatterWriteFields::build_from_args<true>(arg_idx);

    ASSERT(fields.chunk_count == NocUnicastScatterWriteFields::MAX_CHUNKS);
    const auto scatter_header = NocUnicastScatterCommandHeader(
        {
            get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_addresses[0]),
            get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_addresses[1]),
        },
        {fields.chunk_sizes[0]});

    config->packet_header->to_noc_unicast_scatter_write(scatter_header, fields.payload_size_bytes);
    config->noc_fields_.scatter_write_fields = fields;
    config->payload_size_bytes = fields.payload_size_bytes;
}

inline void NocScatterWriteSenderOperations::update_header_impl(SenderKernelTrafficConfig* config) {
    const auto& fields = config->noc_fields_.scatter_write_fields;
    uint32_t buffer_offset = config->payload_buffer_->get_current_offset();
    ASSERT(fields.chunk_count == NocUnicastScatterWriteFields::MAX_CHUNKS);

    const auto scatter_header = NocUnicastScatterCommandHeader(
        {
            get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_addresses[0] + buffer_offset),
            get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_addresses[1] + buffer_offset),
        },
        {fields.chunk_sizes[0]});
    config->packet_header->to_noc_unicast_scatter_write(scatter_header, fields.payload_size_bytes);
}

struct CommonMemoryMap {
    CommonMemoryMap() = default;
    static CommonMemoryMap build_from_args(size_t& arg_idx) { return CommonMemoryMap(arg_idx); }

    uint32_t local_args_base;
    uint32_t local_args_size;
    uint32_t result_buffer_base;
    uint32_t result_buffer_size;
    uint32_t kernel_config_base;
    uint32_t kernel_config_size;
    uint32_t mux_local_addresses_base;
    uint32_t mux_local_addresses_size;
    uint32_t mux_termination_sync_address;

private:
    CommonMemoryMap(size_t& arg_idx) {
        // Extract and initialize local args system first
        local_args_base = get_arg_val<uint32_t>(arg_idx++);
        local_args_size = get_arg_val<uint32_t>(arg_idx++);
        init_local_args(local_args_base, local_args_size);

        // Then parse the rest
        result_buffer_base = get_arg_val<uint32_t>(arg_idx++);
        result_buffer_size = get_arg_val<uint32_t>(arg_idx++);
        kernel_config_base = get_arg_val<uint32_t>(arg_idx++);
        kernel_config_size = get_arg_val<uint32_t>(arg_idx++);
        mux_local_addresses_base = get_arg_val<uint32_t>(arg_idx++);
        mux_local_addresses_size = get_arg_val<uint32_t>(arg_idx++);
        mux_termination_sync_address = get_arg_val<uint32_t>(arg_idx++);
    }
};

/* ****************************************************************************
 * MuxLocalAddresses: Standalone struct for mux connection local semaphores
 *
 * Used by both sender and receiver memory maps to allocate local L1 addresses
 * for mux connection flow control.
 * ****************************************************************************/
struct MuxLocalAddresses {
    uint32_t flow_control_address;
    uint32_t teardown_address;
    uint32_t buffer_index_address;
    uint32_t status_buffer_address;
    uint32_t sync_address;

    static MuxLocalAddresses allocate_from_base(uint32_t base_address, uint32_t address_padding_bytes) {
        uint32_t current_addr = base_address;
        uint32_t flow_control_address = current_addr;
        current_addr += address_padding_bytes;
        uint32_t teardown_address = current_addr;
        current_addr += address_padding_bytes;
        uint32_t buffer_index_address = current_addr;
        current_addr += address_padding_bytes;
        uint32_t status_buffer_address = current_addr;
        current_addr += address_padding_bytes;
        uint32_t sync_address = current_addr;

        // zero initialize all addresses
        auto* base_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(base_address);
        for (uint32_t i = 0; i < (current_addr - base_address) / sizeof(uint32_t); i++) {
            base_ptr[i] = 0;
        }

        return MuxLocalAddresses{
            flow_control_address, teardown_address, buffer_index_address, status_buffer_address, sync_address};
    }

    // Helper to calculate total size needed for one connection
    static constexpr uint32_t size_per_connection(uint32_t address_padding_bytes) {
        constexpr uint32_t num_addresses = sizeof(MuxLocalAddresses) / sizeof(uint32_t);
        return num_addresses * address_padding_bytes;
    }
};

/* ****************************************************************************
 * MuxTerminationManager: Template-based mux termination handler
 *
 * Specializations:
 * - HAS_MUX_CONNECTIONS=false: No-op (not a mux client)
 * - HAS_MUX_CONNECTIONS=true: Runtime master/subordinate role with NUM_MUXES template param
 * ****************************************************************************/
template <bool HAS_MUX_CONNECTIONS, uint8_t NUM_MUXES = 0>
struct MuxTerminationManager;

// Specialization: No mux connections
template <uint8_t NUM_MUXES>
struct MuxTerminationManager<false, NUM_MUXES> {
    MuxTerminationManager(size_t& local_args_idx, uint32_t sync_address) {
        // No args to parse
    }

    FORCE_INLINE void terminate_muxes() {
        // No-op
    }
};

// Specialization: Has mux connections (runtime determines master vs subordinate)
template <uint8_t NUM_MUXES>
struct MuxTerminationManager<true, NUM_MUXES> {
    MuxTerminationManager(size_t& local_args_idx, uint32_t sync_address) {
        is_master_ = get_local_arg_val<uint32_t>(local_args_idx++) != 0;
        total_mux_clients_ = get_local_arg_val<uint32_t>(local_args_idx++);
        uint32_t master_noc_encoding = get_local_arg_val<uint32_t>(local_args_idx++);

        if (is_master_) {
            // Master: setup sync semaphore (should be cleared by host)
            termination_sync_ptr_ = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_address);

            num_muxes_to_terminate_ = get_local_arg_val<uint32_t>(local_args_idx++);
            ASSERT(num_muxes_to_terminate_ <= NUM_MUXES);

            // Parse mux list (x, y, signal_addr triples)
            for (uint8_t i = 0; i < num_muxes_to_terminate_; i++) {
                mux_x_[i] = get_local_arg_val<uint32_t>(local_args_idx++);
                mux_y_[i] = get_local_arg_val<uint32_t>(local_args_idx++);
                mux_signal_addrs_[i] = get_local_arg_val<uint32_t>(local_args_idx++);
            }
        } else {
            // Subordinate: setup NOC address to master's sync semaphore
            master_noc_addr_ = get_noc_addr_helper(master_noc_encoding, sync_address);
        }
    }

    FORCE_INLINE void terminate_muxes() {
        if (is_master_) {
            // Wait for all subordinates (total_clients - 1, excluding self)
            noc_semaphore_wait(termination_sync_ptr_, total_mux_clients_ - 1);

            // Terminate all muxes in sequence
            for (uint8_t i = 0; i < num_muxes_to_terminate_; i++) {
                tt::tt_fabric::fabric_endpoint_terminate(mux_x_[i], mux_y_[i], mux_signal_addrs_[i]);
            }
        } else {
            // Signal the master
            noc_semaphore_inc(master_noc_addr_, 1);
            noc_async_atomic_barrier();
        }
    }

private:
    bool is_master_ = false;
    uint32_t total_mux_clients_ = 0;

    // Master members:
    volatile tt_l1_ptr uint32_t* termination_sync_ptr_ = nullptr;
    uint8_t num_muxes_to_terminate_ = 0;
    uint8_t mux_x_[NUM_MUXES];
    uint8_t mux_y_[NUM_MUXES];
    uint32_t mux_signal_addrs_[NUM_MUXES];

    // Subordinate members:
    uint64_t master_noc_addr_ = 0;
};

struct SenderKernelMemoryMap {
    static constexpr uint32_t address_padding_bytes = 16;
    // Encapsulated common memory map
    CommonMemoryMap common;

    SenderKernelMemoryMap() {}

    static SenderKernelMemoryMap build_from_args(const CommonMemoryMap& common_map, size_t& rt_args_idx) {
        return SenderKernelMemoryMap(common_map, rt_args_idx);
    }

    uint32_t get_packet_header_address() {
        uint32_t addr = curr_packet_header_address_;
        ASSERT(addr + sizeof(PACKET_HEADER_TYPE) < payload_buffer_region_base_);
        curr_packet_header_address_ += sizeof(PACKET_HEADER_TYPE);
        return addr;
    }

    uint32_t get_payload_buffer_address(uint32_t size) {
        uint32_t addr = curr_payload_buffer_address_;
        ASSERT(addr + size < highest_usable_address_);

        // TODO: ensure noc alignment
        curr_payload_buffer_address_ += size;
        return addr;
    }

    // Mux local address allocation (allocates from cursor, then advances it)
    MuxLocalAddresses get_mux_local_addresses_for_connection() {
        auto addrs = MuxLocalAddresses::allocate_from_base(curr_mux_local_address_, address_padding_bytes);
        curr_mux_local_address_ += MuxLocalAddresses::size_per_connection(address_padding_bytes);
        return addrs;
    }

private:
    SenderKernelMemoryMap(const CommonMemoryMap& common_map, size_t& rt_args_idx) {
        // Use pre-parsed common memory map and parse only sender-specific args
        common = common_map;
        packet_header_region_base_ = get_arg_val<uint32_t>(rt_args_idx++);
        payload_buffer_region_base_ = get_arg_val<uint32_t>(rt_args_idx++);
        highest_usable_address_ = get_arg_val<uint32_t>(rt_args_idx++);

        // set the current addresses to the base
        curr_packet_header_address_ = packet_header_region_base_;
        curr_payload_buffer_address_ = payload_buffer_region_base_;
        curr_mux_local_address_ = common.mux_local_addresses_base;
    }

    uint32_t packet_header_region_base_;
    uint32_t payload_buffer_region_base_;
    uint32_t highest_usable_address_;
    uint32_t curr_packet_header_address_;
    uint32_t curr_payload_buffer_address_;
    uint32_t curr_mux_local_address_;
};

// Receiver kernel memory map - for allocating credit return packet headers and mux local addresses
struct ReceiverKernelMemoryMap {
    static constexpr uint32_t address_padding_bytes = 16;

    // Encapsulated common memory map
    CommonMemoryMap common;

    ReceiverKernelMemoryMap() {}

    static ReceiverKernelMemoryMap build_from_args(const CommonMemoryMap& common_map, size_t& rt_args_idx) {
        return ReceiverKernelMemoryMap(common_map, rt_args_idx);
    }

    uint32_t get_credit_header_address() {
        uint32_t addr = curr_credit_header_address_;
        ASSERT(addr + sizeof(PACKET_HEADER_TYPE) <= credit_header_region_end_);
        curr_credit_header_address_ += sizeof(PACKET_HEADER_TYPE);
        return addr;
    }

    // Mux local address allocation (allocates from cursor, then advances it)
    MuxLocalAddresses get_mux_local_addresses_for_connection() {
        auto addrs = MuxLocalAddresses::allocate_from_base(curr_mux_local_address_, address_padding_bytes);
        curr_mux_local_address_ += MuxLocalAddresses::size_per_connection(address_padding_bytes);
        return addrs;
    }

private:
    ReceiverKernelMemoryMap(const CommonMemoryMap& common_map, size_t& rt_args_idx) {
        // Use pre-parsed common memory map and parse only receiver-specific args
        common = common_map;
        credit_header_region_base_ = get_arg_val<uint32_t>(rt_args_idx++);
        credit_header_region_end_ = get_arg_val<uint32_t>(rt_args_idx++);

        // Set the current address to the base
        curr_credit_header_address_ = credit_header_region_base_;
        curr_mux_local_address_ = common.mux_local_addresses_base;
    }

    uint32_t credit_header_region_base_;   // Start of credit header allocation region
    uint32_t credit_header_region_end_;    // End of credit header allocation region
    uint32_t curr_credit_header_address_;  // Current allocation pointer
    uint32_t curr_mux_local_address_;      // Cursor for allocating mux local addresses
};

/* Layout for the run time args for sender
1. Memory map args (unified: common + sender-specific args parsed together)
2. Fabric connection args
3. Traffic config args
3.1. TrafficConfigCommonFields
3.2. Chip send type fields
3.3. Noc send type fields
*/
template <
    uint8_t NUM_TRAFFIC_CONFIGS,
    bool IS_2D_FABRIC,
    bool LINE_SYNC,
    uint8_t NUM_LOCAL_SYNC_CORES>
struct SenderKernelConfig {
    static constexpr bool MASTER_SYNC_CORE = false;

    static SenderKernelConfig build_from_args(
        const CommonMemoryMap& common_map,
        size_t& rt_args_idx,
        size_t& local_args_idx,
        uint8_t num_fabric_connections) {
        return SenderKernelConfig(common_map, rt_args_idx, local_args_idx, num_fabric_connections);
    }

    void open_connections() {
        connections.open_all();
        // Initialize credit management for all traffic configs
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            traffic_config_ptrs[i]->credit_manager_.initialize();
        }
    }

    void local_sync(uint8_t sync_iter) {
        if constexpr (LINE_SYNC) {
            local_sync_config().local_sync(sync_iter);
        }
    }

    void close_connections() {
        // Wait for all credits to be returned before closing
        bool got_all_credits_back = false;
        while (!got_all_credits_back) {
            got_all_credits_back = true;
            for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
                got_all_credits_back &= traffic_config_ptrs[i]->credit_manager_.got_all_credits_back();
            }
        }
        connections.close_all();
    }

    SenderKernelMemoryMap memory_map;

    FabricConnectionArray connections;

    alignas(LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>)
        std::array<char, sizeof(LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>)> local_sync_config_storage;
    std::array<uint8_t, NUM_TRAFFIC_CONFIGS> traffic_config_to_fabric_connection_map;

    using TrafficConfigType = SenderKernelTrafficConfig;

    alignas(
        TrafficConfigType) std::array<char, NUM_TRAFFIC_CONFIGS * sizeof(TrafficConfigType)> traffic_configs_storage;
    std::array<TrafficConfigType*, NUM_TRAFFIC_CONFIGS> traffic_config_ptrs;

    // Helper accessors
    LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>& local_sync_config() {
        return *reinterpret_cast<LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>*>(
            local_sync_config_storage.data());
    }

    TrafficConfigType* traffic_configs(uint8_t idx) {
        return reinterpret_cast<TrafficConfigType*>(traffic_configs_storage.data() + idx * sizeof(TrafficConfigType));
    }

    const std::array<TrafficConfigType*, NUM_TRAFFIC_CONFIGS>& traffic_config_ptrs_array() const {
        return traffic_config_ptrs;
    }

    // Result buffer convenience methods
    uint32_t get_result_buffer_address() const { return memory_map.common.result_buffer_base; }
    uint32_t get_result_buffer_size() const { return memory_map.common.result_buffer_size; }

private:
    SenderKernelConfig(
        const CommonMemoryMap& common_map,
        size_t& rt_args_idx,
        size_t& local_args_idx,
        uint8_t num_fabric_connections) {
        // Parse memory map args from runtime args using pre-parsed common map
        this->memory_map = SenderKernelMemoryMap::build_from_args(common_map, rt_args_idx);

        // Parse all fabric connections using unified array (memory map needed for mux local addresses)
        connections.num_connections = num_fabric_connections;
        connections.template parse_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx, this->memory_map);

        // add line sync initializations here, for each fabric connection, ex, forward and backward connection, run line
        // sync for all.
        if constexpr (LINE_SYNC) {
            uint32_t sync_address = get_local_arg_val<uint32_t>(local_args_idx++);
            uint32_t sync_val = get_local_arg_val<uint32_t>(local_args_idx++);
            new (&local_sync_config()) LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>(sync_address, sync_val);

            // setup core coordinates
            local_sync_config().setup_core_coordinates(local_args_idx);
        }
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            traffic_config_to_fabric_connection_map[i] = get_local_arg_val<uint32_t>(local_args_idx++);
        }

        // Initialize traffic config pointers
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            traffic_config_ptrs[i] = nullptr;
        }

        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            auto metadata = SenderTrafficConfigMetadata::build_from_args(local_args_idx);
            const auto fabric_connection_idx = traffic_config_to_fabric_connection_map[i];
            ASSERT(fabric_connection_idx < connections.num_connections);

            uint32_t packet_header_address = this->memory_map.get_packet_header_address();

            // Get pointer to pre-allocated storage and initialize with placement new
            TrafficConfigType* config_ptr = traffic_configs(i);
            traffic_config_ptrs[i] = config_ptr;

            // Initialize traffic config with connection array pointer and index
            new (config_ptr) TrafficConfigType(&connections, fabric_connection_idx, metadata, packet_header_address);

            traffic_config_ptrs[i]->template parse_and_setup_chip_send_type<IS_2D_FABRIC>(
                local_args_idx, packet_header_address);

            traffic_config_ptrs[i]->parse_and_setup_noc_send_type(local_args_idx);

            // Initialize credit manager (parses credit_management_enabled + SenderCreditInfo)
            traffic_config_ptrs[i]->credit_manager_.init(local_args_idx, metadata.num_packets);

            // the payload buffer size here is the virtual size of the buffer, not the physical size
            // this virtual size is used to keep track of the physical buffer on the receiver side
            // on the sender side, the physical buffer will only be the size of the payload
            uint32_t payload_buffer_size = metadata.payload_buffer_size;
            uint32_t payload_buffer_address =
                this->memory_map.get_payload_buffer_address(traffic_config_ptrs[i]->payload_size_bytes);
            traffic_config_ptrs[i]->setup_payload_buffer(payload_buffer_address, payload_buffer_size);
        }
    };
};

// Helper class to manage credit accumulation and return
// Encapsulates all credit batching logic in one place
// Works with FabricConnectionArray (supports both direct and mux connections)
struct ReceiverCreditManager {
    ReceiverCreditManager() : credit_fields_(0, 0, 0) {}

    template <bool IS_2D_FABRIC>
    void setup_packet_header(size_t& arg_idx, uint32_t packet_header_address) {
        ChipSendTypeHandler<ChipSendType::CHIP_UNICAST, IS_2D_FABRIC>::parse_and_setup(
            arg_idx, packet_header_address, packet_header_);

        credit_fields_ = NocUnicastAtomicIncFields::build_from_args<true>(arg_idx);
        uint64_t noc_addr = get_noc_addr_helper(credit_fields_.dst_noc_encoding, credit_fields_.dst_address);
        packet_header_->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{noc_addr, credit_fields_.atomic_inc_val});
    }

    // Initialize with credit info and fabric connection array
    template <bool IS_2D_FABRIC>
    void init(
        size_t& arg_idx, FabricConnectionArray* connections, uint8_t connection_idx, uint32_t credit_header_address) {
        connection_manager_ = connections;
        connection_idx_ = connection_idx;
        accumulated_credits_ = 0;
        enabled_ = true;

        // Cache connection pointer during initialization
        if (connection_manager_->is_mux[connection_idx_]) {
            connection_ptr_ = &connection_manager_->get_mux_connection(connection_idx_);
        } else {
            connection_ptr_ = &connection_manager_->get_fabric_connection(connection_idx_);
        }

        packet_header_ = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(credit_header_address);
        setup_packet_header<IS_2D_FABRIC>(arg_idx, credit_header_address);
    }

    // Called after each packet is processed
    FORCE_INLINE void accumulate_and_maybe_send() {
        if (!enabled_) {
            return;
        }

        accumulated_credits_++;

        // Send credits in batches for efficiency
        if (accumulated_credits_ >= credit_fields_.atomic_inc_val) {
            send_credits();
            accumulated_credits_ = 0;
        }
    }

    // Called at end to flush remaining credits
    FORCE_INLINE void flush_remaining() {
        if (enabled_ && accumulated_credits_ > 0) {
            send_credits(accumulated_credits_);
            accumulated_credits_ = 0;
        }
    }

private:
    FORCE_INLINE void send_credits() {
        connection_manager_->wait_for_empty_write_slot<false>(connection_ptr_, connection_idx_);
        connection_manager_->send_header_non_blocking<false>(
            connection_ptr_, connection_idx_, (uint32_t)packet_header_);
    }

    FORCE_INLINE void send_credits(uint32_t num_credits) {
        // flush writes before updating the header to avoid race conditions
        noc_async_writes_flushed();

        uint64_t noc_addr = get_noc_addr_helper(credit_fields_.dst_noc_encoding, credit_fields_.dst_address);
        packet_header_->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{noc_addr, num_credits});

        connection_manager_->wait_for_empty_write_slot<false>(connection_ptr_, connection_idx_);
        connection_manager_->send_header_flush_blocking<false>(
            connection_ptr_, connection_idx_, (uint32_t)packet_header_);
    }

    FabricConnectionArray* connection_manager_ = nullptr;
    void* connection_ptr_ = nullptr;  // Cached connection pointer
    uint8_t connection_idx_ = 0;
    uint32_t accumulated_credits_ = 0;
    bool enabled_ = false;
    NocUnicastAtomicIncFields credit_fields_;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header_;
};

struct ReceiverTrafficConfigMetadata {
    static ReceiverTrafficConfigMetadata build_from_args(size_t& arg_idx) {
        return ReceiverTrafficConfigMetadata(arg_idx);
    }

    ReceiverTrafficConfigMetadata(const ReceiverTrafficConfigMetadata& other) :
        num_packets(other.num_packets), seed(other.seed), payload_buffer_size(other.payload_buffer_size) {}

    uint32_t num_packets = 0;
    uint32_t seed = 0;
    uint32_t payload_buffer_size = 0;

private:
    ReceiverTrafficConfigMetadata(size_t& arg_idx) {
        this->num_packets = get_local_arg_val<uint32_t>(arg_idx++);
        this->seed = get_local_arg_val<uint32_t>(arg_idx++);
        this->payload_buffer_size = get_local_arg_val<uint32_t>(arg_idx++);
    }
};

/*
Semantics for data validation: poll() -> validate() -> advance()
*/
struct TrafficValidationConfigBase {
    using PollFunc = bool (*)(TrafficValidationConfigBase*);
    using ValidateFunc = bool (*)(TrafficValidationConfigBase*);
    using UpdateFunc = void (*)(TrafficValidationConfigBase*);

    struct ValidationOps {
        PollFunc poll;
        ValidateFunc validate;
        UpdateFunc update;
    };

    TrafficValidationConfigBase(const ReceiverTrafficConfigMetadata& metadata) : metadata(metadata) {
        // Function pointers will be set by derived classes
        ops.poll = nullptr;
        ops.validate = nullptr;
        ops.update = nullptr;
    }

    bool has_packets_to_validate() const { return num_packets_processed < metadata.num_packets; }

    bool poll() { return ops.poll(this); }

    bool validate() { return ops.validate(this); }

    void advance() {
        num_packets_processed++;
        ops.update(this);

        // Automatically handle credit return after processing packet
        if (credit_manager_ != nullptr) {
            static_cast<ReceiverCreditManager*>(credit_manager_)->accumulate_and_maybe_send();
        }
    }

    ReceiverTrafficConfigMetadata metadata;
    uint32_t num_packets_processed = 0;
    ValidationOps ops;

    // Pointer to credit manager (set by ReceiverKernelConfig during initialization)
    void* credit_manager_ = nullptr;  // Type-erased pointer to ReceiverCreditManager
};

struct AtomicIncValidationConfig : public TrafficValidationConfigBase {
    AtomicIncValidationConfig(
        const NocUnicastAtomicIncFields& atomic_inc_fields, const ReceiverTrafficConfigMetadata& metadata) :
        TrafficValidationConfigBase(metadata) {
        // Set up function pointers
        ops.poll = poll_impl;
        ops.validate = validate_impl;
        ops.update = update_impl;

        poll_address = reinterpret_cast<tt_l1_ptr uint32_t*>(atomic_inc_fields.dst_address);
        value_step_size = atomic_inc_fields.atomic_inc_val;

        // set the initial expected value equal to the step size
        expected_value = value_step_size;
    }

    static bool poll_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<AtomicIncValidationConfig*>(base_config);
        uint32_t current_value = *config->poll_address;
        return current_value >= config->expected_value;
    }

    static bool validate_impl(TrafficValidationConfigBase* base_config) {
        return true;  // no-op for atomic incs
    }

    static void update_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<AtomicIncValidationConfig*>(base_config);
        config->expected_value += config->value_step_size;
    }

    volatile tt_l1_ptr uint32_t* poll_address;
    uint32_t expected_value;
    uint32_t value_step_size;
};

struct WriteValidationConfig : public TrafficValidationConfigBase {
    WriteValidationConfig(const NocUnicastWriteFields& write_fields, const ReceiverTrafficConfigMetadata& metadata) :
        TrafficValidationConfigBase(metadata) {
        // Set up function pointers
        ops.poll = poll_impl;
        ops.validate = validate_impl;
        ops.update = update_impl;

        payload_buffer_ = new (payload_buffer_storage.data()) ReceiverPayloadBuffer(
            write_fields.dst_address, metadata.payload_buffer_size, write_fields.payload_size_bytes);
    }

    static bool poll_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<WriteValidationConfig*>(base_config);
        return config->payload_buffer_->poll_for_data(config->metadata.seed);
    }

    static bool validate_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<WriteValidationConfig*>(base_config);
        return config->payload_buffer_->validate_data(config->metadata.seed);
    }

    static void update_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<WriteValidationConfig*>(base_config);
        config->metadata.seed = prng_next(config->metadata.seed);
        config->payload_buffer_->advance();
    }

    alignas(ReceiverPayloadBuffer) std::array<char, sizeof(ReceiverPayloadBuffer)> payload_buffer_storage;
    ReceiverPayloadBuffer* payload_buffer_;
};

struct WriteAtomicIncValidationConfig : public TrafficValidationConfigBase {
    WriteAtomicIncValidationConfig(
        const NocUnicastWriteAtomicIncFields& write_atomic_inc_fields, const ReceiverTrafficConfigMetadata& metadata) :
        TrafficValidationConfigBase(metadata) {
        // Set up function pointers
        ops.poll = poll_impl;
        ops.validate = validate_impl;
        ops.update = update_impl;

        const auto& write_fields = write_atomic_inc_fields.write_fields;
        const auto& atomic_fields = write_atomic_inc_fields.atomic_inc_fields;

        payload_buffer_ = new (payload_buffer_storage.data()) ReceiverPayloadBuffer(
            write_fields.dst_address, metadata.payload_buffer_size, write_fields.payload_size_bytes);

        atomic_inc_address = reinterpret_cast<tt_l1_ptr uint32_t*>(atomic_fields.dst_address);
        atomic_inc_val = atomic_fields.atomic_inc_val;
        expected_atomic_value = atomic_inc_val;
    }

    static bool poll_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<WriteAtomicIncValidationConfig*>(base_config);

        // Check atomic increment first
        uint32_t atomic_value = *config->atomic_inc_address;
        if (atomic_value < config->expected_atomic_value) {
            return false;
        }

        return config->payload_buffer_->poll_for_data(config->metadata.seed);
    }

    static bool validate_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<WriteAtomicIncValidationConfig*>(base_config);
        // Atomic validation is implicit (polling confirms it completed)
        return config->payload_buffer_->validate_data(config->metadata.seed);
    }

    static void update_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<WriteAtomicIncValidationConfig*>(base_config);
        config->metadata.seed = prng_next(config->metadata.seed);

        config->expected_atomic_value += config->atomic_inc_val;

        config->payload_buffer_->advance();
    }

    alignas(ReceiverPayloadBuffer) std::array<char, sizeof(ReceiverPayloadBuffer)> payload_buffer_storage;
    ReceiverPayloadBuffer* payload_buffer_;
    volatile tt_l1_ptr uint32_t* atomic_inc_address;
    uint32_t atomic_inc_val;
    uint32_t expected_atomic_value;
};

struct ScatterWriteValidationConfig : public TrafficValidationConfigBase {
    ScatterWriteValidationConfig(
        const NocUnicastScatterWriteFields& scatter_write_fields, const ReceiverTrafficConfigMetadata& metadata) :
        TrafficValidationConfigBase(metadata) {
        // Set up function pointers
        ops.poll = poll_impl;
        ops.validate = validate_impl;
        ops.update = update_impl;

        // Store base addresses and chunk sizes
        for (uint32_t i = 0; i < NocUnicastScatterWriteFields::MAX_CHUNKS; i++) {
            base_dst_addresses[i] = scatter_write_fields.dst_addresses[i];
            dst_addresses[i] = scatter_write_fields.dst_addresses[i];
        }
        for (uint32_t i = 0; i < NocUnicastScatterWriteFields::MAX_CHUNKS - 1; i++) {
            chunk_sizes[i] = scatter_write_fields.chunk_sizes[i];
        }

        // Last chunk size is implicit (remaining payload)
        uint32_t chunk_size = 0;
        for (uint32_t i = 0; i < NocUnicastScatterWriteFields::MAX_CHUNKS - 1; i++) {
            chunk_size += chunk_sizes[i];
        }
        last_chunk_size = scatter_write_fields.payload_size_bytes - chunk_size;

        payload_size_bytes = scatter_write_fields.payload_size_bytes;
        current_offset = 0;
    }

    static bool poll_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<ScatterWriteValidationConfig*>(base_config);

        // Check if all chunks have been written by polling the last word of each chunk
        uint32_t offset = 0;
        for (uint32_t i = 0; i < NocUnicastScatterWriteFields::MAX_CHUNKS - 1; i++) {
            uint32_t chunk_size = config->chunk_sizes[i];
            if (!SequentialDataPattern::poll(
                    config->dst_addresses[i], chunk_size, config->metadata.seed + offset / sizeof(uint32_t))) {
                return false;
            }
            offset += chunk_size;
        }

        // Check the last chunk
        if (!SequentialDataPattern::poll(
                config->dst_addresses[NocUnicastScatterWriteFields::MAX_CHUNKS - 1],
                config->last_chunk_size,
                config->metadata.seed + offset / sizeof(uint32_t))) {
            return false;
        }

        return true;
    }

    static bool validate_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<ScatterWriteValidationConfig*>(base_config);

        // Validate all chunks
        uint32_t offset = 0;
        for (uint32_t i = 0; i < NocUnicastScatterWriteFields::MAX_CHUNKS - 1; i++) {
            uint32_t chunk_size = config->chunk_sizes[i];
            if (!SequentialDataPattern::validate(
                    config->dst_addresses[i], chunk_size, config->metadata.seed + offset / sizeof(uint32_t))) {
                return false;
            }
            offset += chunk_size;
        }

        // Validate the last chunk
        if (!SequentialDataPattern::validate(
                config->dst_addresses[NocUnicastScatterWriteFields::MAX_CHUNKS - 1],
                config->last_chunk_size,
                config->metadata.seed + offset / sizeof(uint32_t))) {
            return false;
        }

        return true;
    }

    static void update_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<ScatterWriteValidationConfig*>(base_config);
        config->metadata.seed = prng_next(config->metadata.seed);

        // Advance buffer offset (similar to ReceiverPayloadBuffer::advance())
        // Need to check if we have enough space in the buffer for the next payload
        config->current_offset += config->payload_size_bytes;
        if (config->current_offset + config->payload_size_bytes > config->metadata.payload_buffer_size) {
            config->current_offset = 0;  // Wrap around
        }

        // Update all destination addresses based on new offset
        for (uint32_t i = 0; i < NocUnicastScatterWriteFields::MAX_CHUNKS; i++) {
            config->dst_addresses[i] = config->base_dst_addresses[i] + config->current_offset;
        }
    }

    std::array<uint32_t, NocUnicastScatterWriteFields::MAX_CHUNKS> base_dst_addresses;
    std::array<uint32_t, NocUnicastScatterWriteFields::MAX_CHUNKS> dst_addresses;
    std::array<uint16_t, NocUnicastScatterWriteFields::MAX_CHUNKS - 1> chunk_sizes;
    uint32_t last_chunk_size;
    uint32_t payload_size_bytes;
    uint32_t current_offset;
};

/* Layout for the run time args for receiver
1. Memory map args (unified: result buffer only, as receivers don't allocate memory)
2. Traffic config args
2.1. TrafficConfigCommonFields
2.2. Noc send type fields
*/
template <uint8_t NUM_TRAFFIC_CONFIGS, uint8_t NUM_CREDIT_CONNECTIONS, bool IS_2D_FABRIC>
struct ReceiverKernelConfig {
    static ReceiverKernelConfig build_from_args(
        const CommonMemoryMap& common_map, size_t& rt_args_idx, size_t& local_args_idx) {
        return ReceiverKernelConfig(common_map, rt_args_idx, local_args_idx);
    }

    // Result buffer convenience methods
    uint32_t get_result_buffer_address() const { return memory_map.common.result_buffer_base; }
    uint32_t get_result_buffer_size() const { return memory_map.common.result_buffer_size; }

    // Traffic config accessor
    TrafficValidationConfigBase** traffic_configs() { return traffic_configs_.data(); }

    // Credit connection lifecycle methods
    void open_credit_connections() { credit_connections.open_all(); }

    void close_credit_connections() {
        // Automatically flush any remaining credits before closing
        flush_remaining_credits();
        credit_connections.close_all();
    }

private:
    // Flush any remaining accumulated credits (called automatically by close_credit_connections)
    void flush_remaining_credits() {
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            credit_managers_[i].flush_remaining();
        }
    }

    ReceiverKernelMemoryMap memory_map;
    FabricConnectionArray credit_connections;
    std::array<uint8_t, NUM_TRAFFIC_CONFIGS> traffic_config_to_credit_connection_map;

    // Credit managers - one per traffic config
    std::array<ReceiverCreditManager, NUM_TRAFFIC_CONFIGS> credit_managers_;

    constexpr static size_t MAX_VALIDATION_CONFIG_SIZE = std::max(
        {sizeof(WriteValidationConfig),
         sizeof(AtomicIncValidationConfig),
         sizeof(WriteAtomicIncValidationConfig),
         sizeof(ScatterWriteValidationConfig)});

    alignas(TrafficValidationConfigBase)
        std::array<char, NUM_TRAFFIC_CONFIGS * MAX_VALIDATION_CONFIG_SIZE> validation_configs_storage;
    std::array<TrafficValidationConfigBase*, NUM_TRAFFIC_CONFIGS> traffic_configs_;

private:
    ReceiverKernelConfig(const CommonMemoryMap& common_map, size_t& rt_args_idx, size_t& local_args_idx) {
        // Parse receiver-specific memory map (includes credit header region)
        this->memory_map = ReceiverKernelMemoryMap::build_from_args(common_map, rt_args_idx);

        // Parse credit connections from runtime args (memory map needed for mux local addresses)
        credit_connections.num_connections = NUM_CREDIT_CONNECTIONS;
        credit_connections.parse_from_args(rt_args_idx, this->memory_map);

        // Parse traffic config to credit connection mapping
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            traffic_config_to_credit_connection_map[i] = get_arg_val<uint32_t>(rt_args_idx++);
        }

        // Parse traffic configs from local args (local_args_idx passed from caller)
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            traffic_configs_[i] = nullptr;
        }

        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            const auto metadata = ReceiverTrafficConfigMetadata::build_from_args(local_args_idx);
            NocSendType noc_send_type = static_cast<NocSendType>(get_local_arg_val<uint32_t>(local_args_idx++));

            // Get pointer to pre-allocated storage for this config
            char* config_storage = validation_configs_storage.data() + i * MAX_VALIDATION_CONFIG_SIZE;

            if (noc_send_type == NocSendType::NOC_UNICAST_WRITE) {
                const auto write_fields = NocUnicastWriteFields::build_from_args<false>(local_args_idx);
                traffic_configs_[i] = new (config_storage) WriteValidationConfig(write_fields, metadata);
            } else if (noc_send_type == NocSendType::NOC_UNICAST_ATOMIC_INC) {
                const auto atomic_inc_fields = NocUnicastAtomicIncFields::build_from_args<false>(local_args_idx);
                traffic_configs_[i] = new (config_storage) AtomicIncValidationConfig(atomic_inc_fields, metadata);
            } else if (noc_send_type == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) {
                const auto write_atomic_inc_fields =
                    NocUnicastWriteAtomicIncFields::build_from_args<false>(local_args_idx);
                traffic_configs_[i] =
                    new (config_storage) WriteAtomicIncValidationConfig(write_atomic_inc_fields, metadata);
            } else if (noc_send_type == NocSendType::NOC_UNICAST_SCATTER_WRITE) {
                const auto scatter_write_fields = NocUnicastScatterWriteFields::build_from_args<false>(local_args_idx);
                traffic_configs_[i] = new (config_storage) ScatterWriteValidationConfig(scatter_write_fields, metadata);
            } else {
                ASSERT(false);
            }

            // First parse the presence flag, then conditionally parse the data
            bool has_credit_info = get_local_arg_val<uint32_t>(local_args_idx++) != 0;

            // Initialize credit manager for this traffic config
            if (has_credit_info) {
                // Allocate space for pre-built credit return header using memory map
                const uint32_t credit_header_address = this->memory_map.get_credit_header_address();
                const uint8_t connection_idx = traffic_config_to_credit_connection_map[i];
                credit_managers_[i].template init<IS_2D_FABRIC>(
                    local_args_idx, &credit_connections, connection_idx, credit_header_address);

                // Link the credit manager to this traffic config so advance() can call it automatically
                traffic_configs_[i]->credit_manager_ = &credit_managers_[i];
            }
        }
    }
};

/* ********************
 * SyncKernelConfig   *
 **********************/
template <
    uint8_t NUM_SYNC_FABRIC_CONNECTIONS,
    bool IS_2D_FABRIC,
    uint8_t NUM_LOCAL_SYNC_CORES,
    bool USE_UNICAST_SYNC_PACKETS>
struct SyncKernelConfig {
    static SyncKernelConfig build_from_args(
        const CommonMemoryMap& common_map, size_t& rt_args_idx, size_t& local_args_idx) {
        return SyncKernelConfig(common_map, rt_args_idx, local_args_idx);
    }

    void global_sync(uint8_t sync_iter) {
        // Open all sync connections
        sync_connections.open_all();

        // Send sync start packets
        for (uint8_t i = 0; i < NUM_SYNC_FABRIC_CONNECTIONS; i++) {
            line_sync_configs()[i].global_sync_start();
        }

        // Wait for acks (only need one config to check)
        line_sync_configs()[0].global_sync_finish(sync_iter);

        // Close all sync connections
        sync_connections.close_all();
    }

    void local_sync(uint8_t sync_iter) { local_sync_config().local_sync(sync_iter); }

    // Result buffer convenience methods
    uint32_t get_result_buffer_address() const { return memory_map.common.result_buffer_base; }
    uint32_t get_result_buffer_size() const { return memory_map.common.result_buffer_size; }

    SenderKernelMemoryMap memory_map;

    FabricConnectionArray sync_connections;

    using LineSyncConfigType = LineSyncConfig;
    alignas(LineSyncConfigType)
        std::array<char, NUM_SYNC_FABRIC_CONNECTIONS * sizeof(LineSyncConfigType)> line_sync_configs_storage;
    alignas(LocalSyncConfig<true, NUM_LOCAL_SYNC_CORES>)
        std::array<char, sizeof(LocalSyncConfig<true, NUM_LOCAL_SYNC_CORES>)> local_sync_config_storage;

    // Mapping from sync config index to fabric connection index (same pattern as sender)
    std::array<uint8_t, NUM_SYNC_FABRIC_CONNECTIONS> sync_config_to_fabric_connection_map;

    // Helper accessors
    LineSyncConfigType* line_sync_configs() {
        return reinterpret_cast<LineSyncConfigType*>(line_sync_configs_storage.data());
    }
    LocalSyncConfig<true, NUM_LOCAL_SYNC_CORES>& local_sync_config() {
        return *reinterpret_cast<LocalSyncConfig<true, NUM_LOCAL_SYNC_CORES>*>(local_sync_config_storage.data());
    }

private:
    SyncKernelConfig(const CommonMemoryMap& common_map, size_t& rt_args_idx, size_t& local_args_idx) {
        // Parse memory map args from runtime args using pre-parsed common map
        this->memory_map = SenderKernelMemoryMap::build_from_args(common_map, rt_args_idx);

        // Parse all sync connections using unified array (memory map needed for mux local addresses)
        sync_connections.num_connections = NUM_SYNC_FABRIC_CONNECTIONS;
        sync_connections.parse_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx, this->memory_map);

        // Initialize line sync configurations with connection array
        uint32_t line_sync_val = get_local_arg_val<uint32_t>(local_args_idx++);

        // Parse sync config to fabric connection mapping (same pattern as sender traffic configs)
        for (uint8_t i = 0; i < NUM_SYNC_FABRIC_CONNECTIONS; i++) {
            sync_config_to_fabric_connection_map[i] = get_local_arg_val<uint32_t>(local_args_idx++);
        }

        for (uint8_t i = 0; i < NUM_SYNC_FABRIC_CONNECTIONS; i++) {
            uint32_t packet_header_address = this->memory_map.get_packet_header_address();
            uint8_t connection_idx = sync_config_to_fabric_connection_map[i];
            new (&line_sync_configs()[i])
                LineSyncConfigType(&sync_connections, connection_idx, packet_header_address, line_sync_val);

            // setup packet header fields
            constexpr ChipSendType CHIP_SEND_TYPE =
                USE_UNICAST_SYNC_PACKETS ? ChipSendType::CHIP_UNICAST : ChipSendType::CHIP_MULTICAST;
            line_sync_configs()[i].template setup_packet_header<IS_2D_FABRIC, CHIP_SEND_TYPE>(
                local_args_idx, packet_header_address);
        }

        // Initialize local sync config
        uint32_t sync_address = get_local_arg_val<uint32_t>(local_args_idx++);
        uint32_t sync_val = get_local_arg_val<uint32_t>(local_args_idx++);
        new (&local_sync_config()) LocalSyncConfig<true, NUM_LOCAL_SYNC_CORES>(sync_address, sync_val);

        // setup core coordinates
        local_sync_config().setup_core_coordinates(local_args_idx);
    }
};

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
