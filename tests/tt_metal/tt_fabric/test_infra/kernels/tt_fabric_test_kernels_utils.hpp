// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <algorithm>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
// For WATCHER_RING_BUFFER_PUSH() used by the [SYNC-PROBE] instrumentation below. The macro compiles
// to nothing unless the watcher is enabled (TT_METAL_WATCHER), so it costs nothing in normal runs.
#include "api/debug/ring_buffer.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"

namespace tt::tt_fabric {
namespace fabric_tests {

// Maximum number of fabric connections supported per kernel.
// This is used to size FabricConnectionArray storage without template proliferation.
#ifdef ARCH_BLACKHOLE
// 4 NESW directions + up to 2 Z-link destinations
static constexpr uint8_t MAX_NUM_FABRIC_CONNECTIONS = 6;
#else
static constexpr uint8_t MAX_NUM_FABRIC_CONNECTIONS = 4;
#endif

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

// Per-config result entry for per-endpoint progress tracking.
// Layout must match the host-side definition in tt_fabric_test_memory_map.hpp exactly.
struct PerConfigResult {
    uint32_t packets_low;
    uint32_t packets_high;
};

// Word index in the result buffer where per-config results begin
static constexpr uint32_t PER_CONFIG_RESULT_BASE_WORD_INDEX = 32;

inline tt_l1_ptr PerConfigResult* get_per_config_results(uint32_t result_buffer_base) {
    return reinterpret_cast<tt_l1_ptr PerConfigResult*>(
        result_buffer_base + PER_CONFIG_RESULT_BASE_WORD_INDEX * sizeof(uint32_t));
}

inline void write_per_config_result(tt_l1_ptr PerConfigResult* entry, uint64_t packets) {
    entry->packets_low = static_cast<uint32_t>(packets);
    entry->packets_high = static_cast<uint32_t>(packets >> 32);
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

template <typename EdmSenderT = WorkerToFabricEdmSender>
struct SenderKernelTrafficConfig;

// NOC op structs and function pointer types are all templated on EdmSenderT.
// VC_ID is a compile-time arg so each binary contains exactly one EdmSenderT
// instantiation — function pointer types match call sites exactly, no casting needed.
template <typename EdmSenderT = WorkerToFabricEdmSender>
struct NocOperationTypes {
    using ParseSetupFunc = void (*)(SenderKernelTrafficConfig<EdmSenderT>*, size_t&);
    using UpdateHeaderFunc = void (*)(SenderKernelTrafficConfig<EdmSenderT>*);

    struct Operations {
        ParseSetupFunc parse_and_setup;
        UpdateHeaderFunc update_header;
    };
};

// NOC Operation Class Declarations (implementations after SenderKernelTrafficConfig)
template <typename EdmSenderT = WorkerToFabricEdmSender>
struct NocWriteSenderOperations {
    static void parse_and_setup_impl(SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx);
    static void update_header_impl(SenderKernelTrafficConfig<EdmSenderT>* config);
};

template <typename EdmSenderT = WorkerToFabricEdmSender>
struct NocAtomicSenderOperations {
    static void parse_and_setup_impl(SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx);
    static void update_header_impl(SenderKernelTrafficConfig<EdmSenderT>* config);
};

template <typename EdmSenderT = WorkerToFabricEdmSender>
struct NocFusedSenderOperations {
    static void parse_and_setup_impl(SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx);
    static void update_header_impl(SenderKernelTrafficConfig<EdmSenderT>* config);
};

template <typename EdmSenderT = WorkerToFabricEdmSender>
struct NocScatterWriteSenderOperations {
    static void parse_and_setup_impl(SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx);
    static void update_header_impl(SenderKernelTrafficConfig<EdmSenderT>* config);
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
template <typename EdmSenderT = WorkerToFabricEdmSender>
struct FabricConnectionArray {
    // TODO: get the num buffers more systematically
    static constexpr uint8_t NUM_BUFFERS = 8;

    using MuxConnectionType = tt::tt_fabric::WorkerToFabricMuxSender<NUM_BUFFERS>;
    static constexpr size_t MAX_CONNECTION_SIZE = std::max(sizeof(EdmSenderT), sizeof(MuxConnectionType));

    // Type-erased storage for connections (sized for maximum)
    alignas(std::max(alignof(EdmSenderT), alignof(MuxConnectionType)))
        std::array<char, MAX_NUM_FABRIC_CONNECTIONS * MAX_CONNECTION_SIZE> storage;
    std::array<bool, MAX_NUM_FABRIC_CONNECTIONS> is_mux;

    // Cached mux info for wait_for_fabric_endpoint_ready
    std::array<MuxCachedInfo, MAX_NUM_FABRIC_CONNECTIONS> mux_cached_info;

    // Actual number of connections in use (set at initialization, bounds-checked in kernel)
    uint8_t num_connections = 0;

    // Accessors with proper type casting
    FORCE_INLINE EdmSenderT& get_fabric_connection(uint8_t idx) {
        return *reinterpret_cast<EdmSenderT*>(storage.data() + idx * MAX_CONNECTION_SIZE);
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
                auto conn = EdmSenderT::template build_from_args<core_type>(rt_args_idx);
                new (&get_fabric_connection(i)) EdmSenderT(conn);
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
            static_cast<EdmSenderT*>(conn_ptr)->wait_for_empty_write_slot();
        } else {
            // Normal path: runtime dispatch using cached is_mux array
            if (is_mux[idx]) {
                static_cast<MuxConnectionType*>(conn_ptr)->wait_for_empty_write_slot();
            } else {
                static_cast<EdmSenderT*>(conn_ptr)->wait_for_empty_write_slot();
            }
        }
    }

    // Send header only (used for credit returns)
    template <bool BENCHMARK_MODE = false>
    FORCE_INLINE void send_header_non_blocking(void* conn_ptr, uint8_t idx, uint32_t header_addr) {
        if constexpr (BENCHMARK_MODE) {
            static_cast<EdmSenderT*>(conn_ptr)->send_payload_flush_non_blocking_from_address(
                header_addr, sizeof(PACKET_HEADER_TYPE));
        } else {
            if (is_mux[idx]) {
                static_cast<MuxConnectionType*>(conn_ptr)->send_payload_flush_non_blocking_from_address(
                    header_addr, sizeof(PACKET_HEADER_TYPE));
            } else {
                static_cast<EdmSenderT*>(conn_ptr)->send_payload_flush_non_blocking_from_address(
                    header_addr, sizeof(PACKET_HEADER_TYPE));
            }
        }
    }

    // Send payload without header (used for multi-part sends)
    template <bool BENCHMARK_MODE = false>
    FORCE_INLINE void send_payload_without_header(void* conn_ptr, uint8_t idx, uint32_t payload_addr, size_t size) {
        if constexpr (BENCHMARK_MODE) {
            static_cast<EdmSenderT*>(conn_ptr)->send_payload_without_header_non_blocking_from_address(
                payload_addr, size);
        } else {
            if (is_mux[idx]) {
                static_cast<MuxConnectionType*>(conn_ptr)->send_payload_without_header_non_blocking_from_address(
                    payload_addr, size);
            } else {
                static_cast<EdmSenderT*>(conn_ptr)->send_payload_without_header_non_blocking_from_address(
                    payload_addr, size);
            }
        }
    }

    // Send header with flush (used for completing multi-part sends)
    template <bool BENCHMARK_MODE = false>
    FORCE_INLINE void send_header_flush_blocking(void* conn_ptr, uint8_t idx, uint32_t header_addr) {
        if constexpr (BENCHMARK_MODE) {
            static_cast<EdmSenderT*>(conn_ptr)->send_payload_flush_blocking_from_address(
                header_addr, sizeof(PACKET_HEADER_TYPE));
        } else {
            if (is_mux[idx]) {
                static_cast<MuxConnectionType*>(conn_ptr)->send_payload_flush_blocking_from_address(
                    header_addr, sizeof(PACKET_HEADER_TYPE));
            } else {
                static_cast<EdmSenderT*>(conn_ptr)->send_payload_flush_blocking_from_address(
                    header_addr, sizeof(PACKET_HEADER_TYPE));
            }
        }
    }

    // Combined: send payload + header
    template <bool BENCHMARK_MODE = false>
    FORCE_INLINE void send_payload_with_header(
        void* conn_ptr, uint8_t idx, uint32_t payload_addr, size_t payload_size, uint32_t header_addr) {
        if constexpr (BENCHMARK_MODE) {
            auto* conn = static_cast<EdmSenderT*>(conn_ptr);
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
                auto* conn = static_cast<EdmSenderT*>(conn_ptr);
                if (payload_size > 0) {
                    conn->send_payload_without_header_non_blocking_from_address(payload_addr, payload_size);
                }
                conn->send_payload_flush_non_blocking_from_address(header_addr, sizeof(PACKET_HEADER_TYPE));
            }
        }
    }
};

// ============================================================================
// [SYNC-PROBE] End-of-test sync barrier instrumentation
// ============================================================================
// Watcher-log evidence showed the hung run parked at the end-of-test barrier: sender cores at
// waypoint NSW (noc_semaphore_wait, LocalSyncConfig::local_sync) and sync cores at NSMW
// (noc_semaphore_wait_min, LineSyncConfig::global_sync_finish). Waypoints alone only say "inside a
// semaphore wait" -- they cannot show WHICH barrier iteration, what value was expected, or what the
// semaphore actually holds while stuck. These probes add all three.
//
// Encoding, one uint32 per ring-buffer entry:
//     [31:24] tag   [23:16] sync_iter   [15:0] value
// The tags are grouped so a hexdump reads at a glance: 0xB0/0xB1/0xB2 are the local barrier's
// enter/poll/exit, 0xB4/0xB5/0xB6 the global barrier's.
//
//   ENTER  -> pushed once before spinning; `value` is the value being WAITED FOR.
//   POLL   -> pushed periodically while still spinning; `value` is the value CURRENTLY OBSERVED.
//             This is what makes the semaphore readable on a wedged core -- a stuck barrier leaves a
//             run of POLL entries all carrying the same value, which is the proof it never advanced.
//   EXIT   -> pushed once the wait is satisfied; `value` is the final observed value.
//
// ENTER with no EXIT == wedged in that barrier. The tag says which barrier, sync_iter says which
// iteration of it, and ENTER-vs-POLL says expected-vs-actual.
constexpr uint32_t SYNC_DBG_TAG_LOCAL_ENTER = 0xB0;
constexpr uint32_t SYNC_DBG_TAG_LOCAL_POLL = 0xB1;
constexpr uint32_t SYNC_DBG_TAG_LOCAL_EXIT = 0xB2;
constexpr uint32_t SYNC_DBG_TAG_GLOBAL_ENTER = 0xB4;
constexpr uint32_t SYNC_DBG_TAG_GLOBAL_POLL = 0xB5;
constexpr uint32_t SYNC_DBG_TAG_GLOBAL_EXIT = 0xB6;

// Send-side phases of global_sync(). The B4/B5/B6 tags above only cover the WAIT half
// (global_sync_finish). global_sync_start() calls wait_for_empty_write_slot(), which blocks when the
// local router's sender channel is stalled -- so without these a core wedged while SENDING would emit
// no B4 at all, and would look identical to a core that never reached global_sync(). These make the
// send half observable, so "couldn't push the packet out" and "pushed it but nobody answered" are
// distinguishable.
//   OPEN   value = number of sync fabric connections about to be opened
//   SENT   value = connection index whose atomic-inc packet was just pushed
//   CLOSED value = number of packets pushed this round
constexpr uint32_t SYNC_DBG_TAG_GLOBAL_OPEN = 0xB8;
constexpr uint32_t SYNC_DBG_TAG_GLOBAL_SENT = 0xB9;
constexpr uint32_t SYNC_DBG_TAG_GLOBAL_CLOSED = 0xBA;

// [NOC-DELIVERY PROBE] Did the sync core's NoC write to the router actually land?
//
// send_header_non_blocking() uses EDM_IO_BLOCKING_MODE::NON_BLOCKING, so send_chunk_from_address()
// issues noc_async_write() and returns WITHOUT any flush or barrier. The sync core then goes straight
// into global_sync_finish() and spins on the semaphore. Nothing ever confirms the write reached the
// router's L1 -- it is fire-and-forget, and the last packet before a long blocking wait is exactly
// where that is most dangerous.
//
//   NOCPRE  value = outstanding non-posted writes BEFORE the flush (issued - acked)
//   NOCPOST value = outstanding AFTER the flush; reaching this at all proves the write was acked,
//                   i.e. it landed in the router's L1 and the NoC is NOT the failure point.
// NOCPRE with no NOCPOST == the flush never returned == the write is stuck in the NoC.
constexpr uint32_t SYNC_DBG_TAG_NOC_PRE = 0xBC;
constexpr uint32_t SYNC_DBG_TAG_NOC_POST = 0xBD;

// [WRITE-POINTER PROBE] Which slot does the sync core actually write into?
//
// The router reads from ITS read pointer (recorded in debug word[20]); the worker writes to ITS write
// pointer. These are separate pieces of state, and the round-1 connection is REOPENED after the
// retrain -- open_finish() resyncs the write counter from the router's stored value:
//     buffer_slot_write_counter.counter = *worker_teardown_addr;
//     buffer_slot_index = counter % num_buffers;
// If that resync lands on the wrong slot, the write still succeeds and is still acked, but the router
// is looking somewhere else and never sees it.
//
// Encoding is different from the other tags: [31:24] tag, [23:0] the L1 slot address (addresses here
// are ~0x16ad0, well inside 24 bits). Compare directly against word[20] from the same core:
//   equal    -> pointers agree; if the packet still isn't sent the failure is the credit/stream write
//   differ   -> write-pointer desync; the packet is in a slot the router never reads
constexpr uint32_t SYNC_DBG_TAG_WRADDR = 0xBE;

// [STREAM-ID PROBE, worker side] Which register does the sync core DECREMENT to announce its packet?
//
// Resolved at connection-open time as
//     edm_buffer_remote_free_slots_update_addr = get_stream_reg_write_addr(sender_channel_credits_stream_id)
// and the sync core rebuilds its connection for round 1, after the retrain. The router polls
// sender_channel_free_slots_stream_id (logged in debug word[20], upper 16 bits). Compare the two:
//   same register    -> the decrement genuinely vanished; the bug is in the write path
//   different        -> the rebuilt connection announces on a register nobody reads. That explains
//                       both writes being acked, localfree stuck at num_buffers, and round 0 working
//                       (its connection predates the teardown).
// [23:0] holds the register ADDRESS; convert to a stream id host-side via the STREAM_REG_ADDR layout,
// or just compare round 0's value against round 1's on the same core -- a change across rounds is
// itself the finding.
constexpr uint32_t SYNC_DBG_TAG_CREDITREG = 0xBF;

// [DOORBELL READ-BACK] Does the worker's own decrement actually take effect on the router's counter?
//
// Everything so far measures this from the ROUTER side, minutes later, at teardown. This reads it from
// the WORKER, microseconds after writing it, over NoC -- an independent vantage point at the moment it
// matters. Motivated by the asymmetry in update_edm_buffer_free_slots(): the worker's own free-slot
// view does NOT depend on this remote write (I_USE_STREAM_REG_FOR_CREDIT_RECEIVE is false for this
// connection), so nothing on either side ever verifies the doorbell rang.
//
// The connection stores the UPDATE address (stream reg idx 270). The AVAILABLE counter the router polls
// is idx 297 in the same stream, i.e. +((297-270)*4) = +108 bytes.
//
//   value < num_buffers -> decrement DID apply; the router's later read of 32 is the anomaly
//   value == num_buffers -> decrement did NOT apply, right at the source
constexpr uint32_t SYNC_DBG_TAG_DOORBELL = 0xC1;
constexpr uint32_t STREAM_UPDATE_TO_AVAILABLE_BYTE_OFFSET = (297u - 270u) * 4u;
constexpr uint32_t STREAM_FREE_SLOTS_MASK = (1u << 17) - 1u;  // MEM_WORD_ADDR_WIDTH, same mask get_ptr_val uses

// [ROUTER PROBE AT THE WEDGE] Read the ROUTER's debug slot from the worker, over NoC, while the
// barrier is actually stuck.
//
// Every router-side number in this investigation so far (the send gate, min-free-since-TX, the polled
// stream id) comes from the host's SLOT dump, which the hang handler prints minutes BEFORE the round-1
// barrier develops. So we have precise worker-side data at the moment of failure and only pre-failure
// snapshots of the router. This closes that gap without needing a separate host tool: the worker is
// already doing a remote NoC read into its scratch region, so point it at the router's debug slot too.
//
// Three questions, answered from the same 104-byte read:
//   NOCXY   -- which core is this connection actually pointed at, and is a router even running there
//              (word[0] carries the 0x5E5E.... resume-phase signature; 0 means no router on that core)
//   RFREE   -- what free-slot value does the ROUTER see (word[25]), vs the 31 the worker sees
//   RGATE   -- the send gate (word[16]): receiver_has_space_for_packet / can_send / has_unsent
//
// If RFREE reads 32 while the worker reads 31, the two sides genuinely disagree about one register.
// If RFREE reads 31, the router sees the packet and the blocker is the gate -> read RGATE.
// If word[0] is 0, nothing is running on the core we are writing to.
constexpr uint32_t SYNC_DBG_TAG_NOCXY = 0xC2;
constexpr uint32_t SYNC_DBG_TAG_RFREE = 0xC3;
constexpr uint32_t SYNC_DBG_TAG_RGATE = 0xC4;

// [SAME-REGISTER PROOF] The stream id the target router is actually polling, read at the wedge.
// Everything claiming "both sides use stream 22" so far came from the pre-HUNG host dump, aggregated
// across devices. If this is not 22, the two sides are reading DIFFERENT registers and there is no
// register-visibility question at all -- just a mismatch.
constexpr uint32_t SYNC_DBG_TAG_RSTREAM = 0xC5;

// [LIVENESS AT THE WEDGE] The router's heartbeat, sampled on each probe pass. This is the measurement
// that can collapse the whole premise: if the router is FROZEN, then its recorded free-slots value is
// simply a stale snapshot from before the packet arrived, and there is no disagreement between two
// readers -- just a stopped core. phase=0x11 cannot distinguish these; a frozen core reads identically
// to a looping one. Compare this value across the probe passes:
//   changes  -> router is executing its main loop, so its free-slots record is fresh -> premise holds
//   constant -> router is stopped; the "32" is stale and the investigation moves to why it stopped
constexpr uint32_t SYNC_DBG_TAG_RHB = 0xC6;

// [SAME-INSTANT PAIRING] The worker's own read of the register, taken inside the probe rather than
// back in global_sync_start, so RDOOR and RFREE describe the same moment. Offset 192 keeps it clear
// of the 128-byte slot copy at the start of the scratch region.
constexpr uint32_t SYNC_DBG_TAG_RDOOR = 0xC7;
constexpr uint32_t ERISC_DBG_DOORBELL_SCRATCH_OFF = 192;

// [DEST-CORE PROBE] The NOC (x,y) the worker's doorbell/payload target -- packed (x<<8)|y. Pushed in
// global_sync_start (which survives ring-buffer eviction, unlike probe_router in global_sync_finish).
// Compared against the eth core the router runs on: same core + same stream reg (CREDITREG=stream 22)
// => worker and router address the SAME physical register (cross-core coherency); different => wrong target.
constexpr uint32_t SYNC_DBG_TAG_DESTCORE = 0xC8;

// [ALT-NOC DOORBELL] Read the SAME stream-22 available reg (idx 297) over the OTHER noc than the one the
// decrement + primary doorbell read used. If the same-noc read (0xC1) shows the decrement (31) but this
// shows num_buffers (32), the "31" is a same-NOC ordering/coalescing artifact -- the worker seeing its own
// outstanding write -- and the decrement never actually applied to the register. Same value on both nocs =
// the register genuinely holds 31 for the worker's core (true cross-core disagreement).
constexpr uint32_t SYNC_DBG_TAG_DOORBELL_ALTNOC = 0xC9;
// [DECR NOC] Which noc + sync cmd buf the decrement write used: packed (noc<<8)|sync_noc_cmd_buf.
constexpr uint32_t SYNC_DBG_TAG_DECR_NOC = 0xCA;

// MUST track MEM_AERISC_RESUME_PHASE_BASE in dev_mem_map.h. The region grows DOWNWARD from
// MEM_ERISC_FABRIC_ROUTER_RESERVED_BASE, so enlarging MEM_AERISC_RESUME_PHASE_SIZE MOVES this base.
// The host-side SLOT dump hardcodes the same value with the same warning (test_tt_fabric.cpp).
constexpr uint32_t ERISC_DBG_SLOT_BASE = 0x6F1F8;
constexpr uint32_t ERISC_DBG_SLOT_BYTES = 104;  // MEM_AERISC_RESUME_PHASE_SIZE
constexpr uint32_t ERISC_DBG_WORD_PHASE = 0;
constexpr uint32_t ERISC_DBG_WORD_HB = 14;  // MEM_AERISC_RX_HEARTBEAT_ADDR (base + 56)
constexpr uint32_t ERISC_DBG_WORD_GATE = 16;
constexpr uint32_t ERISC_DBG_WORD_STREAMID = 24;  // MEM_AERISC_POLLED_STREAM_ID_ADDR (base + 96)
constexpr uint32_t ERISC_DBG_WORD_FREE = 25;

// NOC ALIGNMENT. 0x6F1F8 is only 8-byte aligned and 104 is not a multiple of the NOC alignment, so
// reading [base, base+104) directly is rejected:
//   "tried to unicast read 104 bytes ... L1[addr=0x0006f1f8] (invalid address alignment)"
// which aborted the whole run. So read an aligned superset instead: start at the 32-byte boundary at
// or below the slot, and read a 32-byte multiple that covers all 104 bytes. Source and destination
// must share the same alignment, so the destination is bumped up to a 32-byte boundary too.
constexpr uint32_t ERISC_DBG_NOC_ALIGN = 32;
constexpr uint32_t ERISC_DBG_READ_BASE = ERISC_DBG_SLOT_BASE & ~(ERISC_DBG_NOC_ALIGN - 1);
constexpr uint32_t ERISC_DBG_READ_SKEW = ERISC_DBG_SLOT_BASE - ERISC_DBG_READ_BASE;  // bytes to skip
constexpr uint32_t ERISC_DBG_READ_BYTES =
    ((ERISC_DBG_READ_SKEW + ERISC_DBG_SLOT_BYTES + ERISC_DBG_NOC_ALIGN - 1) / ERISC_DBG_NOC_ALIGN) *
    ERISC_DBG_NOC_ALIGN;
static_assert(ERISC_DBG_READ_BASE % ERISC_DBG_NOC_ALIGN == 0, "read base must be NOC aligned");
static_assert(ERISC_DBG_READ_BYTES % ERISC_DBG_NOC_ALIGN == 0, "read size must be NOC aligned");
static_assert(ERISC_DBG_READ_SKEW % 4 == 0, "slot must stay word aligned within the read");
static_assert(ERISC_DBG_READ_BYTES <= 256, "must fit the debug_scratch region carved by the host");

FORCE_INLINE void sync_dbg_push_addr([[maybe_unused]] uint32_t tag, [[maybe_unused]] uint32_t addr) {
    WATCHER_RING_BUFFER_PUSH((tag << 24) | (addr & 0xFFFFFF));
}

// POLL pacing. The ring buffer holds only DEBUG_RING_BUFFER_ELEMENTS (32) entries, so an unbounded
// poll is self-defeating: the sync core legitimately sits in local_sync(1) for the WHOLE run (it is
// waiting for every sender to finish its 100M packets), and a fixed-interval poll fills all 32 slots
// with identical entries and evicts everything else -- including the global-sync markers. Observed
// directly: the sync core's buffer came back as 32 copies of 0xb1010006 and nothing else.
//
// So: at most SYNC_DBG_MAX_POLLS entries per wait, at exponentially growing intervals (2^20, 2^24,
// 2^28 spins). That gives one early sample and one very late one -- enough to show whether the value
// moved -- while costing at most 3 slots per wait. Worst case across all barriers a core executes
// stays inside 32, so the wedged wait's ENTER+POLLs survive at the tail of the buffer.
constexpr uint32_t SYNC_DBG_FIRST_POLL_SPINS = 1u << 20;
constexpr uint32_t SYNC_DBG_POLL_GROWTH_SHIFT = 4;
constexpr uint32_t SYNC_DBG_MAX_POLLS = 3;

// [[maybe_unused]]: with the watcher disabled WATCHER_RING_BUFFER_PUSH expands to nothing, which
// would otherwise leave all three parameters unused and trip -Wunused-parameter in the kernel build.
FORCE_INLINE void sync_dbg_push(
    [[maybe_unused]] uint32_t tag, [[maybe_unused]] uint32_t sync_iter, [[maybe_unused]] uint32_t value) {
    WATCHER_RING_BUFFER_PUSH((tag << 24) | ((sync_iter & 0xFF) << 16) | (value & 0xFFFF));
}

// Instrumented stand-in for noc_semaphore_wait(). Semantics are IDENTICAL to the original -- same
// do/while, same exact-equality (`!=`) test, same invalidate_l1_cache() placement -- so this cannot
// change whether the barrier passes. The WAYPOINT calls are kept as NSW/NSD so existing watcher-log
// analysis keeps working unchanged.
//
// NOTE the equality test is `!=`, not `<`: if the semaphore ever OVERSHOOTS the expected value this
// spins forever. The POLL entries will show that case plainly (observed value > expected).
FORCE_INLINE void sync_dbg_wait_eq(
    volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val, uint32_t tag_base, uint32_t sync_iter) {
    sync_dbg_push(tag_base, sync_iter, val);
    WAYPOINT("NSW");
    uint32_t spins = 0;
    uint32_t polls = 0;
    uint32_t next_poll = SYNC_DBG_FIRST_POLL_SPINS;
    do {
        invalidate_l1_cache();
        if (++spins >= next_poll && polls < SYNC_DBG_MAX_POLLS) {
            sync_dbg_push(tag_base + 1, sync_iter, *sem_addr);
            polls++;
            next_poll <<= SYNC_DBG_POLL_GROWTH_SHIFT;
        }
    } while ((*sem_addr) != val);
    WAYPOINT("NSD");
    sync_dbg_push(tag_base + 2, sync_iter, *sem_addr);
}

// Instrumented stand-in for noc_semaphore_wait_min(). As above, semantics are identical to the
// original (`<` test) and the waypoints stay NSMW/NSMD.
FORCE_INLINE void sync_dbg_wait_min(
    volatile tt_l1_ptr uint32_t* sem_addr, uint32_t val, uint32_t tag_base, uint32_t sync_iter) {
    sync_dbg_push(tag_base, sync_iter, val);
    WAYPOINT("NSMW");
    uint32_t spins = 0;
    uint32_t polls = 0;
    uint32_t next_poll = SYNC_DBG_FIRST_POLL_SPINS;
    do {
        invalidate_l1_cache();
        if (++spins >= next_poll && polls < SYNC_DBG_MAX_POLLS) {
            sync_dbg_push(tag_base + 1, sync_iter, *sem_addr);
            polls++;
            next_poll <<= SYNC_DBG_POLL_GROWTH_SHIFT;
        }
    } while ((*sem_addr) < val);
    WAYPOINT("NSMD");
    sync_dbg_push(tag_base + 2, sync_iter, *sem_addr);
}

// Line sync for each fabric connection (used by SyncKernelConfig)
template <typename EdmSenderT = WorkerToFabricEdmSender>
struct LineSyncConfig {
    LineSyncConfig(
        FabricConnectionArray<EdmSenderT>* connection_array,
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

    void global_sync_start(uint8_t sync_iter = 0, uint32_t debug_scratch_addr = 0) {
        connection_manager_->template wait_for_empty_write_slot<false>(connection_ptr_, connection_idx_);
        // [WRITE-POINTER PROBE] Capture the slot we are ABOUT to write into, before the send advances
        // the pointer. Compared against the router's read slot (debug word[20]) to detect a desync.
        // current_buffer_slot_l1_addr() is private, so read the public member it returns. With
        // EDM_NUM_BUFFER_SLOTS == 0 (USER_DEFINED_NUM_BUFFER_SLOTS false, which is this build)
        // that accessor is exactly `return this->edm_buffer_addr;`.
        sync_dbg_push_addr(SYNC_DBG_TAG_WRADDR, static_cast<EdmSenderT*>(connection_ptr_)->edm_buffer_addr);
        // [STREAM-ID PROBE] The credit register this connection will decrement to announce the packet.
        sync_dbg_push_addr(
            SYNC_DBG_TAG_CREDITREG,
            static_cast<uint32_t>(static_cast<EdmSenderT*>(connection_ptr_)->edm_buffer_remote_free_slots_update_addr));
        // [DEST-CORE PROBE] Which NOC core does this connection's doorbell/payload target? Packed (x<<8)|y.
        // Same-instant with CREDITREG so we know BOTH the core and the register offset the worker addresses.
        sync_dbg_push(
            SYNC_DBG_TAG_DESTCORE,
            sync_iter,
            (static_cast<uint32_t>(static_cast<EdmSenderT*>(connection_ptr_)->edm_noc_x) << 8) |
                static_cast<uint32_t>(static_cast<EdmSenderT*>(connection_ptr_)->edm_noc_y));
        connection_manager_->template send_header_non_blocking<false>(
            connection_ptr_, connection_idx_, (uint32_t)packet_header);
        // [NOC-DELIVERY PROBE] The send above is fire-and-forget (NON_BLOCKING: no flush, no barrier).
        // Record outstanding non-posted writes, then force a flush so we learn whether the write was
        // actually acked by the router's L1. NOTE: this flush is a behaviour change, not a pure
        // observation -- it makes the send synchronous. If it also makes the hang disappear, that is
        // itself the finding.
        // NOTE on what these two markers do and do NOT prove:
        //
        // The (issued - acked) value is USELESS as an outstanding-count. ncrisc_noc_fast_write bumps
        // BOTH software shadows together at issue time:
        //     noc_nonposted_writes_num_issued[noc] += 1;
        //     noc_nonposted_writes_acked[noc]      += num_dests;
        // so the difference is always zero by construction. It is kept only as a cheap sanity value.
        //
        // The load-bearing part is the barrier between them. We use noc_async_write_barrier(), NOT
        // noc_async_writes_flushed():
        //     writes_flushed -> NIU_MST_NONPOSTED_WR_REQ_SENT == num_issued   ("request left the NIU")
        //     write_barrier  -> NIU_MST_WR_ACK_RECEIVED       == acked        ("destination ACKED it")
        // Only the latter proves the bytes actually reached the router's L1. The earlier probe used
        // the weaker one, so "NOC_POST reached" only ever meant the request departed.
        //
        // NOC_PRE with no NOC_POST now means the destination never acked -> the write did not land.
        const uint8_t noc = get_fabric_worker_noc();
        sync_dbg_push(
            SYNC_DBG_TAG_NOC_PRE,
            sync_iter,
            noc_get_nonposted_writes_issued(noc) - noc_get_nonposted_writes_acked(noc));
        noc_async_write_barrier(noc);
        sync_dbg_push(
            SYNC_DBG_TAG_NOC_POST,
            sync_iter,
            noc_get_nonposted_writes_issued(noc) - noc_get_nonposted_writes_acked(noc));

        // [DOORBELL READ-BACK] Read the router's free-slots counter back over NoC, from here, now.
        // See SYNC_DBG_TAG_DOORBELL above for why this vantage point is the one we are missing.
        if (debug_scratch_addr != 0) {
            auto* conn = static_cast<EdmSenderT*>(connection_ptr_);
            const uint32_t available_addr = static_cast<uint32_t>(conn->edm_buffer_remote_free_slots_update_addr) +
                                            STREAM_UPDATE_TO_AVAILABLE_BYTE_OFFSET;
            const uint64_t doorbell_noc_addr = get_noc_addr(conn->edm_noc_x, conn->edm_noc_y, available_addr, noc);
            noc_async_read(doorbell_noc_addr, debug_scratch_addr, sizeof(uint32_t), noc);
            noc_async_read_barrier(noc);
            invalidate_l1_cache();
            const uint32_t free_slots =
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(debug_scratch_addr) & STREAM_FREE_SLOTS_MASK;
            sync_dbg_push(SYNC_DBG_TAG_DOORBELL, sync_iter, free_slots);

            // [DECR NOC] which noc + cmd buf the decrement write used (same 'noc' as the read above).
            sync_dbg_push(SYNC_DBG_TAG_DECR_NOC, sync_iter, (static_cast<uint32_t>(noc) << 8) | conn->sync_noc_cmd_buf);

            // [ALT-NOC DOORBELL] read the SAME reg 297 over the OTHER noc. If this reads num_buffers while the
            // primary (same-noc) read above read the decremented value, the decrement never truly applied and
            // the same-noc read was seeing the worker's own outstanding write.
            const uint8_t alt_noc = 1 - noc;
            const uint64_t doorbell_altnoc_addr =
                get_noc_addr(conn->edm_noc_x, conn->edm_noc_y, available_addr, alt_noc);
            noc_async_read(doorbell_altnoc_addr, debug_scratch_addr + 4, sizeof(uint32_t), alt_noc);
            noc_async_read_barrier(alt_noc);
            invalidate_l1_cache();
            const uint32_t free_slots_altnoc =
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(debug_scratch_addr + 4) & STREAM_FREE_SLOTS_MASK;
            sync_dbg_push(SYNC_DBG_TAG_DOORBELL_ALTNOC, sync_iter, free_slots_altnoc);
        }
    }

    // [ROUTER PROBE AT THE WEDGE] Pull the router's debug slot over NoC and push the three words that
    // matter. See SYNC_DBG_TAG_NOCXY above. Costs 3 ring entries per call, so callers must bound it.
    void probe_router(uint8_t sync_iter, uint32_t debug_scratch_addr) {
        const uint8_t noc = get_fabric_worker_noc();
        auto* conn = static_cast<EdmSenderT*>(connection_ptr_);
        // Aligned superset read; see ERISC_DBG_NOC_ALIGN above. The destination is bumped to the same
        // alignment as the source, and the slot itself starts ERISC_DBG_READ_SKEW bytes into it.
        const uint32_t dst = (debug_scratch_addr + ERISC_DBG_NOC_ALIGN - 1) & ~(ERISC_DBG_NOC_ALIGN - 1);
        const uint64_t slot_noc_addr = get_noc_addr(conn->edm_noc_x, conn->edm_noc_y, ERISC_DBG_READ_BASE, noc);
        noc_async_read(slot_noc_addr, dst, ERISC_DBG_READ_BYTES, noc);
        noc_async_read_barrier(noc);
        invalidate_l1_cache();
        auto* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + ERISC_DBG_READ_SKEW);

        // Which core, and is a router alive on it? word[0] is 0x5E5E00xx when one is running, 0 if not.
        // Coordinate packing: eth cores sit at x=24..31, y=25, so 4 bits per axis TRUNCATES. Earlier
        // probe output read "(13,9)" when the real core was (29,25). 5 bits each, phase in the low 6.
        // The device id is not packed -- the watcher attributes each ring buffer to its own core, and
        // the router is on the same device as the sync core, so it is recoverable from the log.
        const uint32_t phase = w[ERISC_DBG_WORD_PHASE];
        sync_dbg_push(
            SYNC_DBG_TAG_NOCXY,
            sync_iter,
            ((static_cast<uint32_t>(conn->edm_noc_x) & 0x1F) << 11) |
                ((static_cast<uint32_t>(conn->edm_noc_y) & 0x1F) << 6) | (phase & 0x3F));
        sync_dbg_push(SYNC_DBG_TAG_RFREE, sync_iter, w[ERISC_DBG_WORD_FREE]);
        sync_dbg_push(SYNC_DBG_TAG_RGATE, sync_iter, w[ERISC_DBG_WORD_GATE]);
        sync_dbg_push(SYNC_DBG_TAG_RSTREAM, sync_iter, w[ERISC_DBG_WORD_STREAMID]);
        sync_dbg_push(SYNC_DBG_TAG_RHB, sync_iter, w[ERISC_DBG_WORD_HB]);

        // [SAME-INSTANT PAIRING] Read the register ourselves right here, microseconds after reading the
        // router's record of it. The existing 0xC1 probe fires back in global_sync_start, so comparing
        // it against the router's value compares two different moments. This pairs them: RDOOR is what
        // the WORKER sees and RFREE is what the ROUTER saw, taken back to back on the same core.
        const uint32_t available_addr = static_cast<uint32_t>(conn->edm_buffer_remote_free_slots_update_addr) +
                                        STREAM_UPDATE_TO_AVAILABLE_BYTE_OFFSET;
        const uint64_t doorbell_noc_addr = get_noc_addr(conn->edm_noc_x, conn->edm_noc_y, available_addr, noc);
        const uint32_t door_dst = debug_scratch_addr + ERISC_DBG_DOORBELL_SCRATCH_OFF;
        noc_async_read(doorbell_noc_addr, door_dst, sizeof(uint32_t), noc);
        noc_async_read_barrier(noc);
        invalidate_l1_cache();
        sync_dbg_push(
            SYNC_DBG_TAG_RDOOR,
            sync_iter,
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(door_dst) & STREAM_FREE_SLOTS_MASK);
    }

    void global_sync_finish(uint8_t sync_iter, uint32_t debug_scratch_addr = 0) {
        // [ROUTER PROBE AT THE WEDGE] Sample the router twice while this barrier is stuck, before
        // handing off to the blocking wait. The delays are large and growing so both samples land well
        // inside the wedge rather than during normal completion; in a healthy round the barrier is
        // already satisfied and we skip out without spending ring entries.
        const uint32_t target = line_sync_val * (sync_iter + 1);
        if (debug_scratch_addr != 0) {
            // [HEALTHY CONTROL] Probe unconditionally on entry, every round. Round 0 completes quickly
            // and the delayed probes below never fire for it, so without this we only ever observe the
            // two read paths in the BROKEN case -- which leaves "NoC reads of overlay register space
            // don't return the live counter" untestable. Sampling the same code point in a round that
            // works tells us whether worker-read and router-read agree when nothing is wrong.
            // Ring-buffer eviction is not a concern: the watcher dumps every ~5s, so round-0 entries are
            // captured in earlier snapshots even though later rounds overwrite them.
            probe_router(sync_iter, debug_scratch_addr);

            uint32_t delay = 1u << 24;
            for (uint32_t p = 0; p < 2; p++) {
                uint32_t spins = 0;
                while (spins < delay) {
                    invalidate_l1_cache();
                    if (*line_sync_ptr >= target) {
                        break;
                    }
                    spins++;
                }
                invalidate_l1_cache();
                if (*line_sync_ptr >= target) {
                    break;
                }
                probe_router(sync_iter, debug_scratch_addr);
                delay <<= 4;
            }
        }

        // sync wait
        // [SYNC-PROBE] instrumented; identical wait semantics to noc_semaphore_wait_min().
        // [#45872 V3] The stream-22 reconcile now runs in the driver (global_sync) BEFORE this blocking wait, so it
        // can cover EVERY sync config (any direction can be the stranded one), not just config[0]. See
        // reconcile_stream22_once() / poll_barrier() below.
        sync_dbg_wait_min(line_sync_ptr, target, SYNC_DBG_TAG_GLOBAL_ENTER, sync_iter);
    }

    // [#45872 V3] True once this line's barrier semaphore has reached the round's target.
    bool poll_barrier(uint8_t sync_iter) {
        invalidate_l1_cache();
        return *line_sync_ptr >= line_sync_val * (sync_iter + 1);
    }

    // [#45872 V3] One reconcile pass on THIS config's stream 22. The barrier can wedge because a retrain dropped
    // this sync's stream-22 doorbell decrement: the router's free-slots register reads 32/empty and never forwards
    // the sync sitting in the slot. The SYNC connection that owns the slot is the truth -- get_num_free_write_slots
    // = num_buffers - packets_in_flight, so it reads 31 when one sync is genuinely stranded on-chip (lost doorbell)
    // but 32 if the packet was instead lost on the eth forward (a resend problem, not ours). We record both numbers
    // to this config's router debug words so the host SLOT dump shows the sync core's view -- w17=exact-free,
    // w18=register, w19=0xCAFE|delta (the marker proves the write landed) -- then inject the deficit only when it is
    // a small nonzero delta (the sync channel holds a single packet, so |delta|<=4; delta==0 is a no-op). Returns
    // the injected delta (0 if none).
    int32_t reconcile_stream22_once(uint32_t scratch) {
        auto* sc = static_cast<EdmSenderT*>(connection_ptr_);
        const uint8_t rnoc = get_fabric_worker_noc();
        const uint32_t upd_addr = sc->edm_buffer_remote_free_slots_update_addr;
        const uint32_t ef = sc->get_num_free_write_slots();  // sync conn exact free (31 if stranded on-chip)
        noc_async_read(get_noc_addr(sc->edm_noc_x, sc->edm_noc_y, upd_addr + 0x6Cu, rnoc), scratch, sizeof(uint32_t));
        noc_async_read_barrier(rnoc);
        invalidate_l1_cache();
        const uint32_t actual = (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch)) & 0x1FFFFu;
        const int32_t delta = static_cast<int32_t>(ef) - static_cast<int32_t>(actual);
        // telemetry -> this config's router debug words (host SLOT dump reads w0..w25 on the eth core)
        noc_inline_dw_write(
            get_noc_addr(sc->edm_noc_x, sc->edm_noc_y, ERISC_DBG_SLOT_BASE + 17u * 4u, rnoc), ef, 0xf, rnoc);
        noc_inline_dw_write(
            get_noc_addr(sc->edm_noc_x, sc->edm_noc_y, ERISC_DBG_SLOT_BASE + 18u * 4u, rnoc), actual, 0xf, rnoc);
        noc_inline_dw_write(
            get_noc_addr(sc->edm_noc_x, sc->edm_noc_y, ERISC_DBG_SLOT_BASE + 19u * 4u, rnoc),
            0xCAFE0000u | (static_cast<uint32_t>(delta) & 0xFFFFu),
            0xf,
            rnoc);
        int32_t injected = 0;
        if (delta != 0 && delta >= -4 && delta <= 4) {
            noc_inline_dw_write<InlineWriteDst::REG>(
                get_noc_addr(sc->edm_noc_x, sc->edm_noc_y, upd_addr, rnoc),
                pack_value_for_inc_on_write_stream_reg_write(delta),
                0xf,
                rnoc);
            injected = delta;
        }
        noc_async_writes_flushed();
        return injected;
    }

private:
    FabricConnectionArray<EdmSenderT>* connection_manager_;
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
            // [SYNC-PROBE] instrumented; identical wait semantics to noc_semaphore_wait().
            sync_dbg_wait_eq(sync_ptr, expected_val, SYNC_DBG_TAG_LOCAL_ENTER, sync_iter);
        } else {
            uint32_t expected_val = sync_iter + 1;
            // [SYNC-PROBE] instrumented; identical wait semantics to noc_semaphore_wait().
            sync_dbg_wait_eq(sync_ptr, expected_val, SYNC_DBG_TAG_LOCAL_ENTER, sync_iter);
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

template <typename EdmSenderT>
struct SenderKernelTrafficConfig {
    SenderKernelTrafficConfig(
        FabricConnectionArray<EdmSenderT>* connection_array,
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

        switch (noc_send_type_) {
            case NocSendType::NOC_UNICAST_WRITE:
                noc_ops_.parse_and_setup = NocWriteSenderOperations<EdmSenderT>::parse_and_setup_impl;
                noc_ops_.update_header = NocWriteSenderOperations<EdmSenderT>::update_header_impl;
                break;
            case NocSendType::NOC_UNICAST_ATOMIC_INC:
                noc_ops_.parse_and_setup = NocAtomicSenderOperations<EdmSenderT>::parse_and_setup_impl;
                noc_ops_.update_header = NocAtomicSenderOperations<EdmSenderT>::update_header_impl;
                break;
            case NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC:
                noc_ops_.parse_and_setup = NocFusedSenderOperations<EdmSenderT>::parse_and_setup_impl;
                noc_ops_.update_header = NocFusedSenderOperations<EdmSenderT>::update_header_impl;
                break;
            case NocSendType::NOC_UNICAST_SCATTER_WRITE:
                noc_ops_.parse_and_setup = NocScatterWriteSenderOperations<EdmSenderT>::parse_and_setup_impl;
                noc_ops_.update_header = NocScatterWriteSenderOperations<EdmSenderT>::update_header_impl;
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

    FORCE_INLINE void setup_credit_update_noc_state(const EdmSenderT& adapter, uint8_t noc) {
        auto packed_val = pack_value_for_inc_on_write_stream_reg_write(-1);
        const uint64_t noc_sem_addr =
            get_noc_addr(adapter.edm_noc_x, adapter.edm_noc_y, adapter.edm_buffer_remote_free_slots_update_addr, noc);
        noc_inline_dw_write_set_state<false /*posted*/, true /*set_val*/>(
            noc_sem_addr, packed_val, 0xf, adapter.sync_noc_cmd_buf, noc);
    }

    template <bool BENCHMARK_MODE>
    FORCE_INLINE void send_packets_stateful(const uint32_t num_packets, const uint32_t num_warmup) {
        ASSERT(connection_ptr_ != nullptr);
        auto* conn = static_cast<EdmSenderT*>(connection_ptr_);

        // Perform stateful noc send by filling buffers with headers, first, then performing credit-only NOC sends
        // Phase 1: Warmup — send actual headers to fill all buffer slots
        const uint32_t warmup_end = (num_packets < num_warmup) ? num_packets : num_warmup;
        for (uint32_t pkt = 0; pkt < warmup_end; pkt++) {
            this->template send_one_packet<BENCHMARK_MODE, false>();
        }

        setup_credit_update_noc_state(*conn, get_fabric_worker_noc());

        // Phase 2: Steady state — credit-only sends with stateful NOC
        for (uint32_t pkt = warmup_end; pkt < num_packets; pkt++) {
            this->template send_one_packet<BENCHMARK_MODE, true>();
        }
    }

    // [#45872 RECONCILE v2] Correct the router's stream-22 free-slots register to our EXACT counter value.
    // Called at the two quiescent points where a lost decrement strands a packet: (a) a sender STUCK waiting for
    // space (buffer full per our counter, router not draining), and (b) after the final data packet has settled.
    // Reads the router's stream 22 over NoC (read reg 297 == update reg 270 + 0x6C), then injects
    // delta = exact_free - register_free as an ATOMIC increment on the router's UPDATE reg. No-op when delta==0.
    // Uses payload_buffer_ as read scratch -- safe: STEP 3 fill_data() refills it before the next send, and the
    // last-packet caller is done sending. word[17] records the injected delta; the primary signal is completion.
    FORCE_INLINE void reconcile_stream22(uint32_t exact_free) {
        auto* sc = static_cast<EdmSenderT*>(connection_ptr_);
        const uint8_t rnoc = get_fabric_worker_noc();
        const uint32_t scratch = payload_buffer_->get_physical_address();
        const uint32_t upd_addr = sc->edm_buffer_remote_free_slots_update_addr;
        noc_async_read(get_noc_addr(sc->edm_noc_x, sc->edm_noc_y, upd_addr + 0x6Cu, rnoc), scratch, sizeof(uint32_t));
        noc_async_read_barrier(rnoc);
        invalidate_l1_cache();
        const uint32_t actual = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch) & 0x1FFFFu;
        const int32_t delta = static_cast<int32_t>(exact_free) - static_cast<int32_t>(actual);
        noc_inline_dw_write(
            get_noc_addr(sc->edm_noc_x, sc->edm_noc_y, 0x6F1F8u + 68u, rnoc),
            static_cast<uint32_t>(delta));  // word[17]
        noc_async_writes_flushed();
        // [#45872 OPTION2 GUARD] Only inject SMALL deltas. The true off-by-one deficit is 1; a large delta (e.g.
        // -32) means we're reading a MASKED BACKLOG (buffer full of legit packets behind the stuck one), not the
        // real deficit -> skip (word[17] still records the raw delta so we see it was skipped).
        if (delta != 0 && delta >= -4 && delta <= 4) {
            noc_inline_dw_write<InlineWriteDst::REG>(
                get_noc_addr(sc->edm_noc_x, sc->edm_noc_y, upd_addr, rnoc),
                pack_value_for_inc_on_write_stream_reg_write(delta),
                0xf,
                rnoc);
            noc_async_writes_flushed();
        }
    }

    // Send exactly one packet per call (round-robin scheduling)
    // Returns: true if packet was sent, false if blocked (no credits)
    template <bool BENCHMARK_MODE, bool STATEFUL_NOC = false>
    bool send_one_packet() {
        // [#45872 STOP] After the retrain the ERISC raises a stop flag (router word[10] @ 0x6F220, 16B-aligned).
        // Halt sending so a quiescent window forms with the during-down backlog still buffered -> then
        // occupancy (RECV_CUM - TX) shows whether the router drains it on its own. Throttled every 256 sends;
        // the flag is read into the payload buffer (overwritten by fill_data below), keeping the read aligned.
        if ((num_packets_processed & 0xFFu) == 0u && payload_buffer_ != nullptr) {
            auto* sconn = static_cast<EdmSenderT*>(connection_ptr_);
            const uint8_t rnoc = get_fabric_worker_noc();
            const uint32_t sc = payload_buffer_->get_physical_address();
            noc_async_read(
                get_noc_addr(sconn->edm_noc_x, sconn->edm_noc_y, 0x6F220u, rnoc), sc, sizeof(uint32_t), rnoc);
            noc_async_read_barrier(rnoc);
            invalidate_l1_cache();
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc) != 0u) {
                // [#45872 DRAIN-WATCH] Stop sending, but keep EXACT_FREE_LIVE (router word[8]) ALIVE so the
                // ERISC sees the buffer drain to its EXACT final occupancy. We send nothing, but the router
                // keeps advancing the read pointer as it forwards the backlog, so get_num_free_write_slots()
                // (= num_buffers - (write_counter - read_counter), counter-based/EXACT for a worker) rises
                // toward num_buffers as the buffer empties. Spin (writing w8 every 256 iters, breaking once
                // fully drained) so the SLOT dump reads the true post-drain free-slot count -- no TX proxy,
                // residual = num_buffers - w8 is exact and never negative.
                auto* sc2 = static_cast<EdmSenderT*>(connection_ptr_);
                const uint32_t nb = sc2->num_buffers_per_channel;
                const uint64_t w8_addr = get_noc_addr(sc2->edm_noc_x, sc2->edm_noc_y, 0x6F1F8u + 32u);
                uint32_t ef = 0;
                for (uint32_t k = 0; k < 10000000u; ++k) {
                    ef = sc2->get_num_free_write_slots();
                    if ((k & 0xFFu) == 0u) {
                        noc_inline_dw_write(w8_addr, ef);
                        noc_async_writes_flushed();
                    }
                    if (ef >= nb) {
                        break;  // fully drained
                    }
                }
                noc_inline_dw_write(w8_addr, ef);  // final exact free-slot count (post-drain or at cap)
                noc_async_writes_flushed();
                num_packets_processed = metadata.num_packets;  // mark done -> outer send loop exits
                return false;
            }
        }
        // STEP 1: Check credits BEFORE sending (non-benchmark mode only)
        if constexpr (!BENCHMARK_MODE) {
            if (!credit_manager_.has_credits_available(num_packets_processed)) {
                return false;  // No credits available - blocked
            }
        }

        // STEP 2: Wait for space
        if constexpr (BENCHMARK_MODE) {
            connection_manager_->template wait_for_empty_write_slot<BENCHMARK_MODE>(connection_ptr_, connection_idx_);
            // STEP 3: Send packet
            auto* conn = static_cast<EdmSenderT*>(connection_ptr_);
            if (num_packets_processed < conn->num_buffers_per_channel) {
                conn->send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
            } else {
                fabric_detail::update_credits_and_slots<STATEFUL_NOC>(conn);
            }
        } else {
            // [TASK2] reconcile DISABLED -- plain blocking wait (baseline behaviour).
            connection_manager_->template wait_for_empty_write_slot<BENCHMARK_MODE>(connection_ptr_, connection_idx_);
            // STEP 3: Send packet
            if (payload_size_bytes > 0 && payload_buffer_) {
                payload_buffer_->fill_data(metadata.seed);

                // Send payload without header
                connection_manager_->template send_payload_without_header<BENCHMARK_MODE>(
                    connection_ptr_, connection_idx_, payload_buffer_->get_physical_address(), payload_size_bytes);
            }
            // Send header
            connection_manager_->template send_header_non_blocking<BENCHMARK_MODE>(
                connection_ptr_, connection_idx_, (uint32_t)packet_header);
        }

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

        // [#45872 EXACT OCCUPANCY] Snapshot the router's TRUE channel-0 free-slot count from the worker's
        // counter-based flow-control view. get_num_free_write_slots() == num_buffers - (write_counter -
        // read_counter) here (IS_WORKER, i.e. I_USE_STREAM_REG_FOR_CREDIT_RECEIVE=false), so it is EXACT: it
        // counts EVERY channel-0 packet (payload + sync + control) on the SAME side -- no payload-shadow vs
        // global-TX mismatch, no cross-core skew (the two sources of the old +/-1). Write it live to router
        // word[8] EXACT_FREE_LIVE; the ERISC tracks its MIN across the link-down (peak occupancy = 32 - min)
        // and compares it against the stream-register read (FS22). Occupancy = num_buffers - EXACT_FREE.
        {
            auto* sc = static_cast<EdmSenderT*>(connection_ptr_);
            const uint32_t exact_free = sc->get_num_free_write_slots();  // counter-based, EXACT
            noc_inline_dw_write(get_noc_addr(sc->edm_noc_x, sc->edm_noc_y, 0x6F1F8u + 32u), exact_free);  // word[8]
            noc_async_writes_flushed();
        }
        if ((num_packets_processed & 0xFFFFFu) == 0u) {
            sync_dbg_push(0x99, 0, (num_packets_processed >> 16) & 0xFFFF);
        }
        num_packets_processed += 1;  // Always increment by 1

        // [#45872 V3] Payload-sender reconcile REMOVED. The data path completes on its own (combo run: TX=10M,
        // no data hang). The stranded packet is the SYNC (shares stream 22 but is sent from a separate core), so
        // the reconcile now lives in the SYNC path (global_sync_finish), where the SYNC connection's own
        // write-read counters give the correct deficit (-1) for the stuck sync.

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

    friend struct NocWriteSenderOperations<EdmSenderT>;
    friend struct NocAtomicSenderOperations<EdmSenderT>;
    friend struct NocFusedSenderOperations<EdmSenderT>;
    friend struct NocScatterWriteSenderOperations<EdmSenderT>;

private:
    void update_header_for_next_packet() {
        if (payload_buffer_) {
            noc_ops_.update_header(this);
        }
    }

public:
    FabricConnectionArray<EdmSenderT>* connection_manager_;
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
    typename NocOperationTypes<EdmSenderT>::Operations noc_ops_;

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
template <typename EdmSenderT>
inline void NocWriteSenderOperations<EdmSenderT>::parse_and_setup_impl(
    SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx) {
    auto fields = NocUnicastWriteFields::build_from_args<true>(arg_idx);

    uint64_t noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_address);
    config->packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_addr}, fields.payload_size_bytes);

    config->noc_fields_.write_fields = fields;
    config->payload_size_bytes = fields.payload_size_bytes;
}

template <typename EdmSenderT>
inline void NocWriteSenderOperations<EdmSenderT>::update_header_impl(SenderKernelTrafficConfig<EdmSenderT>* config) {
    const auto& fields = config->noc_fields_.write_fields;
    uint32_t buffer_offset = config->payload_buffer_->get_current_offset();
    uint32_t dest_address = fields.dst_address + buffer_offset;
    uint64_t noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, dest_address);
    config->packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_addr}, fields.payload_size_bytes);
}

template <typename EdmSenderT>
inline void NocAtomicSenderOperations<EdmSenderT>::parse_and_setup_impl(
    SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx) {
    auto fields = NocUnicastAtomicIncFields::build_from_args<true>(arg_idx);

    uint64_t noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_address);
    config->packet_header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{noc_addr, fields.atomic_inc_val});

    config->noc_fields_.atomic_inc_fields = fields;
    config->payload_size_bytes = 0;
}

template <typename EdmSenderT>
inline void NocAtomicSenderOperations<EdmSenderT>::update_header_impl(SenderKernelTrafficConfig<EdmSenderT>* config) {
    // No-op - atomic operations use fixed addresses
}

template <typename EdmSenderT>
inline void NocFusedSenderOperations<EdmSenderT>::parse_and_setup_impl(
    SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx) {
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

template <typename EdmSenderT>
inline void NocFusedSenderOperations<EdmSenderT>::update_header_impl(SenderKernelTrafficConfig<EdmSenderT>* config) {
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

template <typename EdmSenderT>
inline void NocScatterWriteSenderOperations<EdmSenderT>::parse_and_setup_impl(
    SenderKernelTrafficConfig<EdmSenderT>* config, size_t& arg_idx) {
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

template <typename EdmSenderT>
inline void NocScatterWriteSenderOperations<EdmSenderT>::update_header_impl(
    SenderKernelTrafficConfig<EdmSenderT>* config) {
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

    // [DEBUG SCRATCH] Aligned, non-overlapping L1 word reserved by the host memory map purely as a NoC
    // read destination. Unlike the packet-header/payload regions it is never handed to the fabric, so
    // reading into it cannot perturb in-flight traffic.
    uint32_t get_debug_scratch_address() const { return debug_scratch_address_; }

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
        debug_scratch_address_ = get_arg_val<uint32_t>(rt_args_idx++);  // [DEBUG SCRATCH]

        // set the current addresses to the base
        curr_packet_header_address_ = packet_header_region_base_;
        curr_payload_buffer_address_ = payload_buffer_region_base_;
        curr_mux_local_address_ = common.mux_local_addresses_base;
    }

    uint32_t packet_header_region_base_;
    uint32_t payload_buffer_region_base_;
    uint32_t highest_usable_address_;
    uint32_t debug_scratch_address_;  // [DEBUG SCRATCH] NoC read destination for the doorbell read-back
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
    uint8_t NUM_LOCAL_SYNC_CORES,
    typename EdmSenderT = WorkerToFabricEdmSender>
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
        // [SENDER-DESTCORE PROBE #45872] Emit each sender fabric connection's target eth core (tag 0xD1,
        // packed (x<<8)|y, same encoding as the sync DESTCORE 0xC8) so the two sets can be compared to
        // confirm whether senders (payload) and the barrier sync share the same eth cores / stream-22.
        for (uint8_t _i = 0; _i < connections.num_connections; _i++) {
            if (!connections.is_mux[_i]) {
                auto& _c = connections.get_fabric_connection(_i);
                WATCHER_RING_BUFFER_PUSH((0xD1u << 24) | (((uint32_t)_c.edm_noc_x << 8) | (uint32_t)_c.edm_noc_y));
            }
        }
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

    FabricConnectionArray<EdmSenderT> connections;

    alignas(LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>)
        std::array<char, sizeof(LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>)> local_sync_config_storage;
    std::array<uint8_t, NUM_TRAFFIC_CONFIGS> traffic_config_to_fabric_connection_map;

    using TrafficConfigType = SenderKernelTrafficConfig<EdmSenderT>;

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
        size_t& arg_idx, FabricConnectionArray<>* connections, uint8_t connection_idx, uint32_t credit_header_address) {
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

    FabricConnectionArray<>* connection_manager_ = nullptr;
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
        const NocUnicastWriteAtomicIncFields& write_atomic_inc_fields,
        const ReceiverTrafficConfigMetadata& metadata,
        uint8_t flow_id) :
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
        flow_id_ = flow_id;
    }

    // [#45872] Read-only stall probe. Fires only after a genuine multi-million-spin wait (transient
    // per-packet waits resolve long before the first threshold), pushing at most SYNC_DBG_MAX_POLLS
    // records per stalled packet at exponentially growing intervals -- can't flood the ring and
    // cannot change control flow. tag: 0xE3=atomic-inc short (completion signal missing by `value`),
    // 0xE4=atomic arrived but payload last-word missing (value=packets remaining). 0xE5 companion =
    // num_packets_processed (how far this flow got). sync_iter byte = flow_id.
    FORCE_INLINE void stall_dbg(uint32_t tag, uint32_t value) {
        if (++stall_spins_ >= stall_next_ && stall_pushes_ < SYNC_DBG_MAX_POLLS) {
            sync_dbg_push(tag, flow_id_, value);
            sync_dbg_push(0xE5, flow_id_, num_packets_processed & 0xFFFF);
            stall_pushes_++;
            stall_next_ <<= SYNC_DBG_POLL_GROWTH_SHIFT;
        }
    }

    static bool poll_impl(TrafficValidationConfigBase* base_config) {
        auto* config = static_cast<WriteAtomicIncValidationConfig*>(base_config);

        // Check atomic increment first
        uint32_t atomic_value = *config->atomic_inc_address;
        if (atomic_value < config->expected_atomic_value) {
            config->stall_dbg(0xE3, (config->expected_atomic_value - atomic_value) & 0xFFFF);
            return false;
        }

        if (!config->payload_buffer_->poll_for_data(config->metadata.seed)) {
            uint32_t remaining = config->metadata.num_packets - config->num_packets_processed;
            config->stall_dbg(0xE4, remaining & 0xFFFF);
            return false;
        }
        return true;
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

        // packet completed -> reset stall throttle so the next packet's wait is measured fresh
        config->stall_spins_ = 0;
        config->stall_pushes_ = 0;
        config->stall_next_ = SYNC_DBG_FIRST_POLL_SPINS;
    }

    alignas(ReceiverPayloadBuffer) std::array<char, sizeof(ReceiverPayloadBuffer)> payload_buffer_storage;
    ReceiverPayloadBuffer* payload_buffer_;
    volatile tt_l1_ptr uint32_t* atomic_inc_address;
    uint32_t atomic_inc_val;
    uint32_t expected_atomic_value;
    uint8_t flow_id_ = 0;
    uint32_t stall_spins_ = 0;
    uint32_t stall_pushes_ = 0;
    uint32_t stall_next_ = SYNC_DBG_FIRST_POLL_SPINS;
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
    FabricConnectionArray<> credit_connections;
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
                    new (config_storage) WriteAtomicIncValidationConfig(write_atomic_inc_fields, metadata, i);
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
        // [SYNC-PROBE] About to open connections. An OPEN with no following SENT means we wedged in
        // open_all(); a SENT run that stops short of NUM_SYNC_FABRIC_CONNECTIONS means we wedged in
        // global_sync_start() (its wait_for_empty_write_slot) on that connection index.
        sync_dbg_push(SYNC_DBG_TAG_GLOBAL_OPEN, sync_iter, NUM_SYNC_FABRIC_CONNECTIONS);

        // Open all sync connections
        sync_connections.open_all();

        // Send sync start packets
        for (uint8_t i = 0; i < NUM_SYNC_FABRIC_CONNECTIONS; i++) {
            line_sync_configs()[i].global_sync_start(sync_iter, get_result_buffer_address());
            sync_dbg_push(SYNC_DBG_TAG_GLOBAL_SENT, sync_iter, i);
        }

        // [#45872 V3] Wait for acks. If the barrier wedges (a retrain stranded a sync packet -> its router's
        // stream-22 doorbell decrement was lost, register reads 32/empty), reconcile EVERY config's stream 22 --
        // the stranded sync can be on any direction, but only config[0] carries the barrier semaphore. Each pass
        // records the sync core's view to that config's router (w17/18/19) so the host SLOT dump is not blind to
        // the tensix side. Bounded so a genuine unrecoverable hang still falls through to the blocking wait below.
        {
            uint32_t rounds = 0;
            while (!line_sync_configs()[0].poll_barrier(sync_iter)) {
                for (volatile uint32_t s = 0; s < 4000000u; ++s) {
                }  // backoff: only act on a real stuck spell
                if (line_sync_configs()[0].poll_barrier(sync_iter)) {
                    break;
                }
                for (uint8_t i = 0; i < NUM_SYNC_FABRIC_CONNECTIONS; i++) {
                    line_sync_configs()[i].reconcile_stream22_once(get_result_buffer_address() + 0x200u + i * 0x40u);
                }
                if (++rounds >= 200u) {
                    break;  // safety cap -> hand off to the blocking wait
                }
            }
        }

        // Blocking wait (satisfied quickly once every stranded sync has been revealed)
        line_sync_configs()[0].global_sync_finish(sync_iter, get_result_buffer_address());

        // Close all sync connections
        sync_connections.close_all();

        // [SYNC-PROBE] Round fully complete (packets out, quorum received, connections closed).
        sync_dbg_push(SYNC_DBG_TAG_GLOBAL_CLOSED, sync_iter, NUM_SYNC_FABRIC_CONNECTIONS);
    }

    void local_sync(uint8_t sync_iter) { local_sync_config().local_sync(sync_iter); }

    // Result buffer convenience methods
    uint32_t get_result_buffer_address() const { return memory_map.common.result_buffer_base; }
    uint32_t get_result_buffer_size() const { return memory_map.common.result_buffer_size; }

    SenderKernelMemoryMap memory_map;

    FabricConnectionArray<> sync_connections;

    using LineSyncConfigType = LineSyncConfig<>;
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
