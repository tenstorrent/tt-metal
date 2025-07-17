// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

namespace tt::tt_fabric {
namespace fabric_tests {

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
        if (current_offset_ >= total_size_) {
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
        this->num_packets = get_arg_val<uint32_t>(arg_idx++);
        this->seed = get_arg_val<uint32_t>(arg_idx++);
        this->payload_buffer_size = get_arg_val<uint32_t>(arg_idx++);
    }
};

struct ChipUnicastFields1D {
    static ChipUnicastFields1D build_from_args(size_t& arg_idx) {
        uint32_t num_hops = get_arg_val<uint32_t>(arg_idx++);
        return ChipUnicastFields1D(num_hops);
    }

    ChipUnicastFields1D(uint32_t num_hops) : num_hops(num_hops) {}

    uint32_t num_hops;
};

struct ChipUnicastFields2D {
    static ChipUnicastFields2D build_from_args(size_t& arg_idx) {
        uint16_t src_device_id = get_arg_val<uint32_t>(arg_idx++);
        uint16_t dst_device_id = get_arg_val<uint32_t>(arg_idx++);
        uint16_t dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
        uint16_t ew_dim = get_arg_val<uint32_t>(arg_idx++);
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
        uint32_t mcast_start_hops = get_arg_val<uint32_t>(arg_idx++);
        uint32_t num_hops = get_arg_val<uint32_t>(arg_idx++);
        return ChipMulticastFields1D(mcast_start_hops, num_hops);
    }

    ChipMulticastFields1D(uint32_t mcast_start_hops, uint32_t num_hops) :
        mcast_start_hops(mcast_start_hops), num_hops(num_hops) {}

    uint32_t mcast_start_hops;
    uint32_t num_hops;
};

struct ChipMulticastFields2D {
    static ChipMulticastFields2D build_from_args(size_t& arg_idx) {
        uint16_t dst_device_id = get_arg_val<uint32_t>(arg_idx++);
        uint16_t dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
        uint16_t num_hops_n = get_arg_val<uint32_t>(arg_idx++);
        uint16_t num_hops_s = get_arg_val<uint32_t>(arg_idx++);
        uint16_t num_hops_e = get_arg_val<uint32_t>(arg_idx++);
        uint16_t num_hops_w = get_arg_val<uint32_t>(arg_idx++);
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
        uint32_t payload_size_bytes = get_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_address = get_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_noc_encoding = 0;
        if constexpr (IS_SOURCE) {
            dst_noc_encoding = get_arg_val<uint32_t>(arg_idx++);
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
        uint16_t atomic_inc_val = get_arg_val<uint32_t>(arg_idx++);
        uint16_t atomic_inc_wrap = get_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_address = get_arg_val<uint32_t>(arg_idx++);
        uint32_t dst_noc_encoding = 0;
        if constexpr (IS_SOURCE) {
            dst_noc_encoding = get_arg_val<uint32_t>(arg_idx++);
        }
        return NocUnicastAtomicIncFields(atomic_inc_val, atomic_inc_wrap, dst_address, dst_noc_encoding);
    }

    NocUnicastAtomicIncFields(
        uint16_t atomic_inc_val, uint16_t atomic_inc_wrap, uint32_t dst_address, uint32_t dst_noc_encoding) :
        atomic_inc_val(atomic_inc_val),
        atomic_inc_wrap(atomic_inc_wrap),
        dst_address(dst_address),
        dst_noc_encoding(dst_noc_encoding) {}

    uint16_t atomic_inc_val;
    uint16_t atomic_inc_wrap;
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

template <typename T>
void setup_2d_unicast_route(
    uint32_t packet_header_address, eth_chan_directions outgoing_direction, const ChipUnicastFields2D& unicast_fields) {
    // Template constraint: T must be MeshPacketHeader or LowLatencyMeshPacketHeader
    fabric_set_unicast_route(
        (T*)packet_header_address,
        outgoing_direction,
        unicast_fields.src_device_id,
        unicast_fields.dst_device_id,
        unicast_fields.dst_mesh_id,
        unicast_fields.ew_dim);
}

template <typename T>
void setup_2d_mcast_route(uint32_t packet_header_address, const ChipMulticastFields2D& mcast_fields) {
    // Template constraint: T must be MeshPacketHeader or LowLatencyMeshPacketHeader
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
 * - Dynamic vs static routing modes
 * - Unicast vs multicast transmission
 */
template <ChipSendType chip_type, bool IS_2D_FABRIC, bool USE_DYNAMIC_ROUTING>
struct ChipSendTypeHandler {
    static void parse_and_setup(
        size_t& arg_idx,
        uint32_t packet_header_address,
        volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
        WorkerToFabricEdmSender* fabric_connection_handle);
};

// 1D Unicast specialization
template <bool USE_DYNAMIC_ROUTING>
struct ChipSendTypeHandler<ChipSendType::CHIP_UNICAST, false, USE_DYNAMIC_ROUTING> {
    static void parse_and_setup(
        size_t& arg_idx,
        uint32_t packet_header_address,
        volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
        WorkerToFabricEdmSender* fabric_connection_handle) {
        const auto unicast_fields = ChipUnicastFields1D::build_from_args(arg_idx);
        packet_header->to_chip_unicast(static_cast<uint8_t>(unicast_fields.num_hops));
    }
};

// 2D Unicast specialization
template <bool USE_DYNAMIC_ROUTING>
struct ChipSendTypeHandler<ChipSendType::CHIP_UNICAST, true, USE_DYNAMIC_ROUTING> {
    static void parse_and_setup(
        size_t& arg_idx,
        uint32_t packet_header_address,
        volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
        WorkerToFabricEdmSender* fabric_connection_handle) {
        const auto unicast_fields = ChipUnicastFields2D::build_from_args(arg_idx);
        const auto outgoing_direction = (eth_chan_directions)fabric_connection_handle->direction;
        if constexpr (USE_DYNAMIC_ROUTING) {
            setup_2d_unicast_route<MeshPacketHeader>(packet_header_address, outgoing_direction, unicast_fields);
        } else {
            setup_2d_unicast_route<LowLatencyMeshPacketHeader>(
                packet_header_address, outgoing_direction, unicast_fields);
        }
    }
};

// 1D Multicast specialization
template <bool USE_DYNAMIC_ROUTING>
struct ChipSendTypeHandler<ChipSendType::CHIP_MULTICAST, false, USE_DYNAMIC_ROUTING> {
    static void parse_and_setup(
        size_t& arg_idx,
        uint32_t packet_header_address,
        volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
        WorkerToFabricEdmSender* fabric_connection_handle) {
        const auto mcast_fields = ChipMulticastFields1D::build_from_args(arg_idx);
        packet_header->to_chip_multicast(MulticastRoutingCommandHeader{
            static_cast<uint8_t>(mcast_fields.mcast_start_hops), static_cast<uint8_t>(mcast_fields.num_hops)});
    }
};

// 2D Multicast specialization
template <bool USE_DYNAMIC_ROUTING>
struct ChipSendTypeHandler<ChipSendType::CHIP_MULTICAST, true, USE_DYNAMIC_ROUTING> {
    static void parse_and_setup(
        size_t& arg_idx,
        uint32_t packet_header_address,
        volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
        WorkerToFabricEdmSender* fabric_connection_handle) {
        const auto mcast_fields = ChipMulticastFields2D::build_from_args(arg_idx);
        if constexpr (USE_DYNAMIC_ROUTING) {
            setup_2d_mcast_route<MeshPacketHeader>(packet_header_address, mcast_fields);
        } else {
            setup_2d_mcast_route<LowLatencyMeshPacketHeader>(packet_header_address, mcast_fields);
        }
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

// line sync for each fabric connection.
struct LineSyncConfig {
    LineSyncConfig(
        WorkerToFabricEdmSender* fabric_connection_handle,
        const uint32_t packet_header_address,
        const uint32_t line_sync_val) :
        fabric_connection_handle(fabric_connection_handle), line_sync_val(line_sync_val) {
        packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_address);
    }

    template <bool IS_2D_FABRIC, bool USE_DYNAMIC_ROUTING>
    void setup_packet_header(size_t& arg_idx, uint32_t packet_header_address) {
        // setup header fields. 2 rt args for 1D
        ChipSendTypeHandler<ChipSendType::CHIP_MULTICAST, IS_2D_FABRIC, USE_DYNAMIC_ROUTING>::parse_and_setup(
            arg_idx, packet_header_address, packet_header, fabric_connection_handle);

        // set up noc fields, 4 rt args
        auto fields = NocUnicastAtomicIncFields::build_from_args<true>(arg_idx);
        line_sync_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fields.dst_address);

        uint64_t noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_address);
        packet_header->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{noc_addr, fields.atomic_inc_val, fields.atomic_inc_wrap});
    }

    void global_sync_start() {
        // send packet to remote devices
        fabric_connection_handle->wait_for_empty_write_slot();
        fabric_connection_handle->send_payload_flush_non_blocking_from_address(
            (uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
    }

    void global_sync_finish() {
        // sync wait
        noc_semaphore_wait(line_sync_ptr, line_sync_val);
    }

private:
    WorkerToFabricEdmSender* fabric_connection_handle;
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
            sync_core_xy_encoding_[i] = get_arg_val<uint32_t>(arg_idx++);
        }
    }

    void local_sync() {
        if constexpr (IS_MASTER_CORE) {
            // Master core: signal all local cores
            for (uint8_t i = 0; i < NUM_LOCAL_CORES; i++) {
                auto dest_noc_addr = get_noc_addr_helper(sync_core_xy_encoding_[i], sync_address);
                noc_semaphore_inc(dest_noc_addr, 1);
            }
            // Wait for all local cores to acknowledge
            noc_semaphore_wait(sync_ptr, NUM_LOCAL_CORES);
        } else {
            noc_semaphore_wait(sync_ptr, 1);
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

struct SenderKernelTrafficConfig {
    SenderKernelTrafficConfig(
        WorkerToFabricEdmSender* fabric_connection_handle,
        const SenderTrafficConfigMetadata& metadata,
        const uint32_t packet_header_address) :
        fabric_connection_handle(fabric_connection_handle),
        metadata(metadata),
        noc_send_type_(static_cast<NocSendType>(0)),
        payload_buffer_(nullptr) {
        packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_address);

        // Initialize function pointers to null (will be set in parse_and_setup_noc_send_type)
        noc_ops_.parse_and_setup = nullptr;
        noc_ops_.update_header = nullptr;
    }

    template <bool IS_2D_FABRIC, bool USE_DYNAMIC_ROUTING>
    void parse_and_setup_chip_send_type(size_t& arg_idx, uint32_t packet_header_address) {
        ChipSendType chip_send_type = static_cast<ChipSendType>(get_arg_val<uint32_t>(arg_idx++));

        if (chip_send_type == ChipSendType::CHIP_UNICAST) {
            ChipSendTypeHandler<ChipSendType::CHIP_UNICAST, IS_2D_FABRIC, USE_DYNAMIC_ROUTING>::parse_and_setup(
                arg_idx, packet_header_address, packet_header, fabric_connection_handle);
        } else if (chip_send_type == ChipSendType::CHIP_MULTICAST) {
            ChipSendTypeHandler<ChipSendType::CHIP_MULTICAST, IS_2D_FABRIC, USE_DYNAMIC_ROUTING>::parse_and_setup(
                arg_idx, packet_header_address, packet_header, fabric_connection_handle);
        } else {
            ASSERT(false);
        }
    }

    void parse_and_setup_noc_send_type(size_t& arg_idx) {
        uint32_t noc_type_raw = get_arg_val<uint32_t>(arg_idx++);
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

    template <bool BENCHMARK_MODE>
    void send_packets() {
        uint32_t num_packets_to_send = 1;
        if constexpr (BENCHMARK_MODE) {
            num_packets_to_send = metadata.num_packets;
        }

        for (uint32_t i = 0; i < num_packets_to_send; i++) {
            fabric_connection_handle->wait_for_empty_write_slot();

            if constexpr (!BENCHMARK_MODE) {
                if (payload_size_bytes > 0 && payload_buffer_) {
                    payload_buffer_->fill_data(metadata.seed);

                    fabric_connection_handle->send_payload_without_header_non_blocking_from_address(
                        payload_buffer_->get_physical_address(), payload_size_bytes);
                }
            }

            fabric_connection_handle->send_payload_flush_non_blocking_from_address(
                (uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));

            if constexpr (!BENCHMARK_MODE) {
                if (payload_size_bytes > 0 && payload_buffer_) {
                    payload_buffer_->advance();
                    update_header_for_next_packet();
                }
                metadata.seed = prng_next(metadata.seed);
            }
        }

        num_packets_processed += num_packets_to_send;
    }

    // Round-robin version: always sends exactly one packet
    template <bool BENCHMARK_MODE>
    void send_one_packet() {
        fabric_connection_handle->wait_for_empty_write_slot();

        if constexpr (!BENCHMARK_MODE) {
            if (payload_size_bytes > 0 && payload_buffer_) {
                payload_buffer_->fill_data(metadata.seed);

                fabric_connection_handle->send_payload_without_header_non_blocking_from_address(
                    payload_buffer_->get_physical_address(), payload_size_bytes);
            }
        }

        fabric_connection_handle->send_payload_flush_non_blocking_from_address(
            (uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));

        if constexpr (!BENCHMARK_MODE) {
            if (payload_size_bytes > 0 && payload_buffer_) {
                payload_buffer_->advance();
                update_header_for_next_packet();
            }
            metadata.seed = prng_next(metadata.seed);
        }

        num_packets_processed += 1;  // Always increment by 1
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

private:
    void update_header_for_next_packet() {
        if (payload_buffer_) {
            noc_ops_.update_header(this);
        }
    }

public:
    WorkerToFabricEdmSender* fabric_connection_handle;
    SenderTrafficConfigMetadata metadata;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header;
    uint32_t payload_size_bytes = 0;
    uint32_t num_packets_processed = 0;
    uint64_t elapsed_cycles = 0;

private:
    NocSendType noc_send_type_;
    NocOperationTypes::Operations noc_ops_;

    union NocFields {
        NocUnicastWriteFields write_fields;
        NocUnicastAtomicIncFields atomic_inc_fields;
        NocUnicastWriteAtomicIncFields write_atomic_inc_fields;

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
    config->packet_header->to_noc_unicast_atomic_inc(
        NocUnicastAtomicIncCommandHeader{noc_addr, fields.atomic_inc_val, fields.atomic_inc_wrap});

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
        NocUnicastAtomicIncFusedCommandHeader{
            write_noc_addr,
            atomic_noc_addr,
            fields.atomic_inc_fields.atomic_inc_val,
            fields.atomic_inc_fields.atomic_inc_wrap},
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
        NocUnicastAtomicIncFusedCommandHeader{
            write_noc_addr,
            atomic_noc_addr,
            fields.atomic_inc_fields.atomic_inc_val,
            fields.atomic_inc_fields.atomic_inc_wrap},
        fields.write_fields.payload_size_bytes);
}

struct CommonMemoryMap {
    CommonMemoryMap() = default;
    static CommonMemoryMap build_from_args(size_t& arg_idx) { return CommonMemoryMap(arg_idx); }

    uint32_t result_buffer_base;
    uint32_t result_buffer_size;

private:
    CommonMemoryMap(size_t& arg_idx) {
        result_buffer_base = get_arg_val<uint32_t>(arg_idx++);
        result_buffer_size = get_arg_val<uint32_t>(arg_idx++);
    }
};

struct SenderKernelMemoryMap {
    // Encapsulated common memory map
    CommonMemoryMap common;

    SenderKernelMemoryMap() {}

    static SenderKernelMemoryMap build_from_args(size_t& arg_idx) { return SenderKernelMemoryMap(arg_idx); }

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

private:
    SenderKernelMemoryMap(size_t& arg_idx) {
        // Parse all memory map arguments in unified call:
        // [result_buffer_base, result_buffer_size, packet_header_base, payload_buffer_base, highest_usable_address]
        common = CommonMemoryMap::build_from_args(arg_idx);  // Parses first 2 args
        packet_header_region_base_ = get_arg_val<uint32_t>(arg_idx++);
        payload_buffer_region_base_ = get_arg_val<uint32_t>(arg_idx++);
        highest_usable_address_ = get_arg_val<uint32_t>(arg_idx++);

        // set the current addresses to the base
        curr_packet_header_address_ = packet_header_region_base_;
        curr_payload_buffer_address_ = payload_buffer_region_base_;
    }

    uint32_t packet_header_region_base_;
    uint32_t payload_buffer_region_base_;
    uint32_t highest_usable_address_;
    uint32_t curr_packet_header_address_;
    uint32_t curr_payload_buffer_address_;
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
    uint8_t NUM_FABRIC_CONNECTIONS,
    uint8_t NUM_TRAFFIC_CONFIGS,
    bool IS_2D_FABRIC,
    bool USE_DYNAMIC_ROUTING,
    bool LINE_SYNC,
    uint8_t NUM_LOCAL_SYNC_CORES>
struct SenderKernelConfig {
    static constexpr bool MASTER_SYNC_CORE = false;
    static SenderKernelConfig build_from_args(size_t& arg_idx) { return SenderKernelConfig(arg_idx); }

    void open_connections() {
        for (uint8_t i = 0; i < NUM_FABRIC_CONNECTIONS; i++) {
            fabric_connections()[i].open();
        }
    }

    void local_sync() {
        if constexpr (LINE_SYNC) {
            local_sync_config().local_sync();
        }
    }

    void close_connections() {
        for (uint8_t i = 0; i < NUM_FABRIC_CONNECTIONS; i++) {
            fabric_connections()[i].close();
        }
    }

    SenderKernelMemoryMap memory_map;
    alignas(WorkerToFabricEdmSender)
        std::array<char, NUM_FABRIC_CONNECTIONS * sizeof(WorkerToFabricEdmSender)> fabric_connections_storage;
    alignas(LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>)
        std::array<char, sizeof(LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>)> local_sync_config_storage;
    std::array<uint8_t, NUM_TRAFFIC_CONFIGS> traffic_config_to_fabric_connection_map;
    alignas(SenderKernelTrafficConfig)
        std::array<char, NUM_TRAFFIC_CONFIGS * sizeof(SenderKernelTrafficConfig)> traffic_configs_storage;
    std::array<SenderKernelTrafficConfig*, NUM_TRAFFIC_CONFIGS> traffic_config_ptrs;

    // Helper accessors
    WorkerToFabricEdmSender* fabric_connections() {
        return reinterpret_cast<WorkerToFabricEdmSender*>(fabric_connections_storage.data());
    }
    LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>& local_sync_config() {
        return *reinterpret_cast<LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>*>(
            local_sync_config_storage.data());
    }
    SenderKernelTrafficConfig* traffic_configs(uint8_t idx) {
        return reinterpret_cast<SenderKernelTrafficConfig*>(
            traffic_configs_storage.data() + idx * sizeof(SenderKernelTrafficConfig));
    }
    SenderKernelTrafficConfig* get_traffic_config(uint8_t idx) { return traffic_config_ptrs[idx]; }

    // Result buffer convenience methods
    uint32_t get_result_buffer_address() const { return memory_map.common.result_buffer_base; }
    uint32_t get_result_buffer_size() const { return memory_map.common.result_buffer_size; }

private:
    SenderKernelConfig(size_t& arg_idx) {
        // Parse unified memory map args (common + sender-specific in one call)
        this->memory_map = SenderKernelMemoryMap::build_from_args(arg_idx);

        // Initialize fabric connections using placement new
        for (uint8_t i = 0; i < NUM_FABRIC_CONNECTIONS; i++) {
            auto connection = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
            new (&fabric_connections()[i]) WorkerToFabricEdmSender(connection);
        }

        // add line sync initializations here, for each fabric connection, ex, forward and backward connection, run line
        // sync for all.
        if constexpr (LINE_SYNC) {
            uint32_t sync_address = get_arg_val<uint32_t>(arg_idx++);
            uint32_t sync_val = get_arg_val<uint32_t>(arg_idx++);
            new (&local_sync_config()) LocalSyncConfig<MASTER_SYNC_CORE, NUM_LOCAL_SYNC_CORES>(sync_address, sync_val);

            // setup core coordinates
            local_sync_config().setup_core_coordinates(arg_idx);
        }

        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            traffic_config_to_fabric_connection_map[i] = get_arg_val<uint32_t>(arg_idx++);
        }

        // Initialize traffic config pointers
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            traffic_config_ptrs[i] = nullptr;
        }

        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            auto metadata = SenderTrafficConfigMetadata::build_from_args(arg_idx);
            const auto fabric_connection_idx = traffic_config_to_fabric_connection_map[i];
            ASSERT(fabric_connection_idx < NUM_FABRIC_CONNECTIONS);

            uint32_t packet_header_address = this->memory_map.get_packet_header_address();
            // Get pointer to pre-allocated storage and initialize with placement new
            SenderKernelTrafficConfig* config_ptr = traffic_configs(i);
            traffic_config_ptrs[i] = config_ptr;

            new (config_ptr) SenderKernelTrafficConfig(
                &fabric_connections()[fabric_connection_idx], metadata, packet_header_address);

            traffic_config_ptrs[i]->template parse_and_setup_chip_send_type<IS_2D_FABRIC, USE_DYNAMIC_ROUTING>(
                arg_idx, packet_header_address);
            traffic_config_ptrs[i]->parse_and_setup_noc_send_type(arg_idx);

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
        this->num_packets = get_arg_val<uint32_t>(arg_idx++);
        this->seed = get_arg_val<uint32_t>(arg_idx++);
        this->payload_buffer_size = get_arg_val<uint32_t>(arg_idx++);
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
    }

    ReceiverTrafficConfigMetadata metadata;
    uint32_t num_packets_processed = 0;
    ValidationOps ops;
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
        wrap_boundary = atomic_inc_fields.atomic_inc_wrap;

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
        if (config->expected_value > config->wrap_boundary - config->value_step_size) {
            config->expected_value = config->value_step_size;  // Wrap around
        } else {
            config->expected_value += config->value_step_size;
        }
    }

    volatile tt_l1_ptr uint32_t* poll_address;
    uint32_t expected_value;
    uint32_t value_step_size;
    uint32_t wrap_boundary;
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
        atomic_inc_wrap = atomic_fields.atomic_inc_wrap;
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

        if (config->expected_atomic_value > config->atomic_inc_wrap - config->atomic_inc_val) {
            config->expected_atomic_value = config->atomic_inc_val;  // Wrap around
        } else {
            config->expected_atomic_value += config->atomic_inc_val;
        }

        config->payload_buffer_->advance();
    }

    alignas(ReceiverPayloadBuffer) std::array<char, sizeof(ReceiverPayloadBuffer)> payload_buffer_storage;
    ReceiverPayloadBuffer* payload_buffer_;
    volatile tt_l1_ptr uint32_t* atomic_inc_address;
    uint32_t atomic_inc_val;
    uint32_t atomic_inc_wrap;
    uint32_t expected_atomic_value;
};

/* Layout for the run time args for receiver
1. Memory map args (unified: result buffer only, as receivers don't allocate memory)
2. Traffic config args
2.1. TrafficConfigCommonFields
2.2. Noc send type fields
*/
template <uint8_t NUM_TRAFFIC_CONFIGS>
struct ReceiverKernelConfig {
    static ReceiverKernelConfig build_from_args(size_t& arg_idx) { return ReceiverKernelConfig(arg_idx); }

    // Result buffer convenience methods
    uint32_t get_result_buffer_address() const { return common_memory_map.result_buffer_base; }
    uint32_t get_result_buffer_size() const { return common_memory_map.result_buffer_size; }

    CommonMemoryMap common_memory_map;
    alignas(TrafficValidationConfigBase)
        std::array<char, NUM_TRAFFIC_CONFIGS * sizeof(WriteAtomicIncValidationConfig)> validation_configs_storage;
    std::array<TrafficValidationConfigBase*, NUM_TRAFFIC_CONFIGS> traffic_configs;

private:
    ReceiverKernelConfig(size_t& arg_idx) {
        // Parse unified memory map args (common only for receivers)
        this->common_memory_map = CommonMemoryMap::build_from_args(arg_idx);

        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            traffic_configs[i] = nullptr;
        }

        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            const auto metadata = ReceiverTrafficConfigMetadata::build_from_args(arg_idx);
            NocSendType noc_send_type = static_cast<NocSendType>(get_arg_val<uint32_t>(arg_idx++));

            // Get pointer to pre-allocated storage for this config
            char* config_storage = validation_configs_storage.data() + i * sizeof(WriteAtomicIncValidationConfig);

            if (noc_send_type == NocSendType::NOC_UNICAST_WRITE) {
                const auto write_fields = NocUnicastWriteFields::build_from_args<false>(arg_idx);
                traffic_configs[i] = new (config_storage) WriteValidationConfig(write_fields, metadata);
            } else if (noc_send_type == NocSendType::NOC_UNICAST_ATOMIC_INC) {
                const auto atomic_inc_fields = NocUnicastAtomicIncFields::build_from_args<false>(arg_idx);
                traffic_configs[i] = new (config_storage) AtomicIncValidationConfig(atomic_inc_fields, metadata);
            } else if (noc_send_type == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) {
                const auto write_atomic_inc_fields = NocUnicastWriteAtomicIncFields::build_from_args<false>(arg_idx);
                traffic_configs[i] =
                    new (config_storage) WriteAtomicIncValidationConfig(write_atomic_inc_fields, metadata);
            } else {
                ASSERT(false);
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
    bool USE_DYNAMIC_ROUTING,
    uint8_t NUM_LOCAL_SYNC_CORES>
struct SyncKernelConfig {
    static SyncKernelConfig build_from_args(size_t& arg_idx) { return SyncKernelConfig(arg_idx); }

    void global_sync() {
        for (uint8_t i = 0; i < NUM_SYNC_FABRIC_CONNECTIONS; i++) {
            sync_fabric_connections()[i].open();
        }
        for (uint8_t i = 0; i < NUM_SYNC_FABRIC_CONNECTIONS; i++) {
            line_sync_configs()[i].global_sync_start();
        }
        // only need one of the config to check for the acks
        line_sync_configs()[0].global_sync_finish();
        for (uint8_t i = 0; i < NUM_SYNC_FABRIC_CONNECTIONS; i++) {
            sync_fabric_connections()[i].close();
        }
    }

    void local_sync() { local_sync_config().local_sync(); }

    // Result buffer convenience methods
    uint32_t get_result_buffer_address() const { return memory_map.result_buffer_base; }
    uint32_t get_result_buffer_size() const { return memory_map.result_buffer_size; }

    CommonMemoryMap memory_map;
    alignas(WorkerToFabricEdmSender)
        std::array<char, NUM_SYNC_FABRIC_CONNECTIONS * sizeof(WorkerToFabricEdmSender)> sync_fabric_connections_storage;
    alignas(LineSyncConfig)
        std::array<char, NUM_SYNC_FABRIC_CONNECTIONS * sizeof(LineSyncConfig)> line_sync_configs_storage;
    alignas(LocalSyncConfig<true, NUM_LOCAL_SYNC_CORES>)
        std::array<char, sizeof(LocalSyncConfig<true, NUM_LOCAL_SYNC_CORES>)> local_sync_config_storage;

    // Helper accessors
    WorkerToFabricEdmSender* sync_fabric_connections() {
        return reinterpret_cast<WorkerToFabricEdmSender*>(sync_fabric_connections_storage.data());
    }
    LineSyncConfig* line_sync_configs() { return reinterpret_cast<LineSyncConfig*>(line_sync_configs_storage.data()); }
    LocalSyncConfig<true, NUM_LOCAL_SYNC_CORES>& local_sync_config() {
        return *reinterpret_cast<LocalSyncConfig<true, NUM_LOCAL_SYNC_CORES>*>(local_sync_config_storage.data());
    }

private:
    SyncKernelConfig(size_t& arg_idx) {
        // Parse memory map args (common only)
        memory_map = CommonMemoryMap::build_from_args(arg_idx);

        // Initialize sync fabric connections using placement new
        for (uint8_t i = 0; i < NUM_SYNC_FABRIC_CONNECTIONS; i++) {
            auto sync_connection = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
            new (&sync_fabric_connections()[i]) WorkerToFabricEdmSender(sync_connection);
        }

        // Initialize line sync configurations
        uint32_t line_sync_val = get_arg_val<uint32_t>(arg_idx++);
        for (uint8_t i = 0; i < NUM_SYNC_FABRIC_CONNECTIONS; i++) {
            // For sync kernel, we allocate packet headers from a simple base address
            // since we don't need the complex memory management of SenderKernelMemoryMap
            uint32_t packet_header_address =
                memory_map.result_buffer_base + memory_map.result_buffer_size + i * sizeof(PACKET_HEADER_TYPE);
            new (&line_sync_configs()[i])
                LineSyncConfig(&sync_fabric_connections()[i], packet_header_address, line_sync_val);

            // setup packet header fields
            line_sync_configs()[i].template setup_packet_header<IS_2D_FABRIC, USE_DYNAMIC_ROUTING>(
                arg_idx, packet_header_address);
        }

        // Initialize local sync config
        uint32_t sync_address = get_arg_val<uint32_t>(arg_idx++);
        uint32_t sync_val = get_arg_val<uint32_t>(arg_idx++);
        new (&local_sync_config()) LocalSyncConfig<true, NUM_LOCAL_SYNC_CORES>(sync_address, sync_val);

        // setup core coordinates
        local_sync_config().setup_core_coordinates(arg_idx);
    }
};

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
