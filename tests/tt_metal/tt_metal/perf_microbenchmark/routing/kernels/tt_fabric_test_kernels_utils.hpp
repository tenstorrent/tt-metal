// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include "dataflow_api.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"

namespace tt::tt_fabric {
namespace fabric_tests {

class PayloadBuffer {
public:
    static constexpr uint32_t WORD_SIZE = sizeof(uint32_t);

    PayloadBuffer(uint32_t base_address, uint32_t total_size, uint32_t payload_size) :
        base_address_(base_address), total_size_(total_size), payload_size_(payload_size) {
        ASSERT(total_size > 0);
        ASSERT(payload_size > 0);
        ASSERT(payload_size <= total_size);
        reset();
    }

    uint32_t current_address() const { return curr_address_; }
    constexpr uint32_t base_address() const { return base_address_; }
    constexpr bool has_wrapped() const { return has_wrapped_; }

    uint32_t advance() {
        uint32_t old_addr = curr_address_;
        curr_address_ += payload_size_;
        if (curr_address_ >= base_address_ + total_size_) {
            curr_address_ = base_address_;
            has_wrapped_ = true;
        }
        return old_addr;
    }

    void reset() {
        curr_address_ = base_address_;
        has_wrapped_ = false;
    }

    void fill_data(uint32_t start_value) {
        auto* addr = reinterpret_cast<tt_l1_ptr uint32_t*>(curr_address_);
        uint32_t num_words = payload_size_ / WORD_SIZE;
        for (uint32_t i = 0; i < num_words; i++) {
            addr[i] = start_value + i;
        }
    }

    bool poll_for_data(uint32_t start_value) {
        // Poll the last word of the current packet slot
        uint32_t expected_value = start_value + payload_size_ / WORD_SIZE - 1;
        uint32_t packet_end_addr = curr_address_ + payload_size_ - WORD_SIZE;
        auto* addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(packet_end_addr);
        return *addr == expected_value;
    }

    bool validate_data(uint32_t start_value) const {
        auto* addr = reinterpret_cast<tt_l1_ptr uint32_t*>(curr_address_);
        uint32_t num_words = payload_size_ / WORD_SIZE;
        for (uint32_t i = 0; i < num_words; i++) {
            if (addr[i] != (start_value + i)) {
                return false;
            }
        }
        return true;
    }

private:
    uint32_t base_address_;
    uint32_t total_size_;
    uint32_t payload_size_;
    uint32_t curr_address_;
    bool has_wrapped_ = false;
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

    NocUnicastWriteFields(const NocUnicastWriteFields& other) {
        this->payload_size_bytes = other.payload_size_bytes;
        this->dst_address = other.dst_address;
        this->dst_noc_encoding = other.dst_noc_encoding;
    }

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

    NocUnicastAtomicIncFields(const NocUnicastAtomicIncFields& other) {
        this->atomic_inc_val = other.atomic_inc_val;
        this->atomic_inc_wrap = other.atomic_inc_wrap;
        this->dst_address = other.dst_address;
        this->dst_noc_encoding = other.dst_noc_encoding;
    }

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
    static_assert(
        std::is_same_v<T, MeshPacketHeader> || std::is_same_v<T, LowLatencyMeshPacketHeader>,
        "T must be MeshPacketHeader or LowLatencyMeshPacketHeader");
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
    static_assert(
        std::is_same_v<T, MeshPacketHeader> || std::is_same_v<T, LowLatencyMeshPacketHeader>,
        "T must be MeshPacketHeader or LowLatencyMeshPacketHeader");
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
        packet_header->to_chip_multicast(
            MulticastRoutingCommandHeader{mcast_fields.mcast_start_hops, static_cast<uint8_t>(mcast_fields.num_hops)});
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

/**
 * NOC Send Type handlers - specialized template classes for different NoC operations.
 * Each specialization handles:
 * - Parsing arguments from runtime args
 * - Setting up packet headers
 * - Creating appropriate buffers
 * - Updating headers for subsequent packets
 */
template <NocSendType noc_type>
struct NocSendTypeHandler {
    using FieldType = void;  // Will be specialized
    static FieldType parse_and_setup(
        size_t& arg_idx,
        volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
        PayloadBuffer*& payload_buffer,
        uint32_t payload_buffer_address,
        uint32_t payload_buffer_size) {
        return FieldType{};
    }
};

// NOC_UNICAST_WRITE specialization
template <>
struct NocSendTypeHandler<NocSendType::NOC_UNICAST_WRITE> {
    using FieldType = NocUnicastWriteFields;

    static FieldType parse_and_setup(
        size_t& arg_idx,
        volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
        PayloadBuffer*& payload_buffer,
        uint32_t payload_buffer_address,
        uint32_t payload_buffer_size) {
        // Parse fields
        auto fields = FieldType::build_from_args<true>(arg_idx);

        payload_buffer = new PayloadBuffer(payload_buffer_address, payload_buffer_size, fields.payload_size_bytes);

        uint64_t noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_address);
        packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_addr}, fields.payload_size_bytes);

        return fields;
    }

    static void update_header(
        volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header, const FieldType& fields, PayloadBuffer* payload_buffer) {
        // Calculate destination address offset based on buffer position
        uint32_t buffer_offset = payload_buffer->current_address() - payload_buffer->base_address();
        uint32_t dest_address = fields.dst_address + buffer_offset;
        uint64_t noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, dest_address);
        packet_header->to_noc_unicast_write(NocUnicastCommandHeader{noc_addr}, fields.payload_size_bytes);
    }
};

// NOC_UNICAST_ATOMIC_INC specialization
template <>
struct NocSendTypeHandler<NocSendType::NOC_UNICAST_ATOMIC_INC> {
    using FieldType = NocUnicastAtomicIncFields;

    static FieldType parse_and_setup(
        size_t& arg_idx,
        volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
        PayloadBuffer*& payload_buffer,
        uint32_t payload_buffer_address,
        uint32_t payload_buffer_size) {
        // Parse fields
        auto fields = FieldType::build_from_args<true>(arg_idx);

        // No buffer needed for atomic operations
        payload_buffer = nullptr;

        // Setup header
        uint64_t noc_addr = get_noc_addr_helper(fields.dst_noc_encoding, fields.dst_address);
        packet_header->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{noc_addr, fields.atomic_inc_val, fields.atomic_inc_wrap});

        return fields;
    }

    static void update_header(
        volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header, const FieldType& fields, PayloadBuffer* payload_buffer) {
        // No-op - atomic operations use fixed addresses
    }
};

// NOC_FUSED_UNICAST_ATOMIC_INC specialization
template <>
struct NocSendTypeHandler<NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC> {
    using FieldType = NocUnicastWriteAtomicIncFields;

    static FieldType parse_and_setup(
        size_t& arg_idx,
        volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
        PayloadBuffer*& payload_buffer,
        uint32_t payload_buffer_address,
        uint32_t payload_buffer_size) {
        // Parse fields
        auto fields = FieldType::build_from_args<true>(arg_idx);

        payload_buffer =
            new PayloadBuffer(payload_buffer_address, payload_buffer_size, fields.write_fields.payload_size_bytes);

        // Setup header
        uint64_t write_noc_addr =
            get_noc_addr_helper(fields.write_fields.dst_noc_encoding, fields.write_fields.dst_address);
        uint64_t atomic_noc_addr =
            get_noc_addr_helper(fields.atomic_inc_fields.dst_noc_encoding, fields.atomic_inc_fields.dst_address);

        packet_header->to_noc_fused_unicast_write_atomic_inc(
            NocUnicastAtomicIncFusedCommandHeader{
                write_noc_addr,
                atomic_noc_addr,
                fields.atomic_inc_fields.atomic_inc_val,
                fields.atomic_inc_fields.atomic_inc_wrap},
            fields.write_fields.payload_size_bytes);

        return fields;
    }

    static void update_header(
        volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header, const FieldType& fields, PayloadBuffer* payload_buffer) {
        // Calculate write destination address offset based on buffer position
        uint32_t buffer_offset = payload_buffer->current_address() - payload_buffer->base_address();
        uint32_t write_dest_address = fields.write_fields.dst_address + buffer_offset;
        uint64_t write_noc_addr = get_noc_addr_helper(fields.write_fields.dst_noc_encoding, write_dest_address);
        uint64_t atomic_noc_addr =
            get_noc_addr_helper(fields.atomic_inc_fields.dst_noc_encoding, fields.atomic_inc_fields.dst_address);

        packet_header->to_noc_fused_unicast_write_atomic_inc(
            NocUnicastAtomicIncFusedCommandHeader{
                write_noc_addr,
                atomic_noc_addr,
                fields.atomic_inc_fields.atomic_inc_val,
                fields.atomic_inc_fields.atomic_inc_wrap},
            fields.write_fields.payload_size_bytes);
    }
};

struct SenderKernelTrafficConfig {
    SenderKernelTrafficConfig(
        WorkerToFabricEdmSender* fabric_connection_handle,
        const SenderTrafficConfigMetadata& metadata,
        const uint32_t packet_header_address,
        const uint32_t payload_buffer_address,
        const uint32_t payload_buffer_size) :
        fabric_connection_handle(fabric_connection_handle),
        metadata(metadata),
        noc_send_type_(static_cast<NocSendType>(0)),
        payload_buffer_(nullptr),
        payload_buffer_address_(payload_buffer_address),
        payload_buffer_size_(payload_buffer_size) {
        this->packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_address);
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
        noc_send_type_ = static_cast<NocSendType>(get_arg_val<uint32_t>(arg_idx++));

        if (noc_send_type_ == NocSendType::NOC_UNICAST_WRITE) {
            auto fields = NocSendTypeHandler<NocSendType::NOC_UNICAST_WRITE>::parse_and_setup(
                arg_idx, packet_header, payload_buffer_, payload_buffer_address_, payload_buffer_size_);
            noc_fields_.write_fields = fields;
        } else if (noc_send_type_ == NocSendType::NOC_UNICAST_ATOMIC_INC) {
            auto fields = NocSendTypeHandler<NocSendType::NOC_UNICAST_ATOMIC_INC>::parse_and_setup(
                arg_idx, packet_header, payload_buffer_, payload_buffer_address_, payload_buffer_size_);
            noc_fields_.atomic_inc_fields = fields;
        } else if (noc_send_type_ == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) {
            auto fields = NocSendTypeHandler<NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC>::parse_and_setup(
                arg_idx, packet_header, payload_buffer_, payload_buffer_address_, payload_buffer_size_);
            noc_fields_.write_atomic_inc_fields = fields;
        } else {
            ASSERT(false);
        }

        this->payload_size_bytes = this->packet_header->get_payload_size_including_header();
    }

    bool has_packets_to_send() const { return this->num_packets_processed < this->metadata.num_packets; }

    template <bool BENCHMARK_MODE>
    void send_packets() {
        uint32_t num_packets_to_send = 1;
        if constexpr (BENCHMARK_MODE) {
            num_packets_to_send = this->metadata.num_packets;
        }

        uint64_t start_timestamp = get_timestamp();
        for (uint32_t i = 0; i < num_packets_to_send; i++) {
            this->fabric_connection_handle->wait_for_empty_write_slot();

            if constexpr (!BENCHMARK_MODE) {
                if (this->payload_size_bytes > 0 && payload_buffer_) {
                    payload_buffer_->fill_data(this->metadata.seed);

                    this->fabric_connection_handle->send_payload_without_header_non_blocking_from_address(
                        payload_buffer_->current_address(), this->payload_size_bytes);
                }
            }

            this->fabric_connection_handle->send_payload_flush_non_blocking_from_address(
                (uint32_t)this->packet_header, sizeof(PACKET_HEADER_TYPE));

            if constexpr (!BENCHMARK_MODE) {
                if (this->payload_size_bytes > 0 && payload_buffer_) {
                    payload_buffer_->advance();
                    update_header_for_next_packet();
                }
                this->metadata.seed = prng_next(this->metadata.seed);
            }
        }

        this->elapsed_cycles += get_timestamp() - start_timestamp;
        this->num_packets_processed += num_packets_to_send;
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

private:
    void update_header_for_next_packet() {
        if (noc_send_type_ == NocSendType::NOC_UNICAST_WRITE && payload_buffer_) {
            NocSendTypeHandler<NocSendType::NOC_UNICAST_WRITE>::update_header(
                packet_header, noc_fields_.write_fields, payload_buffer_);
        } else if (noc_send_type_ == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC && payload_buffer_) {
            NocSendTypeHandler<NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC>::update_header(
                packet_header, noc_fields_.write_atomic_inc_fields, payload_buffer_);
        }
        // NOC_UNICAST_ATOMIC_INC: No update needed - uses fixed address
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

    union NocFields {
        NocUnicastWriteFields write_fields;
        NocUnicastAtomicIncFields atomic_inc_fields;
        NocUnicastWriteAtomicIncFields write_atomic_inc_fields;
    } noc_fields_;

    PayloadBuffer* payload_buffer_;
    uint32_t payload_buffer_address_;
    uint32_t payload_buffer_size_;
};

struct SenderKernelMemoryAllocator {
    static SenderKernelMemoryAllocator build_from_args(size_t& arg_idx) { return SenderKernelMemoryAllocator(arg_idx); }

    uint32_t get_packet_header_address() {
        uint32_t addr = curr_packet_header_address_;
        ASSERT(addr + sizeof(PACKET_HEADER_TYPE) < payload_buffer_region_base_);
        curr_packet_header_address_ += sizeof(PACKET_HEADER_TYPE);
        return addr;
    }

    uint32_t get_payload_buffer_address(uint32_t size) {
        uint32_t addr = curr_payload_buffer_address_;
        ASSERT(addr + size < highest_usable_address_);
        // TODO: make sure the addresses follow noc alignment
        curr_payload_buffer_address_ += size;
        return addr;
    }

private:
    SenderKernelMemoryAllocator(size_t& arg_idx) {
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
1. Memory map args
2. Fabric connection args
3. Traffic config args
3.1. TrafficConfigCommonFields
3.2. Chip send type fields
3.3. Noc send type fields
*/
template <uint8_t NUM_FABRIC_CONNECTIONS, uint8_t NUM_TRAFFIC_CONFIGS, bool IS_2D_FABRIC, bool USE_DYNAMIC_ROUTING>
struct SenderKernelConfig {
    static SenderKernelConfig build_from_args(size_t& arg_idx) { return SenderKernelConfig(arg_idx); }

    void open_connections() {
        for (uint8_t i = 0; i < NUM_FABRIC_CONNECTIONS; i++) {
            fabric_connections[i].open();
        }
    }

    void close_connections() {
        for (uint8_t i = 0; i < NUM_FABRIC_CONNECTIONS; i++) {
            fabric_connections[i].close();
        }
    }

    SenderKernelMemoryAllocator memory_allocator;
    std::array<WorkerToFabricEdmSender, NUM_FABRIC_CONNECTIONS> fabric_connections;
    std::array<uint8_t, NUM_TRAFFIC_CONFIGS> traffic_config_to_fabric_connection_map;
    std::array<SenderKernelTrafficConfig*, NUM_TRAFFIC_CONFIGS> traffic_configs;

private:
    SenderKernelConfig(size_t& arg_idx) {
        this->memory_allocator = SenderKernelMemoryAllocator::build_from_args(arg_idx);

        for (uint8_t i = 0; i < NUM_FABRIC_CONNECTIONS; i++) {
            fabric_connections[i] = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
        }

        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            traffic_config_to_fabric_connection_map[i] = get_arg_val<uint32_t>(arg_idx++);
        }

        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            auto metadata = SenderTrafficConfigMetadata::build_from_args(arg_idx);
            const auto fabric_connection_idx = traffic_config_to_fabric_connection_map[i];
            ASSERT(fabric_connection_idx < NUM_FABRIC_CONNECTIONS);

            uint32_t packet_header_address = this->memory_map.get_packet_header_address();
            uint32_t payload_buffer_size = metadata.payload_buffer_size;
            uint32_t payload_buffer_address = this->memory_map.get_payload_buffer_address(payload_buffer_size);

            traffic_configs[i] = new SenderKernelTrafficConfig(
                &fabric_connections[fabric_connection_idx],
                metadata,
                packet_header_address,
                payload_buffer_address,
                payload_buffer_size);

            traffic_configs[i]->template parse_and_setup_chip_send_type<IS_2D_FABRIC, USE_DYNAMIC_ROUTING>(
                arg_idx, packet_header_address);
            traffic_configs[i]->parse_and_setup_noc_send_type(arg_idx);
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

    TrafficValidationConfigBase(const ReceiverTrafficConfigMetadata& metadata, const ValidationOps& ops) :
        metadata(metadata), ops(ops) {}

    bool has_packets_to_validate() const { return this->num_packets_processed < this->metadata.num_packets; }

    bool poll() { return ops.poll(this); }

    bool validate() { return ops.validate(this); }

    void advance() {
        this->num_packets_processed++;
        ops.update(this);
    }

    ReceiverTrafficConfigMetadata metadata;
    uint32_t num_packets_processed = 0;
    ValidationOps ops;
};

struct AtomicIncValidationConfig : public TrafficValidationConfigBase {
    AtomicIncValidationConfig(
        const NocUnicastAtomicIncFields& atomic_inc_fields, const ReceiverTrafficConfigMetadata& metadata) :
        TrafficValidationConfigBase(metadata, {poll_impl, validate_impl, update_impl}) {
        this->poll_address = reinterpret_cast<tt_l1_ptr uint32_t*>(atomic_inc_fields.dst_address);
        this->value_step_size = atomic_inc_fields.atomic_inc_val;
        this->wrap_boundary = atomic_inc_fields.atomic_inc_wrap;

        // set the initial expected value equal to the step size
        this->expected_value = this->value_step_size;
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
        TrafficValidationConfigBase(metadata, {poll_impl, validate_impl, update_impl}) {
        payload_buffer_ =
            new PayloadBuffer(write_fields.dst_address, metadata.payload_buffer_size, write_fields.payload_size_bytes);
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

    PayloadBuffer* payload_buffer_;
};

struct WriteAtomicIncValidationConfig : public TrafficValidationConfigBase {
    WriteAtomicIncValidationConfig(
        const NocUnicastWriteAtomicIncFields& write_atomic_inc_fields, const ReceiverTrafficConfigMetadata& metadata) :
        TrafficValidationConfigBase(metadata, {poll_impl, validate_impl, update_impl}) {
        const auto& write_fields = write_atomic_inc_fields.write_fields;
        const auto& atomic_fields = write_atomic_inc_fields.atomic_inc_fields;

        payload_buffer_ =
            new PayloadBuffer(write_fields.dst_address, metadata.payload_buffer_size, write_fields.payload_size_bytes);

        this->atomic_inc_address = reinterpret_cast<tt_l1_ptr uint32_t*>(atomic_fields.dst_address);
        this->atomic_inc_val = atomic_fields.atomic_inc_val;
        this->atomic_inc_wrap = atomic_fields.atomic_inc_wrap;
        this->expected_atomic_value = atomic_inc_val;
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

    PayloadBuffer* payload_buffer_;
    volatile tt_l1_ptr uint32_t* atomic_inc_address;
    uint32_t atomic_inc_val;
    uint32_t atomic_inc_wrap;
    uint32_t expected_atomic_value;
};

template <uint8_t NUM_TRAFFIC_CONFIGS>
struct ReceiverKernelConfig {
    static ReceiverKernelConfig build_from_args(size_t& arg_idx) { return ReceiverKernelConfig(arg_idx); }

    std::array<TrafficValidationConfigBase*, NUM_TRAFFIC_CONFIGS> traffic_configs;

private:
    ReceiverKernelConfig(size_t& arg_idx) {
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            traffic_configs[i] = nullptr;
        }

        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            const auto metadata = ReceiverTrafficConfigMetadata::build_from_args(arg_idx);
            NocSendType noc_send_type = static_cast<NocSendType>(get_arg_val<uint32_t>(arg_idx++));

            if (noc_send_type == NocSendType::NOC_UNICAST_WRITE) {
                const auto write_fields = NocUnicastWriteFields::build_from_args<false>(arg_idx);
                this->traffic_configs[i] = new WriteValidationConfig(write_fields, metadata);
            } else if (noc_send_type == NocSendType::NOC_UNICAST_ATOMIC_INC) {
                const auto atomic_inc_fields = NocUnicastAtomicIncFields::build_from_args<false>(arg_idx);
                this->traffic_configs[i] = new AtomicIncValidationConfig(atomic_inc_fields, metadata);
            } else if (noc_send_type == NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC) {
                const auto write_atomic_inc_fields = NocUnicastWriteAtomicIncFields::build_from_args<false>(arg_idx);
                this->traffic_configs[i] = new WriteAtomicIncValidationConfig(write_atomic_inc_fields, metadata);
            } else {
                ASSERT(false);
            }
        }
    }
};

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
