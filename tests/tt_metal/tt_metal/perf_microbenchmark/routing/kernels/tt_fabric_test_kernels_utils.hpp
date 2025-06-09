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

struct SenderTrafficConfigMetadata {
    static SenderTrafficConfigMetadata build_from_args(size_t& arg_idx) { return SenderTrafficConfigMetadata(arg_idx); }

    SenderTrafficConfigMetadata(const SenderTrafficConfigMetadata& other) :
        num_packets(other.num_packets), seed(other.seed) {}

    uint32_t num_packets;
    uint32_t seed;

private:
    SenderTrafficConfigMetadata(size_t& arg_idx) {
        this->num_packets = get_arg_val<uint32_t>(arg_idx++);
        this->seed = get_arg_val<uint32_t>(arg_idx++);
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
        if constexpr (IS_SOURCE) {
            uint32_t dst_noc_encoding = get_arg_val<uint32_t>(arg_idx++);
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
        if constexpr (IS_SOURCE) {
            uint32_t dst_noc_encoding = get_arg_val<uint32_t>(arg_idx++);
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
    // TODO: static assert on T types
    fabric_set_unicast_route(
        (T*)packet_header_address,
        outgoing_direction,
        unicast_fields.src_device_id,
        unicast_fields.dst_device_id,
        unicast_fields.dst_mesh_id,
        unicast_fields.ew_dim);
}

template <typename T>
void setup_2d_mcast_routet(uint32_t packet_header_address, const ChipMulticastFields2D& mcast_fields) {
    // TODO: static assert on T types
    fabric_set_mcast_route(
        (T*)packet_header_address,
        mcast_fields.dst_device_id,
        mcast_fields.dst_mesh_id,
        mcast_fields.num_hops_e,
        mcast_fields.num_hops_w,
        mcast_fields.num_hops_n,
        mcast_fields.num_hops_s);
}

template <bool IS_2D_FABRIC, bool USE_DYNAMIC_ROUTING>
void setup_header_chip_send_type(
    WorkerToFabricEdmSender* fabric_connection_handle, const uint32_t packet_header_address, size_t& arg_idx) {
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_address);

    ChipSendType chip_send_type = static_cast<ChipSendType>(get_arg_val<uint32_t>(arg_idx++));
    switch (chip_send_type) {
        case ChipSendType::CHIP_UNICAST:
            if constexpr (IS_2D_FABRIC) {
                const auto unicast_fields = ChipUnicastFields2D::build_from_args(arg_idx);
                const auto outgoing_direction = (eth_chan_directions)fabric_connection_handle->direction;
                if constexpr (USE_DYNAMIC_ROUTING) {
                    setup_2d_unicast_route<MeshPacketHeader>(packet_header_address, outgoing_direction, unicast_fields);
                } else {
                    setup_2d_unicast_route<LowLatencyMeshPacketHeader>(
                        packet_header_address, outgoing_direction, unicast_fields);
                }
            } else {
                const auto unicast_fields = ChipUnicastFields1D::build_from_args(arg_idx);
                packet_header->to_chip_unicast(static_cast<uint8_t>(unicast_fields.num_hops));
            }
            break;
        case ChipSendType::CHIP_MULTICAST:
            if constexpr (IS_2D_FABRIC) {
                const auto mcast_fields = ChipMulticastFields2D::build_from_args(arg_idx);
                if constexpr (USE_DYNAMIC_ROUTING) {
                    setup_2d_mcast_route<MeshPacketHeader>(packet_header, mcast_fields);
                } else {
                    setup_2d_mcast_route<LowLatencyMeshPacketHeader>(packet_header, mcast_fields);
                }
            } else {
                const auto mcast_fields = ChipMulticastFields1D::build_from_args(arg_idx);
                packet_header->to_chip_multicast(MulticastRoutingCommandHeader{
                    mcast_fields.mcast_start_hops, static_cast<uint8_t>(mcast_fields.num_hops)});
            }
        default: ASSERT(false); break;
    }
}

// returns the reset dst address currently
uint64_t setup_header_noc_send_type(const uint32_t packet_header_address, size_t& arg_idx) {
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header =
        reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_address);
    uint64_t reset_dst_address = 0;

    NocSendType noc_send_type = static_cast<NocSendType>(get_arg_val<uint32_t>(arg_idx++));
    switch (noc_send_type) {
        case NocSendType::NOC_UNICAST_WRITE:
            const auto write_fields = NocUnicastWriteFields::build_from_args(arg_idx);
            uint64_t write_dst_addr = get_noc_addr_helper(write_fields.dst_noc_encoding, write_fields.dst_address);
            packet_header->to_noc_unicast_write(
                NocUnicastCommandHeader{write_dst_addr}, write_fields.payload_size_bytes);
            reset_dst_address = write_dst_addr;
            break;
        case NocSendType::NOC_UNICAST_ATOMIC_INC:
            const auto atomic_inc_fields = NocUnicastAtomicIncFields::build_from_args(arg_idx);
            uint64_t inc_dst_addr =
                get_noc_addr_helper(atomic_inc_fields.dst_noc_encoding, atomic_inc_fields.dst_address);
            packet_header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
                inc_dst_addr, atomic_inc_fields.atomic_inc_val, atomic_inc_fields.atomic_inc_wrap});
            break;
        case NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC:
            const auto write_atomic_inc_fields = NocUnicastWriteAtomicIncFields::build_from_args(arg_idx);
            const auto& write_fields = write_atomic_inc_fields.write_fields;
            const auto& atomic_inc_fields = write_atomic_inc_fields.atomic_inc_fields;
            uint64_t write_dst_addr = get_noc_addr_helper(write_fields.dst_noc_encoding, write_fields.dst_address);
            uint64_t inc_dst_addr =
                get_noc_addr_helper(atomic_inc_fields.dst_noc_encoding, atomic_inc_fields.dst_address);
            packet_header->to_noc_fused_unicast_write_atomic_inc(
                NocUnicastAtomicIncFusedCommandHeader{
                    write_dst_addr, inc_dst_addr, atomic_inc_fields.atomic_inc_val, atomic_inc_fields.atomic_inc_wrap},
                write_fields.payload_size_bytes);
            reset_dst_address = write_dst_addr;
            break;
        default: ASSERT(false); break;
    }
}

struct SenderKernelTrafficConfig {
    template <bool IS_2D_FABRIC, bool USE_DYNAMIC_ROUTING>
    static SenderKernelTrafficConfig build_from_args(
        size_t& arg_idx,
        WorkerToFabricEdmSender* fabric_connection_handle,
        const SenderTrafficConfigMetadata& metadata,
        const uint32_t packet_header_address,
        const uint32_t payload_start_address) {
        uint64_t reset_dst_address = 0;

        setup_header_chip_send_type<IS_2D_FABRIC, USE_DYNAMIC_ROUTING>(
            fabric_connection_handle, packet_header_address, arg_idx);
        reset_dst_address = setup_header_noc_send_type(packet_header_address, arg_idx);
        return SenderKernelTrafficConfig(
            fabric_connection_handle, metadata, packet_header_address, payload_start_address, reset_dst_address);
    }

    SenderKernelTrafficConfig(
        WorkerToFabricEdmSender* fabric_connection_handle,
        const SenderTrafficConfigMetadata& metadata,
        const uint32_t packet_header_address,
        const uint32_t payload_start_address,
        const uint64_t reset_dst_address) :
        fabric_connection_handle(fabric_connection_handle),
        metadata(metadata),
        reset_dst_address(reset_dst_address),
        payload_start_address(payload_start_address) {
        this->packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_address);
        // TODO: find a cleaner way to handle this
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
                // only transmit the payload when not in benchmark mode
                if (this->payload_size_bytes > 0) {
                    // fill packet
                    this->fabric_connection_handle->send_payload_without_header_non_blocking_from_address(
                        this->payload_start_address, this->payload_size_bytes);
                }
            }
            this->fabric_connection_handle->send_payload_flush_non_blocking_from_address(
                (uint32_t)this->packet_header, sizeof(PACKET_HEADER_TYPE));
        }
        this->elapsed_cycles += get_timestamp() - start_timestamp;

        this->num_packets_processed += num_packets_to_send;
    }

    void advance_dst_address() {}

    void reset_dst_address() {}

    WorkerToFabricEdmSender* fabric_connection_handle;
    SenderTrafficConfigMetadata metadata;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header;
    uint64_t reset_dst_address;  // used for resetting the dst address when we run out of buffer space on the receiver
    uint32_t payload_start_address;
    uint32_t payload_size_bytes = 0;
    uint32_t num_packets_processed = 0;
    uint64_t cycles_elapsed = 0;
};

struct SenderKernelMemoryMap {
    static SenderKernelMemoryMap build_from_args(size_t& arg_idx) {
        uint32_t packet_header_buffer_address = get_arg_val<uint32_t>(arg_idx++);
        uint32_t payload_buffer_address = get_arg_val<uint32_t>(arg_idx++);
        uint32_t payload_buffer_chunk_size = get_arg_val<uint32_t>(arg_idx++);
        return SenderKernelMemoryMap(packet_header_buffer_address, payload_buffer_address, payload_buffer_chunk_size);
    }

    SenderKernelMemoryMap(
        uint32_t packet_header_buffer_address, uint32_t payload_buffer_address, uint32_t payload_buffer_chunk_size) :
        packet_header_buffer_address(packet_header_buffer_address),
        payload_buffer_address(payload_buffer_address),
        payload_buffer_chunk_size(payload_buffer_chunk_size) {}

    uint32_t packet_header_buffer_address;
    uint32_t payload_buffer_address;
    uint32_t payload_buffer_chunk_size;
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

    SenderKernelMemoryMap memory_map;
    std::array<WorkerToFabricEdmSender, NUM_FABRIC_CONNECTIONS> fabric_connections;
    std::array<uint8_t, NUM_TRAFFIC_CONFIGS> traffic_config_to_fabric_connection_map;
    std::array<SenderKernelTrafficConfig, NUM_TRAFFIC_CONFIGS> traffic_configs;

private:
    SenderKernelConfig(size_t& arg_idx) {
        this->memory_map = SenderKernelMemoryMap::build_from_args(arg_idx);

        for (uint8_t i = 0; i < NUM_FABRIC_CONNECTIONS; i++) {
            fabric_connections[i] = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx);
        }

        // TODO: optimize this to use fewer rt args maybe?
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            traffic_config_to_fabric_connection_map[i] = get_arg_val<uint32_t>(arg_idx++);
        }

        uint32_t curr_packet_header_address = this->memory_map.packet_header_buffer_address;
        uint32_t curr_payload_start_address = this->memory_map.payload_buffer_address;
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            auto metadata = SenderTrafficConfigMetadata::build_from_args(arg_idx);
            const auto fabric_connection_idx = traffic_config_to_fabric_connection_map[i];
            ASSERT(fabric_connection_idx < NUM_FABRIC_CONNECTIONS);

            traffic_configs[i] = SenderKernelTrafficConfig::build_from_args<IS_2D_FABRIC, USE_DYNAMIC_ROUTING>(
                arg_idx,
                &fabric_connections[fabric_connection_idx],
                metadata,
                curr_packet_header_address,
                curr_payload_start_address);

            curr_packet_header_address += sizeof(PACKET_HEADER_TYPE);
            curr_payload_start_address += this->memory_map.payload_buffer_chunk_size;
        }
    };
};

struct ReceiverTrafficConfigMetadata {
    static ReceiverTrafficConfigMetadata build_from_args(size_t& arg_idx) {
        return ReceiverTrafficConfigMetadata(arg_idx);
    }

    ReceiverTrafficConfigMetadata(const ReceiverTrafficConfigMetadata& other) :
        num_packets(other.num_packets), seed(other.seed) {}

    uint32_t num_packets = 0;
    uint32_t seed = 0;

private:
    ReceiverTrafficConfigMetadata(size_t& arg_idx) {
        this->num_packets = get_arg_val<uint32_t>(arg_idx++);
        this->seed = get_arg_val<uint32_t>(arg_idx++);
    }
};

/*
Semantics for data validation: poll() -> validate() -> advance()
*/
struct BaseTrafficValidationConfig {
    BaseTrafficValidationConfig(const ReceiverTrafficConfigMetadata& metadata) : metadata(metadata) {}

    [[nodiscard]] bool has_packets_to_validate() { return this->num_packets_processed < this->metadata.num_packets; }

    [[nodiscard]] virtual bool poll() = 0;
    [[nodiscard]] virtual bool validate() = 0;
    virtual void update() = 0;

    void advance() {
        this->num_packets_processed++;
        this->update();
    }

    ReceiverTrafficConfigMetadata metadata;
    uint32_t num_packets_processed = 0;
};

struct AtomicIncValidationConfig : public BaseTrafficValidationConfig {
    // TODO need to pass metadata as well for num packets, seed etc
    AtomicIncValidationConfig(
        const NocUnicastAtomicIncFields& atomic_inc_fields, const ReceiverTrafficConfigMetadata& metadata) :
        BaseTrafficValidationConfig(metadata) {
        this->poll_address = reinterpret_cast<tt_l1_ptr uint32_t*>(atomic_inc_fields.dst_address);
        this->value_step_size = atomic_inc_fields.atomic_inc_val;
        this->wrap_boundary = atomic_inc_fields.atomic_inc_wrap;

        // set the initial expected value equal to the step size
        this->expected_value = this->value_step_size;
    }

    bool poll() override {
        uint32_t current_value = *poll_address;
        if (current_value >= expected_value) {
            return true;
        }

        return false;
    }

    // no-op for atomic incs
    bool validate() override { return true; }

    void update() override {
        expected_value += value_step_size;
        if (expected_value > wrap_boundary) {
            // TODO: update the wrap around logic
        }
    }

    volatile tt_l1_ptr uint32_t* poll_address;
    uint32_t expected_value;
    uint32_t value_step_size;
    uint32_t wrap_boundary;
};

struct WriteValidationConfig : public BaseTrafficValidationConfig {
    WriteValidationConfig(const NocUnicastWriteFields& write_fields, const ReceiverTrafficConfigMetadata& metadata) :
        BaseTrafficValidationConfig(metadata) {
        this->poll_address = reinterpret_cast<tt_l1_ptr uint32_t*>(write_fields.dst_address);
        this->value_step_size = write_fields.payload_size_bytes;
    }

    bool poll() override {
        uint32_t current_value = *poll_address;
        if (current_value == expected_value) {
            return true;
        }

        return false;
    }

    bool validate() override {
        return check_packet_data(poll_address, value_step_size, mismatch_address, mismatch_value, expected_value);
    }

    void update() override {
        this->metadata.seed = prng_next(this->metadata.seed);
        expected_value = seed + value_step_size;
        poll_address += address_step_size;

        // wrap around
    }

    volatile tt_l1_ptr uint32_t* poll_address;
    uint32_t address_step_size;
    uint32_t expected_value;
    uint32_t value_step_size;
    uint32_t mismatch_address;
    uint32_t mismatch_value;
};

struct WriteAtomicIncValidationConfig : public BaseTrafficValidationConfig {
    WriteAtomicIncValidationConfig(
        const NocUnicastWriteAtomicIncFields& write_atomic_inc_fields, const ReceiverTrafficConfigMetadata& metadata) :
        BaseTrafficValidationConfig(metadata),
        write_config(write_atomic_inc_fields.write_fields),
        atomic_inc_config(write_atomic_inc_fields.atomic_inc_fields) {}

    bool poll() override {
        bool atomic_inc_done = atomic_inc_config.poll();
        if (!atomic_inc_done) {
            return false;
        }

        bool write_done = write_config.poll();
        return write_done;
    }

    bool validate() override {
        bool atomic_inc_valid = atomic_inc_config.validate();
        if (!atomic_inc_valid) {
            return false;
        }

        bool write_valid = write_config.validate();
        return write_valid;
    }

    void update() override {
        atomic_inc_config.update();
        write_config.update();
    }

    struct WriteValidationConfig write_config;
    struct AtomicIncValidationConfig atomic_inc_config;
};

template <uint8_t NUM_TRAFFIC_CONFIGS>
struct ReceiverKernelConfig {
    static ReceiverKernelConfig build_from_args(size_t& arg_idx) { return ReceiverKernelConfig(arg_idx); }

    std::array<BaseTrafficValidationConfig*, NUM_TRAFFIC_CONFIGS> traffic_configs;

private:
    ReceiverKernelConfig(size_t& arg_idx) {
        for (uint8_t i = 0; i < NUM_TRAFFIC_CONFIGS; i++) {
            const auto metadata = ReceiverTrafficConfigMetadata::build_from_args(arg_idx);
            NocSendType noc_send_type = static_cast<NocSendType>(get_arg_val<uint32_t>(arg_idx++));
            switch (noc_send_type) {
                case NocSendType::NOC_UNICAST_WRITE:
                    const auto write_fields = NocUnicastWriteFields::build_from_args<false>(arg_idx);
                    this->traffic_configs[i] = new WriteValidationConfig(write_fields, metadata);
                    break;
                case NocSendType::NOC_UNICAST_ATOMIC_INC:
                    const auto atomic_inc_fields = NocUnicastAtomicIncFields::build_from_args<false>(arg_idx);
                    this->traffic_configs[i] = new AtomicIncValidationConfig(atomic_inc_fields, metadata);
                    break;
                case NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC:
                    const auto write_atomic_inc_fields =
                        NocUnicastWriteAtomicIncFields::build_from_args<false>(arg_idx);
                    this->traffic_configs[i] = new WriteAtomicIncValidationConfig(write_atomic_inc_fields, metadata);
                    break;
                default: ASSERT(false); break;
            }
        }
    }
};

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
