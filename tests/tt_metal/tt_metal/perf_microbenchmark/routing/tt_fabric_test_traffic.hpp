// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/mesh_graph.hpp>

namespace tt::tt_fabric {
namespace fabric_tests {

struct FieldsBase {
    virtual std::vector<uint32_t> get_args() const = 0;
};

struct SenderMetadataFields : public FieldsBase {
    SenderMetadataFields(uint32_t outgoing_direction, uint32_t num_packets, uint32_t seed) :
        outgoing_direction(outgoing_direction), num_packets(num_packets), seed(seed) {}

    std::vector<uint32_t> get_args() const override {
        std::vector<uint32_t> args = {outgoing_direction, num_packets, seed};
        return args;
    }

    uint32_t outgoing_direction;
    uint32_t num_packets;
    uint32_t seed;
};

struct ReceiverMetadataFields : public FieldsBase {
    ReceiverMetadataFields(uint32_t num_packets, uint32_t seed, uint32_t sender_id) :
        num_packets(num_packets), seed(seed), sender_id(sender_id) {}

    std::vector<uint32_t> get_args() const override {
        std::vector<uint32_t> args = {num_packets, seed};
        return args;
    }

    uint32_t num_packets;
    uint32_t seed;
    uint32_t sender_id;
};

struct ChipUnicastFields1D : public FieldsBase {
    ChipUnicastFields1D(uint32_t num_hops) : num_hops(num_hops) {}

    std::vector<uint32_t> get_args() const override {
        std::vector<uint32_t> args = {num_hops};
        return args;
    }

    uint32_t num_hops;
};

struct ChipUnicastFields2D : public FieldsBase {
    ChipUnicastFields2D(uint16_t src_device_id, uint16_t dst_device_id, uint16_t dst_mesh_id, uint16_t ew_dim) :
        src_device_id(src_device_id), dst_device_id(dst_device_id), dst_mesh_id(dst_mesh_id), ew_dim(ew_dim) {}

    std::vector<uint32_t> get_args() const override {
        std::vector<uint32_t> args = {src_device_id, dst_device_id, dst_mesh_id, ew_dim};
        return args;
    }

    uint16_t src_device_id;
    uint16_t dst_device_id;
    uint16_t dst_mesh_id;
    uint16_t ew_dim;
};

struct ChipMulticastFields1D : public FieldsBase {
    static constexpr default_mcast_start_hops = 1;

    ChipMulticastFields1D(uint32_t num_hops) : num_hops(num_hops) {}

    void set_mcast_start_hops(uint32_t value) { this->mcast_start_hops = value; }

    std::vector<uint32_t> get_args() const override {
        std::vector<uint32_t> args = { mcast_start_hops.value_or(default_mcast_start_hops);
        num_hops
    };
    return args;
}

std::optional<uint32_t>
    mcast_start_hops;
uint32_t num_hops;
};  // namespace fabric_tests

struct ChipMulticastFields2D : public FieldsBase {
    ChipMulticastFields2D(
        uint16_t dst_device_id, uint16_t dst_mesh_id, std::unordered_map<RoutingDirection, uint32_t> hops) :
        dst_device_id(dst_device_id), dst_mesh_id(dst_mesh_id) {
        this->num_hops_n = hops[RoutingDirection::N];
        this->num_hops_s = hops[RoutingDirection::S];
        this->num_hops_e = hops[RoutingDirection::E];
        this->num_hops_w = hops[RoutingDirection::W];
    }

    std::vector<uint32_t> get_args() const override {
        std::vector<uint32_t> args = {dst_device_id, dst_mesh_id, num_hops_n, num_hops_s, num_hops_e, num_hops_w};
        return args;
    }

    uint16_t dst_device_id;
    uint16_t dst_mesh_id;
    uint16_t num_hops_n;
    uint16_t num_hops_s;
    uint16_t num_hops_e;
    uint16_t num_hops_w;
};

struct NocUnicastWriteFields : public FieldsBase {
    template <bool IS_SOURCE>
    NocUnicastWriteFields(uint32_t payload_size_bytes, uint32_t dst_address, std::optional<uint32_t> dst_noc_encoding) :
        payload_size_bytes(payload_size_bytes), dst_address(dst_address), dst_noc_encoding(dst_noc_encoding) {
        if constexpr (IS_SOURCE) {
            if (!this->dst_noc_encoding.has_value()) {
                tt::log_fatal(tt::LogTest, "dst_noc_encoding must be set for source");
                throw std::runtime_error("Unexpected NocUnicastWriteFields");
            }
        }
    }

    std::vector<uint32_t> get_args() const override {
        std::vector<uint32_t> args = {payload_size_bytes, dst_address};
        if (dst_noc_encoding.has_value()) {
            args.push_back(dst_noc_encoding.value());
        }
    }

    uint32_t payload_size_bytes;
    uint32_t dst_address;
    std::optional<uint32_t> dst_noc_encoding;
};

struct NocUnicastAtomicIncFields : public FieldsBase {
    static constexpr default_atomic_inc_val = 1;
    static constexpr default_atomic_inc_wrap = std::numeric_limits<uint16_t>::max();

    template <bool IS_SOURCE>
    NocUnicastAtomicIncFields(uint32_t dst_address, std::optional<uint32_t> dst_noc_encoding) :
        dst_address(dst_address), dst_noc_encoding(dst_noc_encoding) {
        if constexpr (IS_SOURCE) {
            if (!this->dst_noc_encoding.has_value()) {
                tt::log_fatal(tt::LogTest, "dst_noc_encoding must be set for source");
                throw std::runtime_error("Unexpected NocUnicastAtomicIncFields");
            }
        }
    }

    void set_atomic_inc_val(uint16_t value) { this->atomic_inc_val = value; }
    void set_atomic_inc_wrap(uint16_t value) { this->atomic_inc_wrap = value; }

    std::vector<uint32_t> get_args() const override {
        std::vector<uint32_t> args = {
            atomic_inc_val.value_or(default_atomic_inc_val),
            atomic_inc_wrap.value_or(default_atomic_inc_wrap),
            dst_address};
        if (dst_noc_encoding.has_value()) {
            args.push_back(dst_noc_encoding.value());
        }
    }

    std::optional<uint16_t> atomic_inc_val;
    std::optional<uint16_t> atomic_inc_wrap;
    uint32_t dst_address;
    std::optional<uint32_t> dst_noc_encoding;
};

struct NocUnicastWriteAtomicIncFields : public FieldsBase {
    NocUnicastWriteAtomicIncFields(NocUnicastWriteFields write_fields, NocUnicastAtomicIncFields atomic_inc_fields) :
        write_fields(write_fields), atomic_inc_fields(atomic_inc_fields) {}

    std::vector<uint32_t> get_args() const override {
        std::vector<uint32_t> args;
        const auto write_args = write_fields.get_args();
        const auto atomic_inc_args = atomic_inc_fields.get_args();
        args.insert(args.end(), write_args.begin(), write_args.end());
        args.insert(args.end(), atomic_inc_args.begin(), atomic_inc_args.end());
        return args;
    }

    NocUnicastWriteFields write_fields;
    NocUnicastAtomicIncFields atomic_inc_fields;
};

// create memory maps

// this also has to consider the memory map which has addresses for synchronization etc

struct TestTrafficDataConfig {
    ChipSendType chip_send_type;
    NocSendType noc_send_type;
    uint32_t seed;
    size_t num_packets;
    size_t payload_size_bytes;

    void validate() const;
};

struct TestTrafficConfig {
    TestTrafficDataConfig data_config;
    chip_id_t src_phys_chip_id;
    std::optional<std::vector<chip_id_t>> dst_phys_chip_ids;
    std::optional<std::unordered_map<RoutingDirection, uint32_t>> hops;
    std::optional<CoreCoord> src_logical_core;
    std::optional<CoreCoord> dst_logical_core;
    std::optional<std::string_view> sender_kernel_src;
    std::optional<std::string_view> receiver_kernel_src;
    // TODO: add later
    // mode - BW, latency etc

    void validate() const;
};

struct TestTrafficSenderConfig {
    TestTrafficDataConfig data_config;
    std::vector<chip_id_t> dst_phys_chip_ids;
    std::unordered_map<RoutingDirection, uint32_t> hops;
    CoreCoord dst_logical_core;
    size_t target_address;

    std::vector<uint32_t> get_args();
};

struct TestTrafficReceiverConfig {
    TestTrafficDataConfig data_config;
    uint32_t sender_id;
    size_t target_address;

    std::vector<uint32_t> get_args();
};

inline void TestTrafficDataConfig::validate() const {
    // TODO remove these once converted to enum classes
    if (this->chip_send_type > ChipSendType::CHIP_SEND_TYPE_LAST) {
        tt::log_fatal(tt::LogTest, "Unknown chip send type: {}", this->chip_send_type);
        throw std::runtime_error("Unexpected traffic data config");
    }
    if (this->noc_send_type > NocSendType::NOC_SEND_TYPE_LAST) {
        tt::log_fatal(tt::LogTest, "Unknown noc send type: {}", this->noc_send_type);
        throw std::runtime_error("Unexpected traffic data config");
    }

    if (this->num_packets == 0) {
        tt::log_fatal(tt::LogTest, "Number of packets should be greater than 0");
        throw std::runtime_error("Unexpected traffic data config");
    }

    // TODO: get the max payload size
    size_t max_payload_size_bytes = 0;
    if (this->payload_size_bytes > max_payload_size_bytes) {
        tt::log_fatal(
            tt::LogTest,
            "Max allowed payload size is: {}, but got: {}",
            max_payload_size_bytes,
            this->payload_size_bytes);
        throw std::runtime_error("Unexpected traffic data config");
    }

    // payload size can only be 0 for NOC_UNICAST_ATOMIC_INC
    if (this->payload_size_bytes == 0 && this->noc_send_type != NocSendType::NOC_UNICAST_ATOMIC_INC) {
        tt::log_fatal(tt::LogTest, "Payload size can only be 0 for NOC_UNICAST_ATOMIC_INC");
        throw std::runtime_error("Unexpected traffic data config");
    }
}

inline void TestTrafficConfig::validate() const {
    this->data_config.validate();

    // validate only one of dst chip ids or hops present
    if (!this->dst_phys_chip_ids.has_value() && !this->hops.has_value()) {
        tt::log_fatal(tt::LogTest, "One of dst phys chip ids or num hops should be present, none specified");
        throw std::runtime_error("Unexpected traffic config");
    } else if (this->dst_phys_chip_ids.has_value() && this->hops.has_value()) {
        tt::log_fatal(tt::LogTest, "Only one of dst phys chip ids or num hops should be present, both specified");
        throw std::runtime_error("Unexpected traffic config");
    }

    // validate for mcast dst chip ids should not be present -> only for ucast
    if (this->data_config.chip_send_type == ChipSendType::CHIP_MULTICAST && this->dst_phys_chip_ids.has_value()) {
        tt::log_fatal(tt::LogTest, "For multicast dst chip ids shouldnt be specified, only hops");
        throw std::runtime_error("Unexpected traffic config");
    }
}

inline std::vector<uint32_t> TestTrafficSenderConfig::get_args() {
    // for now only expect hops in a single direction
    uint32_t hops = 0;
    eth_chan_directions outgoing_direction;
    for (const auto& [dir, hops_in_dir] : this->hops) {
        if (hops_in_dir > 0) {
            hops = hops_in_dir;
            eth_chan_directions = ;  // query the interface to get outgoing direction from routing direction
            break;
        }
    }

    if (hops == 0) {
        tt::log_fatal(tt::LogTest, "Expected non-zero hops only in one direction for 1D");
        throw std::runtime_error("Unexpected traffic config");
    }

    std::vector<uint32_t> args;

    const auto metadata = SenderMetadataFields(
        static_cast<uint32_t>(outgoing_direction), this->data_config.num_packets, this->data_config.seed);
    const auto metadata_args = metadata.get_args();
    args.insert(args.end(), metadata_args.begin(), metadata_args.end());

    // TODO: get the topology
    bool is_2d_fabric = false;  // query the interface to get outgoing direction from routing direction

    // push chip send type
    args.push_back(this->data_config.chip_send_type);

    if (is_2d_fabric) {
        tt::log_fatal(tt::LogTest, "2D not supported yet");
        throw std::runtime_error("Unexpected traffic config");
    } else {
        if (this->data_config.chip_send_type == ChipSendType::CHIP_UNICAST) {
            const auto chip_unicast_fields = ChipUnicastFields1D(hops);
            const auto chip_unicast_args = chip_unicast_fields.get_args();
            args.insert(args.end(), chip_unicast_args.begin(), chip_unicast_args.end());
        } else {
            tt::log_fatal(tt::LogTest, "Other chip send types not supported yet");
            throw std::runtime_error("Unexpected traffic config");
        }
    }

    // push noc send type
    args.push_back(this->data_config.noc_send_type);

    if (this->data_config.noc_send_type == NocSendType::NOC_UNICAST_WRITE) {
        uint32_t dst_noc_encoding = ;  // TODO: get this from the interface
        const auto unicast_write_fields =
            NocUnicastWriteFields<true>(this->data_config.payload_size_bytes, this->target_address, dst_noc_encoding);
        const auto unicast_write_args = unicast_write_fields.get_args();
        args.insert(args.end(), unicast_write_args.begin(), unicast_write_args.end());
    } else {
        tt::log_fatal(tt::LogTest, "Other noc send types not supported yet");
        throw std::runtime_error("Unexpected traffic config");
    }

    return args;
}

inline std::vector<uint32_t> TestTrafficReceiverConfig::get_args() {
    std::vector<uint32_t> args;

    const auto metadata =
        ReceiverMetadataFields(this->data_config.num_packets, this->data_config.seed, this->sender_id);
    const auto metadata_args = metadata.get_args();
    args.insert(args.end(), metadata_args.begin(), metadata_args.end());

    // push noc send type
    args.push_back(this->data_config.noc_send_type);

    if (this->data_config.noc_send_type == NocSendType::NOC_UNICAST_WRITE) {
        const auto unicast_write_fields =
            NocUnicastWriteFields<false>(this->data_config.payload_size_bytes, this->target_address);
        const auto unicast_write_args = unicast_write_fields.get_args();
        args.insert(args.end(), unicast_write_args.begin(), unicast_write_args.end());
    } else {
        tt::log_fatal(tt::LogTest, "Other noc send types not supported yet");
        throw std::runtime_error("Unexpected traffic config");
    }

    return args;
}

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
