// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/mesh_graph.hpp>

namespace tt::tt_fabric {
namespace fabric_tests {

struct SenderMetadataFields {
    SenderMetadataFields(uint32_t num_packets, uint32_t seed, uint32_t payload_buffer_size) :
        num_packets(num_packets), seed(seed), payload_buffer_size(payload_buffer_size) {}

    std::vector<uint32_t> get_args() const {
        std::vector<uint32_t> args = {num_packets, seed, payload_buffer_size};
        return args;
    }

    uint32_t num_packets;
    uint32_t seed;
    uint32_t payload_buffer_size;
};

struct ReceiverMetadataFields {
    // TODO: add sender id to these fields
    ReceiverMetadataFields(uint32_t num_packets, uint32_t seed, uint32_t payload_buffer_size) :
        num_packets(num_packets), seed(seed), payload_buffer_size(payload_buffer_size) {}

    std::vector<uint32_t> get_args() const {
        std::vector<uint32_t> args = {num_packets, seed, payload_buffer_size};
        return args;
    }

    uint32_t num_packets;
    uint32_t seed;
    uint32_t payload_buffer_size;
};

struct ChipUnicastFields1D {
    ChipUnicastFields1D(uint32_t num_hops) : num_hops(num_hops) {}

    std::vector<uint32_t> get_args() const {
        std::vector<uint32_t> args = {num_hops};
        return args;
    }

    uint32_t num_hops;
};

struct ChipUnicastFields2D {
    ChipUnicastFields2D(uint16_t src_device_id, uint16_t dst_device_id, uint16_t dst_mesh_id, uint16_t ew_dim) :
        src_device_id(src_device_id), dst_device_id(dst_device_id), dst_mesh_id(dst_mesh_id), ew_dim(ew_dim) {}

    std::vector<uint32_t> get_args() const {
        std::vector<uint32_t> args = {src_device_id, dst_device_id, dst_mesh_id, ew_dim};
        return args;
    }

    uint16_t src_device_id;
    uint16_t dst_device_id;
    uint16_t dst_mesh_id;
    uint16_t ew_dim;
};

struct ChipMulticastFields1D {
    static constexpr uint32_t default_mcast_start_hops = 1;

    ChipMulticastFields1D(uint32_t num_hops) : num_hops(num_hops) {}

    void set_mcast_start_hops(uint32_t value) { this->mcast_start_hops = value; }

    std::vector<uint32_t> get_args() const {
        std::vector<uint32_t> args = {mcast_start_hops.value_or(default_mcast_start_hops), num_hops};
        return args;
    }

    std::optional<uint32_t> mcast_start_hops;
    uint32_t num_hops;
};

struct ChipMulticastFields2D {
    ChipMulticastFields2D(
        uint16_t dst_device_id, uint16_t dst_mesh_id, std::unordered_map<RoutingDirection, uint32_t> hops) :
        dst_device_id(dst_device_id), dst_mesh_id(dst_mesh_id) {
        this->num_hops_n = hops[RoutingDirection::N];
        this->num_hops_s = hops[RoutingDirection::S];
        this->num_hops_e = hops[RoutingDirection::E];
        this->num_hops_w = hops[RoutingDirection::W];
    }

    std::vector<uint32_t> get_args() const {
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

struct NocUnicastWriteFields {
    NocUnicastWriteFields(
        uint32_t payload_size_bytes, uint32_t dst_address, std::optional<uint32_t> dst_noc_encoding = std::nullopt) :
        payload_size_bytes(payload_size_bytes), dst_address(dst_address), dst_noc_encoding(dst_noc_encoding) {}

    template <bool IS_SOURCE>
    std::vector<uint32_t> get_args() const {
        if constexpr (IS_SOURCE) {
            if (!this->dst_noc_encoding.has_value()) {
                log_fatal(tt::LogTest, "dst_noc_encoding must be set for source");
                throw std::runtime_error("Unexpected NocUnicastWriteFields");
            }
        }
        std::vector<uint32_t> args = {payload_size_bytes, dst_address};
        if (dst_noc_encoding.has_value()) {
            args.push_back(dst_noc_encoding.value());
        }
        return args;
    }

    uint32_t payload_size_bytes;
    uint32_t dst_address;
    std::optional<uint32_t> dst_noc_encoding;
};

struct NocUnicastAtomicIncFields {
    static constexpr uint32_t default_atomic_inc_val = 1;
    static constexpr uint16_t default_atomic_inc_wrap = std::numeric_limits<uint16_t>::max();

    NocUnicastAtomicIncFields(uint32_t dst_address, std::optional<uint32_t> dst_noc_encoding = std::nullopt) :
        dst_address(dst_address), dst_noc_encoding(dst_noc_encoding) {}

    void set_atomic_inc_val(uint16_t value) { this->atomic_inc_val = value; }
    void set_atomic_inc_wrap(uint16_t value) { this->atomic_inc_wrap = value; }

    template <bool IS_SOURCE>
    std::vector<uint32_t> get_args() const {
        if constexpr (IS_SOURCE) {
            if (!this->dst_noc_encoding.has_value()) {
                log_fatal(tt::LogTest, "dst_noc_encoding must be set for source");
                throw std::runtime_error("Unexpected NocUnicastAtomicIncFields");
            }
        }
        std::vector<uint32_t> args = {
            atomic_inc_val.value_or(default_atomic_inc_val),
            atomic_inc_wrap.value_or(default_atomic_inc_wrap),
            dst_address};
        if (dst_noc_encoding.has_value()) {
            args.push_back(dst_noc_encoding.value());
        }
        return args;
    }

    std::optional<uint16_t> atomic_inc_val;
    std::optional<uint16_t> atomic_inc_wrap;
    uint32_t dst_address;
    std::optional<uint32_t> dst_noc_encoding;
};

struct NocUnicastWriteAtomicIncFields {
    NocUnicastWriteAtomicIncFields(NocUnicastWriteFields write_fields, NocUnicastAtomicIncFields atomic_inc_fields) :
        write_fields(write_fields), atomic_inc_fields(atomic_inc_fields) {}

    template <bool IS_SOURCE>
    std::vector<uint32_t> get_args() const {
        std::vector<uint32_t> args;
        const auto write_args = write_fields.get_args<IS_SOURCE>();
        const auto atomic_inc_args = atomic_inc_fields.get_args<IS_SOURCE>();
        args.insert(args.end(), write_args.begin(), write_args.end());
        args.insert(args.end(), atomic_inc_args.begin(), atomic_inc_args.end());
        return args;
    }

    NocUnicastWriteFields write_fields;
    NocUnicastAtomicIncFields atomic_inc_fields;
};

// create memory maps

// this also has to consider the memory map which has addresses for synchronization etc

struct TrafficParameters {
    // from TrafficPatternConfig
    ChipSendType chip_send_type;
    NocSendType noc_send_type;
    size_t payload_size_bytes;
    size_t num_packets;
    std::optional<uint16_t> atomic_inc_val;
    std::optional<uint16_t> atomic_inc_wrap;
    std::optional<uint32_t> mcast_start_hops;

    // Global context
    uint32_t seed;
    Topology topology;
    RoutingType routing_type;
    tt::tt_metal::distributed::MeshShape mesh_shape;
};

struct TestTrafficConfig {
    TrafficParameters parameters;
    FabricNodeId src_node_id;
    std::optional<std::vector<FabricNodeId>> dst_node_ids;
    std::optional<std::unordered_map<RoutingDirection, uint32_t>> hops;
    std::optional<CoreCoord> src_logical_core;
    std::optional<CoreCoord> dst_logical_core;
    std::optional<uint32_t> target_address;
    std::optional<uint32_t> atomic_inc_address;
    // TODO: add later
    // mode - BW, latency etc
};

struct TestTrafficSenderConfig {
    TrafficParameters parameters;
    FabricNodeId src_node_id;
    std::vector<FabricNodeId> dst_node_ids;
    std::unordered_map<RoutingDirection, uint32_t> hops;
    CoreCoord dst_logical_core;
    size_t target_address;
    std::optional<size_t> atomic_inc_address;
    uint32_t dst_noc_encoding;  // TODO: decide if we should keep it here or not
    uint32_t payload_buffer_size;  // Add payload buffer size field

    std::vector<uint32_t> get_args(bool is_sync_config = false) const;
};

struct TestTrafficReceiverConfig {
    TrafficParameters parameters;
    uint32_t sender_id;
    size_t target_address;
    std::optional<size_t> atomic_inc_address;
    uint32_t payload_buffer_size;  // Add payload buffer size field

    std::vector<uint32_t> get_args() const;
};

inline std::vector<uint32_t> TestTrafficSenderConfig::get_args(bool is_sync_config) const {
    std::vector<uint32_t> args;
    args.reserve(20);  // Reserve a reasonable upper bound to avoid reallocations

    if (!is_sync_config) {
        const auto metadata =
            SenderMetadataFields(this->parameters.num_packets, this->parameters.seed, this->payload_buffer_size);
        const auto metadata_args = metadata.get_args();
        args.insert(args.end(), metadata_args.begin(), metadata_args.end());
    }

    bool is_2d_fabric = (this->parameters.topology == Topology::Mesh);

    // push chip send type
    if (!is_sync_config) {
        args.push_back(this->parameters.chip_send_type);
    }

    if (is_2d_fabric) {
        if (this->parameters.chip_send_type == ChipSendType::CHIP_UNICAST) {
            TT_FATAL(this->dst_node_ids.size() == 1, "2D unicast should have exactly one destination node.");
            const auto& dst_node_id = this->dst_node_ids[0];
            const auto& mesh_shape = this->parameters.mesh_shape;
            // TODO: move this out of here
            const uint32_t EW_DIM = 1;
            const auto unicast_fields = ChipUnicastFields2D(
                this->src_node_id.chip_id, dst_node_id.chip_id, *dst_node_id.mesh_id, mesh_shape[EW_DIM]);
            const auto unicast_args = unicast_fields.get_args();
            args.insert(args.end(), unicast_args.begin(), unicast_args.end());
        } else if (this->parameters.chip_send_type == ChipSendType::CHIP_MULTICAST) {
            TT_FATAL(!this->dst_node_ids.empty(), "2D multicast should have at least one destination node.");
            const auto& dst_rep_node_id = this->dst_node_ids[0];  // Representative destination
            auto adjusted_hops = this->hops;

            // Handle dynamic routing by adjusting hops
            bool is_dynamic_routing = (this->parameters.routing_type == RoutingType::Dynamic);
            if (is_dynamic_routing) {
                auto north_hops = hops.count(RoutingDirection::N) > 0 ? hops.at(RoutingDirection::N) : 0;
                auto south_hops = hops.count(RoutingDirection::S) > 0 ? hops.at(RoutingDirection::S) : 0;
                auto east_hops = hops.count(RoutingDirection::E) > 0 ? hops.at(RoutingDirection::E) : 0;
                auto west_hops = hops.count(RoutingDirection::W) > 0 ? hops.at(RoutingDirection::W) : 0;
                // for dynamic routing, decrement north/south hops by 1, since the start dst node is accounted as one
                // hop.
                if (north_hops > 0) {
                    adjusted_hops[RoutingDirection::N] -= 1;
                }
                if (south_hops > 0) {
                    adjusted_hops[RoutingDirection::S] -= 1;
                }
                // for dynamic routing, decrement east/west hops by 1, since the start dst node is accounted as one hop.
                if (north_hops == 0 && south_hops == 0 && east_hops > 0) {
                    adjusted_hops[RoutingDirection::E] -= 1;
                }
                if (north_hops == 0 && south_hops == 0 && west_hops > 0) {
                    adjusted_hops[RoutingDirection::W] -= 1;
                }
            }

            // chip_id and mesh_id is unused for low latency 2d mesh mcast
            const auto mcast_fields =
                ChipMulticastFields2D(dst_rep_node_id.chip_id, *dst_rep_node_id.mesh_id, adjusted_hops);
            const auto mcast_args = mcast_fields.get_args();
            args.insert(args.end(), mcast_args.begin(), mcast_args.end());
        } else {
            TT_FATAL(false, "Unsupported chip send type for 2D fabric");
        }
    } else {  // 1D logic
        uint32_t num_hops_1d = 0;
        for (const auto& [_, hops_in_dir] : this->hops) {
            if (hops_in_dir > 0) {
                num_hops_1d = hops_in_dir;
                break;
            }
        }

        if (this->parameters.chip_send_type == ChipSendType::CHIP_UNICAST) {
            const auto chip_unicast_fields = ChipUnicastFields1D(num_hops_1d);
            const auto chip_unicast_args = chip_unicast_fields.get_args();
            args.insert(args.end(), chip_unicast_args.begin(), chip_unicast_args.end());
        } else if (this->parameters.chip_send_type == ChipSendType::CHIP_MULTICAST) {
            auto mcast_fields = ChipMulticastFields1D(num_hops_1d);
            if (this->parameters.mcast_start_hops.has_value()) {
                mcast_fields.set_mcast_start_hops(this->parameters.mcast_start_hops.value());
            }
            const auto mcast_args = mcast_fields.get_args();
            args.insert(args.end(), mcast_args.begin(), mcast_args.end());
        } else {
            TT_FATAL(false, "Unsupported chip send type for 1D fabric");
        }
    }

    // push noc send type
    if (!is_sync_config) {
        args.push_back(this->parameters.noc_send_type);
    }

    switch (this->parameters.noc_send_type) {
        case NocSendType::NOC_UNICAST_WRITE: {
            const auto unicast_write_fields = NocUnicastWriteFields(
                this->parameters.payload_size_bytes, this->target_address, this->dst_noc_encoding);
            const auto unicast_write_args = unicast_write_fields.get_args<true>();
            args.insert(args.end(), unicast_write_args.begin(), unicast_write_args.end());
        } break;
        case NocSendType::NOC_UNICAST_ATOMIC_INC: {
            auto atomic_inc_fields =
                NocUnicastAtomicIncFields(this->atomic_inc_address.value(), this->dst_noc_encoding);
            if (this->parameters.atomic_inc_val.has_value()) {
                atomic_inc_fields.set_atomic_inc_val(this->parameters.atomic_inc_val.value());
            }
            if (this->parameters.atomic_inc_wrap.has_value()) {
                atomic_inc_fields.set_atomic_inc_wrap(this->parameters.atomic_inc_wrap.value());
            }
            const auto atomic_inc_args = atomic_inc_fields.get_args<true>();
            args.insert(args.end(), atomic_inc_args.begin(), atomic_inc_args.end());
        } break;
        case NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC: {
            const auto write_fields = NocUnicastWriteFields(
                this->parameters.payload_size_bytes, this->target_address, this->dst_noc_encoding);
            auto atomic_inc_fields =
                NocUnicastAtomicIncFields(this->atomic_inc_address.value(), this->dst_noc_encoding);
            if (this->parameters.atomic_inc_val.has_value()) {
                atomic_inc_fields.set_atomic_inc_val(this->parameters.atomic_inc_val.value());
            }
            if (this->parameters.atomic_inc_wrap.has_value()) {
                atomic_inc_fields.set_atomic_inc_wrap(this->parameters.atomic_inc_wrap.value());
            }
            const auto fused_fields = NocUnicastWriteAtomicIncFields(write_fields, atomic_inc_fields);
            const auto fused_args = fused_fields.get_args<true>();
            args.insert(args.end(), fused_args.begin(), fused_args.end());
        } break;
        default: TT_FATAL(false, "Unsupported noc send type");
    }

    return args;
}

inline std::vector<uint32_t> TestTrafficReceiverConfig::get_args() const {
    std::vector<uint32_t> args;
    args.reserve(10);  // Reserve a reasonable upper bound to avoid reallocations

    const auto metadata =
        ReceiverMetadataFields(this->parameters.num_packets, this->parameters.seed, this->payload_buffer_size);
    const auto metadata_args = metadata.get_args();
    args.insert(args.end(), metadata_args.begin(), metadata_args.end());

    // push noc send type
    args.push_back(this->parameters.noc_send_type);

    switch (this->parameters.noc_send_type) {
        case NocSendType::NOC_UNICAST_WRITE: {
            const auto unicast_write_fields =
                NocUnicastWriteFields(this->parameters.payload_size_bytes, this->target_address);
            const auto unicast_write_args = unicast_write_fields.get_args<false>();
            args.insert(args.end(), unicast_write_args.begin(), unicast_write_args.end());
            break;
        }
        case NocSendType::NOC_UNICAST_ATOMIC_INC: {
            auto atomic_inc_fields = NocUnicastAtomicIncFields(this->atomic_inc_address.value());
            if (this->parameters.atomic_inc_val.has_value()) {
                atomic_inc_fields.set_atomic_inc_val(this->parameters.atomic_inc_val.value());
            }
            if (this->parameters.atomic_inc_wrap.has_value()) {
                atomic_inc_fields.set_atomic_inc_wrap(this->parameters.atomic_inc_wrap.value());
            }
            const auto atomic_inc_args = atomic_inc_fields.get_args<false>();
            args.insert(args.end(), atomic_inc_args.begin(), atomic_inc_args.end());
            break;
        }
        case NocSendType::NOC_FUSED_UNICAST_ATOMIC_INC: {
            const auto write_fields = NocUnicastWriteFields(this->parameters.payload_size_bytes, this->target_address);
            auto atomic_inc_fields = NocUnicastAtomicIncFields(this->atomic_inc_address.value());
            if (this->parameters.atomic_inc_val.has_value()) {
                atomic_inc_fields.set_atomic_inc_val(this->parameters.atomic_inc_val.value());
            }
            if (this->parameters.atomic_inc_wrap.has_value()) {
                atomic_inc_fields.set_atomic_inc_wrap(this->parameters.atomic_inc_wrap.value());
            }
            const auto fused_fields = NocUnicastWriteAtomicIncFields(write_fields, atomic_inc_fields);
            const auto fused_args = fused_fields.get_args<false>();
            args.insert(args.end(), fused_args.begin(), fused_args.end());
            break;
        }
        default: TT_FATAL(false, "Unsupported noc send type");
    }

    return args;
}

}  // namespace fabric_tests
}  // namespace tt::tt_fabric
