// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt_stl/reflection.hpp>

#include <umd/device/tt_core_coordinates.h>  // CoreType

namespace tt::tt_metal {

enum DispatchWorkerType : uint32_t {
    PREFETCH = 0,
    PREFETCH_HD = 1,
    PREFETCH_H = 2,
    PREFETCH_D = 3,
    DISPATCH = 4,
    DISPATCH_HD = 5,
    DISPATCH_H = 6,
    DISPATCH_D = 7,
    DISPATCH_S = 8,
    MUX = 9,
    MUX_D = 10,
    DEMUX = 11,
    DEMUX_D = 12,
    US_TUNNELER_LOCAL = 13,
    US_TUNNELER_REMOTE = 14,
    PACKET_ROUTER_MUX = 15,
    PACKET_ROUTER_DEMUX = 16,
    FABRIC_MUX = 17,         // Downstream from MMIO to remote mux. Tunnel index is required.
    RETURN_FABRIC_MUX = 18,  // Upstream from remote to MMIO mux. Tunnel index will be determined from the device id.
    COUNT,
};

enum class DispatchCoreType : uint32_t { WORKER, ETH, COUNT };

enum class DispatchCoreAxis { ROW, COL, COUNT };

class DispatchCoreConfig {
private:
    DispatchCoreType type_;
    std::optional<DispatchCoreAxis> axis_;

    static DispatchCoreAxis get_default_axis();

public:
    DispatchCoreConfig() : type_(DispatchCoreType::WORKER) {}

    DispatchCoreConfig(DispatchCoreType type) : type_(type) {}

    DispatchCoreConfig(DispatchCoreType type, DispatchCoreAxis axis) : type_(type), axis_(axis) {}

    static constexpr auto attribute_names = std::forward_as_tuple("type", "axis");
    auto attribute_values() const { return std::forward_as_tuple(this->type_, this->axis_); }

    CoreType get_core_type() const {
        switch (type_) {
            case DispatchCoreType::WORKER: return CoreType::WORKER;
            case DispatchCoreType::ETH: return CoreType::ETH;
            default: TT_THROW("invalid dispatch core type");
        }
    }

    DispatchCoreType get_dispatch_core_type() const { return type_; }

    void set_dispatch_core_type(DispatchCoreType new_type) { type_ = new_type; }

    DispatchCoreAxis get_dispatch_core_axis() const { return axis_.value_or(get_default_axis()); }

    void set_dispatch_core_axis(DispatchCoreAxis new_axis) { axis_ = new_axis; }

    bool operator==(const DispatchCoreConfig& other) const { return (type_ == other.type_) && (axis_ == other.axis_); }
};

// Helper functions to get the dispatch core config/type
DispatchCoreConfig get_dispatch_core_config();
CoreType get_dispatch_core_type();

}  // namespace tt::tt_metal

namespace std {

template <>
struct hash<tt::tt_metal::DispatchCoreConfig> {
    std::size_t operator()(const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) const {
        return tt::stl::hash::hash_objects_with_default_seed(dispatch_core_config.attribute_values());
    }
};

}  // namespace std
