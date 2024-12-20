// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/core_descriptor.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/llrt/get_platform_architecture.hpp"
#include "tt_metal/tt_stl/reflection.hpp"

namespace tt::tt_metal {

enum DispatchWorkerType : uint32_t {
    PREFETCH = 0,
    PREFETCH_D = 1,
    DISPATCH = 2,
    DISPATCH_D = 3,
    DISPATCH_S = 4,
    MUX = 5,
    MUX_D = 6,
    DEMUX = 7,
    DEMUX_D = 8,
    US_TUNNELER_LOCAL = 9,
    US_TUNNELER_REMOTE = 10,
    DS_TUNNELER_LOCAL = 11,
    DS_TUNNELER_REMOTE = 12,
    COUNT = 13
};

enum class DispatchCoreType : uint32_t { WORKER, ETH, COUNT };

enum class DispatchCoreAxis { ROW, COL, COUNT };

class DispatchCoreConfig {
private:
    DispatchCoreType type_;
    DispatchCoreAxis axis_;

    static DispatchCoreAxis get_default_axis() {
        return (tt::tt_metal::get_platform_architecture() == tt::ARCH::BLACKHOLE) ? DispatchCoreAxis::COL
                                                                                  : DispatchCoreAxis::ROW;
    }

public:
    DispatchCoreConfig() : type_(DispatchCoreType::WORKER), axis_(get_default_axis()) {}

    DispatchCoreConfig(DispatchCoreType type) : type_(type), axis_(get_default_axis()) {}

    DispatchCoreConfig(DispatchCoreType type, DispatchCoreAxis axis) : type_(type), axis_(axis) {}

    static constexpr auto attribute_names = std::forward_as_tuple("type", "axis");
    const auto attribute_values() const { return std::forward_as_tuple(this->type_, this->axis_); }

    CoreType get_core_type() const {
        switch (type_) {
            case DispatchCoreType::WORKER: return CoreType::WORKER;
            case DispatchCoreType::ETH: return CoreType::ETH;
            default: TT_THROW("invalid dispatch core type");
        }
    }

    DispatchCoreType get_dispatch_core_type() const { return type_; }

    void set_dispatch_core_type(DispatchCoreType new_type) { type_ = new_type; }

    DispatchCoreAxis get_dispatch_core_axis() const { return axis_; }

    void set_dispatch_core_axis(DispatchCoreAxis new_axis) { axis_ = new_axis; }

    bool operator==(const DispatchCoreConfig& other) const { return (type_ == other.type_) && (axis_ == other.axis_); }
};

}  // namespace tt::tt_metal

namespace std {

template <>
struct hash<tt::tt_metal::DispatchCoreConfig> {
    std::size_t operator()(const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) const {
        return tt::stl::hash::hash_objects_with_default_seed(dispatch_core_config.attribute_values());
    }
};

}  // namespace std
