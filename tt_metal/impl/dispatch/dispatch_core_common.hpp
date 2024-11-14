// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/core_descriptor.hpp"
#include "tt_metal/common/core_coord.hpp"
#include "tt_metal/llrt/get_platform_architecture.hpp"
#include <list>

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

enum DispatchCoreType: uint32_t {
    WORKER = 0,
    ETH = 1,
};

enum class DispatchCoreAxis {
    ROW,
    COL,
};

struct DispatchCoreConfig {
private:
    DispatchCoreType type;
    DispatchCoreAxis axis;

public:
    DispatchCoreConfig()
    : type(DispatchCoreType::WORKER),
      axis(tt::tt_metal::get_platform_architecture() == tt::ARCH::BLACKHOLE ? DispatchCoreAxis::COL : DispatchCoreAxis::ROW) {}

    DispatchCoreConfig(DispatchCoreType type)
        : type(type),
          axis(tt::tt_metal::get_platform_architecture() == tt::ARCH::BLACKHOLE ? DispatchCoreAxis::COL : DispatchCoreAxis::ROW) {}

    DispatchCoreConfig(DispatchCoreType type, DispatchCoreAxis axis) : type(type), axis(axis) {}

    CoreType get_core_type() const {
        const std::unordered_map<DispatchCoreType, CoreType> dispatch_core_type_map = {
            {DispatchCoreType::WORKER, CoreType::WORKER},
            {DispatchCoreType::ETH, CoreType::ETH}
        };
        return dispatch_core_type_map.at(type);
    }

    DispatchCoreType get_dispatch_core_type() const {
        return type;
    }

    void set_dispatch_core_type(DispatchCoreType new_type) {
        type = new_type;
    }

    DispatchCoreAxis get_dispatch_core_axis() const {
        return axis;
    }

    void set_dispatch_core_axis(DispatchCoreAxis new_axis) {
        axis = new_axis;
    }

    bool operator==(const DispatchCoreConfig& other) const {
        return (type == other.type) && (axis == other.axis);
    }
};

}   // namespace tt::tt_metal
