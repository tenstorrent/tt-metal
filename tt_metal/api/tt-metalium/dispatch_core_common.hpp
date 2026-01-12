// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/assert.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt_stl/reflection.hpp>

namespace tt::tt_metal {

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

    DispatchCoreType get_dispatch_core_type() const { return type_; }

    void set_dispatch_core_type(DispatchCoreType new_type) { type_ = new_type; }

    DispatchCoreAxis get_dispatch_core_axis() const { return axis_.value_or(get_default_axis()); }

    void set_dispatch_core_axis(DispatchCoreAxis new_axis) { axis_ = new_axis; }

    bool operator==(const DispatchCoreConfig& other) const = default;
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
