// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt_stl/assert.hpp>
#include <tt-metalium/core_coord.hpp>

namespace tt::tt_metal {

enum class DispatchCoreType : uint32_t { WORKER, ETH, COUNT };

enum class DispatchCoreAxis { ROW, COL, COUNT };

class DispatchCoreConfig {
private:
    DispatchCoreType type_;
    std::optional<DispatchCoreAxis> axis_;

public:
    DispatchCoreConfig() : type_(DispatchCoreType::WORKER) {}

    DispatchCoreConfig(DispatchCoreType type) : type_(type) {}

    DispatchCoreConfig(DispatchCoreType type, DispatchCoreAxis axis) : type_(type), axis_(axis) {}

    static constexpr auto attribute_names = std::forward_as_tuple("type", "axis");
    auto attribute_values() const { return std::forward_as_tuple(this->type_, this->axis_); }

    DispatchCoreType get_dispatch_core_type() const { return type_; }

    void set_dispatch_core_type(DispatchCoreType new_type) { type_ = new_type; }

    DispatchCoreAxis get_dispatch_core_axis() const {
        TT_FATAL(
            axis_.has_value(),
            "Dispatch core axis has not been resolved. Set it explicitly or call resolve_dispatch_core_axis().");
        return axis_.value();
    }

    void set_dispatch_core_axis(DispatchCoreAxis new_axis) { axis_ = new_axis; }

    bool operator==(const DispatchCoreConfig& other) const = default;
};

}  // namespace tt::tt_metal

namespace std {

template <>
struct hash<tt::tt_metal::DispatchCoreConfig> {
    std::size_t operator()(const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) const;
};

}  // namespace std
