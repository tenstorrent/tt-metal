// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

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

    /// Create default dispatch core config,
    ///
    /// Default type: ETH for N300, T3K, N300_2x2 clusters; else WORKER
    /// Default axis: Blackhole without MUX -> COL; otherwise ROW
    static DispatchCoreConfig create_dispatch_core_config(
        std::optional<DispatchCoreType> dispatch_core_type = std::nullopt,
        std::optional<DispatchCoreAxis> dispatch_core_axis = std::nullopt,
        std::optional<tt::tt_fabric::FabricTensixConfig> fabric_tensix_config = std::nullopt);

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
    std::size_t operator()(const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) const;
};

}  // namespace std
