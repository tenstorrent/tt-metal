// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <functional>
#include <string>
#include <tt_stl/strong_type.hpp>

namespace tt::tt_fabric {

enum class FabricConfig : uint32_t {
    DISABLED = 0,
    FABRIC_1D = 1,                    // 1D routing and no deadlock avoidance
    FABRIC_1D_RING = 2,               // 1D routing and deadlock avoidance using datelines
    FABRIC_2D = 3,                    // 2D routing
    FABRIC_2D_TORUS_X = 4,            // 2D routing and deadlock avoidance along X axis
    FABRIC_2D_TORUS_Y = 5,            // 2D routing and deadlock avoidance along Y axis
    FABRIC_2D_TORUS_XY = 6,           // 2D routing and deadlock avoidance along XY axes
    FABRIC_2D_DYNAMIC = 7,            // 2D routing with dynamic routing
    FABRIC_2D_DYNAMIC_TORUS_X = 8,    // 2D routing with dynamic routing and deadlock avoidance along X axis
    FABRIC_2D_DYNAMIC_TORUS_Y = 9,    // 2D routing with dynamic routing and deadlock avoidance along Y axis
    FABRIC_2D_DYNAMIC_TORUS_XY = 10,  // 2D routing with dynamic routing and deadlock avoidance along XY axes
    CUSTOM = 11
};

// tensix extension for fabric routers, used to build connections between worker - fabric router, upstream fabric router
// - downstream fabric router.
class FabricTensixConfig {
public:
    enum Status : uint32_t {
        DISABLED = 0,  // not using tensix extension
        ENABLED = 1,   // using tensix extension
    };

    enum class SenderChannelExtension : uint32_t {
        DISABLED = 0,  // not using sender channel extension
        MUX = 1,       // using mux kernel as sender channel extension
    };

    // Member variables
    Status status = DISABLED;
    SenderChannelExtension sender_channel = SenderChannelExtension::DISABLED;

    // Constructors
    FabricTensixConfig() = default;
    FabricTensixConfig(Status s, SenderChannelExtension sc = SenderChannelExtension::DISABLED) :
        status(s), sender_channel(sc) {}

    // Helper methods
    bool is_enabled() const { return status == ENABLED; }
    bool is_disabled() const { return status == DISABLED; }
    bool has_mux() const { return is_enabled() && sender_channel == SenderChannelExtension::MUX; }

    // Equality operators
    bool operator==(const FabricTensixConfig& other) const {
        return status == other.status && sender_channel == other.sender_channel;
    }
    bool operator!=(const FabricTensixConfig& other) const { return !(*this == other); }

    // Implicit conversion to int for switch statements (based on whether mux is enabled)
    operator int() const {
        if (is_disabled()) {
            return DISABLED_TYPE;
        }
        if (has_mux()) {
            return MUX_TYPE;
        }
        return DISABLED_TYPE;  // default to disabled
    }

    // String conversion for logging/debugging
    std::string to_string() const {
        if (is_disabled()) {
            return "DISABLED";
        }
        if (has_mux()) {
            return "MUX";
        }
        if (is_enabled()) {
            return "ENABLED";
        }
        return "UNKNOWN";
    }

    // Implicit conversion to string for fmt formatting
    operator std::string() const { return to_string(); }

    // Direct construction is preferred over factory methods to avoid explosion of methods

    // Enum values for switch statements
    enum ConfigType : int {
        DISABLED_TYPE = 0,
        MUX_TYPE = 1,
    };
};

enum class FabricReliabilityMode : uint32_t {

    // When fabric is initialized, user expects live links/devices to exactly match the mesh graph descriptor.
    // Any downed devices/links will result in some sort of error condition being reported.
    STRICT_SYSTEM_HEALTH_SETUP_MODE = 0,

    // When fabric is initialized, user is flexible towards downed links/devices. This mode specifies that fabric
    // can be initialized with fewer routing planes than are in the mesh graph descriptor, according to the number
    // of live links in the system
    RELAXED_SYSTEM_HEALTH_SETUP_MODE = 1,

    // Unsupported - fabric can be setup at runtime. Placeholder
    DYNAMIC_RECONFIGURATION_SETUP_MODE = 2,
};

}  // namespace tt::tt_metal

namespace tt::tt_fabric {

using MeshId = tt::stl::StrongType<uint32_t, struct MeshIdTag>;
using MeshHostRankId = tt::stl::StrongType<uint32_t, struct HostRankTag>;

}  // namespace tt::tt_fabric

// Hash specialization for std::unordered_map support
namespace std {
template <>
struct hash<tt::tt_fabric::FabricTensixConfig> {
    std::size_t operator()(const tt::tt_fabric::FabricTensixConfig& config) const noexcept {
        // Combine hash values of both members
        std::size_t h1 = std::hash<uint32_t>{}(static_cast<uint32_t>(config.status));
        std::size_t h2 = std::hash<uint32_t>{}(static_cast<uint32_t>(config.sender_channel));
        return h1 ^ (h2 << 1);
    }
};
}  // namespace std
