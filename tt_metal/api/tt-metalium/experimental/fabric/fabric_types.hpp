// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <functional>
#include <ostream>
#include <optional>
#include <tt_stl/strong_type.hpp>
#include <tt_stl/reflection.hpp>

#include <fmt/format.h>

namespace tt::tt_fabric {

enum class FabricConfig : uint32_t {
    DISABLED = 0,
    FABRIC_1D_NEIGHBOR_EXCHANGE = 1,  // 1D topology with no forwarding between non-adjacent devices
    FABRIC_1D = 2,                    // 1D routing and no deadlock avoidance
    FABRIC_1D_RING = 3,               // 1D routing and deadlock avoidance using datelines
    FABRIC_2D = 4,                    // 2D routing
    FABRIC_2D_TORUS_X = 5,            // 2D routing and deadlock avoidance along X axis
    FABRIC_2D_TORUS_Y = 6,            // 2D routing and deadlock avoidance along Y axis
    FABRIC_2D_TORUS_XY = 7,           // 2D routing and deadlock avoidance along XY axes
    CUSTOM = 8
};

// tensix extension for fabric routers, used to build connections between worker - fabric router, upstream fabric router
// - downstream fabric router.
enum class FabricTensixConfig : uint32_t {
    DISABLED = 0,  // not using tensix extension
    MUX = 1,       // using mux kernel as tensix extension
    UDM = 2,       // in udm (unified datamovement) mode, we build both mux and relay kernels as tensix extension
};

// Unified Datamovement knob for configuring fabric with different parameters
enum class FabricUDMMode : uint32_t {
    DISABLED = 0,
    ENABLED = 1,
};

// Fabric manager mode configuration
enum class FabricManagerMode : uint32_t {
    INIT_FABRIC = 1 << 0,
    TERMINATE_FABRIC = 1 << 1,
    ENABLED = (INIT_FABRIC & TERMINATE_FABRIC),
    DEFAULT =
        (INIT_FABRIC |
         TERMINATE_FABRIC),  // Maintains behaviour of Metal runtime, which fully initializes and terminates fabric
};
FabricManagerMode operator|(FabricManagerMode lhs, FabricManagerMode rhs);
FabricManagerMode operator&(FabricManagerMode lhs, FabricManagerMode rhs);
bool has_flag(FabricManagerMode flags, FabricManagerMode test_flag);

// Configuration for router-level parameters
// Extensible for future router tuning (buffer counts, VC settings, etc.)
struct FabricRouterConfig {
    // Optional override for maximum packet payload size (bytes)
    // If not set, uses architecture and routing mode defaults
    std::optional<size_t> max_packet_payload_size_bytes = std::nullopt;
};

enum class FabricType {
    MESH = 1 << 0,
    TORUS_X = 1 << 1,  // Connections along mesh_coord[1]
    TORUS_Y = 1 << 2,  // Connections along mesh_coord[0]
    TORUS_XY = (TORUS_X | TORUS_Y),
};

FabricType operator|(FabricType lhs, FabricType rhs);
FabricType operator&(FabricType lhs, FabricType rhs);
bool has_flag(FabricType flags, FabricType test_flag);

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

}  // namespace tt::tt_fabric

namespace tt::tt_fabric {

using MeshId = tt::stl::StrongType<uint32_t, struct MeshIdTag>;
using MeshHostRankId = tt::stl::StrongType<uint32_t, struct HostRankTag>;
using SwitchId = tt::stl::StrongType<uint32_t, struct SwitchIdTag>;

/**
 * @brief Represents a fabric node identifier combining mesh ID and chip ID
 */
class FabricNodeId {
public:
    explicit FabricNodeId(MeshId mesh_id_val, std::uint32_t chip_id_val);
    MeshId mesh_id{0};
    std::uint32_t chip_id = 0;
};

bool operator==(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator!=(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator<(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator>(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator<=(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator>=(const FabricNodeId& lhs, const FabricNodeId& rhs);
std::ostream& operator<<(std::ostream& os, const FabricNodeId& fabric_node_id);

}  // namespace tt::tt_fabric

namespace std {
template <>
struct hash<tt::tt_fabric::FabricNodeId> {
    size_t operator()(const tt::tt_fabric::FabricNodeId& fabric_node_id) const noexcept;
};
}  // namespace std

template <>
struct fmt::formatter<tt::tt_fabric::FabricNodeId> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::tt_fabric::FabricNodeId& node_id, format_context& ctx) const -> format_context::iterator;
};
namespace tt::tt_metal {

using AsicID = tt::stl::StrongType<uint64_t, struct AsicIDTag>;
using TrayID = tt::stl::StrongType<uint32_t, struct TrayIDTag>;
using ASICLocation = tt::stl::StrongType<uint32_t, struct ASICLocationTag>;
using ASICPosition = std::pair<TrayID, ASICLocation>;
using RackID = tt::stl::StrongType<uint32_t, struct RackIDTag>;
using UID = tt::stl::StrongType<uint32_t, struct UIDTag>;
using HallID = tt::stl::StrongType<uint32_t, struct HallIDTag>;
using AisleID = tt::stl::StrongType<uint32_t, struct AisleIDTag>;

}  // namespace tt::tt_metal
