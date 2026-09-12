// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <functional>
#include <ostream>
#include <optional>
#include <vector>
#include <tt_stl/assert.hpp>
#include <tt_stl/strong_type.hpp>

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

// A declared torus dimension realizes a distinct wrap edge only at size three or
// larger. Size-one and size-two dimensions retain ordinary mesh links.
constexpr bool is_genuine_torus_dim(uint32_t dim_size) { return dim_size > 2; }

inline std::vector<int32_t> row_major_coords_from_linear_index(
    uint32_t linear_index, const std::vector<int32_t>& dims) {
    std::vector<int32_t> coords(dims.size());
    int32_t remaining = static_cast<int32_t>(linear_index);
    for (int32_t dim_idx = static_cast<int32_t>(dims.size()) - 1; dim_idx >= 0; --dim_idx) {
        const int32_t dim_size = dims[static_cast<size_t>(dim_idx)];
        coords[static_cast<size_t>(dim_idx)] = remaining % dim_size;
        remaining /= dim_size;
    }
    return coords;
}

inline uint32_t row_major_linear_index_from_coords(
    const std::vector<int32_t>& coords, const std::vector<int32_t>& dims) {
    uint32_t linear_index = 0;
    uint32_t multiplier = 1;
    for (int32_t dim_idx = static_cast<int32_t>(dims.size()) - 1; dim_idx >= 0; --dim_idx) {
        linear_index += static_cast<uint32_t>(coords[static_cast<size_t>(dim_idx)]) * multiplier;
        multiplier *= static_cast<uint32_t>(dims[static_cast<size_t>(dim_idx)]);
    }
    return linear_index;
}

// Row-major chip index in device_dims -> host-partition index in host_dims. Matches MeshGraph host-rank tiling.
inline uint32_t host_partition_index_for_row_major_chip(
    uint32_t chip_index, const std::vector<int32_t>& device_dims, const std::vector<int32_t>& host_dims) {
    TT_FATAL(
        device_dims.size() == host_dims.size(),
        "Device topology dims {} do not match host topology dims {}",
        device_dims.size(),
        host_dims.size());
    const std::vector<int32_t> device_coords = row_major_coords_from_linear_index(chip_index, device_dims);
    std::vector<int32_t> host_coords(host_dims.size());
    for (size_t dim_idx = 0; dim_idx < device_dims.size(); ++dim_idx) {
        TT_FATAL(
            host_dims[dim_idx] > 0 && device_dims[dim_idx] % host_dims[dim_idx] == 0,
            "Device dim {} is not divisible by host dim {}",
            device_dims[dim_idx],
            host_dims[dim_idx]);
        const int32_t tiles_per_host = device_dims[dim_idx] / host_dims[dim_idx];
        host_coords[dim_idx] = device_coords[dim_idx] / tiles_per_host;
    }
    return row_major_linear_index_from_coords(host_coords, host_dims);
}

// MeshShape axis 0 (north/south) maps to TORUS_Y; axis 1 (east/west) maps to TORUS_X.
constexpr FabricType torus_flag_for_axis(uint32_t axis) {
    return axis == 0 ? FabricType::TORUS_Y : FabricType::TORUS_X;
}

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

using MeshId = ttsl::StrongType<uint32_t, struct MeshIdTag>;
using MeshHostRankId = ttsl::StrongType<uint32_t, struct HostRankTag>;
using SwitchId = ttsl::StrongType<uint32_t, struct SwitchIdTag>;

// Sentinel value indicating that TT_MESH_HOST_RANK environment variable is unset
constexpr MeshHostRankId MESH_HOST_RANK_UNSET{UINT32_MAX};

// Mesh-local logical chip id (row-major node within a single mesh), matching FabricNodeId::chip_id. The full
// FabricNodeId (mesh_id + chip_id) is only known once a logical MeshId is assigned, so pre-assignment contexts
// carry just the chip id.
using LogicalChipId = uint32_t;

// Node id within one PGD grouping's adjacency graph (proto Instance.id). Named so it is not
// conflated with LogicalChipId (a mesh-local MGD chip) or a physical chip id.
using GroupingChipId = uint32_t;

// Stable numeric handle of one resolved PGD grouping instance (groupings are otherwise identified
// by their name/type strings).
using PhysicalGroupingId = uint32_t;

/**
 * @brief Represents a fabric node identifier combining mesh ID and logical chip ID
 */
class FabricNodeId {
public:
    explicit FabricNodeId(MeshId mesh_id_val, LogicalChipId chip_id_val);
    MeshId mesh_id{0};
    LogicalChipId chip_id = 0;
};

bool operator==(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator!=(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator<(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator>(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator<=(const FabricNodeId& lhs, const FabricNodeId& rhs);
bool operator>=(const FabricNodeId& lhs, const FabricNodeId& rhs);
std::ostream& operator<<(std::ostream& os, const MeshId& mesh_id);
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

template <>
struct fmt::formatter<tt::tt_fabric::MeshId> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::tt_fabric::MeshId& mesh_id, format_context& ctx) const -> format_context::iterator;
};

namespace tt::tt_metal {

// Physical port / cable type for ethernet connections
enum class PortType {
    UNKNOWN,
    TRACE,
    QSFP_DD,
    WARP100,
    WARP400,
    LINKING_BOARD_1,
    LINKING_BOARD_2,
    LINKING_BOARD_3,
};

using AsicID = ttsl::StrongType<uint64_t, struct AsicIDTag>;
using TrayID = ttsl::StrongType<uint32_t, struct TrayIDTag>;
using ASICLocation = ttsl::StrongType<uint32_t, struct ASICLocationTag>;
using ASICPosition = std::pair<TrayID, ASICLocation>;
using RackID = ttsl::StrongType<uint32_t, struct RackIDTag>;
using UID = ttsl::StrongType<uint32_t, struct UIDTag>;
using HallID = ttsl::StrongType<uint32_t, struct HallIDTag>;
using AisleID = ttsl::StrongType<uint32_t, struct AisleIDTag>;

// Stream operators for StrongType types
std::ostream& operator<<(std::ostream& os, const AsicID& asic_id);
std::ostream& operator<<(std::ostream& os, const TrayID& tray_id);
std::ostream& operator<<(std::ostream& os, const ASICLocation& asic_location);

}  // namespace tt::tt_metal

template <>
struct fmt::formatter<tt::tt_metal::AsicID> {
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator { return ctx.end(); }

    auto format(const tt::tt_metal::AsicID& asic_id, format_context& ctx) const -> format_context::iterator;
};
