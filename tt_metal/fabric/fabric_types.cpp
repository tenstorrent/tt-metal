// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/fabric/fabric_types.hpp>

namespace tt::tt_fabric {

FabricManagerMode operator|(FabricManagerMode lhs, FabricManagerMode rhs) {
    return static_cast<FabricManagerMode>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

FabricManagerMode operator&(FabricManagerMode lhs, FabricManagerMode rhs) {
    return static_cast<FabricManagerMode>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

bool has_flag(FabricManagerMode flags, FabricManagerMode test) { return (flags & test) == test; }

FabricType operator|(FabricType lhs, FabricType rhs) {
    return static_cast<FabricType>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}

FabricType operator&(FabricType lhs, FabricType rhs) {
    return static_cast<FabricType>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}

bool has_flag(FabricType flags, FabricType test) { return (flags & test) == test; }

}  // namespace tt::tt_fabric
