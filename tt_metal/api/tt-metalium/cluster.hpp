// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

namespace tt::tt_metal {

/**
 * @brief Represents different types of hardware clusters
 *
 * @warning SERIALIZATION NOTE: This enum is exposed to Python bindings and may be
 * serialized in various contexts (Python pickle, JSON, configuration files, etc.).
 * The explicit integer values assigned to each enum member are part of the stable
 * API contract.
 *
 * ORDERING CONSTRAINTS:
 * - DO NOT change existing enum values (breaks backward compatibility)
 * - DO NOT reorder existing entries (breaks serialized data)
 * - New enum values MUST be added at the end before the closing brace
 * - Use the next sequential integer value for new entries
 * - Mark deprecated entries with comments but DO NOT remove them
 */
enum class ClusterType : std::uint8_t {
    INVALID = 0,
    N150 = 1,                    // Production N150
    N300 = 2,                    // Production N300
    T3K = 3,                     // Production T3K, built with 4 N300s
    GALAXY = 4,                  // Production Galaxy, all chips with mmio
    TG = 5,                      // Will be deprecated
    P100 = 6,                    // Blackhole single card, ethernet disabled
    P150 = 7,                    // Blackhole single card, ethernet enabled
    P150_X2 = 8,                 // 2 Blackhole single card, ethernet connected
    P150_X4 = 9,                 // 4 Blackhole single card, ethernet connected
    SIMULATOR_WORMHOLE_B0 = 10,  // Simulator Wormhole B0
    SIMULATOR_BLACKHOLE = 11,    // Simulator Blackhole
    N300_2x2 = 12,               // 2 N300 cards, ethernet connected to form 2x2
    P150_X8 = 13,                // 8 Blackhole single card, ethernet connected
    P300 = 14,                   // Production P300
    SIMULATOR_QUASAR = 15,       // Simulator Quasar
    BLACKHOLE_GALAXY = 16,       // Blackhole Galaxy, all chips with mmio
    P300_X2 = 17,                // 2 P300 cards
    CUSTOM = 18,  // Custom cluster type, used boards with custom fabric mesh graph descriptor path specified
};

/**
 * @brief Get the type of the current cluster
 *
 * @return ClusterType The cluster type detected at runtime
 */
tt::tt_metal::ClusterType GetClusterType();

/**
 * @brief Serialize the cluster descriptor to a file and return the path
 *
 * @return std::string Path to the serialized cluster descriptor file
 */
std::string SerializeClusterDescriptor();

}  // namespace tt::tt_metal
