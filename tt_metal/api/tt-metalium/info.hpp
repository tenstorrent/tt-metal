// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

// UMD: re-exports tt::ARCH (the return type of info::architecture).
#include <umd/device/types/arch.hpp>

/**
 * @file
 * @brief Tags for the device query API (`get_info<InfoType>()`).
 *
 * A tag is an empty struct whose `return_type` is the type of the property it names. Tags are shared by every object
 * that can answer a query (currently MeshDevice), so a property has one name and one meaning everywhere.
 *
 * @code
 * uint32_t alignment = mesh_device->get_info<tt::tt_metal::info::l1_alignment>();
 * @endcode
 *
 * To add a property, add a tag here and implement it for each object that supports it.
 */

/**
 * @brief Tags naming the properties that can be queried with `get_info<InfoType>()`.
 *
 * `InfoType::return_type` is the type of the property's value.
 */
namespace tt::tt_metal::info {

/**
 * @brief Required address alignment, in bytes, for L1 allocations.
 */
struct l1_alignment {
    using return_type = std::uint32_t;
};

/**
 * @brief Required address alignment, in bytes, for DRAM allocations.
 */
struct dram_alignment {
    using return_type = std::uint32_t;
};

/**
 * @brief Architecture of the device (e.g. `tt::ARCH::WORMHOLE_B0`).
 */
struct architecture {
    using return_type = tt::ARCH;
};

/**
 * @brief Lowercase name of the architecture of the device (e.g. `"wormhole_b0"`).
 */
struct architecture_name {
    using return_type = std::string;
};

}  // namespace tt::tt_metal::info
