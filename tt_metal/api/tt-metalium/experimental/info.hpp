// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <string_view>

// UMD: re-exports tt::ARCH (the return type of info::architecture).
#include <umd/device/types/arch.hpp>

/**
 * @file
 * These experimental tags may change without notice.
 *
 * @brief Tags for the device query API (`get_info<InfoType>()`).
 *
 * A tag is an empty struct with a `return_type` for its value and a static `name` for diagnostics. Tags are shared by
 * every object that can answer a query (currently MeshDevice), so a property has one name and one meaning everywhere.
 *
 * @code
 * namespace info = tt::tt_metal::experimental::info;
 * using tt::tt_metal::experimental::mesh_device::get_info;
 * uint32_t alignment = get_info<info::l1_alignment>(*mesh_device);
 * @endcode
 *
 * To add a property, declare it with METALIUM_INFO here, add it to `all_tags`, and implement it for each object that
 * supports it.
 */

// Variadic so return types containing commas can be passed directly.
#define METALIUM_INFO(Name, ...)                        \
    struct Name {                                       \
        using return_type = __VA_ARGS__;                \
        static constexpr std::string_view name = #Name; \
    }

/**
 * @brief Tags naming the properties that can be queried with `get_info<InfoType>()`.
 *
 * `InfoType::return_type` is the type of the property's value.
 */

namespace tt::tt_metal::experimental::info {

/**
 * @brief Required address alignment, in bytes, for L1 allocations.
 */
METALIUM_INFO(l1_alignment, std::uint32_t);

/**
 * @brief Required address alignment, in bytes, for DRAM allocations.
 */
METALIUM_INFO(dram_alignment, std::uint32_t);

/**
 * @brief Architecture of the device (e.g. `tt::ARCH::WORMHOLE_B0`).
 */
METALIUM_INFO(architecture, tt::ARCH);

/**
 * @brief Lowercase name of the architecture of the device (e.g. `"wormhole_b0"`).
 */
METALIUM_INFO(architecture_name, std::string);

/**
 * @brief A list of tags, so that bindings and generic tooling can visit every property.
 */
template <class... Tags>
struct tag_list {};

/**
 * @brief Every tag declared above. Add each new tag here as well as declaring it.
 */
using all_tags = tag_list<l1_alignment, dram_alignment, architecture, architecture_name>;

}  // namespace tt::tt_metal::experimental::info

#undef METALIUM_INFO
