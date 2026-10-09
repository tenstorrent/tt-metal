// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

// UMD: re-exports tt::ARCH (the return type of info::architecture).
#include <umd/device/types/arch.hpp>

// Tags for the device query API (`get_info<P>()`).
//
// A tag is an empty struct whose `return_type` is the type of the property. Tags are shared by every object that can
// answer a query (MeshDevice today, MetalEnv later), so a property has one name and one documented meaning
// everywhere. Adding a property means adding a tag here and an implementation for each object that supports it.
//
//     uint32_t alignment = mesh_device->get_info<info::l1_alignment>();
//
namespace tt::tt_metal::info {

/// Required address alignment in bytes for L1 allocations.
struct l1_alignment {
    using return_type = std::uint32_t;
};

/// Required address alignment in bytes for DRAM allocations.
struct dram_alignment {
    using return_type = std::uint32_t;
};

/// Architecture of the device.
struct architecture {
    using return_type = tt::ARCH;
};

/// Lowercase name of the architecture of the device (e.g. "wormhole_b0").
struct architecture_name {
    using return_type = std::string;
};

}  // namespace tt::tt_metal::info
