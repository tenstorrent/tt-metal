// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

#include <tt-umd/types/cluster_descriptor_types.hpp>  // ChipId

namespace tt::tt_metal::experimental {
class PinnedMemory;
}

namespace tt::tt_metal::distributed {

class MeshDevice;
class MeshCoordinateRangeSet;

// Host memory pinned and mapped to the NoC, and how a core of one chip addresses it: the PCIe core's NoC encoding and
// the address through it. Device kernels write host-visible words here (socket credits, router credits) and read rings.
struct NocPinnedBuffer {
    std::shared_ptr<tt::tt_metal::experimental::PinnedMemory> pinned_memory;
    uint32_t pcie_xy_enc = 0;
    uint64_t addr = 0;
};

// Pins `num_words` words at `words` (the pin keeps them alive) for the devices of `device_range` on `mesh_device`,
// mapped to the NoC, and returns how chip `chip` addresses them. Throws, naming `what`, when the memory is not mapped
// to that chip's NoC: the chip must be PCIe-attached and own the mapping (a remote chip cannot reach it).
NocPinnedBuffer pin_host_memory_for_noc(
    MeshDevice& mesh_device,
    const MeshCoordinateRangeSet& device_range,
    const std::shared_ptr<uint32_t[]>& words,
    size_t num_words,
    ChipId chip,
    std::string_view what);

}  // namespace tt::tt_metal::distributed
