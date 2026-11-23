// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/distributed/noc_pinned_buffer.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>

namespace tt::tt_metal::distributed {

NocPinnedBuffer pin_host_memory_for_noc(
    MeshDevice& mesh_device,
    const MeshCoordinateRangeSet& device_range,
    const std::shared_ptr<uint32_t[]>& words,
    size_t num_words,
    ChipId chip,
    std::string_view what) {
    tt::tt_metal::HostBuffer view(ttsl::Span<uint32_t>(words.get(), num_words), tt::tt_metal::MemoryPin(words));
    NocPinnedBuffer buffer;
    buffer.pinned_memory =
        tt::tt_metal::experimental::PinnedMemory::Create(mesh_device, device_range, view, /*map_to_noc=*/true);
    const auto noc_addr = buffer.pinned_memory->get_noc_addr(chip);
    TT_FATAL(noc_addr.has_value(), "{}: failed to get a NOC address for its pinned memory on device {}.", what, chip);
    TT_FATAL(
        noc_addr->device_id == chip,
        "{}: its pinned memory must be mapped to the NOC of device {}, the device whose cores use it, but it is mapped "
        "to device {}. It cannot be used from a remote device.",
        what,
        chip,
        noc_addr->device_id);
    buffer.pcie_xy_enc = noc_addr->pcie_xy_enc;
    buffer.addr = noc_addr->addr;
    return buffer;
}

}  // namespace tt::tt_metal::distributed
