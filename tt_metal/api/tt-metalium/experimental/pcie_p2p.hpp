// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Chip-to-chip PCIe peer-to-peer windows for Blackhole: lets a Tensix core on one chip NoC-write straight
// into another chip's L1 with no host and no Ethernet involved. Mechanism (all userspace, tt-kmd untouched):
//   outbound  the writer's PCIe core forwards NoC addresses with bit 60 set to its outbound iATU (BAR2 +
//             0x1000, 16 regions); one region per peer maps a NoC range onto the peer's whole BAR0.
//   inbound   the peer's BAR0 is 2 MiB TLB windows; one window per participating worker core points at that
//             core's L1 (offset 0), so BAR0 + window*2MiB + l1_addr lands on the core.
// Measured on this hardware: ~1 us one-way, 27.9 GB/s per direction on a Gen4 x16 pair.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace tt::tt_metal::experimental::pcie_p2p {

struct Config {
    // NoC-local base address of the outbound windows. Peer with mesh index p lives at noc_base + p * 512 MiB.
    // Only 36 local address bits reach the NoC, so noc_base + num_devices * 512 MiB must stay below 64 GiB.
    uint64_t noc_base = 32ull << 30;
    // First outbound iATU region to use (region 0 is tt-metal's hugepage; the pcie_p2p microbenchmarks use
    // 13..15). One region per peer.
    uint32_t first_iatu_region = 8;
    // Inbound TLB ordering: "relaxed" (default, full rate), "posted", or "strict" (1.2 GB/s, do not use).
    std::string ordering = "relaxed";
};

// Declare `chips` (physical chip ids) as a 1xN mesh with no Ethernet between them, so one process can open
// them as a MeshDevice. Must run before any device is opened. Requires the control plane to tolerate
// logically adjacent chips without an Ethernet link (control_plane.cpp turns that TT_FATAL into a warning).
void use_isolated_mesh(const std::vector<int>& chips);

// Program the windows for every device in `mesh` and every logical worker core in `logical_cores` (same
// core list on every device). Idempotent for an identical request; a different request tears down first.
void setup(distributed::MeshDevice& mesh, const std::vector<CoreCoord>& logical_cores, const Config& cfg = {});

bool is_setup();
void teardown();

// The 64-bit NoC address a Tensix core on `from_chip` writes to in order to land on `l1_addr` of logical worker
// core `core` on `to_chip`. Valid after setup(); `core` must be one of the setup cores.
uint64_t remote_l1_noc_addr(ChipId from_chip, ChipId to_chip, const CoreCoord& core, uint32_t l1_addr);

// Mesh index (position in MeshDevice::get_devices()) of a physical chip, as used for the per-peer window base.
uint32_t mesh_index(ChipId chip);

const std::vector<CoreCoord>& cores();

}  // namespace tt::tt_metal::experimental::pcie_p2p
