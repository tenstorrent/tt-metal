// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>

namespace tt::tt_fabric {
class ControlPlane;
class FabricNodeId;
}  // namespace tt::tt_fabric

namespace tt::umd {
class SWEmuleChip;
}  // namespace tt::umd

namespace tt_emule {
class RankState;
}  // namespace tt_emule

namespace tt::tt_metal::emule::multi_rank {

// Process-local chip ids that remain meaningful for peer-owned chips.
int global_chip_for_node(tt::tt_fabric::ControlPlane& control_plane, const tt::tt_fabric::FabricNodeId& node);
bool node_for_global_chip(tt::tt_fabric::ControlPlane& control_plane, uint32_t chip, tt::tt_fabric::FabricNodeId& node);
std::optional<uint32_t> global_chip_for_asic(tt::tt_fabric::ControlPlane& control_plane, uint64_t asic_id);

// Resolve a peer-owned worker-L1 address through the rank's shared chip segment.
uint8_t* resolve_peer_l1(uint32_t destination_chip, uint64_t noc_address, tt::umd::SWEmuleChip& local_chip);

// Shared rank-state ownership and scheduler probes.
tt_emule::RankState& rank_state();
void begin_dispatch();
bool peer_liveness();
bool peer_probes_installed();
void note_deliveries(uint32_t count);

// The driver owns the wait/deadline policy. Runner callbacks retain only operations that must touch
// its private suspended-run state.
struct PeerDriverCallbacks {
    std::function<bool()> needs_peer_pump;
    std::function<uint64_t()> run_sequence;
    std::function<void()> invalidate_run_sequence;
    std::mutex* run_mutex = nullptr;
    std::function<void()> pump_locked;
    std::function<void()> clear_suspended_state;
};

void ensure_peer_wait_driver(PeerDriverCallbacks callbacks);
void notify_peer_wait_driver();

}  // namespace tt::tt_metal::emule::multi_rank
