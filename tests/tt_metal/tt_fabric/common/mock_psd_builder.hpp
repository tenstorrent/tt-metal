// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/fabric/physical_system_descriptor.hpp>

namespace tt::tt_fabric::test {

// An undirected link between two 0-based asic indices, carrying `channels` eth channels per direction.
struct MockLink {
    int a;
    int b;
    uint32_t channels;
};

// Build an in-memory mock PhysicalSystemDescriptor from a topology, replacing hand-written PSD textprotos.
//   host_of_asic[i] = host name of asic i (asic_id = base_asic_id + i); its size is the asic count.
//   links           = undirected connections with a per-link channel count.
//   positions[i]    = (tray_id, asic_location) of asic i; if empty, defaults to (0, i).
// Links whose endpoints sit on different hosts are marked cross-host (is_local=false) and also populate the
// host-connectivity + exit-node tables, so multi-host PSDs are complete. Hosts are ranked in first-seen order.
tt::tt_metal::PhysicalSystemDescriptor build_mock_psd(
    const std::vector<std::string>& host_of_asic,
    const std::vector<MockLink>& links,
    const std::vector<std::pair<uint32_t, uint32_t>>& positions = {},
    uint64_t base_asic_id = 100);

// Uniform-channel convenience: every edge gets `channels` channels (positions default to (0, i)).
tt::tt_metal::PhysicalSystemDescriptor build_mock_psd(
    const std::vector<std::string>& host_of_asic,
    const std::vector<std::pair<int, int>>& edges,
    uint32_t channels = 2,
    uint64_t base_asic_id = 100);

// Row-major rows x cols grid PSD. Matches the hand-written grid PSDs: asic (r,c) gets tray_id=r+1,
// asic_location=c+1, host = host_of_asic[r*cols + c]. `extra_links` adds non-grid edges (e.g. torus wraps).
tt::tt_metal::PhysicalSystemDescriptor build_grid_mock_psd(
    int rows,
    int cols,
    const std::vector<std::string>& host_of_asic,
    uint32_t channels = 2,
    const std::vector<std::pair<int, int>>& extra_links = {},
    uint64_t base_asic_id = 100);

// Topology-shape edge lists (0-based indices), to hand to build_mock_psd.
std::vector<std::pair<int, int>> line_edges(int num_asics);       // 0-1-2-...-(n-1)
std::vector<std::pair<int, int>> ring_edges(int num_asics);       // line + wrap (n-1)-0
std::vector<std::pair<int, int>> grid_edges(int rows, int cols);  // row-major 2D mesh
std::vector<std::pair<int, int>> star_edges(int num_leaves);      // center 0 to leaves 1..n

}  // namespace tt::tt_fabric::test
