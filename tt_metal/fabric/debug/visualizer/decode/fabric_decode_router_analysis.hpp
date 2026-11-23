// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// What decode says about one router before analysis, beside its activity: what capture read of it
// (analysis.capture), and each VC's connected and live directions (analysis.directions). Neither needs another
// router's values.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_interpret.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_snapshot.hpp"
#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_values.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_model.hpp"

namespace tt::tt_fabric::decode {

// What capture read of a router: its full read's status, and how many of its regions decode has values for. This is
// coverage, which sets the exit status; a failed round read is a finding, not coverage.
struct CaptureCoverage {
    RouterReadStatus status = RouterReadStatus::NOT_READ;
    // Why the router is NOT_READ or PARTIAL. Null when it's COMPLETE.
    std::optional<Unread> reason;
    size_t num_decoded = 0;
    size_t num_inactive = 0;
    size_t num_unread = 0;
};

CaptureCoverage summarize_capture(const RouterValues& values);

// The router's analysis.capture.
json make_capture_json(const CaptureCoverage& coverage);

// One VC's directions, one bit per eth_chan_directions, as Routing2DCodec's action bits are.
struct VcDirections {
    uint32_t vc = 0;
    // The siblings the router has a downstream connection to (intra_chip_downstream_edges).
    uint8_t connected = 0;
    // The directions the router forwards packets to (live_downstreams).
    uint8_t live = 0;
};

// Each VC with a downstream array, in VC order. Null in 1D, where the router has one downstream and no live mask.
std::optional<std::vector<VcDirections>> find_router_directions(const manifest::Router& router);

// The router's analysis.directions: each VC's connected and live directions, and the two differences, which
// legitimately happen (decision 20). Null in 1D.
json make_directions_json(const std::optional<std::vector<VcDirections>>& directions);

}  // namespace tt::tt_fabric::decode
