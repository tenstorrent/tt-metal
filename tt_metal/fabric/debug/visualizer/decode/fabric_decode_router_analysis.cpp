// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/fabric/debug/visualizer/decode/fabric_decode_router_analysis.hpp"

#include <algorithm>
#include <variant>

#include <hostdevcommon/fabric_common.h>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_fields.hpp"
#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"

namespace tt::tt_fabric::decode {

namespace {

// The direction the router faces: its my_direction field, which every ERISC of a router is fed alike.
eth_chan_directions find_router_facing(const manifest::Router& router) {
    const manifest::Field* field = manifest::find_erisc_field(router.fields_by_erisc, 0, "my_direction");
    const auto* direction = field != nullptr && field->content.has_value()
                                ? std::get_if<manifest::content::Enum>(&*field->content)
                                : nullptr;
    TT_FATAL(direction != nullptr, "Fabric decode: a 2D router's manifest entry has no my_direction field");
    return static_cast<eth_chan_directions>(direction->value);
}

// The direction of a downstream edge: the kernel's EDGE_<n> is the sibling's compact index among the router's other
// directions, plus one (Routing2DCodec::compact_index_on).
eth_chan_directions find_edge_direction(eth_chan_directions facing, const manifest::DownstreamEdge& edge) {
    for (uint32_t value = 0; value < eth_chan_directions::COUNT; ++value) {
        const auto direction = static_cast<eth_chan_directions>(value);
        if (direction != facing && edge.edge == Routing2DCodec::compact_index_on(facing, direction) + 1) {
            return direction;
        }
    }
    TT_THROW(
        "Fabric decode: downstream edge {} names no direction of a router facing {}",
        edge.edge,
        static_cast<int>(facing));
}

// The names of the directions whose bits are set in `bits`, in eth_chan_directions order.
json make_direction_names_json(uint8_t bits) {
    json names = json::array();
    for (uint32_t value = 0; value < eth_chan_directions::COUNT; ++value) {
        if ((bits & (1u << value)) != 0) {
            names.push_back(manifest::lower_enum_name(static_cast<eth_chan_directions>(value)));
        }
    }
    return names;
}

}  // namespace

CaptureCoverage summarize_capture(const RouterValues& values) {
    const RouterSnapshot& snapshot = values.get_snapshot();
    return CaptureCoverage{
        .status = snapshot.get_read_status(),
        .reason = snapshot.get_unread_reason(),
        .num_decoded = values.get_num_read(),
        .num_inactive = values.get_num_inactive(),
        .num_unread = values.get_num_unread(),
    };
}

json make_capture_json(const CaptureCoverage& coverage) {
    json out;
    out["status"] = manifest::lower_enum_name(coverage.status);
    if (coverage.reason.has_value()) {
        out["reason"] = {
            {"code", manifest::lower_enum_name(coverage.reason->code)}, {"message", coverage.reason->message}};
    } else {
        out["reason"] = nullptr;
    }
    out["regions"] = {
        {"decoded", coverage.num_decoded}, {"inactive", coverage.num_inactive}, {"unread", coverage.num_unread}};
    return out;
}

std::optional<std::vector<VcDirections>> find_router_directions(const manifest::Router& router) {
    if (!router.live_downstreams.has_value()) {
        return std::nullopt;
    }
    const eth_chan_directions facing = find_router_facing(router);
    std::vector<VcDirections> directions;
    for (uint32_t vc = 0; vc < manifest::k_num_vcs_with_downstream_array; ++vc) {
        VcDirections vc_directions{.vc = vc, .live = (*router.live_downstreams)[vc]};
        if (vc < router.intra_chip_downstream_edges.size()) {
            for (const manifest::DownstreamEdge& edge : router.intra_chip_downstream_edges[vc]) {
                vc_directions.connected |= static_cast<uint8_t>(1u << find_edge_direction(facing, edge));
            }
        }
        directions.push_back(vc_directions);
    }
    return directions;
}

json make_directions_json(const std::optional<std::vector<VcDirections>>& directions) {
    if (!directions.has_value()) {
        return json(nullptr);
    }
    json out = json::array();
    for (const VcDirections& vc : *directions) {
        json vc_json;
        vc_json["vc"] = vc.vc;
        vc_json["connected"] = make_direction_names_json(vc.connected);
        vc_json["live"] = make_direction_names_json(vc.live);
        vc_json["connected_not_live"] = make_direction_names_json(static_cast<uint8_t>(vc.connected & ~vc.live));
        vc_json["live_not_connected"] = make_direction_names_json(static_cast<uint8_t>(vc.live & ~vc.connected));
        out.push_back(std::move(vc_json));
    }
    return out;
}

}  // namespace tt::tt_fabric::decode
