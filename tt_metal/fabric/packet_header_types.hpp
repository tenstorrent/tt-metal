// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The packet header types fabric can choose, and how it picks one for a run. FabricContext sizes the run's header from
// these, and everything that enumerates header types (such as the struct layouts) iterates them, so each header's
// template argument is written once. Host only: kernels get the chosen argument as a define.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

#include <hostdevcommon/fabric_common.h>
#include <tt_stl/assert.hpp>

#include "tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp"

namespace tt::tt_fabric {

// 1D: routing words after the base word. Each word holds RoutingFieldsConstants::LowLatency::BASE_HOPS (16) hops, so
// 0 to 3 extension words reach 16 to 64 hops, in a 48 B header with none and a 64 B header with 1 to 3.
inline constexpr std::array<uint32_t, 4> PKT_HDR_1D_EXTENSION_WORDS = {0, 1, 2, 3};

// 2D: route buffer bytes, one per header size. Each header is 60 B of fixed fields plus its route buffer, so each size
// fills a 16 B-aligned header.
inline constexpr std::array<int, 3> PKT_HDR_2D_ROUTE_BUFFER_SIZES = {
    // NOTE: 80B header size de-stabilized some Mesh benchmarks for 8X4 mesh, so disabling for now
    // 20,  // 80 B header (60 + 20)
    36,  // 96 B header (60 + 36)
    52,  // 112 B header (60 + 52)
    68,  // 128 B header (60 + 68)
};

// compute_1d_pkt_hdr_extension_words can return any count from 0 up to the largest, so the 1D list must hold each one
// in order: 0, 1, 2, and so on.
static_assert(
    [] {
        for (std::size_t i = 0; i < PKT_HDR_1D_EXTENSION_WORDS.size(); ++i) {
            if (PKT_HDR_1D_EXTENSION_WORDS[i] != i) {
                return false;
            }
        }
        return true;
    }(),
    "compute_1d_pkt_hdr_extension_words can return any count up to the largest, so each must be offered");
// The largest 2D header must hold the action map of the largest mesh the routing codec supports (64x4), so any mesh
// that routes has a header.
static_assert(
    PKT_HDR_2D_ROUTE_BUFFER_SIZES.back() == Routing2DCodec::MAX_AXIS_SIZE + Routing2DCodec::MAX_ORTHOGONAL_AXIS_SIZE,
    "the largest 2D header must hold the action map of the largest mesh (64x4)");
// compute_2d_pkt_hdr_route_buffer_size takes the first size that fits, so the 2D list must be in ascending order for it
// to pick the smallest.
static_assert(
    std::ranges::is_sorted(PKT_HDR_2D_ROUTE_BUFFER_SIZES),
    "compute_2d_pkt_hdr_route_buffer_size takes the first size that fits, so the sizes must be in ascending order");

// The number of extension words a 1D header needs to carry routes of up to max_hops hops (at least 1). The base routing
// word holds the first BASE_HOPS hops, and each extension word holds BASE_HOPS more, so 16 hops need none and 17 need
// one. FabricContext calls this with the longest route in the topology.
constexpr uint32_t compute_1d_pkt_hdr_extension_words(uint32_t max_hops) {
    return (max_hops - 1) / RoutingFieldsConstants::LowLatency::BASE_HOPS;
}

// The route buffer size of the smallest 2D header that holds an action map of required_bytes. A smaller header leaves
// more of each packet for payload, so the first size in PKT_HDR_2D_ROUTE_BUFFER_SIZES that fits is chosen. If
// none fits, it returns the largest size, and FabricContext has already rejected an action map that large.
constexpr uint32_t compute_2d_pkt_hdr_route_buffer_size(uint32_t required_bytes) {
    for (const int size : PKT_HDR_2D_ROUTE_BUFFER_SIZES) {
        if (required_bytes <= static_cast<uint32_t>(size)) {
            return static_cast<uint32_t>(size);
        }
    }
    return static_cast<uint32_t>(PKT_HDR_2D_ROUTE_BUFFER_SIZES.back());
}

// Which packet header a run uses, as FabricContext chose it: the routing mode, and the template argument of that mode's
// header. A 1D run uses LowLatencyPacketHeaderT<extension_words>, and a 2D run uses
// HybridMeshPacketHeaderT<route_buffer_size>. The field for the other mode is ignored.
struct PacketHeaderConfig {
    bool is_2d = false;
    uint32_t extension_words = 0;    // 1D
    uint32_t route_buffer_size = 0;  // 2D

    // Whether this config and other name the same header type: the same routing mode, and the same template argument
    // for that mode. The other mode's field isn't compared, since neither config uses it.
    constexpr bool picks_same_header(const PacketHeaderConfig& other) const {
        if (is_2d != other.is_2d) {
            return false;
        }
        return is_2d ? route_buffer_size == other.route_buffer_size : extension_words == other.extension_words;
    }
};

// Calls f once for every packet header type fabric can choose, passing in the header type as well as the corresponding
// PacketHeaderConfig.
template <typename F>
constexpr void for_each_packet_header_type(const F& f) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        (f(std::type_identity<LowLatencyPacketHeaderT<PKT_HDR_1D_EXTENSION_WORDS[I]>>{},
           PacketHeaderConfig{.is_2d = false, .extension_words = PKT_HDR_1D_EXTENSION_WORDS[I]}),
         ...);
    }(std::make_index_sequence<PKT_HDR_1D_EXTENSION_WORDS.size()>{});
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        (f(std::type_identity<HybridMeshPacketHeaderT<PKT_HDR_2D_ROUTE_BUFFER_SIZES[I]>>{},
           PacketHeaderConfig{
               .is_2d = true, .route_buffer_size = static_cast<uint32_t>(PKT_HDR_2D_ROUTE_BUFFER_SIZES[I])}),
         ...);
    }(std::make_index_sequence<PKT_HDR_2D_ROUTE_BUFFER_SIZES.size()>{});
}

// Finds the appropriate packet header type based on config, calls f with the header type, and returns what f returns.
template <typename F>
auto visit_packet_header_type(const PacketHeaderConfig& config, const F& f) {
    // Dummy call to f just to figure out what type f returns
    using Result = decltype(f(std::type_identity<LowLatencyPacketHeaderT<PKT_HDR_1D_EXTENSION_WORDS[0]>>{}));

    // Empty until the loop reaches the header config picks
    std::optional<Result> result;
    for_each_packet_header_type([&](auto type, const PacketHeaderConfig& candidate) {
        static_assert(
            std::is_same_v<decltype(f(type)), Result>, "f must return the same type for every packet header type");
        if (!result.has_value() && config.picks_same_header(candidate)) {
            result.emplace(f(type));
        }
    });

    // Ensure a matching header was found
    TT_FATAL(
        result.has_value(),
        "No packet header for {} routing with {} {}",
        config.is_2d ? "2D" : "1D",
        config.is_2d ? config.route_buffer_size : config.extension_words,
        config.is_2d ? "route buffer bytes" : "extension words");

    return *std::move(result);
}

}  // namespace tt::tt_fabric
