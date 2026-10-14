// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// The shadow fabric header against tt-metal's own packet header type, on the
// fields the two share: size, alignment, the base fields, the routing fields and
// the route buffer's start. The shadow keeps a 35-byte route buffer where
// tt-metal's has 36, so the fields after it are x280's own and are pinned only by
// the shadow's static assertions. Compile-time only: the TU defines no test.

#include "fabric_shadow_header.hpp"

#include <tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp>

#include <cstddef>

namespace {

using Upstream = tt::tt_fabric::HybridMeshPacketHeader;
using UpstreamBase = tt::tt_fabric::PacketHeaderBase<Upstream>;

static_assert(sizeof(fs_hybrid_mesh_packet_header_t) == sizeof(Upstream));
static_assert(alignof(fs_hybrid_mesh_packet_header_t) == alignof(Upstream));
static_assert(FS_PACKET_HEADER_BYTES == sizeof(Upstream));
static_assert(sizeof(fs_mesh_routing_fields_t) == sizeof(tt::tt_fabric::LowLatencyMeshRoutingFields));

static_assert(FS_PAYLOAD_SIZE_BYTES_OFFSET == offsetof(UpstreamBase, payload_size_bytes));
static_assert(FS_SRC_CH_ID_OFFSET == offsetof(UpstreamBase, src_ch_id));

// The derived type is not standard-layout (its base has members too), which
// GCC accepts for offsetof with a warning; single non-virtual inheritance
// leaves the offsets well defined.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
static_assert(offsetof(fs_hybrid_mesh_packet_header_t, routing_fields) == offsetof(Upstream, routing_fields));
static_assert(offsetof(fs_hybrid_mesh_packet_header_t, route_buffer) == offsetof(Upstream, route_buffer));
#pragma GCC diagnostic pop

}  // namespace
