// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/api_common.h"
#include "accessor/tensor_accessor.h"
#include "accessor/tensor_accessor_args.h"
#include "kernel_common.hpp"

using namespace tt;
using namespace tt::tt_fabric;
using namespace tt::tt_fabric::mesh::experimental;
using namespace tt::tt_fabric::common::experimental;

// Helper function to handle route variant directional fanout logic for FusedAtomicInc
template <ApiVariant api_variant, typename DstAccT>
FORCE_INLINE void send_route_directional_fanout_fused(
    uint16_t hops,
    tt::tt_fabric::RoutingPlaneConnectionManager& cm,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_l1_addr,
    const DstAccT& dst_acc,
    uint32_t i,
    uint64_t sem_noc) {
    if (hops == 0) {
        return;
    }

    if constexpr (api_variant == ApiVariant::RouteBasic) {
        fabric_multicast_noc_fused_unicast_with_atomic_inc(cm, route_id, ranges, src_l1_addr, dst_acc, i, sem_noc, 1);
    } else {
        fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
            cm, route_id, src_l1_addr, dst_acc, i, sem_noc, 1);
    }
}

// Helper function for RouteSetState pre-loop setup for FusedAtomicInc
template <typename DstAccT>
FORCE_INLINE void setup_route_set_state_for_direction_fused(
    bool has_hops,
    tt::tt_fabric::RoutingPlaneConnectionManager& cm,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    const DstAccT& dst_acc,
    uint64_t sem_noc) {
    if (!has_hops) {
        return;
    }
    fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(cm, route_id, ranges, dst_acc, 0, sem_noc, 1);
}

// Helper function for RouteWithState pre-loop setup for FusedAtomicInc
FORCE_INLINE void setup_route_with_state_for_direction_fused(
    bool has_hops, tt::tt_fabric::RoutingPlaneConnectionManager& cm, uint8_t route_id, const MeshMcastRange& ranges) {
    if (!has_hops) {
        return;
    }

    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = cm.get(i);
        fabric_set_mcast_route(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
        packet_header->noc_send_type = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
        packet_header->payload_size_bytes = static_cast<uint16_t>(FABRIC_MAX_PACKET_SIZE);
    });
}

//
// Multicast writer (fabric sender) kernel for route variants - FusedAtomicInc only.
// Sends pages from CB c_0 to multiple destination devices using RoutingPlaneConnectionManager.
//   - API_VARIANT: RouteBasic, RouteWithState, or RouteSetState
//
// CT args:
//   0: API_VARIANT (ApiVariant enum: RouteBasic, RouteWithState, RouteSetState)
//   1: TOTAL_PAGES
//   2: PAGE_SIZE (actual data size to transfer)
//   3: ALIGNED_PAGE_SIZE (destination buffer spacing for address calculation)
//   4: SRC_ALIGNED_PAGE_SIZE (source CB stride - unused for FusedAtomicInc but kept for consistency)
//
// RT args (must match host):
//   0:  dst_base       (u32)
//   1:  rx_noc_x       (u32)   // same worker on every chip
//   2:  rx_noc_y       (u32)
//   3:  sem_l1_addr    (u32)   // same L1 offset on every chip
//   4:  dir_mask       (u32)   // directional bitmask (W=1,E=2,N=4,S=8)
//   … routing plane connection manager args per direction (inserted by append_routing_plane_connection_manager_rt_args)
//   … then multicast hop counts:
//      e_hops (u32), w_hops (u32), n_hops (u32), s_hops (u32)

void kernel_main() {
    constexpr auto ta_args = TensorAccessorArgs<0>();
    constexpr uint32_t CTA_BASE = ta_args.next_compile_time_args_offset();
    constexpr uint32_t API_VARIANT = get_compile_time_arg_val(CTA_BASE + 0);
    constexpr uint32_t TOTAL_PAGES = get_compile_time_arg_val(CTA_BASE + 1);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 2);
    constexpr uint32_t ALIGNED_PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 3);
    constexpr uint32_t SRC_ALIGNED_PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 4);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    // Cast to enum type for clearer comparisons
    constexpr auto api_variant = static_cast<ApiVariant>(API_VARIANT);

    size_t idx = 0;
    const uint32_t dst_base = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_x = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr = get_arg_val<uint32_t>(idx++);

    // Parse directional bitmask
    const uint32_t dir_mask = get_arg_val<uint32_t>(idx++);
    const bool hasW = (dir_mask & 0x1u) != 0;
    const bool hasE = (dir_mask & 0x2u) != 0;
    const bool hasN = (dir_mask & 0x4u) != 0;
    const bool hasS = (dir_mask & 0x8u) != 0;

    // Per-direction connection managers (1 connection each)
    tt::tt_fabric::RoutingPlaneConnectionManager cm_w, cm_e, cm_n, cm_s;
    uint8_t route_id_w = 0, route_id_e = 0, route_id_n = 0, route_id_s = 0;

    // Build connection managers for each active direction
    if (hasW) {
        route_id_w = PacketHeaderPool::allocate_header_n(1);
        open_connections(cm_w, 1, idx);
    }
    if (hasE) {
        route_id_e = PacketHeaderPool::allocate_header_n(1);
        open_connections(cm_e, 1, idx);
    }
    if (hasN) {
        route_id_n = PacketHeaderPool::allocate_header_n(1);
        open_connections(cm_n, 1, idx);
    }
    if (hasS) {
        route_id_s = PacketHeaderPool::allocate_header_n(1);
        open_connections(cm_s, 1, idx);
    }

    // Multicast hop counts (E/W/N/S) appended by the host
    const uint16_t e_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t w_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t n_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t s_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));

    // Use ALIGNED_PAGE_SIZE (dst) for address calculation
    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/ALIGNED_PAGE_SIZE);

    // FusedAtomicInc: compute semaphore NOC address before loop
    ASSERT(sem_l1_addr != 0);
    const uint64_t sem_noc = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*NOC_INDEX=*/0);

    // Pre-loop setup for RouteWithState and RouteSetState variants
    if constexpr (api_variant == ApiVariant::RouteWithState) {
        // Route variant WithState: set route and noc_send_type for each direction's headers
        MeshMcastRange ranges_w_init{0, static_cast<uint8_t>(w_hops), 0, 0};
        MeshMcastRange ranges_e_init{static_cast<uint8_t>(e_hops), 0, 0, 0};
        MeshMcastRange ranges_n_init{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), static_cast<uint8_t>(n_hops), 0};
        MeshMcastRange ranges_s_init{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), 0, static_cast<uint8_t>(s_hops)};

        setup_route_with_state_for_direction_fused(hasW, cm_w, route_id_w, ranges_w_init);
        setup_route_with_state_for_direction_fused(hasE, cm_e, route_id_e, ranges_e_init);
        setup_route_with_state_for_direction_fused(hasN, cm_n, route_id_n, ranges_n_init);
        setup_route_with_state_for_direction_fused(hasS, cm_s, route_id_s, ranges_s_init);
        noc_async_writes_flushed();
    } else if constexpr (api_variant == ApiVariant::RouteSetState) {
        // Route variant SetState: use set_state addrgen route variant for each direction
        MeshMcastRange ranges_w_init{0, static_cast<uint8_t>(w_hops), 0, 0};
        MeshMcastRange ranges_e_init{static_cast<uint8_t>(e_hops), 0, 0, 0};
        MeshMcastRange ranges_n_init{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), static_cast<uint8_t>(n_hops), 0};
        MeshMcastRange ranges_s_init{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), 0, static_cast<uint8_t>(s_hops)};

        setup_route_set_state_for_direction_fused(hasW, cm_w, route_id_w, &ranges_w_init, dst_acc, sem_noc);
        setup_route_set_state_for_direction_fused(hasE, cm_e, route_id_e, &ranges_e_init, dst_acc, sem_noc);
        setup_route_set_state_for_direction_fused(hasN, cm_n, route_id_n, &ranges_n_init, dst_acc, sem_noc);
        setup_route_set_state_for_direction_fused(hasS, cm_s, route_id_s, &ranges_s_init, dst_acc, sem_noc);
    }

    // Main loop - process pages (FusedAtomicInc: increment by 1)
    constexpr uint32_t loop_increment = 1;
    constexpr uint32_t cb_wait_count = 1;

    // Per-direction multicast ranges
    MeshMcastRange ranges_w{0, static_cast<uint8_t>(w_hops), 0, 0};
    MeshMcastRange ranges_e{static_cast<uint8_t>(e_hops), 0, 0, 0};
    MeshMcastRange ranges_n{
        static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), static_cast<uint8_t>(n_hops), 0};
    MeshMcastRange ranges_s{
        static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), 0, static_cast<uint8_t>(s_hops)};

    for (uint32_t i = 0; i < TOTAL_PAGES; i += loop_increment) {
        cb_wait_front(CB_ID, cb_wait_count);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        // Directional fanout using route-based APIs
        send_route_directional_fanout_fused<api_variant>(
            w_hops, cm_w, route_id_w, &ranges_w, src_l1_addr, dst_acc, i, sem_noc);
        send_route_directional_fanout_fused<api_variant>(
            e_hops, cm_e, route_id_e, &ranges_e, src_l1_addr, dst_acc, i, sem_noc);
        send_route_directional_fanout_fused<api_variant>(
            n_hops, cm_n, route_id_n, &ranges_n, src_l1_addr, dst_acc, i, sem_noc);
        send_route_directional_fanout_fused<api_variant>(
            s_hops, cm_s, route_id_s, &ranges_s, src_l1_addr, dst_acc, i, sem_noc);

        cb_pop_front(CB_ID, cb_wait_count);
    }

    noc_async_writes_flushed();

    // FusedAtomicInc does NOT need post-loop atomic inc (it's fused with each write)

    // Close all active direction connections
    if (hasW) {
        close_connections(cm_w);
    }
    if (hasE) {
        close_connections(cm_e);
    }
    if (hasN) {
        close_connections(cm_n);
    }
    if (hasS) {
        close_connections(cm_s);
    }
}
