// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/api_common.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/tensor_accessor_args.h"
#include "kernel_common.hpp"

using namespace tt;
using namespace tt::tt_fabric;
using namespace tt::tt_fabric::mesh::experimental;
using namespace tt::tt_fabric::common::experimental;

// Build per-direction MeshMcastRange from hop counts
FORCE_INLINE MeshMcastRange build_range_for_dir(
    uint32_t dir, uint16_t e_hops, uint16_t w_hops, uint16_t n_hops, uint16_t s_hops) {
    constexpr uint32_t DIR_W = 0, DIR_E = 1, DIR_N = 2;
    if (dir == DIR_W) {
        return MeshMcastRange{0, static_cast<uint8_t>(w_hops), 0, 0};
    } else if (dir == DIR_E) {
        return MeshMcastRange{static_cast<uint8_t>(e_hops), 0, 0, 0};
    } else if (dir == DIR_N) {
        return MeshMcastRange{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), static_cast<uint8_t>(n_hops), 0};
    } else {  // DIR_S
        return MeshMcastRange{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), 0, static_cast<uint8_t>(s_hops)};
    }
}

// Unified directional fanout for all operation types with connection manager
template <OperationType operation_type, ApiVariant api_variant, typename DstAccT, typename ScatterAccT>
FORCE_INLINE void send_conn_mgr_directional_fanout(
    uint16_t hops,
    tt::tt_fabric::RoutingPlaneConnectionManager& cm,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_l1_addr,
    const DstAccT& dst_acc,
    const ScatterAccT& scatter_acc,
    uint32_t i,
    uint64_t sem_noc) {
    if (hops == 0) {
        return;
    }

    if constexpr (operation_type == OperationType::BasicWrite) {
        if constexpr (api_variant == ApiVariant::ConnMgrBasic) {
            fabric_multicast_noc_unicast_write(cm, route_id, ranges, src_l1_addr, dst_acc, i, 0);
        } else {
            fabric_multicast_noc_unicast_write_with_state(cm, route_id, src_l1_addr, dst_acc, i, 0);
        }
    } else if constexpr (operation_type == OperationType::Scatter) {
        if constexpr (api_variant == ApiVariant::ConnMgrBasic) {
            fabric_multicast_noc_scatter_write(cm, route_id, ranges, src_l1_addr, scatter_acc, i, i + 1, 0, 0);
        } else {
            fabric_multicast_noc_scatter_write_with_state(cm, route_id, src_l1_addr, scatter_acc, i, i + 1, 0, 0);
        }
    } else if constexpr (operation_type == OperationType::FusedAtomicInc) {
        if constexpr (api_variant == ApiVariant::ConnMgrBasic) {
            fabric_multicast_noc_fused_unicast_with_atomic_inc(
                cm, route_id, ranges, src_l1_addr, dst_acc, i, sem_noc, 1);
        } else {
            fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
                cm, route_id, src_l1_addr, dst_acc, i, sem_noc, 1);
        }
    }
}

// Unified ConnMgrSetState pre-loop setup
template <OperationType operation_type, typename DstAccT, typename ScatterAccT>
FORCE_INLINE void setup_conn_mgr_set_state_for_direction(
    bool has_hops,
    tt::tt_fabric::RoutingPlaneConnectionManager& cm,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    const DstAccT& dst_acc,
    const ScatterAccT& scatter_acc,
    uint64_t sem_noc) {
    if (!has_hops) {
        return;
    }
    if constexpr (operation_type == OperationType::BasicWrite) {
        fabric_multicast_noc_unicast_write_set_state(cm, route_id, ranges, dst_acc, 0, 0);
    } else if constexpr (operation_type == OperationType::Scatter) {
        fabric_multicast_noc_scatter_write_set_state(cm, route_id, ranges, scatter_acc, 0, 1, 0, 0);
    } else if constexpr (operation_type == OperationType::FusedAtomicInc) {
        fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(cm, route_id, ranges, dst_acc, 0, sem_noc, 1);
    }
}

// Unified ConnMgrWithState pre-loop setup
template <OperationType operation_type>
FORCE_INLINE void setup_conn_mgr_with_state_for_direction(
    bool has_hops,
    tt::tt_fabric::RoutingPlaneConnectionManager& cm,
    uint8_t route_id,
    const MeshMcastRange& ranges) {
    if (!has_hops) {
        return;
    }

    auto noc_send_type_for_op = tt::tt_fabric::NOC_UNICAST_WRITE;
    if constexpr (operation_type == OperationType::Scatter) {
        noc_send_type_for_op = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
    } else if constexpr (operation_type == OperationType::FusedAtomicInc) {
        noc_send_type_for_op = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
    }

    PacketHeaderPool::for_each_header(route_id, [&](volatile PACKET_HEADER_TYPE* packet_header, uint8_t i) {
        auto& slot = cm.get(i);
        fabric_set_mcast_route(
            packet_header, slot.dst_dev_id, slot.dst_mesh_id, ranges.e, ranges.w, ranges.n, ranges.s);
        packet_header->noc_send_type = noc_send_type_for_op;
        packet_header->payload_size_bytes = static_cast<uint16_t>(FABRIC_MAX_PACKET_SIZE);
    });
}

//
// Unified multicast writer kernel for connection manager variants.
// Consolidates basic/scatter/fused into a single kernel via OPERATION_TYPE.
//
// CT args:
//   0: OPERATION_TYPE (OperationType enum: BasicWrite, Scatter, FusedAtomicInc)
//   1: API_VARIANT (ApiVariant enum: ConnMgrBasic, ConnMgrWithState, ConnMgrSetState)
//   2: TOTAL_PAGES
//   3: PAGE_SIZE (actual data size to transfer)
//   4: ALIGNED_PAGE_SIZE (destination buffer spacing for address calculation)
//   5: SRC_ALIGNED_PAGE_SIZE (source CB stride for scatter operations)
//
// RT args (must match host):
//   0:  dst_base       (u32)
//   1:  rx_noc_x       (u32)
//   2:  rx_noc_y       (u32)
//   3:  sem_l1_addr    (u32)
//   4:  dir_mask       (u32)
//   … routing plane connection manager args per direction
//   … then multicast hop counts: e_hops, w_hops, n_hops, s_hops

void kernel_main() {
    constexpr auto ta_args = TensorAccessorArgs<0>();
    constexpr uint32_t CTA_BASE = ta_args.next_compile_time_args_offset();
    constexpr uint32_t OPERATION_TYPE = get_compile_time_arg_val(CTA_BASE + 0);
    constexpr uint32_t API_VARIANT = get_compile_time_arg_val(CTA_BASE + 1);
    constexpr uint32_t TOTAL_PAGES = get_compile_time_arg_val(CTA_BASE + 2);
    constexpr uint32_t PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 3);
    constexpr uint32_t ALIGNED_PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 4);
    constexpr uint32_t SRC_ALIGNED_PAGE_SIZE = get_compile_time_arg_val(CTA_BASE + 5);
    constexpr uint32_t CB_ID = tt::CBIndex::c_0;

    constexpr auto operation_type = static_cast<OperationType>(OPERATION_TYPE);
    constexpr auto api_variant = static_cast<ApiVariant>(API_VARIANT);

    if constexpr (operation_type == OperationType::Scatter) {
        ASSERT((TOTAL_PAGES % 2) == 0);
    }

    size_t idx = 0;
    const uint32_t dst_base = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_x = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr = get_arg_val<uint32_t>(idx++);

    const uint32_t dir_mask = get_arg_val<uint32_t>(idx++);
    constexpr uint32_t NUM_DIRECTIONS = 4;

    tt::tt_fabric::RoutingPlaneConnectionManager cm[NUM_DIRECTIONS];
    uint8_t route_id[NUM_DIRECTIONS] = {0, 0, 0, 0};
    bool has_dir[NUM_DIRECTIONS] = {
        (dir_mask & 0x1u) != 0,  // W
        (dir_mask & 0x2u) != 0,  // E
        (dir_mask & 0x4u) != 0,  // N
        (dir_mask & 0x8u) != 0   // S
    };

    for (uint32_t dir = 0; dir < NUM_DIRECTIONS; ++dir) {
        if (has_dir[dir]) {
            route_id[dir] = PacketHeaderPool::allocate_header_n(1);
            open_connections(cm[dir], 1, idx);
        }
    }

    const uint16_t e_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t w_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t n_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t s_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t hops[NUM_DIRECTIONS] = {w_hops, e_hops, n_hops, s_hops};

    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/ALIGNED_PAGE_SIZE);
    const auto scatter_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/SRC_ALIGNED_PAGE_SIZE);

    // FusedAtomicInc: compute semaphore NOC address before loop
    uint64_t sem_noc = 0;
    if constexpr (operation_type == OperationType::FusedAtomicInc) {
        ASSERT(sem_l1_addr != 0);
        sem_noc = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*NOC_INDEX=*/0);
    }

    // Pre-loop setup for ConnMgrWithState and ConnMgrSetState variants
    if constexpr (api_variant == ApiVariant::ConnMgrWithState) {
        for (uint32_t dir = 0; dir < NUM_DIRECTIONS; ++dir) {
            MeshMcastRange range_init = build_range_for_dir(dir, e_hops, w_hops, n_hops, s_hops);
            setup_conn_mgr_with_state_for_direction<operation_type>(has_dir[dir], cm[dir], route_id[dir], range_init);
        }
        noc_async_writes_flushed();
    } else if constexpr (api_variant == ApiVariant::ConnMgrSetState) {
        for (uint32_t dir = 0; dir < NUM_DIRECTIONS; ++dir) {
            MeshMcastRange range_init = build_range_for_dir(dir, e_hops, w_hops, n_hops, s_hops);
            setup_conn_mgr_set_state_for_direction<operation_type>(
                has_dir[dir], cm[dir], route_id[dir], &range_init, dst_acc, scatter_acc, sem_noc);
        }
    }

    // Main loop
    constexpr uint32_t loop_increment = (operation_type == OperationType::Scatter) ? 2 : 1;
    constexpr uint32_t cb_wait_count = (operation_type == OperationType::Scatter) ? 2 : 1;

    MeshMcastRange ranges[NUM_DIRECTIONS];
    for (uint32_t dir = 0; dir < NUM_DIRECTIONS; ++dir) {
        ranges[dir] = build_range_for_dir(dir, e_hops, w_hops, n_hops, s_hops);
    }

    for (uint32_t i = 0; i < TOTAL_PAGES; i += loop_increment) {
        cb_wait_front(CB_ID, cb_wait_count);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        for (uint32_t dir = 0; dir < NUM_DIRECTIONS; ++dir) {
            send_conn_mgr_directional_fanout<operation_type, api_variant>(
                hops[dir], cm[dir], route_id[dir], &ranges[dir], src_l1_addr, dst_acc, scatter_acc, i, sem_noc);
        }

        cb_pop_front(CB_ID, cb_wait_count);
    }

    noc_async_writes_flushed();

    // Post-loop completion: BasicWrite and Scatter need separate atomic inc
    // FusedAtomicInc does NOT (it's fused with each write)
    if constexpr (operation_type != OperationType::FusedAtomicInc) {
        ASSERT(sem_l1_addr != 0);
        const uint64_t sem_noc_final = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*NOC_INDEX=*/0);

        for (uint32_t dir = 0; dir < NUM_DIRECTIONS; ++dir) {
            if (hops[dir] > 0) {
                fabric_multicast_noc_unicast_atomic_inc(
                    cm[dir],
                    route_id[dir],
                    &ranges[dir],
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_final, /*inc=*/1, /*width_bits=*/32));
            }
        }
    }

    for (uint32_t dir = 0; dir < NUM_DIRECTIONS; ++dir) {
        if (has_dir[dir]) {
            close_connections(cm[dir]);
        }
    }
}
