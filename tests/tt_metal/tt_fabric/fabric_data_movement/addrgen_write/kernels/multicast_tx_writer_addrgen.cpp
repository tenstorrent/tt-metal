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

// Helper function to handle directional fanout logic
template <OperationType operation_type, ApiVariant api_variant, typename DstAccT, typename ScatterAccT>
inline void send_directional_fanout(
    uint16_t hops,
    WorkerToFabricEdmSender* sender,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    const MeshMcastRange& ranges,
    uint32_t src_l1_addr,
    const DstAccT& dst_acc,
    const ScatterAccT& scatter_acc,
    uint32_t i,
    uint64_t sem_noc) {
    if (hops > 0) {
        if constexpr (operation_type == OperationType::BasicWrite) {
            if constexpr (api_variant == ApiVariant::Basic) {
                fabric_multicast_noc_unicast_write(sender, packet_header, 0, 0, ranges, src_l1_addr, dst_acc, i);
            } else {  // WithState or SetState
                fabric_multicast_noc_unicast_write_with_state(
                    sender, packet_header, 0, 0, ranges, src_l1_addr, dst_acc, i);
            }
        } else if constexpr (operation_type == OperationType::Scatter) {
            // Use scatter_acc with SRC_ALIGNED_PAGE_SIZE to match CB stride
            if constexpr (api_variant == ApiVariant::Basic) {
                fabric_multicast_noc_scatter_write(
                    sender, packet_header, 0, 0, ranges, src_l1_addr, scatter_acc, i, i + 1);
            } else {  // WithState or SetState
                fabric_multicast_noc_scatter_write_with_state(
                    sender, packet_header, 0, 0, ranges, src_l1_addr, scatter_acc, i, i + 1);
            }
        } else if constexpr (operation_type == OperationType::FusedAtomicInc) {
            if constexpr (api_variant == ApiVariant::Basic) {
                fabric_multicast_noc_fused_unicast_with_atomic_inc(
                    sender, packet_header, 0, 0, ranges, src_l1_addr, dst_acc, i, sem_noc, 1);
            } else {  // WithState or SetState
                fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
                    sender, packet_header, src_l1_addr, dst_acc, i, sem_noc, 1);
            }
        }
    }
}

// Helper function to send completion atomic increment for a direction
inline void send_completion_atomic_inc(
    uint16_t hops,
    WorkerToFabricEdmSender& sender,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint16_t e_hops_for_route,
    uint16_t w_hops_for_route,
    uint16_t n_hops_for_route,
    uint16_t s_hops_for_route,
    uint64_t sem_noc_final) {
    if (hops > 0) {
        fabric_set_mcast_route(
            packet_header, 0, 0, e_hops_for_route, w_hops_for_route, n_hops_for_route, s_hops_for_route);
        packet_header->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader(sem_noc_final, /*inc=*/1, /*width_bits=*/32));
        sender.wait_for_empty_write_slot();
        sender.send_payload_flush_non_blocking_from_address((uint32_t)packet_header, sizeof(PACKET_HEADER_TYPE));
    }
}

// Helper function for SetState pre-loop setup for a direction
template <OperationType operation_type, typename DstAccT, typename ScatterAccT>
inline void setup_set_state_for_direction(
    uint16_t hops,
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    const MeshMcastRange& ranges,
    const DstAccT& dst_acc,
    const ScatterAccT& scatter_acc,
    uint64_t sem_noc) {
    if (hops > 0) {
        if constexpr (operation_type == OperationType::BasicWrite) {
            fabric_multicast_noc_unicast_write_set_state(packet_header, 0, 0, ranges, dst_acc, 0);
        } else if constexpr (operation_type == OperationType::Scatter) {
            // Use scatter_acc with SRC_ALIGNED_PAGE_SIZE to match CB stride
            fabric_multicast_noc_scatter_write_set_state(packet_header, 0, 0, ranges, scatter_acc, 0, 1);
        } else if constexpr (operation_type == OperationType::FusedAtomicInc) {
            fabric_multicast_noc_fused_unicast_with_atomic_inc_set_state(
                packet_header, 0, 0, ranges, dst_acc, 0, sem_noc, 1);
        }
    }
}

// Helper function to handle route variant directional fanout logic
template <OperationType operation_type, ApiVariant api_variant, typename DstAccT, typename ScatterAccT>
FORCE_INLINE void send_route_directional_fanout(
    uint16_t hops,
    tt::tt_fabric::RoutingPlaneConnectionManager& cm,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    uint32_t src_l1_addr,
    const DstAccT& dst_acc,
    const ScatterAccT& scatter_acc,
    uint32_t i,
    uint64_t sem_noc = 0) {
    if (hops == 0) {
        return;
    }

    if constexpr (operation_type == OperationType::BasicWrite) {
        if constexpr (api_variant == ApiVariant::RouteBasic) {
            fabric_multicast_noc_unicast_write(cm, route_id, ranges, src_l1_addr, dst_acc, i, 0);
        } else {
            fabric_multicast_noc_unicast_write_with_state(cm, route_id, src_l1_addr, dst_acc, i, 0);
        }
    } else if constexpr (operation_type == OperationType::Scatter) {
        if constexpr (api_variant == ApiVariant::RouteBasic) {
            fabric_multicast_noc_scatter_write(cm, route_id, ranges, src_l1_addr, scatter_acc, i, i + 1, 0, 0);
        } else {
            fabric_multicast_noc_scatter_write_with_state(cm, route_id, src_l1_addr, scatter_acc, i, i + 1, 0, 0);
        }
    } else if constexpr (operation_type == OperationType::FusedAtomicInc) {
        if constexpr (api_variant == ApiVariant::RouteBasic) {
            fabric_multicast_noc_fused_unicast_with_atomic_inc(
                cm, route_id, ranges, src_l1_addr, dst_acc, i, sem_noc, 1);
        } else {
            fabric_multicast_noc_fused_unicast_with_atomic_inc_with_state(
                cm, route_id, src_l1_addr, dst_acc, i, sem_noc, 1);
        }
    }
}

// Helper function for RouteSetState pre-loop setup for a direction
template <OperationType operation_type, typename DstAccT, typename ScatterAccT>
FORCE_INLINE void setup_route_set_state_for_direction(
    bool has_hops,
    tt::tt_fabric::RoutingPlaneConnectionManager& cm,
    uint8_t route_id,
    const MeshMcastRange* ranges,
    const DstAccT& dst_acc,
    const ScatterAccT& scatter_acc,
    uint64_t sem_noc = 0) {
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

// Helper function for RouteWithState pre-loop setup for a direction
template <OperationType operation_type>
FORCE_INLINE void setup_route_with_state_for_direction(
    bool has_hops, tt::tt_fabric::RoutingPlaneConnectionManager& cm, uint8_t route_id, const MeshMcastRange& ranges) {
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
// Unified multicast writer (fabric sender) kernel — consolidates 18 variants (3 operation types × 6 API variants).
// Sends pages from CB c_0 to multiple destination devices using compile-time parameters:
//   - OPERATION_TYPE: BasicWrite, Scatter, or FusedAtomicInc
//   - API_VARIANT: Basic, WithState, SetState, RouteBasic, RouteWithState, or RouteSetState
//
// CT args:
//   0: OPERATION_TYPE (OperationType enum: BasicWrite, Scatter, FusedAtomicInc)
//   1: API_VARIANT (ApiVariant enum: Basic, WithState, SetState, RouteBasic, RouteWithState, RouteSetState)
//   2: TOTAL_PAGES
//   3: PAGE_SIZE (actual data size to transfer)
//   4: ALIGNED_PAGE_SIZE (destination buffer spacing for address calculation)
//   5: SRC_ALIGNED_PAGE_SIZE (source CB stride for scatter operations)
//
// RT args (same structure for all variants):
//   0:  dst_base       (u32)
//   1:  rx_noc_x       (u32)   // same worker on every chip
//   2:  rx_noc_y       (u32)
//   3:  sem_l1_addr    (u32)   // same L1 offset on every chip
//   4:  dir_mask       (u32)   // directional bitmask (W=1,E=2,N=4,S=8)
//
// For Basic/WithState/SetState:
//   … fabric-connection args per direction (inserted by append_fabric_connection_rt_args)
//
// For RouteBasic:
//   … routing plane connection manager args per direction (inserted by append_routing_plane_connection_manager_rt_args)
//
// Then for all variants:
//   e_hops (u32), w_hops (u32), n_hops (u32), s_hops (u32)

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

    // Cast to enum types for clearer comparisons
    constexpr auto operation_type = static_cast<OperationType>(OPERATION_TYPE);
    constexpr auto api_variant = static_cast<ApiVariant>(API_VARIANT);

    // Scatter requires even number of pages
    if constexpr (operation_type == OperationType::Scatter) {
        ASSERT((TOTAL_PAGES % 2) == 0);
    }

    // Check if this is a route variant
    constexpr bool is_route_variant =
        (api_variant == ApiVariant::RouteBasic || api_variant == ApiVariant::RouteWithState ||
         api_variant == ApiVariant::RouteSetState);

    size_t idx = 0;
    const uint32_t dst_base = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_x = get_arg_val<uint32_t>(idx++);
    const uint32_t rx_noc_y = get_arg_val<uint32_t>(idx++);
    const uint32_t sem_l1_addr = get_arg_val<uint32_t>(idx++);

    // Parse directional bitmask (same for both route and non-route variants)
    const uint32_t dir_mask = get_arg_val<uint32_t>(idx++);
    const bool hasW = (dir_mask & 0x1u) != 0;
    const bool hasE = (dir_mask & 0x2u) != 0;
    const bool hasN = (dir_mask & 0x4u) != 0;
    const bool hasS = (dir_mask & 0x8u) != 0;

    // Route variant: per-direction connection managers (1 connection each)
    tt::tt_fabric::RoutingPlaneConnectionManager cm_w, cm_e, cm_n, cm_s;
    uint8_t route_id_w = 0, route_id_e = 0, route_id_n = 0, route_id_s = 0;

    // Non-route variant: per-direction senders
    WorkerToFabricEdmSender senderW{}, senderE{}, senderN{}, senderS{};

    volatile tt_l1_ptr PACKET_HEADER_TYPE* left_packet_header = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* right_packet_header = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* north_packet_header = nullptr;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* south_packet_header = nullptr;

    if constexpr (is_route_variant) {
        // Route variant: build connection managers for each active direction
        // Each direction has 1 connection
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
    } else {
        // Non-route variant: build senders for each active direction
        if (hasW) {
            senderW = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
        }
        if (hasE) {
            senderE = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
        }
        if (hasN) {
            senderN = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
        }
        if (hasS) {
            senderS = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(idx);
        }
    }

    // Multicast hop counts (E/W/N/S) appended by the host (same for both variants)
    const uint16_t e_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t w_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t n_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));
    const uint16_t s_hops = static_cast<uint16_t>(get_arg_val<uint32_t>(idx++));

    if constexpr (!is_route_variant) {
        // Non-route variant: allocate headers and open senders
        left_packet_header = PacketHeaderPool::allocate_header();
        right_packet_header = PacketHeaderPool::allocate_header();
        north_packet_header = PacketHeaderPool::allocate_header();
        south_packet_header = PacketHeaderPool::allocate_header();

        if (hasW) {
            senderW.open<true>();
        }
        if (hasE) {
            senderE.open<true>();
        }
        if (hasN) {
            senderN.open<true>();
        }
        if (hasS) {
            senderS.open<true>();
        }
    }

    // For non-scatter: Use ALIGNED_PAGE_SIZE (dst) for address calculation
    // For scatter: Use SRC_ALIGNED_PAGE_SIZE to match CB stride (less BW efficient but correct)
    const auto dst_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/ALIGNED_PAGE_SIZE);
    const auto scatter_acc = TensorAccessor(ta_args, /*bank_base=*/dst_base, /*page_size=*/SRC_ALIGNED_PAGE_SIZE);

    // FusedAtomicInc: compute semaphore NOC address before loop
    uint64_t sem_noc = 0;
    if constexpr (operation_type == OperationType::FusedAtomicInc) {
        ASSERT(sem_l1_addr != 0);
        sem_noc = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*NOC_INDEX=*/0);
    }

    // Pre-loop setup for WithState and SetState variants (non-route only)
    if constexpr (!is_route_variant && api_variant == ApiVariant::WithState) {
        // Determine the appropriate NOC send type based on operation
        auto noc_send_type_for_op = tt::tt_fabric::NOC_UNICAST_WRITE;
        if constexpr (operation_type == OperationType::Scatter) {
            noc_send_type_for_op = tt::tt_fabric::NOC_UNICAST_SCATTER_WRITE;
        } else if constexpr (operation_type == OperationType::FusedAtomicInc) {
            noc_send_type_for_op = tt::tt_fabric::NOC_FUSED_UNICAST_ATOMIC_INC;
        }

        // Manually configure routing for each direction
        if (w_hops > 0) {
            fabric_set_mcast_route(left_packet_header, 0, 0, 0, w_hops, 0, 0);
            left_packet_header->noc_send_type = noc_send_type_for_op;
        }
        if (e_hops > 0) {
            fabric_set_mcast_route(right_packet_header, 0, 0, e_hops, 0, 0, 0);
            right_packet_header->noc_send_type = noc_send_type_for_op;
        }
        if (n_hops > 0) {
            fabric_set_mcast_route(north_packet_header, 0, 0, e_hops, w_hops, n_hops, 0);
            north_packet_header->noc_send_type = noc_send_type_for_op;
        }
        if (s_hops > 0) {
            fabric_set_mcast_route(south_packet_header, 0, 0, e_hops, w_hops, 0, s_hops);
            south_packet_header->noc_send_type = noc_send_type_for_op;
        }
    } else if constexpr (!is_route_variant && api_variant == ApiVariant::SetState) {
        // Initialize all header fields for each direction
        MeshMcastRange ranges_w{0, static_cast<uint8_t>(w_hops), 0, 0};
        setup_set_state_for_direction<operation_type>(
            w_hops, left_packet_header, ranges_w, dst_acc, scatter_acc, sem_noc);

        MeshMcastRange ranges_e{static_cast<uint8_t>(e_hops), 0, 0, 0};
        setup_set_state_for_direction<operation_type>(
            e_hops, right_packet_header, ranges_e, dst_acc, scatter_acc, sem_noc);

        MeshMcastRange ranges_n{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), static_cast<uint8_t>(n_hops), 0};
        setup_set_state_for_direction<operation_type>(
            n_hops, north_packet_header, ranges_n, dst_acc, scatter_acc, sem_noc);

        MeshMcastRange ranges_s{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), 0, static_cast<uint8_t>(s_hops)};
        setup_set_state_for_direction<operation_type>(
            s_hops, south_packet_header, ranges_s, dst_acc, scatter_acc, sem_noc);
    } else if constexpr (api_variant == ApiVariant::RouteWithState) {
        // Route variant WithState: set route and noc_send_type for each direction's headers
        MeshMcastRange ranges_w_init{0, static_cast<uint8_t>(w_hops), 0, 0};
        MeshMcastRange ranges_e_init{static_cast<uint8_t>(e_hops), 0, 0, 0};
        MeshMcastRange ranges_n_init{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), static_cast<uint8_t>(n_hops), 0};
        MeshMcastRange ranges_s_init{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), 0, static_cast<uint8_t>(s_hops)};

        setup_route_with_state_for_direction<operation_type>(hasW, cm_w, route_id_w, ranges_w_init);
        setup_route_with_state_for_direction<operation_type>(hasE, cm_e, route_id_e, ranges_e_init);
        setup_route_with_state_for_direction<operation_type>(hasN, cm_n, route_id_n, ranges_n_init);
        setup_route_with_state_for_direction<operation_type>(hasS, cm_s, route_id_s, ranges_s_init);
        noc_async_writes_flushed();
    } else if constexpr (api_variant == ApiVariant::RouteSetState) {
        // Route variant SetState: use set_state addrgen route variant for each direction
        MeshMcastRange ranges_w_init{0, static_cast<uint8_t>(w_hops), 0, 0};
        MeshMcastRange ranges_e_init{static_cast<uint8_t>(e_hops), 0, 0, 0};
        MeshMcastRange ranges_n_init{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), static_cast<uint8_t>(n_hops), 0};
        MeshMcastRange ranges_s_init{
            static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), 0, static_cast<uint8_t>(s_hops)};

        setup_route_set_state_for_direction<operation_type>(
            hasW, cm_w, route_id_w, &ranges_w_init, dst_acc, scatter_acc, sem_noc);
        setup_route_set_state_for_direction<operation_type>(
            hasE, cm_e, route_id_e, &ranges_e_init, dst_acc, scatter_acc, sem_noc);
        setup_route_set_state_for_direction<operation_type>(
            hasN, cm_n, route_id_n, &ranges_n_init, dst_acc, scatter_acc, sem_noc);
        setup_route_set_state_for_direction<operation_type>(
            hasS, cm_s, route_id_s, &ranges_s_init, dst_acc, scatter_acc, sem_noc);
    }

    // Main loop - process pages
    constexpr uint32_t loop_increment = (operation_type == OperationType::Scatter) ? 2 : 1;
    constexpr uint32_t cb_wait_count = (operation_type == OperationType::Scatter) ? 2 : 1;

    // Per-direction multicast ranges (same for both route and non-route)
    MeshMcastRange ranges_w{0, static_cast<uint8_t>(w_hops), 0, 0};
    MeshMcastRange ranges_e{static_cast<uint8_t>(e_hops), 0, 0, 0};
    MeshMcastRange ranges_n{
        static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), static_cast<uint8_t>(n_hops), 0};
    MeshMcastRange ranges_s{
        static_cast<uint8_t>(e_hops), static_cast<uint8_t>(w_hops), 0, static_cast<uint8_t>(s_hops)};

    for (uint32_t i = 0; i < TOTAL_PAGES; i += loop_increment) {
        cb_wait_front(CB_ID, cb_wait_count);
        const uint32_t src_l1_addr = get_read_ptr(CB_ID);

        if constexpr (is_route_variant) {
            // Route variant: directional fanout using route-based APIs
            send_route_directional_fanout<operation_type, api_variant>(
                w_hops, cm_w, route_id_w, &ranges_w, src_l1_addr, dst_acc, scatter_acc, i, sem_noc);
            send_route_directional_fanout<operation_type, api_variant>(
                e_hops, cm_e, route_id_e, &ranges_e, src_l1_addr, dst_acc, scatter_acc, i, sem_noc);
            send_route_directional_fanout<operation_type, api_variant>(
                n_hops, cm_n, route_id_n, &ranges_n, src_l1_addr, dst_acc, scatter_acc, i, sem_noc);
            send_route_directional_fanout<operation_type, api_variant>(
                s_hops, cm_s, route_id_s, &ranges_s, src_l1_addr, dst_acc, scatter_acc, i, sem_noc);
        } else {
            // Non-route variant: directional fanout
            send_directional_fanout<operation_type, api_variant>(
                w_hops, &senderW, left_packet_header, ranges_w, src_l1_addr, dst_acc, scatter_acc, i, sem_noc);
            send_directional_fanout<operation_type, api_variant>(
                e_hops, &senderE, right_packet_header, ranges_e, src_l1_addr, dst_acc, scatter_acc, i, sem_noc);
            send_directional_fanout<operation_type, api_variant>(
                n_hops, &senderN, north_packet_header, ranges_n, src_l1_addr, dst_acc, scatter_acc, i, sem_noc);
            send_directional_fanout<operation_type, api_variant>(
                s_hops, &senderS, south_packet_header, ranges_s, src_l1_addr, dst_acc, scatter_acc, i, sem_noc);
        }

        cb_pop_front(CB_ID, cb_wait_count);
    }

    noc_async_writes_flushed();

    // Post-loop completion: BasicWrite and Scatter need separate atomic inc
    // FusedAtomicInc does NOT (it's fused with each write)
    if constexpr (operation_type != OperationType::FusedAtomicInc) {
        ASSERT(sem_l1_addr != 0);
        const uint64_t sem_noc_final = safe_get_noc_addr(rx_noc_x, rx_noc_y, sem_l1_addr, /*NOC_INDEX=*/0);

        if constexpr (is_route_variant) {
            // Route variant: send completion per active branch using route-based atomic inc
            if (w_hops > 0) {
                fabric_multicast_noc_unicast_atomic_inc(
                    cm_w,
                    route_id_w,
                    &ranges_w,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_final, /*inc=*/1, /*width_bits=*/32));
            }
            if (e_hops > 0) {
                fabric_multicast_noc_unicast_atomic_inc(
                    cm_e,
                    route_id_e,
                    &ranges_e,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_final, /*inc=*/1, /*width_bits=*/32));
            }
            if (n_hops > 0) {
                fabric_multicast_noc_unicast_atomic_inc(
                    cm_n,
                    route_id_n,
                    &ranges_n,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_final, /*inc=*/1, /*width_bits=*/32));
            }
            if (s_hops > 0) {
                fabric_multicast_noc_unicast_atomic_inc(
                    cm_s,
                    route_id_s,
                    &ranges_s,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader(sem_noc_final, /*inc=*/1, /*width_bits=*/32));
            }
        } else {
            // Send a completion per active branch so every sub-tree gets the semaphore bump.
            send_completion_atomic_inc(w_hops, senderW, left_packet_header, 0, w_hops, 0, 0, sem_noc_final);
            send_completion_atomic_inc(e_hops, senderE, right_packet_header, e_hops, 0, 0, 0, sem_noc_final);
            send_completion_atomic_inc(n_hops, senderN, north_packet_header, e_hops, w_hops, n_hops, 0, sem_noc_final);
            send_completion_atomic_inc(s_hops, senderS, south_packet_header, e_hops, w_hops, 0, s_hops, sem_noc_final);
        }
    }

    if constexpr (is_route_variant) {
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
    } else {
        if (hasW) {
            senderW.close();
        }
        if (hasE) {
            senderE.close();
        }
        if (hasN) {
            senderN.close();
        }
        if (hasS) {
            senderS.close();
        }
    }
}
