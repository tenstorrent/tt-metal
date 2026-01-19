// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "cq_common.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "internal/risc_attribs.h"
#include "api/debug/waypoint.h"
#include "noc/noc_parameters.h"

#if !defined(FD_CORE_TYPE)
#define FD_CORE_TYPE 0
#endif

#if !defined(FABRIC_2D)
#define FABRIC_2D 0
#endif

template <uint32_t mux_num_buffers_per_channel, uint32_t mux_channel_buffer_size_bytes, uint32_t header_rb>
class CQRelayClient {
private:
    constexpr static ProgrammableCoreType fd_core_type = static_cast<ProgrammableCoreType>(FD_CORE_TYPE);

    tt::tt_fabric::WorkerToFabricMuxSender<mux_num_buffers_per_channel> edm;

public:
    CQRelayClient() = default;

    template <
        uint8_t noc_index,
        uint32_t mux_x,
        uint32_t mux_y,
        uint32_t mux_worker_credits_stream_id,
        uint32_t mux_channel_base_address,
        uint32_t mux_connection_handshake_address,
        uint32_t mux_connection_info_address,
        uint32_t mux_buffer_index_address,
        uint32_t worker_flow_control_sem,
        uint32_t worker_teardown_sem,
        uint32_t worker_buffer_index_sem,
        uint32_t mux_status_address,
        uint32_t local_mux_status_address,
        uint32_t to_mesh_id,
        uint32_t ew_dim,
        uint32_t packet_header_addr,
        uint8_t num_hops,
        uint8_t downstream_cmd_buf>
    FORCE_INLINE void init(
        uint64_t downstream_noc_addr, uint32_t my_dev_id, uint32_t to_dev_id, uint32_t router_direction) {
        WAYPOINT("FMCW");
#if defined(FABRIC_RELAY)
        edm.template init<fd_core_type>(
            true /*connected_to_persistent_fabric*/,
            mux_x,
            mux_y,
            mux_channel_base_address,
            mux_num_buffers_per_channel,
            mux_connection_handshake_address,
            mux_connection_info_address,
            mux_channel_buffer_size_bytes,
            mux_buffer_index_address,
            (volatile uint32_t* const)get_semaphore<fd_core_type>(worker_flow_control_sem),
            (volatile uint32_t* const)get_semaphore<fd_core_type>(worker_teardown_sem),
            get_semaphore<fd_core_type>(worker_buffer_index_sem),
            mux_worker_credits_stream_id,
            StreamId{0}  // my stream id -- As a sender I currently do NOT get acks over stream regs
        );

        tt::tt_fabric::wait_for_fabric_endpoint_ready(mux_x, mux_y, mux_status_address, local_mux_status_address);
        tt::tt_fabric::fabric_client_connect<mux_num_buffers_per_channel>(edm);

        if constexpr (FABRIC_2D) {
#if defined(GALAXY_CLUSTER)
            tt::tt_fabric::fabric_set_route(
                (tt::tt_fabric::HybridMeshPacketHeader*)packet_header_addr,
                (eth_chan_directions)router_direction,
                0,  // branch forward
                0,  // start hop
                num_hops,
                true);
#else
            tt::tt_fabric::fabric_set_unicast_route(
                (tt::tt_fabric::HybridMeshPacketHeader*)packet_header_addr, to_dev_id, to_mesh_id);
#endif
        } else {
            auto header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(packet_header_addr);
            fabric_set_unicast_route<false>((LowLatencyPacketHeader*)header, num_hops);
        }
#else
        init_write_state_only<noc_index, downstream_cmd_buf>(downstream_noc_addr);
#endif
        WAYPOINT("FMCD");
    }

    template <uint8_t noc_index, uint8_t downstream_cmd_buf>
    FORCE_INLINE void init_write_state_only(uint64_t downstream_noc_addr) {
#if !defined(FABRIC_RELAY)
        cq_noc_async_write_init_state<CQ_NOC_sNdl, false, false, downstream_cmd_buf>(
            0, downstream_noc_addr, 0, noc_index);
#endif
    }

    template <uint8_t noc_index>
    FORCE_INLINE void init_inline_write_state_only(uint64_t downstream_noc_addr) {
#if !defined(FABRIC_RELAY)
        cq_noc_inline_dw_write_init_state<CQ_NOC_INLINE_Ndvb>(downstream_noc_addr);
#endif
    }

    template <uint8_t noc_index, uint64_t noc_xy, uint32_t sem_id>
    FORCE_INLINE void teardown() {
#if defined(FABRIC_RELAY)
        tt::tt_fabric::fabric_client_disconnect<mux_num_buffers_per_channel>(edm);
#else
        constexpr uint32_t k_PacketQueueTeardownFlag = 0x80000000;
        noc_semaphore_inc(
            get_noc_addr_helper(noc_xy, get_semaphore<fd_core_type>(sem_id)), k_PacketQueueTeardownFlag, noc_index);
#endif
    }

    template <uint8_t noc_idx, bool count = true>
    FORCE_INLINE void write_inline(uint64_t dst, uint32_t val) {
#if defined(FABRIC_RELAY)
        auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_rb);
        packet_header->to_noc_unicast_inline_write(
            tt::tt_fabric::NocUnicastInlineWriteCommandHeader{.noc_address = dst, .value = val});
        // Use the fabric_atomic_inc helper to send the header
        tt::tt_fabric::fabric_atomic_inc<mux_num_buffers_per_channel>(edm, packet_header);
#else
        cq_noc_inline_dw_write_with_state<CQ_NOC_INLINE_nDVB>(dst, val, 0xF, noc_idx);
        if constexpr (count) {
            noc_nonposted_writes_num_issued[noc_idx]++;
            noc_nonposted_writes_acked[noc_idx]++;
        }
#endif
    }

    template <uint8_t noc_idx, bool wait, uint8_t downstream_cmd_buf, bool count = true>
    FORCE_INLINE void write_any_len(uint32_t data_ptr, uint64_t dst_ptr, uint32_t length) {
#if defined(FABRIC_RELAY)
        // Writing to a HEADER only buffer is wrong. This function requires a FULL SIZE buffer
        ASSERT(mux_channel_buffer_size_bytes > sizeof(PACKET_HEADER_TYPE));
        constexpr uint32_t k_FabricMaxBurstSize = mux_channel_buffer_size_bytes - sizeof(PACKET_HEADER_TYPE);

        auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_rb);
        while (length > k_FabricMaxBurstSize) {
            packet_header->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst_ptr}, k_FabricMaxBurstSize);

            tt::tt_fabric::fabric_async_write<mux_num_buffers_per_channel>(
                edm, packet_header, data_ptr, k_FabricMaxBurstSize);

            dst_ptr += k_FabricMaxBurstSize;
            data_ptr += k_FabricMaxBurstSize;
            length -= k_FabricMaxBurstSize;
        }

        packet_header->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst_ptr}, length);

        tt::tt_fabric::fabric_async_write<mux_num_buffers_per_channel>(edm, packet_header, data_ptr, length);
#else
        if constexpr (wait) {
            cq_noc_async_write_with_state_any_len<true, count, CQNocWait::CQ_NOC_WAIT, downstream_cmd_buf>(
                data_ptr, dst_ptr, length, 1, noc_idx);
        } else {
            cq_noc_async_write_with_state_any_len<true, count, CQNocWait::CQ_NOC_wait, downstream_cmd_buf>(
                data_ptr, dst_ptr, length, 1, noc_idx);
        }
#endif
    }

    template <uint8_t noc_idx, bool wait, uint8_t downstream_cmd_buf, bool count = true>
    FORCE_INLINE void write(uint32_t data_ptr, uint64_t dst_ptr, uint32_t length) {
#if defined(FABRIC_RELAY)
        write_any_len<noc_idx, wait, downstream_cmd_buf, count>(data_ptr, dst_ptr, length);
#else
        if constexpr (wait) {
            cq_noc_async_write_with_state<
                CQ_NOC_SnDL,
                CQNocWait::CQ_NOC_WAIT,
                CQNocSend::CQ_NOC_SEND,
                downstream_cmd_buf>(data_ptr, dst_ptr, length, 1, noc_idx);
        } else {
            cq_noc_async_write_with_state<
                CQ_NOC_SnDL,
                CQNocWait::CQ_NOC_wait,
                CQNocSend::CQ_NOC_SEND,
                downstream_cmd_buf>(data_ptr, dst_ptr, length, 1, noc_idx);
        }
        if (count) {
            noc_nonposted_writes_num_issued[noc_idx]++;
            noc_nonposted_writes_acked[noc_idx]++;
        }
#endif
    }

    template <uint8_t noc_idx, uint32_t dest_noc_xy, uint32_t dest_sem_id>
    FORCE_INLINE void release_pages(uint32_t n) {
#if defined(FABRIC_RELAY)
        auto sem_addr = get_semaphore<fd_core_type>(dest_sem_id);
        uint64_t noc_dest_addr = get_noc_addr_helper(dest_noc_xy, sem_addr);

        auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_rb);
        packet_header->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{noc_dest_addr, n});
        tt::tt_fabric::fabric_atomic_inc<mux_num_buffers_per_channel>(edm, packet_header);
#else
        noc_semaphore_inc(get_noc_addr_helper(dest_noc_xy, get_semaphore<fd_core_type>(dest_sem_id)), n, noc_idx);
#endif
    }

    template <
        uint32_t downstream_noc_idx,
        uint32_t downstream_noc_xy,
        uint32_t downstream_sem_id,
        bool wait,
        uint8_t downstream_cmd_buf>
    FORCE_INLINE void write_atomic_inc_any_len(uint32_t data_ptr, uint64_t dst_ptr, uint32_t length, uint32_t n) {
#if defined(FABRIC_RELAY)
        // Writing to a HEADER only buffer is wrong. This function requires a FULL SIZE buffer
        ASSERT(mux_channel_buffer_size_bytes > sizeof(PACKET_HEADER_TYPE));
        constexpr uint32_t k_FabricMaxBurstSize = mux_channel_buffer_size_bytes - sizeof(PACKET_HEADER_TYPE);

        auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_rb);
        while (length > k_FabricMaxBurstSize) {
            packet_header->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst_ptr}, k_FabricMaxBurstSize);

            tt::tt_fabric::fabric_async_write<mux_num_buffers_per_channel>(
                edm, packet_header, data_ptr, k_FabricMaxBurstSize);

            dst_ptr += k_FabricMaxBurstSize;
            data_ptr += k_FabricMaxBurstSize;
            length -= k_FabricMaxBurstSize;
        }

        packet_header->to_noc_fused_unicast_write_atomic_inc(
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                dst_ptr, get_noc_addr_helper(downstream_noc_xy, get_semaphore<fd_core_type>(downstream_sem_id)), n},
            length);

        tt::tt_fabric::fabric_async_write<mux_num_buffers_per_channel>(edm, packet_header, data_ptr, length);
#else
        write_any_len<downstream_noc_idx, wait, downstream_cmd_buf>(data_ptr, dst_ptr, length);
        release_pages<downstream_noc_idx, downstream_noc_xy, downstream_sem_id>(n);
#endif
    }
};
