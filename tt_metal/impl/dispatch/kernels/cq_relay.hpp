// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "dataflow_api.h"
#include "risc_attribs.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "fabric_edm_packet_header.hpp"
#include "debug/waypoint.h"

//
// This class maintains an open connection to the Fabric Mux
// Once it goes out of scope the connection is torn down
//
template <
    uint32_t mux_x,
    uint32_t mux_y,
    uint32_t mux_channel_base_address,
    uint32_t mux_num_buffers_per_channel,
    uint32_t mux_flow_control_address,
    uint32_t mux_connection_handshake_address,
    uint32_t mux_connection_info_address,
    uint32_t mux_channel_buffer_size_bytes,
    uint32_t mux_buffer_index_address,
    uint32_t worker_flow_control_sem,
    uint32_t worker_teardown_sem,
    uint32_t worker_buffer_index_sem,
    uint32_t mux_status_address,
    uint32_t local_mux_status_address,
    bool is_hd_variant>
struct FDFabricMuxConnectionScope {
    constexpr static ProgrammableCoreType fd_core_type = static_cast<ProgrammableCoreType>(FD_CORE_TYPE);
    tt::tt_fabric::WorkerToFabricMuxSender<mux_num_buffers_per_channel>* edm_ptr;

    FORCE_INLINE FDFabricMuxConnectionScope(tt::tt_fabric::WorkerToFabricMuxSender<mux_num_buffers_per_channel>& edm) {
        if constexpr (is_hd_variant) {
            return;
        }

        WAYPOINT("FMCW");

        edm.init(
            true /*connected_to_persistent_fabric*/,
            0,
            mux_x,
            mux_y,
            mux_channel_base_address,
            mux_num_buffers_per_channel,
            mux_flow_control_address,
            mux_connection_handshake_address,
            mux_connection_info_address,
            mux_channel_buffer_size_bytes,
            mux_buffer_index_address,
            (volatile uint32_t* const)get_semaphore<fd_core_type>(worker_flow_control_sem),
            (volatile uint32_t* const)get_semaphore<fd_core_type>(worker_teardown_sem),
            get_semaphore<fd_core_type>(worker_buffer_index_sem));

        tt::tt_fabric::wait_for_fabric_endpoint_ready(mux_x, mux_y, mux_status_address, local_mux_status_address);
        tt::tt_fabric::fabric_client_connect<mux_num_buffers_per_channel>(edm);

        WAYPOINT("FMCD");

        edm_ptr = &edm;
    }

    FORCE_INLINE ~FDFabricMuxConnectionScope() {
        if constexpr (is_hd_variant) {
            return;
        }
        DPRINT << "MUX DISCONNECT" << ENDL();
        tt::tt_fabric::fabric_client_disconnect<mux_num_buffers_per_channel>(*edm_ptr);
    }
};

// Use the EDM Client to write data in packets
template <
    uint32_t fabric_mux_channel_buffer_size_bytes,
    uint32_t fabric_mux_num_buffers_per_channel,
    uint32_t header_rb,
    typename T>
inline void cq_fabric_write_any_len(T& edm, uint32_t data_ptr, uint64_t dst_ptr, uint32_t length) {
    // DPRINT << "CQ WRITE ANY LEN " << length << ENDL();
    // Writing to a HEADER only buffer is wrong. This function requires a FULL SIZE buffer
    ASSERT(fabric_mux_channel_buffer_size_bytes > sizeof(PACKET_HEADER_TYPE));
    constexpr uint32_t k_FabricMaxBurstSize = fabric_mux_channel_buffer_size_bytes - sizeof(PACKET_HEADER_TYPE);

    auto packet_header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(header_rb);
    packet_header->to_chip_unicast(static_cast<uint8_t>(1));  // Defeault to 1 Hop for N300
    while (length > k_FabricMaxBurstSize) {
        // DPRINT << "Send write of length " << DEC() << k_FabricMaxBurstSize << ENDL();
        packet_header->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst_ptr}, k_FabricMaxBurstSize);

        tt::tt_fabric::fabric_async_write<fabric_mux_num_buffers_per_channel>(
            edm, packet_header, data_ptr, k_FabricMaxBurstSize);

        dst_ptr += k_FabricMaxBurstSize;
        data_ptr += k_FabricMaxBurstSize;
        length -= k_FabricMaxBurstSize;
    }

    packet_header->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst_ptr}, length);

    tt::tt_fabric::fabric_async_write<fabric_mux_num_buffers_per_channel>(edm, packet_header, data_ptr, length);
}

// Use the EDM Client to write data in packets and then atomic increment at the end
template <
    uint32_t downstream_noc_xy,
    uint32_t downstream_sem_id,
    uint32_t fabric_mux_channel_buffer_size_bytes,
    uint32_t fabric_mux_num_buffers_per_channel,
    uint32_t header_rb,
    typename T>
inline void cq_fabric_write_atomic_inc_any_len(
    T& edm, uint32_t data_ptr, uint64_t dst_ptr, uint32_t length, uint16_t n) {
    // Writing to a HEADER only buffer is wrong. This function requires a FULL SIZE buffer
    ASSERT(fabric_mux_channel_buffer_size_bytes > sizeof(PACKET_HEADER_TYPE));
    constexpr uint32_t k_FabricMaxBurstSize = fabric_mux_channel_buffer_size_bytes - sizeof(PACKET_HEADER_TYPE);

    auto packet_header = reinterpret_cast<tt_l1_ptr PACKET_HEADER_TYPE*>(header_rb);
    packet_header->to_chip_unicast(static_cast<uint8_t>(1));  // Defeault to 1 Hop for N300
    while (length > k_FabricMaxBurstSize) {
        packet_header->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst_ptr}, k_FabricMaxBurstSize);

        tt::tt_fabric::fabric_async_write<fabric_mux_num_buffers_per_channel>(
            edm, packet_header, data_ptr, k_FabricMaxBurstSize);

        dst_ptr += k_FabricMaxBurstSize;
        data_ptr += k_FabricMaxBurstSize;
        length -= k_FabricMaxBurstSize;
    }

    packet_header->to_noc_fused_unicast_write_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
            dst_ptr,
            get_noc_addr_helper(
                downstream_noc_xy, get_semaphore<static_cast<ProgrammableCoreType>(FD_CORE_TYPE)>(downstream_sem_id)),
            n,
            std::numeric_limits<uint16_t>::max()},
        length);

    tt::tt_fabric::fabric_async_write<fabric_mux_num_buffers_per_channel>(edm, packet_header, data_ptr, length);
}

// Use the EDM Client to release pages
template <
    uint32_t dest_noc_xy,
    uint32_t dest_sem_id,
    uint32_t fabric_mux_num_buffers_per_channel,
    uint32_t header_rb,
    typename T>
inline void cq_fabric_release_pages(T& edm, uint16_t n) {
    auto sem_addr = get_semaphore<static_cast<ProgrammableCoreType>(FD_CORE_TYPE)>(dest_sem_id);
    uint64_t noc_dest_addr = get_noc_addr_helper(dest_noc_xy, sem_addr);

    auto packet_header = reinterpret_cast<volatile tt_l1_ptr PACKET_HEADER_TYPE*>(header_rb);
    packet_header->to_chip_unicast(static_cast<uint8_t>(1));  // Default to 1 Hop for N300
    packet_header->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        noc_dest_addr,
        n,
        std::numeric_limits<uint16_t>::max(),
    });
    tt::tt_fabric::fabric_atomic_inc<fabric_mux_num_buffers_per_channel>(edm, packet_header);
}

template <uint32_t mux_x, uint32_t mux_y, uint32_t mux_termination_signal_address>
inline void cq_fabric_terminate() {
    noc_inline_dw_write(
        get_noc_addr(mux_x, mux_y, mux_termination_signal_address),
        static_cast<uint32_t>(tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE));
}
