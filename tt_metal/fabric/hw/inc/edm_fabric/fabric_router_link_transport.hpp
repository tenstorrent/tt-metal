// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Link transport for the fabric router. All sends and credit updates to the peer router go through
// SenderLinkTransport / ReceiverLinkTransport instead of the eth txq APIs directly, so other transports
// (e.g. a host bridge) can be swapped in at compile time without touching the router logic.
//
// Interface:
//   is_busy()                        - transport can't accept a new command yet
//   send_packet(src, dest, bytes)    - send bytes to the remote address. Requires !is_busy()
//   send_with_context_switch(...)    - same, but blocks (context switching) until the transport accepts it.
//                                      For the handshake and other cold paths.
//   remote_update_ptr_val<id>(val)   - increment remote stream reg. Requires !is_busy()
//   remote_update_ptr_val(id, val)   - same, runtime stream id

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

#include "internal/ethernet/tunneling.h"
#ifdef FABRIC_HOST_BRIDGE
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_router_host_bridge_transport.hpp"
#endif

#include <cstdint>

// Ethernet txq transport
template <uint32_t TXQ_ID>
struct EthTxqLinkTransport {

    static FORCE_INLINE bool is_busy() { return internal_::eth_txq_is_busy(TXQ_ID); }

    // Assumes !is_busy() -- PLEASE CHECK BEFORE CALLING
    static FORCE_INLINE void send_packet(uint32_t src_addr, uint32_t dest_addr, uint32_t num_bytes) {
        internal_::eth_send_packet_bytes_unsafe(TXQ_ID, src_addr, dest_addr, num_bytes);
    }

    static FORCE_INLINE void send_with_context_switch(uint32_t src_addr, uint32_t dest_addr, uint32_t num_bytes) {
        internal_::eth_send_packet<true>(TXQ_ID, src_addr >> 4, dest_addr >> 4, num_bytes >> 4);
    }

    // Assumes !is_busy()
    template <uint32_t stream_id>
    static FORCE_INLINE void remote_update_ptr_val(int32_t val) {
        ::remote_update_ptr_val<stream_id, TXQ_ID>(val);
    }

    // Assumes !is_busy()
    static FORCE_INLINE void remote_update_ptr_val(uint32_t stream_id, int32_t val) {
        ::remote_update_ptr_val<TXQ_ID>(stream_id, val);
    }
};

// Transport selection. Sender side carries packets + packets-sent updates, receiver side carries acks + completions.
// FABRIC_HOST_BRIDGE (builder define): both go through the host instead of the wire.
#ifdef FABRIC_HOST_BRIDGE
using SenderLinkTransport = HostBridgeLinkTransport<
    host_bridge_block_addr + tt::tt_fabric::host_bridge::kTxSocketOffset,
    host_bridge_block_addr + tt::tt_fabric::host_bridge::kStatusOffset,
    host_bridge_page_bytes,
    host_bridge_tx_noc,
    host_bridge_tx_cmd_buf>;
using ReceiverLinkTransport = HostBridgeLinkTransport<
    host_bridge_block_addr + tt::tt_fabric::host_bridge::kRxSocketOffset,
    host_bridge_block_addr + tt::tt_fabric::host_bridge::kStatusOffset,
    host_bridge_page_bytes,
    host_bridge_rx_noc,
    host_bridge_rx_cmd_buf>;
#else
using SenderLinkTransport = EthTxqLinkTransport<sender_txq_id>;
using ReceiverLinkTransport = EthTxqLinkTransport<receiver_txq_id>;
#endif
