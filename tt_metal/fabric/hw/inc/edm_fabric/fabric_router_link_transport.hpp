// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The router's link transport: the one place its send and credit paths touch the hardware that
// carries bytes and stream-register updates to the peer router. Everything in the router that
// moves data or credits across the link goes through SenderLinkTransport / ReceiverLinkTransport
// rather than the ethernet txq primitives directly, so a link that is bridged through the host
// (E2H -> H2H -> H2E) becomes another policy here, selected at compile time, and the router's
// flow control and channel logic stay as they are.
//
// A policy provides:
//   is_busy()                         the transport cannot accept another operation right now
//   send_packet(src, dest, bytes)     move bytes from local L1 to the peer's address; requires !is_busy()
//   remote_update_ptr_val<id>(val)    add val to the peer's stream register `id`; requires !is_busy()
//   remote_update_ptr_val(id, val)    the same with a runtime stream id
//
// Every router call site already has the direction fixed at compile time (packets and the packets-sent
// notification go out on the sender side, acks and completions on the receiver side), which is what
// lets the two directions carry different policies without any runtime check.

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

#include "internal/ethernet/tunneling.h"

#include <cstdint>

// Ethernet transmit queue `TXQ_ID` on this core's link.
template <uint32_t TXQ_ID>
struct EthTxqLinkTransport {
    static constexpr uint32_t txq_id = TXQ_ID;

    static FORCE_INLINE bool is_busy() { return internal_::eth_txq_is_busy(TXQ_ID); }

    // Assumes !is_busy() -- the caller spins on is_busy() first, so that a site which already
    // sampled the busy bit for other reasons does not pay for it twice.
    static FORCE_INLINE void send_packet(uint32_t src_addr, uint32_t dest_addr, uint32_t num_bytes) {
        internal_::eth_send_packet_bytes_unsafe(TXQ_ID, src_addr, dest_addr, num_bytes);
    }

    // Assumes !is_busy().
    template <uint32_t stream_id>
    static FORCE_INLINE void remote_update_ptr_val(int32_t val) {
        ::remote_update_ptr_val<stream_id, TXQ_ID>(val);
    }

    // Assumes !is_busy().
    static FORCE_INLINE void remote_update_ptr_val(uint32_t stream_id, int32_t val) {
        ::remote_update_ptr_val<TXQ_ID>(stream_id, val);
    }
};

// The policies this router is built with. Both directions ride the ethernet link today; a bridged
// link swaps one or both of these for a host-bound policy with the same interface.
using SenderLinkTransport = EthTxqLinkTransport<sender_txq_id>;
using ReceiverLinkTransport = EthTxqLinkTransport<receiver_txq_id>;
