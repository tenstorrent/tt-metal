// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "internal/ethernet/tt_eth_api.h"
#include "internal/ethernet/tunneling.h"

// A router's software credits sent to its Ethernet peer (STREAM_REGISTERS_AVAILABLE 0). The router names these sends,
// rather than sending them as plain data, so an architecture with a cheaper path for a credit supplies its own version
// of this header on its include path, as with fabric_software_credit.hpp; callers include it by this root-relative path.
//
// This version: Ethernet data packets; on a Keraunos CCE, the snooped sends chippy compat makes.

#ifdef KERAUNOS_CCE
// Send @p value into the peer's 16-byte slot at @p dest_addr on TX queue @p txq, after the sends before it on that
// queue, as an 8-byte inline write the peer snoops (compat cce_eth_send_counter); @p src_addr is not read.
// Assumes !eth_txq_is_busy(txq).
FORCE_INLINE void fabric_send_credit_word(uint32_t txq, uint32_t /*src_addr*/, uint32_t dest_addr, uint32_t value) {
    ASSERT(!internal_::eth_txq_is_busy(txq));
    internal_::cce_eth::send_dword(txq, dest_addr, value, /*snoop=*/true);
}

// Send @p num_bytes of credit totals from @p src_addr into the peer at @p dest_addr on TX queue @p txq, after the sends
// before it on that queue, as a copy the peer snoops (compat cce_eth_send). Not inline writes: an inline value carries
// 52 bits, and the block packs 32-bit totals 4 bytes apart. Assumes !eth_txq_is_busy(txq).
FORCE_INLINE void fabric_send_credit_block(uint32_t txq, uint32_t src_addr, uint32_t dest_addr, uint32_t num_bytes) {
    ASSERT(!internal_::eth_txq_is_busy(txq));
    internal_::cce_eth::send_bytes(txq, src_addr, dest_addr, num_bytes, /*snoop=*/true);
}
#else
// Send the router's free-running 32-bit credit total @p value, also stored at the start of the 16-byte slot at
// @p src_addr, into the peer's slot at @p dest_addr on TX queue @p txq, after the sends before it on that queue.
// Assumes !eth_txq_is_busy(txq).
FORCE_INLINE void fabric_send_credit_word(uint32_t txq, uint32_t src_addr, uint32_t dest_addr, uint32_t /*value*/) {
    internal_::eth_send_packet_bytes_unsafe(txq, src_addr, dest_addr, ETH_WORD_SIZE_BYTES);
}

// Send @p num_bytes (a multiple of 16) of credit totals from @p src_addr into the peer at @p dest_addr on TX queue
// @p txq, after the sends before it on that queue. Assumes !eth_txq_is_busy(txq).
FORCE_INLINE void fabric_send_credit_block(uint32_t txq, uint32_t src_addr, uint32_t dest_addr, uint32_t num_bytes) {
    internal_::eth_send_packet_bytes_unsafe(txq, src_addr, dest_addr, num_bytes);
}
#endif
