// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "internal/risc_attribs.h"
#include "noc_overlay_parameters.h"

// A fabric credit counter of another core on the same chip, added to over the NOC.
//
// Fabric code names such a counter only through FabricCreditCounter and the functions below, never by a register
// address, so an architecture whose counters are not stream registers supplies its own version of this header on its
// include path. The local and Ethernet-peer operations stay in fabric_stream_regs.hpp.
//
// This version: a counter is a stream register, and an add of n is an inline write of n << REMOTE_DEST_BUF_WORDS_FREE_INC
// to the stream's increment register.

struct FabricCreditCounter {
    uint32_t reg_addr;  // the stream's STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE register
};

// The counter of stream @p stream_id, on whichever core a later add targets.
FORCE_INLINE constexpr FabricCreditCounter make_fabric_credit_counter(uint32_t stream_id) {
    return FabricCreditCounter{STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX)};
}

// Add @p val (|val| below 2^(31 - REMOTE_DEST_BUF_WORDS_FREE_INC)) to @p counter on core (@p noc_x, @p noc_y).
template <bool posted = false>
FORCE_INLINE void fabric_credit_counter_add(
    uint8_t noc_x, uint8_t noc_y, FabricCreditCounter counter, int32_t val, uint8_t noc = noc_index) {
    noc_inline_dw_write<InlineWriteDst::REG, posted>(
        get_noc_addr(noc_x, noc_y, counter.reg_addr, noc), val << REMOTE_DEST_BUF_WORDS_FREE_INC, 0xF, noc);
}

// Program @p cmd_buf to add @p val to @p counter on core (@p noc_x, @p noc_y); each
// fabric_credit_counter_add_with_state() on it then issues that add. Other traffic on @p cmd_buf in between clobbers
// the state.
template <bool posted = false>
FORCE_INLINE void fabric_credit_counter_add_set_state(
    uint8_t noc_x,
    uint8_t noc_y,
    FabricCreditCounter counter,
    int32_t val,
    uint8_t cmd_buf,
    uint8_t noc = noc_index,
    uint8_t vc = NOC_UNICAST_WRITE_VC) {
    noc_inline_dw_write_set_state<posted, /*set_val=*/true>(
        get_noc_addr(noc_x, noc_y, counter.reg_addr, noc),
        val << REMOTE_DEST_BUF_WORDS_FREE_INC,
        0xF,
        cmd_buf,
        noc,
        vc);
}

// Issue the add programmed on @p cmd_buf by fabric_credit_counter_add_set_state().
template <bool update_counter = true, bool posted = false>
FORCE_INLINE void fabric_credit_counter_add_with_state(uint8_t cmd_buf, uint8_t noc = noc_index) {
    noc_inline_dw_write_with_state<
        /*update_addr_lo=*/false,
        update_counter,
        posted,
        /*update_addr_hi=*/false,
        /*update_val=*/false,
        InlineWriteDst::REG>(0, 0, cmd_buf, noc);
}

// As above, retargeted to @p counter on the core and with the value programmed on @p cmd_buf.
template <bool update_counter = true, bool posted = false>
FORCE_INLINE void fabric_credit_counter_add_with_state(
    FabricCreditCounter counter, uint8_t cmd_buf, uint8_t noc = noc_index) {
    noc_inline_dw_write_with_state<
        /*update_addr_lo=*/true,
        update_counter,
        posted,
        /*update_addr_hi=*/false,
        /*update_val=*/false,
        InlineWriteDst::REG>(0, counter.reg_addr, cmd_buf, noc);
}
