// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/hw/inc/risc_attribs.h"
#include "noc_overlay_parameters.h"
#include "tt_metal/hw/inc/ethernet/tunneling.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/named_types.hpp"

using StreamId = tt::tt_fabric::NamedType<uint32_t, struct StreamIdType>;

// This will be an atomic register read to the register
template <uint32_t stream_id>
FORCE_INLINE int32_t get_ptr_val() {
#ifdef ARCH_WORMHOLE
    return NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
#else
    return (
        NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) &
        ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1));
#endif
}
FORCE_INLINE int32_t get_ptr_val(uint8_t stream_id) {
#ifdef ARCH_WORMHOLE
    return NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
#else
    return (
        NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX) &
        ((1 << REMOTE_DEST_WORDS_FREE_WIDTH) - 1));
#endif
}

// Writing to this register will leverage the built-in stream hardware which will automatically perform an atomic
// increment on the register. This can save precious erisc cycles by offloading a lot of pointer manipulation.
// Additionally, these registers are accessible via eth_reg_write calls which can be used to write a value,
// inline the eth command (without requiring source L1)
template <uint32_t stream_id>
FORCE_INLINE void increment_local_update_ptr_val(int32_t val) {
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX, val << REMOTE_DEST_BUF_WORDS_FREE_INC);
}
FORCE_INLINE void increment_local_update_ptr_val(uint8_t stream_id, int32_t val) {
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX, val << REMOTE_DEST_BUF_WORDS_FREE_INC);
}

template <uint32_t stream_id>
constexpr FORCE_INLINE uint32_t get_stream_reg_read_addr() {
    return STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
}

FORCE_INLINE uint32_t get_stream_reg_read_addr(uint8_t stream_id) {
    return STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
}

template <uint32_t stream_id>
constexpr FORCE_INLINE uint32_t get_stream_reg_write_addr() {
    return STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
}

FORCE_INLINE uint32_t get_stream_reg_write_addr(uint8_t stream_id) {
    return STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
}

template <uint32_t stream_id, uint32_t txq_id>
FORCE_INLINE void remote_update_ptr_val(int32_t val) {
    constexpr uint32_t addr = STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
    internal_::eth_write_remote_reg_no_txq_check(txq_id, addr, val << REMOTE_DEST_BUF_WORDS_FREE_INC);
}
template <uint32_t txq_id>
FORCE_INLINE void remote_update_ptr_val(uint32_t stream_id, int32_t val) {
    const uint32_t addr = STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
    internal_::eth_write_remote_reg_no_txq_check(txq_id, addr, val << REMOTE_DEST_BUF_WORDS_FREE_INC);
}

template <uint32_t stream_id>
FORCE_INLINE void init_ptr_val(int32_t val) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, val);
}

FORCE_INLINE void init_ptr_val(uint32_t stream_id, int32_t val) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, val);
}
