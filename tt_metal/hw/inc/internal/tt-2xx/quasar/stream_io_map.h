// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _STREAM_IO_MAP_
#define _STREAM_IO_MAP_

#include <stdint.h>

#include "internal/risc_attribs.h"

// Note: Quasar doesn't have streaming registers - these definitions exist for code compatibility
const uint32_t OPERAND_START_STREAM = 0;

// Indexed with operand = kernel operand ID (0-31) per the table above
// Used for tile push/pop operations.
inline __attribute__((always_inline)) uint32_t get_operand_stream_id(int operand) {
    return OPERAND_START_STREAM + operand;
}

// Pointers to stream scratch registers (implemented using don't-care functional registers) that are used for CB
// synchronization

inline __attribute__((always_inline)) volatile uint32_t* get_cb_tiles_received_ptr(int operand) {
    return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(
        get_operand_stream_id(operand), STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
}

inline __attribute__((always_inline)) volatile uint32_t* get_cb_tiles_acked_ptr(int operand) {
    return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(
        get_operand_stream_id(operand), STREAM_REMOTE_DEST_BUF_START_REG_INDEX));
}

inline __attribute__((always_inline)) volatile uint32_t* get_cq_finish_ptr() {
    return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(
        get_operand_stream_id(0), STREAM_REMOTE_DEST_BUF_START_REG_INDEX));
}

inline __attribute__((always_inline)) volatile uint32_t* get_sync_register_ptr() {
    return (volatile uint32_t*)(uintptr_t)(STREAM_REG_ADDR(0, STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX));
}
#endif
