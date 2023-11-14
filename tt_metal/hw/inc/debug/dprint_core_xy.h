/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// TODO(AP): temporary debugging API for passing current core's XY from NCRISC to TRISCS
// since compute doesn't currently know it's core id we can pass it via L1 in NC
// Also note that there is currently no synchronization guaranteeing the validity of write
// by the time TRISC code is executed. This is not an issue if it's used after any cb wait in TRISCS.
inline void nc_set_core_xy() {
#if defined(COMPILE_FOR_NCRISC)
    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(0, 0, NOC_NODE_ID);
    uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    auto nc_buffer = reinterpret_cast<uint8_t*>(PRINT_BUFFER_NC);
    reinterpret_cast<DebugPrintMemLayout*>(nc_buffer)->aux.core_x = my_x;
    reinterpret_cast<DebugPrintMemLayout*>(nc_buffer)->aux.core_y = my_y;
#endif
}

inline uint32_t core_x() {
    auto nc_buffer = reinterpret_cast<uint8_t*>(PRINT_BUFFER_NC);
    // we are deliberately looking at NC print buffer on all cores, since NC will be writing there
    // TODO: this can be at a single location or look for a different solution
    return reinterpret_cast<DebugPrintMemLayout*>(nc_buffer)->aux.core_x;
}

inline uint32_t core_y() {
    auto nc_buffer = reinterpret_cast<uint8_t*>(PRINT_BUFFER_NC);
    return reinterpret_cast<DebugPrintMemLayout*>(nc_buffer)->aux.core_y;
}
