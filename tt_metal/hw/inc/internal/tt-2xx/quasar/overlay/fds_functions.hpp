// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
// Version: FFN1.3.0

#ifndef __FDS_FUNCTIONS_HPP__
#define __FDS_FUNCTIONS_HPP__

#include <cstdint>
#include "meta/fds_registers/tt_fds_dispatch_reg.h"
#include "meta/fds_registers/tt_fds_tensixneo_reg.h"
#include "rocc_instructions.hpp"

#define TRAP_INT_MASK 0x8000000000000000  // Mask to identify interrupt traps
#define CORE_OFFSET 0x1000                // Offset between mhartid cores

// Definitions live in this header: these are one-instruction ROCC register accesses, and JIT-compiled
// kernels link only against their own translation unit plus the firmware wrapper, so an out-of-line
// implementation would be unresolvable from kernel code.
namespace overlay {

namespace FdsDispatch {
// Configure filter length: how many cycles a done signal from a NEO must be stable (through the deglitcher)
inline void fds_config_filter_length(uint32_t threshold) {
    FDS_INTF_WRITE(TT_FDS_DISPATCH_FILTER_COUNT_THRESHOLD_REG_ADDR, threshold);
}

// Configure groups: enable specific neos (mask) for each group ID, set count threshold for the groupID
inline void fds_config_groupid(uint32_t group_id, uint32_t mask, uint32_t threshold) {
    FDS_INTF_WRITE((TT_FDS_DISPATCH_GROUPID_ENABLE_0__REG_ADDR + (group_id * sizeof(uint32_t))), mask);
    FDS_INTF_WRITE((TT_FDS_DISPATCH_GROUPID_COUNT_THRESHOLD_0__REG_ADDR + (group_id * sizeof(uint32_t))), threshold);
}

// Configure interrupts for groupIDs: set bit for the groupIDs to generate interrupts
inline void fds_config_interrupt_en(uint32_t mask) { FDS_INTF_WRITE(TT_FDS_DISPATCH_INTERRUPT_ENABLE_REG_ADDR, mask); }

// Configure auto dispatch: enable feature, set cycle count and outbox address
inline void fds_config_auto_dispatch(bool enable, uint32_t cycle_count, uint32_t address) {
    if (enable) {
        FDS_INTF_WRITE(TT_FDS_DISPATCH_AUTO_DISPATCH_EN_REG_ADDR, 0x1);
    }
    FDS_INTF_WRITE(TT_FDS_DISPATCH_AUTO_DISPATCH_CYCLE_COUNT_REG_ADDR, cycle_count);
    FDS_INTF_WRITE(TT_FDS_DISPATCH_AUTO_DISPATCH_OUTBOX_ADDRESS_REG_ADDR, address);
}

// Send go signal to from Dispatch to NEO Tiles
inline void fds_go(bool ad_enable, uint16_t group_id) {
    if (ad_enable) {
        while (FDS_INTF_READ(TT_FDS_DISPATCH_AUTO_DISPATCH_FIFO_FULL_REG_ADDR));
    }
    FDS_INTF_WRITE(TT_FDS_DISPATCH_DISPATCH_TO_TENSIX_REG_ADDR, group_id);
}

// When AD is enabled: send go signal from Dispatch to NEO Tiles; if fifo is full, will automatically block until space
// is available
inline void fds_go_blocking(uint16_t group_id) {
    FDS_INTF_WRITE(TT_FDS_DISPATCH_DISPATCH_TO_TENSIX_REG_ADDR, group_id);
}

// Clear interrupt on FDS interface side by writing a 0 to specified input bus register
inline void fds_clear_neo_status(uint32_t neo_inst) {
    FDS_INTF_WRITE(TT_FDS_DISPATCH_TENSIX_TO_DISPATCH_0__REG_ADDR + (neo_inst * sizeof(uint32_t)), 0x0);
}

// Read how many enabled NEOs have signalled done for the specified group ID
inline uint32_t fds_read_group_count(uint32_t group_id) {
    return FDS_INTF_READ(TT_FDS_DISPATCH_GROUPID_COUNT_0__REG_ADDR + (group_id * sizeof(uint32_t)));
}

// Poll for count threshold reached for specified group ID
inline void fds_poll(uint32_t group_id, uint32_t count_threshold) {
    while (fds_read_group_count(group_id) < count_threshold);
}
}  // namespace FdsDispatch

namespace FdsNeo {
// Configure filter length: how many cycles a go signal from a Dispatch must be stable (through the deglitcher)
inline void fds_config_filter_length(uint32_t threshold) {
    FDS_INTF_WRITE(TT_FDS_TENSIXNEO_FILTER_COUNT_THRESHOLD_REG_ADDR, threshold);
}

// Configure groups: enable specific dispatches (mask) for each group ID
inline void fds_config_groupid(uint32_t group_id, uint32_t mask, uint32_t threshold) {
    FDS_INTF_WRITE((TT_FDS_TENSIXNEO_GROUPID_ENABLE_0__REG_ADDR + (group_id * sizeof(uint32_t))), mask);
    FDS_INTF_WRITE((TT_FDS_TENSIXNEO_GROUPID_COUNT_THRESHOLD_0__REG_ADDR + (group_id * sizeof(uint32_t))), threshold);
}

// Configure auto dispatch: enable feature, set cycle count and outbox address
inline void fds_config_auto_dispatch(bool enable, uint32_t cycle_count, uint32_t address) {
    if (enable) {
        FDS_INTF_WRITE(TT_FDS_TENSIXNEO_AUTO_DISPATCH_EN_REG_ADDR, 0x1);
    }
    FDS_INTF_WRITE(TT_FDS_TENSIXNEO_AUTO_DISPATCH_CYCLE_COUNT_REG_ADDR, cycle_count);
    FDS_INTF_WRITE(TT_FDS_TENSIXNEO_AUTO_DISPATCH_OUTBOX_ADDRESS_REG_ADDR, address);
}

// Read the latched go status for the specified group ID
inline uint32_t fds_read_group_status(uint32_t group_id) {
    return FDS_INTF_READ(TT_FDS_TENSIXNEO_GROUPID_STATUS_0__REG_ADDR + (group_id * sizeof(uint32_t)));
}

// Read the raw go value the specified dispatch instance is driving into this NEO
inline uint32_t fds_read_de_status(uint32_t dispatch_inst) {
    return FDS_INTF_READ(TT_FDS_TENSIXNEO_DISPATCH_TO_TENSIX_0__REG_ADDR + (dispatch_inst * sizeof(uint32_t)));
}

// Poll for go signal from specified dispatch
inline void fds_poll(uint32_t group_id, uint32_t dispatch_inst) {
    while (!fds_read_group_status(group_id));
    while (fds_read_de_status(dispatch_inst) != group_id);
}

// Send done signal from NEO Tiles to Dispatch
inline void fds_done(bool ad_enable, uint32_t group_id) {
    if (ad_enable) {
        while (FDS_INTF_READ(TT_FDS_TENSIXNEO_AUTO_DISPATCH_FIFO_FULL_REG_ADDR));
    }
    FDS_INTF_WRITE(TT_FDS_TENSIXNEO_TENSIX_TO_DISPATCH_REG_ADDR, group_id);
}

// When AD is enabled: send done signal from NEO Tiles to Dispatch; if fifo is full, will automatically block until
// space is available
inline void fds_done_blocking(uint16_t group_id) {
    FDS_INTF_WRITE(TT_FDS_TENSIXNEO_TENSIX_TO_DISPATCH_REG_ADDR, group_id);
}

// Clear done signal (use between done signals of same group ID)
inline void fds_clear_done() { FDS_INTF_WRITE(TT_FDS_TENSIXNEO_TENSIX_TO_DISPATCH_REG_ADDR, 0); }

// Clear interrupt on FDS interface side by writing a 0 to specified input bus register
inline void fds_clear_de_status(uint32_t dispatch_inst) {
    FDS_INTF_WRITE(TT_FDS_TENSIXNEO_DISPATCH_TO_TENSIX_0__REG_ADDR + (dispatch_inst * sizeof(uint32_t)), 0x0);
}

// Configure interrupts for groupIDs: set bit for the groupIDs to generate interrupts
inline void fds_config_interrupt_en(uint32_t mask) { FDS_INTF_WRITE(TT_FDS_TENSIXNEO_INTERRUPT_ENABLE_REG_ADDR, mask); }
}  // namespace FdsNeo

}  // namespace overlay

#endif
