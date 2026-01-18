// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef __FDS_FUNCTIONS_HPP__
#define __FDS_FUNCTIONS_HPP__

#include <cstdint>
#include "tt_fds_dispatch_reg.h"
#include "tt_fds_tensixneo_reg.h"
#include "rocc_instructions.hpp"

#define TRAP_INT_MASK 0x8000000000000000  // Mask to identify interrupt traps
#define CORE_OFFSET 0x1000                // Offset between mhartid cores

namespace FdsDispatch {
// Configure filter length: how many cycles a done signal from a NEO must be stable (through the deglitcher)
void fds_config_filter_length(uint32_t threshold);

// Configure groups: enable specific neos (mask) for each group ID, set count threshold for the groupID
void fds_config_groupid(uint32_t group_id, uint32_t mask, uint32_t threshold);

// Configure interrupts for groupIDs: set bit for the groupIDs to generate interrupts
void fds_config_interrupt_en(uint32_t mask);

// Configure auto dispatch: enable feature, set cycle count and outbox address
void fds_config_auto_dispatch(bool enable, uint32_t cycle_count, uint32_t address);

// Send go signal to from Dispatch to NEO Tiles
void fds_go(bool ad_enable, uint16_t group_id);

// When AD is enabled: send go signal from Dispatch to NEO Tiles; if fifo is full, will automatically block until space
// is available
void fds_go_blocking(uint16_t group_id);

// Clear interrupt on FDS interface side by writing a 0 to specified input bus register
void fds_clear_neo_status(uint32_t neo_inst);

// Poll for count threshold reached for specified group ID
void fds_poll(uint32_t group_id, uint32_t count_threshold);
}  // namespace FdsDispatch

namespace FdsNeo {
// Configure filter length: how many cycles a done signal from a NEO must be stable (through the deglitcher)
void fds_config_filter_length(uint32_t threshold);

// Configure groups: enable specific dispatches (mask) for each group ID
void fds_config_groupid(uint32_t group_id, uint32_t mask, uint32_t threshold);

// Configure auto dispatch: enable feature, set cycle count and outbox address
void fds_config_auto_dispatch(bool enable, uint32_t cycle_count, uint32_t address);

// Poll for go signal from specified Dispatch
void fds_poll(uint32_t group_id, uint32_t dispatch_inst);

// Send done signal from NEO Tiles to Dispatch
void fds_done(bool ad_enable, uint32_t group_id);

// When AD is enabled: send done signal from NEO Tiles to Dispatch; if fifo is full, will automatically block until
// space is available
void fds_done_blocking(uint16_t group_id);

// Clear done signal (use between done signals of same group ID)
void fds_clear_done();

// Clear interrupt on FDS interface side by writing a 0 to specified input bus register
void fds_clear_de_status(uint32_t dispatch_inst);

// Configure interrupts for groupIDs: set bit for the groupIDs to generate interrupts
void fds_config_interrupt_en(uint32_t mask);
}  // namespace FdsNeo
#endif
