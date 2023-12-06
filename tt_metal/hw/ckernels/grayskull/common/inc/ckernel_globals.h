// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel_structs.h"
#include "risc_attribs.h"

extern uint32_t cfg_state_id;
extern uint32_t unp_cfg_context;
extern uint32_t gl_alu_format_spec_reg;

extern volatile uint32_t l1_buffer[16];

extern uint32_t pack_sync_tile_dst_ptr;
extern uint32_t math_sync_tile_dst_index;
