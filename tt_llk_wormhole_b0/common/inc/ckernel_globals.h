// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_structs.h"
#include "risc_attribs.h"

extern uint32_t cfg_state_id;
extern uint32_t unp_cfg_context;

extern uint32_t volatile tt_l1_ptr l1_buffer[16];

extern uint32_t pack_sync_tile_dst_ptr;
extern uint32_t math_sync_tile_dst_index;

extern uint32_t __local_mem_rodata_start_addr[];
extern uint32_t __local_mem_rodata_end_addr[];
extern uint32_t __firmware_start[];
