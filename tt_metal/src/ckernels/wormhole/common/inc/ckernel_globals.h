/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include "ckernel_structs.h"

extern uint32_t cfg_state_id;
extern uint32_t unp_cfg_context;

extern volatile uint32_t l1_buffer[16];

//extern const int32_t unpack_src_format[24];
//extern const int32_t unpack_dst_format[24];
//extern const int32_t pack_src_format[16];
//extern const int32_t pack_dst_format[16];

extern uint32_t pack_sync_tile_dst_ptr;
extern uint32_t math_sync_tile_dst_index;

extern ckernel::operand_u operands[24];
extern ckernel::output_u outputs[16];

extern uint32_t __local_mem_rodata_start_addr[];
extern uint32_t __local_mem_rodata_end_addr[];
extern uint32_t __firmware_start[];
