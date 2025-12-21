// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

////

int get_epoch_table_x(int my_x, int my_y) __attribute__((const));
int get_epoch_table_y(int my_x, int my_y) __attribute__((const));

inline uint16_t op_pack_tiles_ptr_add(uint16_t a, uint16_t b) { return (a + b) & 0x3FF; }

inline uint16_t op_pack_tiles_ptr_sub(uint16_t a, uint16_t b) { return (a - b) & 0x3FF; }
