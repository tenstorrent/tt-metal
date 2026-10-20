// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/risc_attribs.h"

#include "api/compile_time_args.h"
#include "risc_common.h"

uint32_t fake_arg(int arg_idx);
uint32_t fake_common_arg(int arg_idx);

template <typename T>
T get_arg_val(int arg_idx) {
    static_assert(sizeof(T) == 4);
    return static_cast<T>(fake_arg(arg_idx));
}

template <typename T>
T get_common_arg_val(int arg_idx) {
    static_assert(sizeof(T) == 4);
    return static_cast<T>(fake_common_arg(arg_idx));
}

inline uint64_t get_noc_addr(uint32_t addr, uint8_t /*noc*/ = 0) { return addr; }

// Copies SRAM to SRAM, complete on return.
void noc_async_write(
    uint32_t src_local_l1_addr, uint64_t dst_noc_addr, uint32_t size, uint8_t noc = 0, uint32_t vc = 2);
inline void noc_async_write_barrier(uint8_t /*noc*/ = 0) {}
