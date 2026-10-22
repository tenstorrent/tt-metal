// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Address generator 0 on a CCE (quasar_addrgen / quasar_im2col port): program one of the upstream example patterns,
// pop its addresses and store them in own TL1 for the host to compare with the loop model. The kernel issues no
// transfers, so results and the status line go straight through the uncached alias; the status is written last.

#include "cce_addrgen_abi.h"
#include "cce_test_common.h"
#include "dev_mem_map.h"
#include "internal/tt-2xx/quasar/overlay/addrgen_api.hpp"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"

namespace abi = cce_addrgen_abi;
using cce_test::own_tl1_words;
using cce_test::read_cycle_counter;
using overlay::ADDRGEN_0;
using overlay::CMDBUF_0;

static void setup_pattern(const abi::PatternSpec& pattern, bool src_strided, bool dest_strided) {
    overlay::reset_cmdbuf<CMDBUF_0>();
    overlay::reset_addrgen<ADDRGEN_0>();
    overlay::setup_src_base_start_addrgen<ADDRGEN_0>(pattern.src.base);
    if (src_strided) {
        overlay::setup_src_inner_loop_addrgen<ADDRGEN_0>(pattern.src.inner.stride, pattern.src.inner.end);
        overlay::setup_src_outer_loop_addrgen<ADDRGEN_0>(pattern.src.outer.stride, pattern.src.outer.end);
        overlay::setup_src_face_size_addrgen<ADDRGEN_0>(pattern.src.face_size);
    }
    overlay::setup_dest_base_start_addrgen<ADDRGEN_0>(pattern.dest.base);
    if (dest_strided) {
        overlay::setup_dest_inner_loop_addrgen<ADDRGEN_0>(pattern.dest.inner.stride, pattern.dest.inner.end);
        overlay::setup_dest_outer_loop_addrgen<ADDRGEN_0>(pattern.dest.outer.stride, pattern.dest.outer.end);
        overlay::setup_dest_face_size_addrgen<ADDRGEN_0>(pattern.dest.face_size);
    }
}

// Pops pass an explicit count of 1: the count-less pop encodes rs1 = x1 with xs1 set, so the hardware pops as many
// addresses as ra holds.
static void pop_one(volatile uint64_t* entry) {
    entry[abi::kResultSrc] = overlay::pop_src_addrgen<ADDRGEN_0>(1);
    entry[abi::kResultDest] = overlay::pop_dest_addrgen<ADDRGEN_0>(1);
}

static void pop_pattern(
    const abi::PatternSpec& pattern, bool src_strided, uint32_t num_addresses, volatile uint64_t* results) {
    if (pattern.patch_pops == 0) {
        for (uint32_t address = 0; address < num_addresses; ++address) {
            pop_one(results + address * abi::kResultEntryWords64);
        }
        return;
    }
    uint32_t address = 0;
    for (uint32_t patch_row = 0; patch_row < pattern.patch_row_count; ++patch_row) {
        for (uint32_t patch_column = 0; patch_column < pattern.patch_column_count; ++patch_column) {
            if (src_strided) {
                overlay::setup_src_base_start_addrgen<ADDRGEN_0>(
                    pattern.src.base + patch_row * pattern.patch_row_stride +
                    patch_column * pattern.patch_column_stride);
            }
            for (uint32_t pop = 0; pop < pattern.patch_pops && address < num_addresses; ++pop, ++address) {
                pop_one(results + address * abi::kResultEntryWords64);
            }
        }
    }
}

void kernel_main() {
    const uint32_t mode = get_arg_val<uint32_t>(abi::kArgMode);
    const uint32_t status_offset = get_arg_val<uint32_t>(abi::kArgStatusOffset);
    const uint32_t result_offset = get_arg_val<uint32_t>(abi::kArgResultOffset);
    const uint32_t pattern_index = get_arg_val<uint32_t>(abi::kArgPattern);
    const bool src_strided = get_arg_val<uint32_t>(abi::kArgSrcStrideEnable) != 0;
    const bool dest_strided = get_arg_val<uint32_t>(abi::kArgDestStrideEnable) != 0;
    const uint32_t num_addresses = get_arg_val<uint32_t>(abi::kArgNumAddresses);
    const bool args_valid = mode == abi::kModePattern && pattern_index < abi::kNumPatterns &&
                            num_addresses <= abi::kMaxAddresses && status_offset % abi::kLineBytes == 0 &&
                            result_offset % abi::kLineBytes == 0 && status_offset != result_offset;

    uint32_t code = abi::kCodeDone;
    uint32_t pop_cycles = 0;
    if (!args_valid) {
        code = abi::kCodeBadArgs;
    } else {
        const abi::PatternSpec& pattern = abi::kPatterns[pattern_index];
        setup_pattern(pattern, src_strided, dest_strided);
        const uint64_t pop_start = read_cycle_counter();
        pop_pattern(pattern, src_strided, num_addresses, own_tl1_words<uint64_t>(result_offset));
        pop_cycles = static_cast<uint32_t>(read_cycle_counter() - pop_start);
    }

    volatile uint32_t* status = own_tl1_words(status_offset);
    status[abi::kStatusCode] = code;
    status[abi::kStatusPopCycles] = pop_cycles;
    asm volatile("fence w, w" ::: "memory");
    status[abi::kStatusMagic] = abi::kDoneMagic;
}
