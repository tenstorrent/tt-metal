// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstring>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

inline __attribute__((always_inline)) void fill_pad_dfb_with_val(
    const DataflowBuffer& dfb, const uint32_t num_bytes, const uint32_t val) {
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb.get_write_ptr());
    for (uint32_t i = 0; i < num_bytes / sizeof(uint32_t); ++i) {
        ptr[i] = val;
    }
}

void kernel_main() {
    uint32_t num_sticks_per_core = get_arg(args::num_sticks_per_core);
    uint32_t num_sticks_per_barrier = get_arg(args::num_sticks_per_barrier);
    uint32_t start_id = get_arg(args::start_id);
    uint32_t front_pad_n = get_arg(args::front_pad_n);
    uint32_t front_pad_c = get_arg(args::front_pad_c);
    uint32_t front_pad_h = get_arg(args::front_pad_h);

    constexpr uint32_t N = get_arg(args::N);
    constexpr uint32_t H = get_arg(args::H);
    constexpr uint32_t C = get_arg(args::C);
    constexpr uint32_t stick_size_bytes = get_arg(args::stick_size_bytes);
    constexpr uint32_t N_padded = get_arg(args::N_padded);
    constexpr uint32_t H_padded = get_arg(args::H_padded);
    constexpr uint32_t C_padded = get_arg(args::C_padded);
    constexpr uint32_t stick_size_padded = get_arg(args::stick_size_padded);
    constexpr uint32_t stick_size_padded_front = get_arg(args::stick_size_padded_front);
    constexpr uint32_t stick_size_padded_end = get_arg(args::stick_size_padded_end);
    constexpr uint32_t num_zero_pad_sticks_read = get_arg(args::num_zero_pad_sticks_read);
    constexpr uint32_t last_zero_stick_size = get_arg(args::last_zero_stick_size);
    constexpr uint32_t stick_size_padded_aligned = get_arg(args::stick_size_padded_aligned);

    constexpr bool not_pad_by_zero = get_arg(args::not_pad_by_zero) == 1;
    constexpr uint32_t front_padding = get_arg(args::stick_size_padded_front);
    constexpr bool unaligned = get_arg(args::unaligned) == 1;

    uint32_t packed_pad_value = 0;
    uint32_t row_major_min_bytes = 0;
    uint32_t num_front_pad_sticks_read = 0;
    uint32_t num_end_pad_sticks_read = 0;
    uint32_t num_sticks_padded_read = 0;
    if constexpr (not_pad_by_zero) {
        packed_pad_value = get_arg(args::packed_pad_value);
        row_major_min_bytes = get_arg(args::row_major_min_bytes);
        num_front_pad_sticks_read = get_arg(args::num_front_pad_sticks_read);
        num_end_pad_sticks_read = get_arg(args::num_end_pad_sticks_read);
        num_sticks_padded_read = get_arg(args::num_sticks_padded_read);
    }

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_pad(dfb::pad);
    DataflowBuffer dfb_pad_align(dfb::pad_align);

    const auto s = TensorAccessor(tensor::src);

    // Prepare pad pattern
    fill_pad_dfb_with_val(dfb_pad, stick_size_padded, packed_pad_value);
    uint64_t pad_val_addr = dfb_pad.get_write_ptr();
    uint64_t pad_val_noc_addr = get_noc_addr(pad_val_addr);

    uint64_t pad_align_addr = dfb_pad_align.get_read_ptr();
    uint64_t pad_align_noc_addr = get_noc_addr(pad_align_addr);

    uint32_t i_stick = start_id;
    uint32_t curr_c = 0, curr_h = 0, curr_n = 0;
    for (uint32_t iter = 0; iter < num_sticks_per_core;) {
        dfb_in0.reserve_back(num_sticks_per_barrier);
        uint32_t l1_write_addr = dfb_in0.get_write_ptr();
        for (uint32_t i = 0; i < num_sticks_per_barrier && iter < num_sticks_per_core; ++i, ++iter) {
            bool read_stick = (curr_h >= front_pad_h and curr_h < H) and (curr_c >= front_pad_c and curr_c < C) and
                              (curr_n >= front_pad_n and curr_n < N);
            uint64_t read_noc_addr = s.get_noc_addr(i_stick);
            // Seed pad value in first word to guarantee padding when writer writes only stick_size_bytes
            *((volatile tt_l1_ptr uint32_t*)l1_write_addr) = packed_pad_value;
            if (read_stick) {
                if constexpr (front_padding) {
                    noc_async_read(read_noc_addr, dfb_pad_align.get_write_ptr(), stick_size_bytes);
                    noc_async_read_barrier();
                    memmove(
                        (void*)(l1_write_addr + stick_size_padded_front),
                        (void*)(dfb_pad_align.get_read_ptr()),
                        (size_t)(stick_size_bytes));
                } else if constexpr (unaligned) {
                    noc_async_read(read_noc_addr, dfb_pad_align.get_write_ptr(), stick_size_bytes);
                    noc_async_read_barrier();
                    noc_async_read(pad_align_noc_addr, l1_write_addr, stick_size_bytes);
                    // Drain this loop-back read out of dfb_pad_align before the next stick's read-in
                    // overwrites that shared scratch, else the read-in can land while this read is still
                    // sourcing from it (WAR on dfb_pad_align).
                    noc_async_read_barrier();
                } else {
                    noc_async_read(read_noc_addr, l1_write_addr, stick_size_bytes);
                }
                i_stick++;
            }
            l1_write_addr += stick_size_padded_aligned;
            curr_h++;
            if (curr_h == H_padded) {
                curr_c++;
                curr_h = 0;
                if (curr_c == C_padded) {
                    curr_n++;
                    curr_c = 0;
                }
            }
        }
        noc_async_read_barrier();
        dfb_in0.push_back(num_sticks_per_barrier);
    }
}
