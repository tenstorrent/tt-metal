// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "internal/risc_attribs.h"

extern "C" void wzerorange(uint32_t* start, uint32_t* end);
void l1_to_local_mem_copy(uint32_t* dst, uint32_t tt_l1_ptr* src, L1WordCount len);

inline void do_crt1(uint32_t tt_l1_ptr* data_image) {
    // Clear bss.
    extern uint32_t __ldm_bss_start[];
    extern uint32_t __ldm_bss_end[];
    wzerorange(__ldm_bss_start, __ldm_bss_end);

    // Copy initialized data.
    extern uint32_t __ldm_data_start[];
    extern uint32_t __ldm_data_end[];
    if (__ldm_data_start != data_image) {
        l1_to_local_mem_copy(__ldm_data_start, data_image, l1_word_count_from_range(__ldm_data_start, __ldm_data_end));
    }
}

inline void do_thread_crt1(uint32_t tt_l1_ptr* data_image) {
    // Clear thread bss.
    extern thread_local uint32_t __ldm_tbss_start[];
    extern thread_local uint32_t __ldm_tbss_end[];
    wzerorange(__ldm_tbss_start, __ldm_tbss_end);

    // Copy thread initialized data.
    extern thread_local uint32_t __ldm_tdata_start[];
    extern thread_local uint32_t __ldm_tdata_end[];
    l1_to_local_mem_copy(__ldm_tdata_start, data_image, l1_word_count_from_range(__ldm_tdata_start, __ldm_tdata_end));
}
