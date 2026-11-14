// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Hy4-Preview skinny linear TRISC (8 DRAM-bank cores): out[:, PNC b + j] = x [K = KT tiles] @ W[:, PNC b + j] for j < PNC, bank-streamed bfp8 weights,
// custom_mm (tiny in0 tiles [R, 32], LoFi, fp32 dest accumulation).
// CT args: 0 cb_x, 1 cb_w, 2 cb_o, 3 KT, 4 fp32, 5 SBK (K tiles per weight column wait; = KT), 6 PNC, 7 SBW, 8 MT

#include <cstdint>
#include "hsk_mm.hpp"
#include "api/compute/compute_kernel_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_o = get_compile_time_arg_val(2);
    constexpr uint32_t KT = get_compile_time_arg_val(3);
    constexpr bool fp32 = get_compile_time_arg_val(4) != 0;
    constexpr uint32_t SBK = get_compile_time_arg_val(5);
    constexpr uint32_t PNC = get_compile_time_arg_val(6);
    constexpr uint32_t SBW = get_compile_time_arg_val(7);
    constexpr uint32_t MT = get_compile_time_arg_val(8);

    deepseek_compute_kernel_init();
    cb_wait_front(cb_x, KT * MT);
    {
        DeviceZoneScopedN("T_MM");
        sk_mm<cb_x, cb_w, cb_o, SBK, PNC, SBW, KT / SBK, MT, fp32>(0, KT);
    }
    cb_pop_front(cb_x, KT * MT);
}
