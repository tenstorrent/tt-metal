// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "tt_metal/tools/profiler/kernel_profiler.hpp"

/**
 * LOOP_COUNT and LOOP_SIZE provide the ability to decide how many cycles this kernel takes.
 * With a large enough LOOP_COUNT and a LOOP_SIZEs within icache size, cycle count will be
 * very close to LOOP_COUNT x (LOOP_SIZE + loop_overhead). loop_overhead is 2 cycle 1 for
 * addi and 1 for branch if not zero.
 *
 * Keeping LOOP_SIZE constant and suitable for all 5 risc ichahes, The diff between to runs
 * with LOOP_COUNT and LOOP_COUNT + 1 should be the same across all riscs and it should be
 * LOOP_COUNT + 2 cycles
 *
 * More info on tt-metal issue #515
 *
 * https://github.com/tenstorrent/tt-metal/issues/515#issuecomment-1548434301
 */

void kernel_main() {
    kernel_profiler::debug_timestamper_configure(
        /*buf0_start_word=*/0x8000 / 16,
        /*buf0_end_word=*/0x8100 / 16,
        /*buf1_start_word=*/0x8100 / 16,
        /*buf1_end_word=*/0x8200 / 16,
        /*enable_buf0=*/true,
        /*enable_buf1=*/false,
        /*reset_streams=*/true);

    constexpr uint32_t kInnerLoopStartToken = 0x000001;
    constexpr uint32_t kInnerLoopEndToken = 0x000002;

    for (int i = 0; i < 2; i++) {
// Max unroll size
#pragma GCC unroll 65534
        for (int j = 0; j < 5; j++) {
            {
                DeviceZoneScopedN("InnerLoopStart");
                kernel_profiler::debug_timestamper_append(
                    kInnerLoopStartToken, kernel_profiler::DebugTimestampEventSize::Bits64);
            }
            asm("nop");
            {
                DeviceZoneScopedN("InnerLoopStop");
                kernel_profiler::debug_timestamper_append(
                    kInnerLoopEndToken, kernel_profiler::DebugTimestampEventSize::Bits64);
            }
        }
    }

    {
        DeviceZoneScopedN("InnerLoopFlush");
        kernel_profiler::debug_timestamper_flush(kernel_profiler::DebugTimestampEventSize::Bits64);
    }
}
