// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reproducer: ckernel::read_tile_value / ckernel::get_tile_address (api/compute/cb_api.h) broadcast the
// UNPACK thread's L1 read only to MATH and PACK. On Quasar the isolated-SFPU TRISC also runs kernel_main
// and gets the function's initial value (0).
//
// Drop-in replacement for dfb_read_tile_value_compute.cpp (same args, same 7-word-per-thread result layout),
// so the existing DataflowBufferReadTileValue gtest can drive it:
//   results[0..3] = ckernel::read_tile_value(cb, {0,0}, {0,1}, {1,0}, {1,1})   <- API under test
//   results[4]    = *ckernel::get_tile_address(cb, 1)                           <- API under test
//   results[5..6] = DataflowBuffer::read_tile_value<uint16_t>(1, {0,1})         <- control (DFB API)

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"  // dummy_unpack (TEN-4746)
#include "api/dataflow/dataflow_buffer.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

#include <cstdint>

void kernel_main() {
    constexpr std::uint32_t num_entries_per_consumer = get_arg(args::num_entries_per_consumer);
    const std::uint32_t result_l1_addr = get_arg(args::result_l1_addr);

    DataflowBuffer dfb(dfb::in);
    const std::uint32_t cb_id = dfb.get_id();
    compute_kernel_hw_startup(cb_id, cb_id);

    constexpr std::uint32_t k_num_results = 7;
    std::uint32_t results[k_num_results] = {};

    dfb.wait_front(num_entries_per_consumer);

    // On craq-sim the first L1 reads right after wait_front can still return 0 (the unmodified
    // dfb_read_tile_value_compute.cpp fails the same way on all 4 threads). Spin, using the DFB API
    // (broadcast to all 4 TRISCs, so every thread takes the same trip count), until the known nonzero
    // words of both tiles are visible, so the only remaining difference is the API under test.
    for (std::uint32_t spin = 0; spin < (1u << 20); ++spin) {
        if (dfb.read_tile_value<std::uint32_t>(0, 1) != 0 && dfb.read_tile_value<std::uint32_t>(1, 1) != 0) {
            break;
        }
    }

    // API under test: ckernel:: (cb_api.h) mailbox broadcast.
    results[0] = ckernel::read_tile_value(cb_id, 0, 0);
    results[1] = ckernel::read_tile_value(cb_id, 0, 1);
    results[2] = ckernel::read_tile_value(cb_id, 1, 0);
    results[3] = ckernel::read_tile_value(cb_id, 1, 1);
    const std::uint32_t tile_addr = ckernel::get_tile_address(cb_id, 1);
    // Do not dereference a bogus 0 address on the thread that did not receive it.
    results[4] = tile_addr != 0 ? *reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(tile_addr) : 0u;

    // Control: DataflowBuffer API also broadcasts to IsolateSfpuThreadId.
    results[5] = static_cast<std::uint32_t>(dfb.read_tile_value<std::uint16_t>(1, 0));
    results[6] = static_cast<std::uint32_t>(dfb.read_tile_value<std::uint16_t>(1, 1));

    for (std::uint32_t i = 0; i < num_entries_per_consumer; ++i) {
        dummy_unpack(cb_id);
        dfb.pop_front(1);
    }

#if defined(TRISC_UNPACK)
    constexpr std::uint32_t result_slot = 0;
#elif defined(TRISC_MATH)
    constexpr std::uint32_t result_slot = 1;
#elif defined(TRISC_PACK)
    constexpr std::uint32_t result_slot = 2;
#elif defined(TRISC_ISOLATE_SFPU)
    constexpr std::uint32_t result_slot = 3;
#endif
#if defined(TRISC_UNPACK) || defined(TRISC_MATH) || defined(TRISC_PACK) || defined(TRISC_ISOLATE_SFPU)
#ifdef ARCH_QUASAR
    const std::uint32_t result_l1_ptr_addr = result_l1_addr + MEM_L1_UNCACHED_BASE;
#else
    const std::uint32_t result_l1_ptr_addr = result_l1_addr;
#endif
    volatile tt_l1_ptr std::uint32_t* const out =
        reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(result_l1_ptr_addr);
    const std::uint32_t slot_base = result_slot * k_num_results;
    for (std::uint32_t i = 0; i < k_num_results; ++i) {
        out[slot_base + i] = results[i];
    }
#endif

    dfb.finish();
}
