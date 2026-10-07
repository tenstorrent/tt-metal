// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute side of the scratch CB test (experimental/scratch_cb_api.h). `pattern` selects:
//
//   A  datacopy: UNPACK waits on channel 0 and copies ring A slot (i % capacity) into DST, then PACK reserves
//      channel 1, packs DST into ring B slot (i % capacity) and publishes it. Both channels are live at once.
//      Data access uses the id-free 2.0 LLKOperand API. `nosync`=1 drops wait/reserve (negative control).
//   B  bounded producer: UNPACK is a deliberately slow consumer of channel 0, one page at a time.
//   C  ping-pong: the same copy as A at capacity 1; the DM side sends every result back as the next input.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/pack.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"
#include "experimental/kernel_args.h"
#include "experimental/scratch_cb_api.h"

namespace {

constexpr std::uint32_t PATTERN_A = 0;
constexpr std::uint32_t PATTERN_B = 1;
constexpr std::uint32_t PATTERN_C = 2;

// Pattern B: RISC busy-loop per consumed page, so the producer is always ahead of the consumer.
constexpr std::uint32_t kSlowConsumerSpins = 2000;

// LLKOperand addresses are 16B words with the tile-header bias, exactly what cb_read_address yields.
inline std::uint32_t operand_addr(std::uint32_t byte_addr) { return (byte_addr >> 4) - 1u; }

inline void spin(std::uint32_t iterations) {
    for (std::uint32_t i = 0; i < iterations; ++i) {
        asm volatile("nop");
    }
}

}  // namespace

void kernel_main() {
    constexpr std::uint32_t pattern = get_arg(args::pattern);
    constexpr std::uint32_t capacity = get_arg(args::capacity);
    constexpr std::uint32_t nosync = get_arg(args::nosync);
    const std::uint32_t num_iters = get_arg(args::num_iters);
    const std::uint32_t tile_bytes = get_arg(args::tile_bytes);
    const std::uint32_t ring_a_addr = get_arg(args::ring_a_addr);
    const std::uint32_t ring_b_addr = get_arg(args::ring_b_addr);

    if constexpr (pattern == PATTERN_A || pattern == PATTERN_C) {
        using Op = ckernel::experimental::LLKOperand<DataFormat::Float16_b, ckernel::DEFAULT_TENSOR_SHAPE>;
        const Op a(operand_addr(ring_a_addr));
        const Op b(operand_addr(ring_b_addr));

        compute_kernel_hw_startup(a, b);
        ckernel::experimental::copy_init(a);

        for (std::uint32_t i = 0; i < num_iters; ++i) {
            const std::uint32_t slot = i % capacity;
            const std::uint32_t a_slot_addr = ring_a_addr + slot * tile_bytes;
            const std::uint32_t b_slot_addr = ring_b_addr + slot * tile_bytes;

            tile_regs_acquire();
            if constexpr (!nosync) {
                ::experimental::scratch_wait_front<0, capacity>(1, a_slot_addr);
            }
            ckernel::experimental::copy_tile(a, slot, 0);
            ::experimental::scratch_pop_front<0, capacity>(1, a_slot_addr);
            tile_regs_commit();

            tile_regs_wait();
            if constexpr (!nosync) {
                ::experimental::scratch_reserve_back<1, capacity>(1, b_slot_addr);
            }
            ckernel::experimental::pack_tile(b, slot, 0);
            ::experimental::scratch_push_back<1, capacity>(1, b_slot_addr);
            tile_regs_release();
        }
    }

    if constexpr (pattern == PATTERN_B) {
        for (std::uint32_t i = 0; i < num_iters; ++i) {
            ::experimental::scratch_wait_front<0, capacity>(1);
            UNPACK((spin(kSlowConsumerSpins)));
            ::experimental::scratch_pop_front<0, capacity>(1);
        }
    }
}
