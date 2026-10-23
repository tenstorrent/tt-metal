// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Runs on ERISC0 of the chip's wall-clock tile. That idle eth tile measures the chip's wall clock against its refclk
// for the clock sync between chips. This kernel catches nearly every refclk update and writes the new refclk count and
// the wall clock at that moment to a ring in L1, which eth_clock_model.cpp reads on ERISC1.

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "internal/ethernet/eth_ptp.hpp"
#include "internal/firmware_common.h"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock.hpp"

namespace eth_clock {
// Each pass starts after a pad, a random delay of 0 to kPadRange - 1 nops. kPadRange is a block's length in cycles, so
// passes start at a uniform phase relative to the blocks.
constexpr uint32_t kPadRange = 6;
constexpr uint32_t kPadSlots = 256;
// Each pass's random pad, stored as the nops() target that runs that many nops. They are drawn once up front, so a
// pass gets its pad with a single load.
uint32_t pads[kPadSlots];

// Samples refclk changes from sample number `tail` on, sample n in slot n & mask, and returns the sample number after
// the last one stored. It publishes the number of finished samples to *tail_word, and rereads the model's *head when
// the tail reaches `limit`. Each pass first waits a random number of nops, so passes start at random phases of the
// refclk, then reads blocks of refclk, refclk, wall clock, refclk until the refclk changes. A change is kept only if
// its block and the next took exactly the calibrated period, so a stalled read can't skew it. Calibration (period 0)
// stores the block lengths instead.
//
// The optimize attribute keeps each change handler short enough to catch the next update at low AICLK. It stops GCC
// from merging the handlers' tails, which adds a taken branch to each change, and from hoisting the tail's increment
// above the period check, which adds a register copy.
__attribute__((noinline, aligned(64), optimize("no-crossjumping", "no-code-hoisting"))) inline uint32_t sampler_stream(
    Calibration calibration,
    volatile uint32_t* slots,
    uint32_t mask,
    volatile uint32_t* head,
    uint32_t limit,
    volatile uint32_t* tail_word,
    uint32_t tail) {
    constexpr eth_ptp::Reg<> refclk = eth_ptp::kRefclkLo, wall = eth_ptp::kWallClockLo;
    const uint32_t period = calibration.period;
    // This word holds each gap's correction in one byte, so the store picks its gap's correction with a shift instead
    // of a branch. Shifting by 0, 8 or 16 brings gap 2's, gap 0's or gap 1's correction to the low byte.
    const uint32_t gap_word = calibration.gap_positions.word;
    const uint32_t positions = ((gap_word << 8) | (gap_word >> 16)) & 0xFFFFFFu;
    const auto read = [&]() __attribute__((always_inline)) {
        Block block;
        block.first_refclk = refclk.read();
        block.second_refclk = refclk.read();
        block.wall = wall.read();
        block.last_refclk = refclk.read();
        return block;
    };
    uint32_t pad = pads[tail & (kPadSlots - 1)];
    // The pass starts on a cache line so its blocks have the same layout in every build, because layout alone can shift
    // the sampled times by about a tick. Passing pad through the asm keeps its load before the alignment.
    asm volatile(".p2align 6" : "+r"(pad));
    // Stores a change that happened between the final refclk reads of blocks `before` and `last`. The change is timed
    // with `last`'s reads, and `now`, the block after `last`, measures how long that block took. Returns whether the
    // stream must wait for the model or stop.
    const auto store = [&](const Block& before, const Block& last, const Block& now)
                           __attribute__((always_inline)) -> bool {
        volatile uint32_t* const slot = slots + 2 * (tail & mask);
        // A block takes at least one tick, so in calibration (period 0) the period check always fails, and calibration
        // stores the next block's length from that cold path.
        if (last.wall - before.wall == period && now.wall - last.wall == period) {
            // Exactly one gap holds the change: before this block's first refclk read (gap 0), between its first and
            // second (gap 1), or between its second and last (gap 2). Each refclk difference is 0 or 4, so
            // 2 * (first - previous last) + 4 * (second - first) is 8, 16 or 0: the shift to that gap's byte.
            static_assert(kernel_profiler::kRefclkTicksPerUpdate == 4);
            const uint32_t shift =
                2 * (last.first_refclk - before.last_refclk) + 4 * (last.second_refclk - last.first_refclk);
            slot[0] = last.last_refclk;
            slot[1] = (last.wall << kernel_profiler::kWallEighthBits) +
                      static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(positions >> shift)));
            // The published tail excludes the slot just written, so the model reads finished slots without a fence.
            *tail_word = tail;
            if (++tail != limit) {
                return false;
            }
            // The tail reached the limit set from the last head read, so reread the head here.
            const uint32_t model_head = *head;
            if (tail - model_head >= mask) {
                return true;
            }
            limit = model_head + mask;
            return false;
        }
        if (period == 0) {
            slot[0] = now.wall - last.wall;
            return ++tail == limit;
        }
        return false;
    };
    // Reads blocks until the refclk steps, then stores the step and returns store()'s result. A pass that ends without
    // a step returns false. Rotating three named blocks keeps each check's reads in the registers they were loaded
    // into. A single rotating variable or an array makes the compiler copy them into one handler's registers.
    const auto run = [&]() __attribute__((always_inline)) -> bool {
        Block a = read(), b = read(), c;
#pragma GCC unroll 6
        for (uint32_t i = 0; i < 6; i++) {
            c = read();
            if (b.last_refclk != a.last_refclk) {
                return store(a, b, c);
            }
            a = read();
            if (c.last_refclk != b.last_refclk) {
                return store(b, c, a);
            }
            b = read();
            if (a.last_refclk != c.last_refclk) {
                return store(c, a, b);
            }
        }
        return false;
    };
    while (true) {
        // The pad table holds nop addresses, so a pass starts with a single jump.
        nops<kPadRange - 1>(pad);
        pad = pads[tail & (kPadSlots - 1)];
        if (run()) {
            // The ring is full or the model is stopping the stream.
            uint32_t model_head;
            do {
                model_head = *head;
                if (static_cast<int32_t>(tail - model_head) < 0) {
                    return tail;
                }
            } while (tail - model_head >= mask);
            limit = model_head + mask;
        }
    }
}
}  // namespace eth_clock

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl");
constexpr uint32_t kSampleRingAddr = get_named_compile_time_arg_val("sample_ring");

void kernel_main() {
    // The kernel runs with branch prediction and the L0 data cache off. With the cache off, the stream rereads the
    // model's head straight from L1, so it needs no fence. A fence costs about 16 cycles here, because the store queue
    // holds its last store that long in case another store merges with it.
    eth_clock::disable_branch_predictor();
    set_l1_data_cache<false>();
    volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::ResidentCtrl*>(kCtrlAddr);
    volatile tt_l1_ptr kernel_profiler::SyncSampleRing* ring =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::SyncSampleRing*>(kSampleRingAddr);
    uint32_t random_state = eth_ptp::kWallClockLo.read() | 1u;
    for (uint32_t& pad : eth_clock::pads) {
        pad = reinterpret_cast<uint32_t>(eth_clock::eth_clock_nops_end) -
              4u * eth_clock::draw(random_state, eth_clock::kPadRange);
    }
    // Calibration runs one pass at a time into `slot`. The head is kSyncHeadStop, so the stream returns after one
    // update, and the tail goes to a word the model never reads.
    uint32_t calibration_tail = 0, unpublished = 0;
    ring->head = kernel_profiler::kSyncHeadStop;
    const eth_clock::Calibration calibration = eth_clock::calibrate(
        [&](uint32_t period, eth_clock::GapPositions gap_positions, volatile tt_l1_ptr uint32_t* slot) {
            const uint32_t from = calibration_tail;
            calibration_tail =
                eth_clock::sampler_stream({period, gap_positions}, slot, 0, &ring->head, from + 1, &unpublished, from);
            return calibration_tail - from;
        },
        ctrl);
    if (ctrl->stop == 0u) {
        ring->wall_eighths = eth_ptp::read_instant().wall << kernel_profiler::kWallEighthBits;
        ring->head = 0;
        constexpr uint32_t kMask = kernel_profiler::kSyncSampleRingSamples - 1;
        const uint32_t tail =
            eth_clock::sampler_stream(calibration, &ring->samples[0].refclk, kMask, &ring->head, kMask, &ring->tail, 0);
        std::atomic_thread_fence(std::memory_order_release);
        ring->tail = tail;
    }
    std::atomic_thread_fence(std::memory_order_release);
    ring->done = 1;
    configure_l1_data_cache();
    eth_clock::enable_branch_predictor();
}
