// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "internal/ethernet/eth_ptp.hpp"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock.hpp"

namespace eth_clock {
// The stream's block length in cycles, so each pass starts at a uniform phase against the blocks.
constexpr uint32_t kPadRange = 6;
constexpr uint32_t kPadSlots = 256;
// sampler_stream's arguments. The registers come in as data, so the compiler holds their addresses in registers instead
// of rebuilding them between blocks.
struct StreamArgs {
    eth_ptp::Reg<> refclk;
    eth_ptp::Reg<> wall;
    Calibration calibration;
    volatile uint32_t* slots;
    uint32_t mask;
    volatile uint32_t* head;
    uint32_t limit;
    bool stop_on_reject;
    const uint32_t* pads;
    volatile uint32_t* tail_word;
    uint32_t tail;

    // Sets up one calibration pass, which stores the first update it catches into `slot` and returns. Needs *head at
    // kSyncHeadStop, so the reload after that update ends the call, and tail_word on a word the model never reads.
    // Field by field: copying the structs goes through the stack at -Os.
    FORCE_INLINE void prepare_calibration_pass(uint32_t period, GapPos8 pos8, volatile tt_l1_ptr uint32_t* slot) {
        calibration.period = period;
        calibration.pos8.word = pos8.word;
        slots = slot;
        mask = 0;
        stop_on_reject = true;
        limit = tail + 1;
    }
    // Sets up sampling, which streams every update into `ring` until the model stops it, with no limit on rejected
    // passes.
    FORCE_INLINE void prepare_sampling(
        const Calibration& calibrated, volatile tt_l1_ptr kernel_profiler::SyncSampleRing* ring) {
        calibration.period = calibrated.period;
        calibration.pos8.word = calibrated.pos8.word;
        slots = &ring->samples[0].refclk;
        mask = kernel_profiler::kSyncSampleRingSamples - 1;
        stop_on_reject = false;
        tail_word = &ring->tail;
        tail = 0;
        limit = mask;
    }
};

// Catches refclk updates into args.slots from args.tail on, update n into slot n & args.mask, and returns the tail
// after the last one it stored.
//
// Each pass runs args.pads[tail % kPadSlots] / 4 nops, then reads blocks of refclk, refclk, wall, refclk until the
// refclk steps, so calibration and sampling passes start at the same spread of phases.
//
// With args.calibration.period nonzero, an update whose two blocks each took exactly period wall ticks is stored as its
// new refclk and the wall time of its step in eighths: the block's wall read plus its gap's position from
// args.calibration.pos8. Other updates are dropped. Each stored update also writes the tail before it to
// *args.tail_word, publishing every earlier update, so a published sample was stored a handler earlier.
//
// With args.calibration.period 0, for calibration, every update is stored as its block's wall length and nothing is
// published.
//
// At args.limit the stream reloads *args.head, the model's position, and moves the limit args.mask past it, waiting
// there while the ring is full. It returns when the head is kSyncHeadStop past the model's position, or, with
// args.stop_on_reject, after a pass that stored nothing.
//
// -Os lays the calibration path in line and merges the handlers' tails, and the extra taken branches, which cost more
// with the branch predictor off, made the stream miss every other update at 800 MHz.
__attribute__((noinline, aligned(64), optimize("no-crossjumping", "reorder-blocks-algorithm=stc"))) inline uint32_t
sampler_stream(const StreamArgs& args) {
    struct Block {
        uint32_t first_refclk, second_refclk, wall, last_refclk;
    };
    enum class Step : uint8_t { Stored, Refill, Rejected };
    const eth_ptp::Reg<> refclk = args.refclk, wall = args.wall;
    const uint32_t period = args.calibration.period, mask = args.mask;
    const GapPos8 pos8{args.calibration.pos8.word};
    volatile uint32_t* const slots = args.slots;
    volatile uint32_t* const head = args.head;
    uint32_t limit = args.limit;
    const bool stop_on_reject = args.stop_on_reject;
    const uint32_t* const pads = args.pads;
    volatile uint32_t* const tail_word = args.tail_word;
    uint32_t tail = args.tail;
    const auto read = [&]() __attribute__((always_inline)) {
        Block block;
        block.first_refclk = refclk.read();
        block.second_refclk = refclk.read();
        block.wall = wall.read();
        block.last_refclk = refclk.read();
        return block;
    };
    // The pad jumps pad bytes back from the end of its nops. The asm defines the label, so it must be emitted once.
    uint32_t nops_end;
    asm("lla %0, .Lsampler_stream_nops_end" : "=r"(nops_end));
    uint32_t pad = pads[tail & (kPadSlots - 1)];
    disable_branch_predictor();
    // The pass starts on a cache line, so its blocks sit where they do in every build. Passing pad through keeps its
    // load ahead of the alignment, so nothing lands between the alignment and the pass.
    asm volatile(".p2align 6" : "+r"(pad));
    // Odds this low keep a step at the last check a single taken branch into its handler.
    const auto stepped = [](const Block& before, const Block& last) __attribute__((always_inline)) {
        return __builtin_expect_with_probability(last.last_refclk != before.last_refclk, 0, 0.001);
    };
    // A step between before's and last's final reads is placed by last's reads, and now times the block after it.
    const auto store = [&](const Block& before, const Block& last, const Block& now)
                           __attribute__((always_inline)) -> Step {
        const uint32_t step_block_ticks = last.wall - before.wall, next_block_ticks = now.wall - last.wall;
        volatile uint32_t* const slot = slots + 2 * (tail & mask);
        if (__builtin_expect(period != 0, 1)) {
            if (((step_block_ticks ^ period) | (next_block_ticks ^ period)) != 0) {
                return Step::Rejected;
            }
            const uint32_t gap = static_cast<uint32_t>(last.first_refclk == before.last_refclk)
                                 << (last.second_refclk == last.first_refclk);
            slot[0] = last.last_refclk;
            slot[1] = (last.wall << 3) + static_cast<uint32_t>(pos8.at(gap));
            *tail_word = tail;
            return ++tail == limit ? Step::Refill : Step::Stored;
        }
        slot[0] = next_block_ticks;
        return ++tail == limit ? Step::Refill : Step::Stored;
    };
    // Reads blocks until the refclk steps, and stores the step. Three blocks rotating by name keep each check's reads
    // in the registers they landed in; a single rotating site or an array makes the compiler copy them into one
    // handler's registers.
    const auto run = [&]() __attribute__((always_inline)) -> Step {
        Block a = read(), b = read(), c;
#pragma GCC unroll 6
        for (uint32_t i = 0; i < 6; i++) {
            c = read();
            if (stepped(a, b)) {
                return store(a, b, c);
            }
            a = read();
            if (stepped(b, c)) {
                return store(b, c, a);
            }
            b = read();
            if (stepped(c, a)) {
                return store(c, a, b);
            }
        }
        return Step::Rejected;
    };
pass:
    asm volatile(
        ".option push\n\t.option norvc\n\t"
        "sub t0, %[end], %[pad]\n\t"
        "jr t0\n\t"
        ".rept %[n]\n\tnop\n\t.endr\n"
        ".Lsampler_stream_nops_end:\n\t"
        ".option pop"
        :
        : [end] "r"(nops_end), [pad] "r"(pad), [n] "i"(kPadRange - 1)
        : "t0", "memory");
    pad = pads[tail & (kPadSlots - 1)];
    switch (run()) {
        case Step::Stored: goto pass;
        case Step::Refill: goto refill;
        case Step::Rejected: goto reject;
    }
reject:
    if (!stop_on_reject) {
        goto pass;
    }
    goto done;
refill:
    for (;;) {
        invalidate_l1_cache();
        const uint32_t model_head = *head;
        if (__builtin_expect(static_cast<int32_t>(tail - model_head) < 0, 0)) {
            goto done;
        }
        limit = model_head + mask;
        if (limit != tail) {
            break;
        }
    }
    pad = pads[tail & (kPadSlots - 1)];
    goto pass;
done:
    enable_branch_predictor();
    return tail;
}
}  // namespace eth_clock

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl_addr");
constexpr uint32_t kSampleRingAddr = get_named_compile_time_arg_val("sample_ring_addr");

namespace eth_ptp = tt::tt_metal::eth_ptp;

void kernel_main() {
    volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::ResidentCtrl*>(kCtrlAddr);
    volatile tt_l1_ptr kernel_profiler::SyncSampleRing* ring =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::SyncSampleRing*>(kSampleRingAddr);
    // The stream's pads in bytes, 4 per nop, uniform over kPadRange nops, drawn once so an update's pad is one load.
    static uint32_t pads[eth_clock::kPadSlots];
    uint32_t walk = eth_ptp::kWallClockLo.read() | 1u;
    for (uint32_t& pad : pads) {
        pad = 4u * eth_clock::draw(walk, eth_clock::kPadRange);
    }
    uint32_t unpublished = 0;
    eth_clock::StreamArgs args{
        .refclk = eth_ptp::kRefclkLo,
        .wall = eth_ptp::kWallClockLo,
        .head = &ring->head,
        .pads = pads,
        .tail_word = &unpublished};
    ring->head = kernel_profiler::kSyncHeadStop;
    const eth_clock::Calibration calibration = eth_clock::calibrate(
        [&](uint32_t period, eth_clock::GapPos8 pos8, volatile tt_l1_ptr uint32_t* slot) {
            const uint32_t from = args.tail;
            args.prepare_calibration_pass(period, pos8, slot);
            args.tail = sampler_stream(args);
            return args.tail - from;
        },
        ctrl);
    if (ctrl->stop == 0u) {
        const eth_ptp::Instant start = eth_ptp::read_instant();
        ring->refclk = start.refclk;
        ring->wall8 = start.wall << 3;
        ring->head = 0;
        args.prepare_sampling(calibration, ring);
        const uint32_t tail = sampler_stream(args);
        std::atomic_thread_fence(std::memory_order_release);
        ring->tail = tail;
    }
    std::atomic_thread_fence(std::memory_order_release);
    ring->done = 1;
}
