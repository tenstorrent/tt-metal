// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel for the compute semaphore (SemScope::COMPUTE_ATOMIC): the Tensix hardware (Sync Unit)
// semaphore, index UNPACK_OPERAND_SYNC on Blackhole and PACK_UNPACK on Quasar. One kernel, patterns
// selected by the compile-time arg `pattern`:
//
//   A  single-thread up/down/wait/wait_min/set/value self-check (run once per thread via thread_sel)
//   D  two-hop datacopy through a kDepth-slot L1 ring `mid`: in[i] -> mid[slot] (PACK wait_not_full(), up()),
//      then mid[slot] -> out[i] (UNPACK wait_min() before the read, down() after). The semaphore alone orders
//      PACK's write of a slot before UNPACK's read. Blackhole: LLKOperand on host-written L1, no DM kernels.
//      Quasar (PATTERN_D_DFB): DFBs with DM reader/writer; `mid` is never pushed or popped. Host checks
//      out == in bit for bit. `nosync`=1 drops the waits and down (and up on Quasar): must corrupt.
//   E  producer back-pressure: PACK produces num_iters credits in up(batch)s, each gated by
//      wait_not_full(batch); UNPACK is a deliberately slow consumer that records the highest value it ever
//      observes. With the host's max_value = kDepth that high-water mark stays <= kDepth (batch > 1 takes
//      the RISC-poll form of wait_not_full). `nosync`=1 drops the wait_not_full() (negative control: PACK
//      runs to the 15 ceiling on Blackhole, posts are lost, UNPACK times out). On Quasar, SEMPOST
//      back-pressure keeps the value at the programmed maximum and every credit must survive.
//
// The hardware semaphore is 4-bit (0..15) and starts every kernel at 0: compute_kernel_hw_startup()
// seeds it on PACK (pattern D), and every pattern leaves it balanced at 0 for the next program. The
// patterns that do not call hw_startup (A/E) seed it with set(0) on their PRODUCING thread, which
// is safe for the same reason the hw_startup seed is: the producer's first up() is queued behind its
// own SEMINIT, and the consumer reads 0 either way. A/E waits are bounded so a lost update cannot
// hang the part; D uses the API's own wait_min (the point of D is the shipped primitives on a real op).

#include <cstdint>

#include "api/compute/common.h"
// TODO @RT: drop the Quasar includes and use the 2.0 ones after the Quasar compute API has been ported to Metal 2.0.
#ifdef ARCH_QUASAR
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#else
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/pack.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"
#endif
#include "api/semaphore.h"
#include "experimental/kernel_args.h"

namespace {

constexpr std::uint32_t PATTERN_A = 0;
// 1 (cross-thread counter ping-pong) and 2 (raw-LLK packer publish) retired: D covers publish-after-data
// through the Compute API with a no-sync negative control, E covers cross-thread credits and back-pressure.
constexpr std::uint32_t PATTERN_D = 3;
constexpr std::uint32_t PATTERN_E = 4;

constexpr std::uint32_t SEL_UNPACK = 0;
constexpr std::uint32_t SEL_PACK = 2;

constexpr std::uint32_t kPollCap = 5000000;

// Pattern D/E bounded-buffer depth. Well under the 15 hardware max so a few in-flight (uncommitted)
// posts cannot saturate the semaphore and drop an update. Patterns D and E also pass it to the host as
// the semaphore's max_value (capacity).
constexpr std::uint32_t kDepth = 4;

// Pattern E: RISC busy-loop per consumed credit, so the producer is always ahead of the consumer.
constexpr std::uint32_t kSlowConsumerSpins = 2000;

// Quasar pattern D: RISC busy-loop before each ring pack, so the packer lags the unpacker.
constexpr std::uint32_t kPackSkewSpins = 2000;

// Pattern D: full 32x32 Float16_b tiles (2 KB), host-written. Regions are fixed offsets from report_addr,
// mirrored on the host: in[kDMaxTiles], mid[kDepth] ring, out[kDMaxTiles]. 256 KB above the report so
// they are clear of the kernel-config region.
constexpr std::uint32_t kDMaxTiles = 8;
constexpr std::uint32_t kDTileBytes = 32 * 32 * 2;
constexpr std::uint32_t kDInOffset = 0x40000;
constexpr std::uint32_t kDMidOffset = kDInOffset + kDMaxTiles * kDTileBytes;
constexpr std::uint32_t kDOutOffset = kDMidOffset + kDepth * kDTileBytes;

// LLKOperand addresses are 16B words with the tile-header bias, exactly what cb_read_address yields.
inline std::uint32_t operand_addr(std::uint32_t byte_addr) { return (byte_addr >> 4) - 1u; }

inline void store_u32(std::uint32_t addr, std::uint32_t v) {
#ifdef ARCH_QUASAR
    addr += MEM_L1_UNCACHED_BASE;  // UNPACK and PACK share report lines; bypass the TRISC cache
#endif
    *reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(addr) = v;
}

inline std::uint32_t load_u32(std::uint32_t addr) {
#ifdef ARCH_QUASAR
    addr += MEM_L1_UNCACHED_BASE;
#else
    invalidate_l1_cache();  // L0 is not coherent with the other thread's L1 writes
#endif
    return *reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(addr);
}

}  // namespace

void kernel_main() {
    constexpr std::uint32_t pattern = get_arg(args::pattern);
    constexpr std::uint32_t thread_sel = get_arg(args::thread_sel);
    [[maybe_unused]] constexpr std::uint32_t nosync = get_arg(args::nosync);    // patterns D and E
    [[maybe_unused]] constexpr std::uint32_t batch = get_arg(args::batch);      // pattern E up() size
    [[maybe_unused]] const std::uint32_t num_iters = get_arg(args::num_iters);  // unused by pattern A
    const std::uint32_t report_addr = get_arg(args::report_addr);

    Semaphore sem(sem::sem);

    // -----------------------------------------------------------------------------------------------
    // PATTERN A: one thread exercises every primitive against a known sequence of values.
    // -----------------------------------------------------------------------------------------------
    if constexpr (pattern == PATTERN_A) {
        // Same body on either thread; thread_sel picks which one runs it. MATH has no semaphore methods.
#if defined(TRISC_UNPACK)
        constexpr std::uint32_t this_thread = SEL_UNPACK;
#elif defined(TRISC_PACK)
        constexpr std::uint32_t this_thread = SEL_PACK;
#endif
#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
        if constexpr (thread_sel == this_thread) {
            std::uint32_t fail = 0;
            sem.set(0);
            sem.up(5);
            sem.wait_min(5);  // poll until the 5 posts have landed
            const std::uint32_t v1 = sem.value();
            if (v1 != 5) {
                fail = 1;
            }
            sem.down(3);
            sem.wait(2);  // poll until the 3 gets have landed
            const std::uint32_t v2 = sem.value();
            if (v2 != 2) {
                fail = 2;
            }
            sem.up(1);
            sem.wait(3);
            const std::uint32_t v3 = sem.value();
            if (v3 != 3) {
                fail = 3;
            }
            sem.down(3);  // leave it balanced at 0 for the next program
            sem.wait(0);
            store_u32(report_addr + 0u, fail == 0 ? 1u : 0u);
            store_u32(report_addr + 4u, v1);
            store_u32(report_addr + 8u, v2);
            store_u32(report_addr + 12u, v3);
            store_u32(report_addr + 16u, fail);
        }
#endif
    }

    // -----------------------------------------------------------------------------------------------
    // PATTERN D: the real thing. Compute-only two-hop datacopy through a shared L1 ring `mid`, using the
    // 2.0 LLKOperand compute API; the semaphore plays the CB's role for `mid`:
    //   PACK   pack_tile(mid[slot]) ; up(1)          -- push: STALLWAIT(PACK) + SEMPOST
    //   UNPACK wait_min(1) ; copy_tile(mid[slot]) ; down(1)  -- wait_front / pop_front:
    //          SEMGET is blocked by STALLWAIT(UNPACK) until the unpacker has finished reading the slot.
    // The DST handshake (tile_regs_*) already orders round 2 after round 1 per tile; the semaphore is the
    // only thing that orders PACK's L1 writes before UNPACK's L1 reads of the same slot. With nosync=1,
    // UNPACK unpacks mid[slot] right after in[i], long before PACK has written it, so out[i] != in[i].
    // -----------------------------------------------------------------------------------------------
#ifndef ARCH_QUASAR
    if constexpr (pattern == PATTERN_D) {
        using Op = ckernel::experimental::LLKOperand<DataFormat::Float16_b, ckernel::DEFAULT_TENSOR_SHAPE>;
        const Op in(operand_addr(report_addr + kDInOffset));
        const Op mid(operand_addr(report_addr + kDMidOffset));
        const Op out(operand_addr(report_addr + kDOutOffset));

        compute_kernel_hw_startup(in, out);  // also seeds the compute semaphore to 0 (on PACK)
        ckernel::experimental::copy_init(in);

        for (std::uint32_t i = 0; i < num_iters; ++i) {
            const std::uint32_t slot = i % kDepth;

            // Round 1: in[i] -> DST[0] -> mid[slot], then PACK publishes the slot.
            tile_regs_acquire();
            ckernel::experimental::copy_tile(in, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (!nosync) {
                // Ring full (max_value = kDepth): hold the PACR until a slot frees. Dropped in the negative
                // control too -- with no consumer down() it would block forever instead of corrupting.
                PACK(sem.wait_not_full();)
            }
            ckernel::experimental::pack_tile(mid, slot, 0);
            PACK(sem.up(1);)
            tile_regs_release();

            // Round 2: mid[slot] -> DST[0] -> out[i]; UNPACK waits for the slot, reads it, releases it.
            tile_regs_acquire();
            if constexpr (!nosync) {
                UNPACK(sem.wait_min(1);)
            }
            ckernel::experimental::copy_tile(mid, slot, 0);
            if constexpr (!nosync) {
                UNPACK(sem.down(1);)
            }
            tile_regs_commit();
            tile_regs_wait();
            ckernel::experimental::pack_tile(out, i, 0);
            tile_regs_release();
        }

        PACK(store_u32(report_addr + 0u, num_iters);)
        UNPACK(store_u32(report_addr + 8u, sem.value());)
    }
#elif defined(PATTERN_D_DFB)
    // TODO @RT: delete this branch and run the LLKOperand path above on Quasar after the Quasar compute API has
    // been ported to Metal 2.0.
    // Quasar: LLKOperand is Blackhole-only, so `in`, `mid` and `out` are dataflow buffers and the Quasar
    // Compute API addresses them by id.
    //  - Seed: the id-based compute_kernel_hw_startup does not seed the semaphore; PACK set(0)s it.
    //  - Negative control: nosync=1 also drops PACK's up(). Blackhole posts saturate at 15, so its 8
    //    unconsumed posts are harmless; a Quasar post at Max (kDepth) stalls until a down() that never
    //    comes, so the 5th up() would hang the kernel instead of corrupting the output.
    //  - Skew: PACK spins kPackSkewSpins before each ring pack, in both runs, so the packer always lags
    //    the unpacker. Without it UNPACK is not reliably ahead, and the negative control could come out
    //    correct by timing alone.
    //  - Report words: store_u32/load_u32 go through the uncached L1 alias, since UNPACK and PACK write
    //    the same report lines and the TRISC data cache is not coherent between them.
    if constexpr (pattern == PATTERN_D) {
        DataflowBuffer dfb_in(dfb::in);
        DataflowBuffer dfb_out(dfb::out);
        const std::uint32_t in_id = dfb_in.get_id();
        const std::uint32_t mid_id = DataflowBuffer(dfb::mid).get_id();
        const std::uint32_t out_id = dfb_out.get_id();

        compute_kernel_hw_startup(in_id, out_id);
        PACK(sem.set(0);)

        for (std::uint32_t i = 0; i < num_iters; ++i) {
            const std::uint32_t slot = i % kDepth;

            // Round 1: in[i] -> DST[0] -> mid[slot], then PACK publishes the slot.
            dfb_in.wait_front(1);
            tile_regs_acquire();
            copy_init(in_id);
            copy_tile(in_id, /*tile_index=*/0, /*dst_index=*/0);
            tile_regs_commit();
            tile_regs_wait();
            if constexpr (!nosync) {
                PACK(sem.wait_not_full();)
            }
            PACK(for (volatile std::uint32_t d = 0; d < kPackSkewSpins; ++d){})
            pack_init(mid_id);
            pack_tile<true>(/*dst_index=*/0, mid_id, slot);
            if constexpr (!nosync) {
                PACK(sem.up(1);)
            }
            tile_regs_release();
            dfb_in.pop_front(1);

            // Round 2: mid[slot] -> DST[0] -> out[i]; UNPACK waits for the slot, reads it, releases it.
            dfb_out.reserve_back(1);
            tile_regs_acquire();
            if constexpr (!nosync) {
                UNPACK(sem.wait_min(1);)
            }
            copy_init(mid_id);
            copy_tile(mid_id, slot, /*dst_index=*/0);
            if constexpr (!nosync) {
                UNPACK(sem.down(1);)
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_init(out_id);
            pack_tile(/*dst_index=*/0, out_id);
            tile_regs_release();
            dfb_out.push_back(1);
        }

        PACK(store_u32(report_addr + 0u, num_iters);)
        UNPACK(store_u32(report_addr + 8u, sem.value());)
    }
#endif

    // -----------------------------------------------------------------------------------------------
    // PATTERN E: producer back-pressure. PACK: wait_not_full(batch); up(batch), num_iters/batch times. With
    // batch == 1 the SEMWAIT blocks no engine instruction here (pure counter), but the STALLWAIT inside up()
    // is held behind it, so the SEMPOST cannot execute while Value >= Max. UNPACK: slow consumer, polls value() (the
    // settled Sync Unit value) so the high-water mark it records is exact; with wait_not_full() in place it can never
    // exceed kDepth. Balanced, so the final value is 0 iff no post was lost.
    // Quasar: a SEMPOST at Max stalls until a SEMGET makes room (Blackhole saturates at 15 and drops the
    // post instead). So PACK can still be stuck inside up() when UNPACK's poll times out; if UNPACK stopped
    // there, as on Blackhole, nothing would ever down() and PACK would hang forever. UNPACK therefore keeps
    // draining past a timeout until PACK reports done (report word 5), and PACK publishes done only after
    // value() has retired all its posts, including any stalled at capacity.
    // -----------------------------------------------------------------------------------------------
    if constexpr (pattern == PATTERN_E) {
#ifdef TRISC_PACK
        sem.set(0);  // PACK produces: seed on the producer (Max = host max_value)
        for (std::uint32_t i = 0; i < num_iters; i += batch) {
            if constexpr (!nosync) {
                sem.wait_not_full(batch);
            }
            sem.up(batch);
        }
        store_u32(report_addr + 0u, num_iters);
#ifdef ARCH_QUASAR
        (void)sem.value();
        store_u32(report_addr + 20u, 1u);
#endif
#endif
#ifdef TRISC_UNPACK
        std::uint32_t consumed = 0;
        std::uint32_t high_water = 0;
        std::uint32_t timeout = 0;
        for (std::uint32_t i = 0; i < num_iters; ++i) {
            std::uint32_t spins = 0;
            std::uint32_t v;
            while ((v = sem.value()) < 1u) {
                if (++spins >= kPollCap) {
                    timeout = 1;
#ifdef ARCH_QUASAR
                    // Once done is visible, re-read: the last post may have retired after v.
                    if (load_u32(report_addr + 20u) == 0u || (v = sem.value()) != 0u) {
                        spins = 0;
                        continue;
                    }
#endif
                    break;
                }
            }
            if (v == 0u) {  // timed out
                break;
            }
            if (v > high_water) {
                high_water = v;
            }
            for (volatile std::uint32_t d = 0; d < kSlowConsumerSpins; ++d) {
            }
            sem.down(1);
            ++consumed;
        }
        store_u32(report_addr + 4u, consumed);
        store_u32(report_addr + 8u, high_water);
        store_u32(report_addr + 12u, sem.value());
        store_u32(report_addr + 16u, timeout);
#endif
    }
}
