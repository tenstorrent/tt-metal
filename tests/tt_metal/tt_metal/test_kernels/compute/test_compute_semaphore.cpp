// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel for the compute semaphore (SemScope::COMPUTE_ATOMIC): the Tensix hardware (Sync Unit)
// semaphore, index UNPACK_OPERAND_SYNC on Blackhole and PACK_UNPACK on Quasar. One kernel, patterns
// selected by the compile-time arg `pattern`:
//
//   A  single-thread up/down/wait/wait_min/set/value self-check (run once per thread via thread_sel)
//   D  real LLKOperand datacopy, compute only (no data-movement kernels): the host writes the input
//      tiles straight into L1; per tile, round 1 copies in[i] -> DST -> mid[slot] (PACK up()s), round 2
//      copies mid[slot] -> DST -> out[i] (UNPACK wait_min()s before reading, down()s after). The
//      semaphore is the only thing ordering PACK's write of `mid` before UNPACK's read of it. PACK gates
//      each pack with wait_not_full() (capacity = kDepth). Host checks out == in bit for bit. `nosync`=1
//      drops wait_not_full/wait_min/down (and up on Quasar; negative control: must corrupt).
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
#ifdef ARCH_QUASAR
#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
#include "llk_bfd_alloc.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#endif
#ifdef TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack.h"
#include "llk_pack_common.h"
#endif
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
#else

    //  - Negative control: nosync=1 also drops PACK's up(). Blackhole posts saturate at 15, so its 8
    //    unconsumed posts are harmless; a Quasar post at Max (kDepth) stalls until a down() that never
    //    comes, so the 5th up() would hang the kernel instead of corrupting the output.
    //  - Skew: PACK spins kPackSkewSpins before each ring pack, in both runs, so the packer always lags
    //    the unpacker. Without it UNPACK is not reliably ahead, and the negative control could come out
    //    correct by timing alone.
    //  - Report words: store_u32/load_u32 go through the uncached L1 alias, since UNPACK and PACK write
    //    the same report lines and the TRISC data cache is not coherent between them.
    if constexpr (pattern == PATTERN_D) {
        using namespace ckernel;
        using namespace ckernel::trisc;
        [[maybe_unused]] constexpr TensorShape shape = DEFAULT_TENSOR_SHAPE;  // UNPACK and PACK only
        constexpr auto fmt = DataFormat::Float16_b;

#ifdef TRISC_UNPACK
        bfd_alloc_and_program<BfdResource::Unp0>(
            shape, (report_addr + kDInOffset) >> 4, static_cast<std::uint32_t>(fmt));
        _llk_unpack_configure_unary_<p_unpacr::UNP_A>(fmt);
        _llk_unpack_unary_operand_init_<p_unpacr::UNP_A, false, false>(bfd_current<BfdResource::Unp0>(), shape, 1);
        for (std::uint32_t i = 0; i < num_iters; ++i) {
            _llk_unpack_unary_operand_<p_unpacr::UNP_A>(i, shape);  // in[i]
            if constexpr (!nosync) {
                sem.wait_min(1);
            }
            _llk_unpack_unary_operand_<p_unpacr::UNP_A>(kDMaxTiles + i % kDepth, shape);  // mid[slot]
            if constexpr (!nosync) {
                sem.down(1);
            }
        }
        store_u32(report_addr + 8u, sem.value());
#endif
#ifdef TRISC_MATH
        _llk_math_srcAB_hw_configure_<true, false, false>(fmt, fmt);
        _llk_math_pack_sync_init_<DstSync::SyncHalf>();
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, false>(64, 1);
        for (std::uint32_t i = 0; i < 2 * num_iters; ++i) {
            _llk_math_wait_for_dest_available_();
            _llk_math_eltwise_unary_datacopy_(0);
            _llk_math_dest_section_done_<DstSync::SyncHalf, false>();
        }
#endif
#ifdef TRISC_PACK
        sem.set(0);  // no 2.0 hw_startup on Quasar: seed on the producer
        bfd_alloc_and_program<BfdResource::Pack0>(
            shape, (report_addr + kDMidOffset) >> 4, static_cast<std::uint32_t>(fmt));
        _llk_pack_hw_configure_<p_pacr::PACK0, false>(fmt, ReluConfig::none());
        _llk_pack_init_(bfd_current<BfdResource::Pack0>(), shape, 1);
        _llk_pack_dest_init_<p_pacr::PACK0, DstSync::SyncHalf>();
        for (std::uint32_t i = 0; i < num_iters; ++i) {
            // Round 1: DST -> mid[slot], then publish. The delay holds the packer back in both the positive
            // and the negative run, so an early ring read by UNPACK is always visible without the semaphore.
            if constexpr (!nosync) {
                sem.wait_not_full();
            }
            for (volatile std::uint32_t d = 0; d < kPackSkewSpins; ++d) {
            }
            _llk_packer_wait_for_math_done_();
            _llk_pack_(0, i % kDepth, shape);
            _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, DstSync::SyncHalf, false>();
            if constexpr (!nosync) {
                // Without a consumer down(), a Quasar post at capacity can stall forever.
                sem.up(1);
            }

            // Round 2: DST -> out[i].
            _llk_packer_wait_for_math_done_();
            _llk_pack_(0, kDepth + i, shape);
            _llk_pack_dest_semaphore_section_done_<p_pacr::PACK0, DstSync::SyncHalf, false>();
        }
        store_u32(report_addr + 0u, num_iters);
#endif
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
