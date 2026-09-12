// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test kernel for the Blackhole compute semaphore (SemScope::COMPUTE_ATOMIC): the Tensix hardware (Sync
// Unit) semaphore, index UNPACK_OPERAND_SYNC. One kernel, five patterns selected by the compile-time
// arg `pattern`:
//
//   A  single-thread up/down/wait/wait_min/set/value self-check (run once per thread via thread_sel)
//   B  cross-thread producer (UNPACK up) / consumer (PACK wait + down), bounded depth
//   C  real packer-engine writes into an intermediate L1 buffer, semaphore-gated, data verified
//   D  real LLKOperand datacopy, compute only (no data-movement kernels): the host writes the input
//      tiles straight into L1; per tile, round 1 copies in[i] -> DST -> mid[slot] (PACK up()s), round 2
//      copies mid[slot] -> DST -> out[i] (UNPACK wait_min()s before reading, down()s after). The
//      semaphore is the only thing ordering PACK's write of `mid` before UNPACK's read of it. PACK gates
//      each pack with wait_not_full() (capacity = kDepth). Host checks out == in bit for bit. `nosync`=1
//      drops wait_not_full/wait_min/down (negative control: must corrupt).
//   E  producer back-pressure: PACK produces num_iters credits gated by wait_not_full(), UNPACK is a
//      deliberately slow consumer that records the highest value it ever observes. With the host's
//      max_value = kDepth that high-water mark stays <= kDepth. `nosync`=1 drops the wait_not_full()
//      (negative control: PACK runs to the 15 ceiling, posts are lost, UNPACK times out).
//
// The hardware semaphore is 4-bit (0..15) and starts every kernel at 0: compute_kernel_hw_startup()
// seeds it on PACK (pattern D), and every pattern leaves it balanced at 0 for the next program. The
// patterns that do not call hw_startup (A/B/C/E) seed it with set(0) on their PRODUCING thread, which
// is safe for the same reason the hw_startup seed is: the producer's first up() is queued behind its
// own SEMINIT, and the consumer reads 0 either way. A/B/C/E waits are bounded so a lost update cannot
// hang the part; D uses the API's own wait_min (the point of D is the shipped primitives on a real op).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/pack.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"
#include "api/compute/experimental/semaphore.h"
#include "experimental/kernel_args.h"

namespace {

constexpr std::uint32_t PATTERN_A = 0;
constexpr std::uint32_t PATTERN_B = 1;
constexpr std::uint32_t PATTERN_C = 2;
constexpr std::uint32_t PATTERN_D = 3;
constexpr std::uint32_t PATTERN_E = 4;

constexpr std::uint32_t SEL_UNPACK = 0;
constexpr std::uint32_t SEL_PACK = 2;

constexpr std::uint32_t kPollCap = 5000000;

// Pattern B/C/D/E bounded-buffer depth. Well under the 15 hardware max so a few in-flight (uncommitted)
// posts cannot saturate the semaphore and drop an update. Patterns D and E also pass it to the host as
// the semaphore's max_value (capacity).
constexpr std::uint32_t kDepth = 4;

// Pattern E: RISC busy-loop per consumed credit, so the producer is always ahead of the consumer.
constexpr std::uint32_t kSlowConsumerSpins = 2000;

// Pattern C: a real 1-face Float16_b tile. The packer writes 32 words (128 B) per _llk_pack_; each slot is
// exactly that. num_iters slots, one write each (no reuse), so every iteration is a genuine
// publish-after-data test with no consumer-side re-poison hazard.
constexpr std::uint32_t kSlotWords = 32;
constexpr std::uint32_t kBufByteOffset = 256;  // shared buffer starts here, clear of the report words
constexpr std::uint32_t kFmt = 5;              // DataFormat::Float16_b
constexpr std::uint32_t kFaceRDim = 16;
constexpr std::uint32_t kTileCDim = 16;
constexpr std::uint32_t kNumFaces = 1;
constexpr std::uint32_t kTileBytes = 16 * 16 * 2;
constexpr std::uint32_t kTileSize16B = kTileBytes / 16;  // packer tile_size is in 16-byte words
constexpr std::uint32_t kPoison = 0xDEADBEEFu;

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
    *reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(addr) = v;
}

inline std::uint32_t load_u32(std::uint32_t addr) {
    invalidate_l1_cache();  // L0 is not coherent with the other thread's L1 writes
    return *reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(addr);
}

}  // namespace

void kernel_main() {
    constexpr std::uint32_t pattern = get_arg(args::pattern);
    constexpr std::uint32_t thread_sel = get_arg(args::thread_sel);
    [[maybe_unused]] constexpr std::uint32_t nosync = get_arg(args::nosync);  // patterns D and E
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
    // PATTERN B: cross-thread bounded producer/consumer. UNPACK posts num_iters credits (never more
    // than kDepth outstanding); PACK waits for and consumes each. Balanced, so the final value is 0
    // iff no post was lost and no get underflowed.
    // -----------------------------------------------------------------------------------------------
    if constexpr (pattern == PATTERN_B) {
        PACK(
            std::uint32_t consumed = 0;
            std::uint32_t timeout = 0;
            for (std::uint32_t i = 0; i < num_iters; ++i) {
                std::uint32_t spins = 0;
                while (sem.value() < 1u) {
                    if (++spins >= kPollCap) { timeout = 1; break; }
                }
                if (timeout) { break; }
                sem.down(1);
                ++consumed;
            }
            store_u32(report_addr + 8u, consumed);
            store_u32(report_addr + 12u, sem.value());
            store_u32(report_addr + 16u, timeout);)

        UNPACK(
            sem.set(0);  // UNPACK produces here, so UNPACK seeds (its posts queue behind the SEMINIT)
            std::uint32_t produced = 0;
            std::uint32_t timeout = 0;
            for (std::uint32_t i = 0; i < num_iters && !timeout; ++i) {
                std::uint32_t spins = 0;
                while (sem.value() >= kDepth) {  // don't run more than kDepth ahead of the consumer
                    if (++spins >= kPollCap) { timeout = 1; break; }
                }
                if (timeout) { break; }
                sem.up(1);
                ++produced;
            }
            store_u32(report_addr + 0u, produced);
            store_u32(report_addr + 4u, timeout);)
    }

    // -----------------------------------------------------------------------------------------------
    // PATTERN C: PACK packs a real tile (packer engine) into shared-buffer slot i and up()s; UNPACK
    // waits, verifies the slot no longer holds the host-written poison (i.e. the packer's writes are
    // visible once the semaphore says so), and down()s. num_iters slots, written once each, so the
    // buffer is host-poisoned and never re-poisoned: no consumer-side release-ordering hazard. The
    // up()'s STALLWAIT(PACK) is what orders the packer writes before the credit (publish-after-data).
    // -----------------------------------------------------------------------------------------------
    if constexpr (pattern == PATTERN_C) {
        const std::uint32_t buffer_addr = report_addr + kBufByteOffset;

        PACK(
            _llk_pack_hw_configure_<false, PackMode::Default>(
                kFmt, kFmt, kTileSize16B, kFaceRDim, kTileCDim, kNumFaces, false, 0);
            _llk_pack_dest_init_<DstSync::SyncFull, false>();
            _llk_pack_init_<PackMode::Default, true /*zero_output*/>(
                kFmt, kFaceRDim, kTileCDim, kNumFaces, 1, false);

            sem.set(0);  // PACK produces: seed on the producer, no handshake needed

            std::uint32_t timeout = 0;
            for (std::uint32_t i = 0; i < num_iters && !timeout; ++i) {
                const std::uint32_t slot_addr = buffer_addr + i * kSlotWords * 4u;
                // Packer writes the slot. Address in 16B words, biased by -1 for the tile header.
                _llk_pack_<DstSync::SyncFull, false, PackMode::Default>(0, (slot_addr >> 4) - 1u);
                sem.up(1);  // STALLWAIT(PACK) + SEMPOST: credit ordered after the packer writes
            }
            store_u32(report_addr + 0u, num_iters);)

        UNPACK(
            std::uint32_t consumed = 0;
            std::uint32_t mismatches = 0;
            std::uint32_t timeout = 0;
            for (std::uint32_t i = 0; i < num_iters && !timeout; ++i) {
                std::uint32_t spins = 0;
                while (sem.value() < 1u) {
                    if (++spins >= kPollCap) { timeout = 1; break; }
                }
                if (timeout) { break; }
                const std::uint32_t slot_addr = buffer_addr + i * kSlotWords * 4u;
                // Read in reverse: the last word the packer writes is checked first, so a premature
                // credit (packer not yet done) is caught rather than raced past.
                for (std::uint32_t w = kSlotWords; w-- > 0;) {
                    if (load_u32(slot_addr + w * 4u) == kPoison) { ++mismatches; }
                }
                sem.down(1);
                ++consumed;
            }
            store_u32(report_addr + 4u, consumed);
            store_u32(report_addr + 8u, mismatches);
            store_u32(report_addr + 16u, timeout);)
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

    // -----------------------------------------------------------------------------------------------
    // PATTERN E: producer back-pressure. PACK: wait_not_full(); up(1), num_iters times. The SEMWAIT
    // blocks no engine instruction here (pure counter), but the STALLWAIT inside up() is held behind it,
    // so the SEMPOST cannot execute while Value >= Max. UNPACK: slow consumer, polls value() (the settled
    // Sync Unit value) so the high-water mark it records is exact; with wait_not_full() in place it can
    // never exceed kDepth. Balanced, so the final value is 0 iff no post was lost.
    // -----------------------------------------------------------------------------------------------
    if constexpr (pattern == PATTERN_E) {
        PACK(
            sem.set(0);  // PACK produces: seed on the producer (Max = host max_value)
            for (std::uint32_t i = 0; i < num_iters; ++i) {
                if constexpr (!nosync) {
                    sem.wait_not_full();
                }
                sem.up(1);
            }
            store_u32(report_addr + 0u, num_iters);)

        UNPACK(
            std::uint32_t consumed = 0;
            std::uint32_t high_water = 0;
            std::uint32_t timeout = 0;
            for (std::uint32_t i = 0; i < num_iters && !timeout; ++i) {
                std::uint32_t spins = 0;
                std::uint32_t v;
                while ((v = sem.value()) < 1u) {
                    if (++spins >= kPollCap) { timeout = 1; break; }
                }
                if (timeout) { break; }
                if (v > high_water) { high_water = v; }
                for (volatile std::uint32_t d = 0; d < kSlowConsumerSpins; ++d) {
                }
                sem.down(1);
                ++consumed;
            }
            store_u32(report_addr + 4u, consumed);
            store_u32(report_addr + 8u, high_water);
            store_u32(report_addr + 12u, sem.value());
            store_u32(report_addr + 16u, timeout);)
    }
}
