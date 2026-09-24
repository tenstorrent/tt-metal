// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute half of the Blackhole DM <-> compute semaphore test (SemScope::DM_COMPUTE_ATOMICS): one L1
// semaphore word bound by BRISC, NCRISC (test_dm_compute_semaphore_dm.cpp) and this compute kernel. Roles:
// BRISC 0, NCRISC 1, UNPACK 2, PACK 3 (MATH has no semaphore methods and idles). Pattern by compile-time arg:
//
//   A  self-check: only the role `active` runs up/down/wait/wait_min/set/value against known values and logs
//      each observed value; the others stay idle (still bound, so the scope is DM_COMPUTE_ATOMICS).
//   D  contention: all four roles, released together by a ready/go handshake, each do num_iters x up(1).
//      BRISC (DM kernel) logs the final value once every role is done.
//
// Ring patterns (-DDMC_RING; bindings `sem` = full slots, `free` = free slots, both DM_COMPUTE_ATOMICS, and
// `csem`, compute-only COMPUTE_ATOMIC). BRISC and this kernel only; kDepth-slot ring of Float16_b tiles:
//   B  DM -> compute: BRISC NoC-copies in[i] into ring[slot], barrier, sem.up(1); UNPACK sem.wait_min(1),
//      copy_tile(ring[slot]), sem.down(1), free.up(1); PACK packs out[i].
//   C  compute -> DM: UNPACK copy_tile(in[i]); PACK free.wait_min(1), pack_tile(ring[slot]), free.down(1),
//      sem.up(1); BRISC sem.wait_min(1), NoC-copies ring[slot] to out[i], barrier, sem.down(1), free.up(1).
//   E  C behind the COMPUTE_ATOMIC datacopy hop: in[i] -> mid[slot] (csem, UNPACK <-> PACK) -> ring[slot].
//   `nosync` = 1 drops every `sem`/`free` call, 2 (E only) every `csem` call (negative controls: out must be
//   corrupted).
// Remote pattern (-DDMC_REMOTE; bindings sem, rmc, rdst, msem, all DM_COMPUTE_ATOMICS, on nodes X and Y):
//   F  BRISC on X: sem.up(noc, Y) and rmc.inc_multicast(Y..Y) num_iters x 1 each, msem.relay_unicast into
//      Y's rdst, msem.set_multicast into Y's msem. This kernel on Y (UNPACK) polls each word to its expected
//      value (bounded, via value(), the load wait_min() polls) and logs what it saw. `nosync` = 1: X sends
//      nothing, so every poll must time out at the initial 0.
//
// Report layout (words from report_addr, mirrored in the DM kernel and on the host): role r's log at
// r * kSlotWords ([0] = done, [1..] = observed values); ready[r] at kReadyWord + r; go at kGoWord.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/2_0/hw_startup.h"
#include "api/compute/experimental/2_0/pack.h"
#include "api/compute/experimental/2_0/tile_move_copy.h"
#include "api/semaphore.h"
#include "experimental/kernel_args.h"

namespace {

constexpr std::uint32_t PATTERN_A = 0;
constexpr std::uint32_t PATTERN_B = 1;
constexpr std::uint32_t PATTERN_C = 2;
constexpr std::uint32_t PATTERN_D = 3;
constexpr std::uint32_t PATTERN_E = 4;
constexpr std::uint32_t PATTERN_F = 5;

constexpr std::uint32_t kSlotWords = 16;
constexpr std::uint32_t kReadyWord = 64;
constexpr std::uint32_t kGoWord = 68;
constexpr std::uint32_t kPollCap = 50000000;

// Ring patterns: regions at fixed offsets from report_addr, mirrored in the DM kernel and on the host.
constexpr std::uint32_t kDepth = 4;
constexpr std::uint32_t kMaxTiles = 256;
constexpr std::uint32_t kTileBytes = 32 * 32 * 2;
constexpr std::uint32_t kInOffset = 0x40000;
constexpr std::uint32_t kRingOffset = kInOffset + kMaxTiles * kTileBytes;  // DM_COMPUTE_ATOMICS ring
constexpr std::uint32_t kMidOffset = kRingOffset + kDepth * kTileBytes;    // E: COMPUTE_ATOMIC ring
constexpr std::uint32_t kOutOffset = kMidOffset + kDepth * kTileBytes;

// Pattern F: values the relay / set_multicast deliver (mirrored in the DM kernel and on the host).
constexpr std::uint32_t kRemotePollCap = 2000000;
constexpr std::uint32_t kRelayValue = 0xA5A5A;
constexpr std::uint32_t kMcastValue = 0x5A5A5;

// LLKOperand addresses are 16B words with the tile-header bias, exactly what cb_read_address yields.
inline std::uint32_t operand_addr(std::uint32_t byte_addr) { return (byte_addr >> 4) - 1u; }

inline void store_u32(std::uint32_t addr, std::uint32_t v) {
    *reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(addr) = v;
}

inline std::uint32_t load_u32(std::uint32_t addr) {
    invalidate_l1_cache();  // L0 is not coherent with the other cores' L1 writes
    return *reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(addr);
}

}  // namespace

void kernel_main() {
    constexpr std::uint32_t pattern = get_arg(args::pattern);
    [[maybe_unused]] constexpr std::uint32_t active = get_arg(args::active);  // pattern A
    [[maybe_unused]] const std::uint32_t num_iters = get_arg(args::num_iters);
    [[maybe_unused]] const std::uint32_t report_addr = get_arg(args::report_addr);

    static_assert(decltype(sem::sem)::scope == SemScope::DM_COMPUTE_ATOMICS, "a DM + compute binding");
    Semaphore sem(sem::sem);

#if defined(DMC_RING)
    if constexpr (pattern == PATTERN_B || pattern == PATTERN_C || pattern == PATTERN_E) {
        constexpr std::uint32_t nosync = get_arg(args::nosync);
        constexpr bool ring_sync = nosync != 1;  // `sem` / `free`
        constexpr bool hop_sync = nosync != 2;   // `csem`
        static_assert(decltype(sem::free)::scope == SemScope::DM_COMPUTE_ATOMICS, "a DM + compute binding");
        static_assert(decltype(sem::csem)::scope == SemScope::COMPUTE_ATOMIC, "a compute-only binding");
        Semaphore free_slots(sem::free);
        [[maybe_unused]] Semaphore csem(sem::csem);

        using Op = ckernel::experimental::LLKOperand<DataFormat::Float16_b, ckernel::DEFAULT_TENSOR_SHAPE>;
        const Op in(operand_addr(report_addr + kInOffset));
        const Op ring(operand_addr(report_addr + kRingOffset));
        [[maybe_unused]] const Op mid(operand_addr(report_addr + kMidOffset));
        const Op out(operand_addr(report_addr + kOutOffset));
        compute_kernel_hw_startup(in, out);  // also seeds csem (the Sync Unit semaphore) to 0
        ckernel::experimental::copy_init(in);

        for (std::uint32_t i = 0; i < num_iters; ++i) {
            const std::uint32_t slot = i % kDepth;
            if constexpr (pattern == PATTERN_B) {
                // Consumer of BRISC's ring: the canonical `wait_min(n); <read slot>; down(n)`, then the free-slot
                // credit back; both ATINCGETs wait for the unpacker to finish reading the slot.
                tile_regs_acquire();
                if constexpr (ring_sync) {
                    UNPACK(sem.wait_min(1);)
                }
                ckernel::experimental::copy_tile(ring, slot, 0);
                if constexpr (ring_sync) {
                    UNPACK(sem.down(1); free_slots.up(1);)
                }
                tile_regs_commit();
                tile_regs_wait();
                ckernel::experimental::pack_tile(out, i, 0);
                tile_regs_release();
            } else {
                if constexpr (pattern == PATTERN_E) {
                    // COMPUTE_ATOMIC hop, as the compute-only datacopy: in[i] -> mid[slot].
                    tile_regs_acquire();
                    ckernel::experimental::copy_tile(in, i, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    if constexpr (hop_sync) {
                        PACK(csem.wait_not_full();)
                    }
                    ckernel::experimental::pack_tile(mid, slot, 0);
                    if constexpr (hop_sync) {
                        PACK(csem.up(1);)
                    }
                    tile_regs_release();
                }
                tile_regs_acquire();
                if constexpr (pattern == PATTERN_E) {
                    if constexpr (hop_sync) {
                        UNPACK(csem.wait_min(1);)
                    }
                    ckernel::experimental::copy_tile(mid, slot, 0);
                    if constexpr (hop_sync) {
                        UNPACK(csem.down(1);)
                    }
                } else {
                    ckernel::experimental::copy_tile(in, i, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                // Producer into BRISC's ring: wait for a free slot, pack, then consume the free credit and
                // publish the slot; both ATINCGETs wait for the packer to finish writing it.
                if constexpr (ring_sync) {
                    PACK(free_slots.wait_min(1);)
                }
                ckernel::experimental::pack_tile(ring, slot, 0);
                if constexpr (ring_sync) {
                    PACK(free_slots.down(1); sem.up(1);)
                }
                tile_regs_release();
            }
        }
        PACK(store_u32(report_addr + 4, num_iters);)  // retires after the last pack (the RISC store is later)
    }
#endif

#if defined(DMC_REMOTE)
    if constexpr (pattern == PATTERN_F) {
        static_assert(decltype(sem::rmc)::scope == SemScope::DM_COMPUTE_ATOMICS, "a DM + compute binding");
        static_assert(decltype(sem::rdst)::scope == SemScope::DM_COMPUTE_ATOMICS, "a DM + compute binding");
        static_assert(decltype(sem::msem)::scope == SemScope::DM_COMPUTE_ATOMICS, "a DM + compute binding");
#if defined(TRISC_UNPACK)
        // Bounded poll of value() (a settled, fresh read of the L1 word); logs what it last saw.
        auto observe = [](auto s, std::uint32_t expected, std::uint32_t log_addr) {
            std::uint32_t v = s.value();
            for (std::uint32_t spins = 0; v != expected && spins < kRemotePollCap; ++spins) {
                v = s.value();
            }
            store_u32(log_addr, v);
        };
        observe(sem, num_iters, report_addr + 4);
        observe(Semaphore(sem::rmc), num_iters, report_addr + 8);
        observe(Semaphore(sem::rdst), kRelayValue, report_addr + 12);
        observe(Semaphore(sem::msem), kMcastValue, report_addr + 16);
        store_u32(report_addr + 0, 1);
#endif
    }
#endif

#if defined(TRISC_UNPACK) || defined(TRISC_PACK)
#if defined(TRISC_UNPACK)
    constexpr std::uint32_t role = 2;
#else
    constexpr std::uint32_t role = 3;
#endif
    const std::uint32_t slot = report_addr + role * kSlotWords * 4;

    if constexpr (pattern == PATTERN_A) {
        if constexpr (active == role) {
            // Same sequence as the DM kernel; the host starts the word at initial_value = 5.
            store_u32(slot + 4, sem.value());  // 5
            sem.up(3);
            sem.wait(8);
            store_u32(slot + 8, sem.value());  // 8
            sem.down(2);
            sem.wait(6);
            sem.wait_min(4);                    // must return at 6: ">=", not "=="
            store_u32(slot + 12, sem.value());  // 6
            sem.set(0x12345);
            store_u32(slot + 16, sem.value());  // 0x12345 (a 16-bit store would read 0x2345)
            sem.up(1);
            const std::uint32_t v = sem.value();
            store_u32(slot + 20, v);  // 0x12346
            sem.down(v);              // the observed value, so a wrong one cannot hang the part
            sem.wait(0);
            store_u32(slot + 24, sem.value());  // 0
            store_u32(slot + 0, 1);
        }
    }

    if constexpr (pattern == PATTERN_D) {
        store_u32(report_addr + (kReadyWord + role) * 4, 1);
        for (std::uint32_t spins = 0; load_u32(report_addr + kGoWord * 4) == 0 && spins < kPollCap; ++spins) {
        }
        for (std::uint32_t i = 0; i < num_iters; ++i) {
            sem.up(1);
        }
        store_u32(slot + 4, sem.value());  // also retires this thread's posted ATINCGETs before `done`
        store_u32(slot + 0, 1);
    }
#endif
}
