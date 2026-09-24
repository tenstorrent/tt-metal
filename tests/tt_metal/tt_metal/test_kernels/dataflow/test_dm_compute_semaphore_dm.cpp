// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DM half of the Blackhole DM <-> compute semaphore test (SemScope::DM_COMPUTE_ATOMICS); patterns, roles
// and report layout are described in test_kernels/compute/test_dm_compute_semaphore.cpp. `role` is 0 on
// BRISC and 1 on NCRISC. Pattern D with plain_rmw = 1 is the negative control: instead of up(1) (a NoC
// atomic) this kernel does a plain RISC read-modify-write on the word, which must lose updates.
//
// Ring and remote patterns (B, C, E, F): BRISC only. Ring data moves by local NoC write, and the kernel waits
// for it (noc_async_write_barrier) before handing the slot on: a producer publishes only after its data is
// in L1, a consumer returns the free credit only after its copy has read the slot. The NoC write is the
// contract's documented publish path; RISC stores followed by a NoC atomic would add a store-vs-NoC-atomic
// ordering question the API does not answer.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
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
constexpr std::uint32_t kFinalWord = 69;    // BRISC: value() once every role is done
constexpr std::uint32_t kTimeoutWord = 70;  // BRISC: a handshake poll ran out
constexpr std::uint32_t kAddrWord = 71;     // BRISC: the word's L1 address, so the host can read it too
constexpr std::uint32_t kNumRoles = 4;
constexpr std::uint32_t kPollCap = 50000000;

// Ring patterns (mirrors the compute kernel).
constexpr std::uint32_t kDepth = 4;
constexpr std::uint32_t kMaxTiles = 256;
constexpr std::uint32_t kTileBytes = 32 * 32 * 2;
constexpr std::uint32_t kInOffset = 0x40000;
constexpr std::uint32_t kRingOffset = kInOffset + kMaxTiles * kTileBytes;
constexpr std::uint32_t kOutOffset = kRingOffset + 2 * kDepth * kTileBytes;  // past the ring and E's mid ring
constexpr std::uint32_t kRingSemAddrWord = 2;   // the `sem` (full slots) word's L1 address, for the host
constexpr std::uint32_t kRingFreeAddrWord = 3;  // the `free` word's L1 address

// Pattern F (mirrors the compute kernel).
constexpr std::uint32_t kRelayValue = 0xA5A5A;
constexpr std::uint32_t kMcastValue = 0x5A5A5;

// Local NoC copy of one tile, complete (read and written) on return.
inline void noc_copy_tile(std::uint32_t src, std::uint32_t dst) {
    noc_async_write(src, get_noc_addr(dst), kTileBytes);
    noc_async_write_barrier();
}

inline void store_u32(std::uint32_t addr, std::uint32_t v) {
    *reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(addr) = v;
}

inline std::uint32_t load_u32(std::uint32_t addr) {
    invalidate_l1_cache();
    return *reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(addr);
}

// Spin until the report word at `word` is nonzero; false on timeout.
inline bool poll_flag(std::uint32_t report_addr, std::uint32_t word) {
    for (std::uint32_t spins = 0; spins < kPollCap; ++spins) {
        if (load_u32(report_addr + word * 4) != 0) {
            return true;
        }
    }
    return false;
}

}  // namespace

void kernel_main() {
    constexpr std::uint32_t pattern = get_arg(args::pattern);
    constexpr std::uint32_t role = get_arg(args::role);
    [[maybe_unused]] constexpr std::uint32_t active = get_arg(args::active);        // pattern A
    [[maybe_unused]] constexpr std::uint32_t plain_rmw = get_arg(args::plain_rmw);  // pattern D control
    const std::uint32_t num_iters = get_arg(args::num_iters);
    const std::uint32_t report_addr = get_arg(args::report_addr);

    static_assert(decltype(sem::sem)::scope == SemScope::DM_COMPUTE_ATOMICS, "a DM + compute binding");
    Semaphore sem(sem::sem);
    const std::uint32_t slot = report_addr + role * kSlotWords * 4;

#if defined(DMC_RING)
    if constexpr (pattern == PATTERN_B || pattern == PATTERN_C || pattern == PATTERN_E) {
        constexpr bool ring_sync = get_arg(args::nosync) != 1;  // 2 = E's csem-only control
        static_assert(decltype(sem::free)::scope == SemScope::DM_COMPUTE_ATOMICS, "a DM + compute binding");
        Semaphore free_slots(sem::free);
        for (std::uint32_t i = 0; i < num_iters; ++i) {
            const std::uint32_t ring_slot = report_addr + kRingOffset + (i % kDepth) * kTileBytes;
            if constexpr (pattern == PATTERN_B) {
                // Producer: take a free slot, fill it, publish it (data in L1 before the NoC atomic).
                if constexpr (ring_sync) {
                    free_slots.wait_min(1);
                }
                noc_copy_tile(report_addr + kInOffset + i * kTileBytes, ring_slot);
                if constexpr (ring_sync) {
                    free_slots.down(1);
                    sem.up(1);
                }
            } else {
                // Consumer: wait for a full slot, copy it out, release it (read complete before the credit).
                if constexpr (ring_sync) {
                    sem.wait_min(1);
                }
                noc_copy_tile(ring_slot, report_addr + kOutOffset + i * kTileBytes);
                if constexpr (ring_sync) {
                    sem.down(1);
                    free_slots.up(1);
                }
            }
        }
        store_u32(report_addr + kRingSemAddrWord * 4, get_semaphore(decltype(sem::sem)::id));
        store_u32(report_addr + kRingFreeAddrWord * 4, get_semaphore(decltype(sem::free)::id));
        store_u32(report_addr + 0, num_iters);
    }
#endif

#if defined(DMC_REMOTE)
    if constexpr (pattern == PATTERN_F) {
        constexpr std::uint32_t nosync = get_arg(args::nosync);
        static_assert(decltype(sem::rmc)::scope == SemScope::DM_COMPUTE_ATOMICS, "a DM + compute binding");
        static_assert(decltype(sem::rdst)::scope == SemScope::DM_COMPUTE_ATOMICS, "a DM + compute binding");
        static_assert(decltype(sem::msem)::scope == SemScope::DM_COMPUTE_ATOMICS, "a DM + compute binding");
        Semaphore rmc(sem::rmc);
        Semaphore rdst(sem::rdst);
        Semaphore msem(sem::msem);
        const std::uint32_t peer_x = get_arg(args::peer_x);  // node Y, virtual NoC coordinates
        const std::uint32_t peer_y = get_arg(args::peer_y);
        Noc noc(noc_index);
        if constexpr (!nosync) {
            for (std::uint32_t i = 0; i < num_iters; ++i) {
                sem.up(noc, peer_x, peer_y, 1);
            }
            for (std::uint32_t i = 0; i < num_iters; ++i) {
                rmc.inc_multicast(noc, peer_x, peer_y, peer_x, peer_y, 1, 1);
            }
            // msem's own word on X is the relay / multicast source; nothing else updates X's copy.
            msem.set(kRelayValue);
            msem.relay_unicast(noc, rdst, peer_x, peer_y);
            noc_async_write_barrier();  // the relay has read the source word before it is overwritten
            msem.set(kMcastValue);
            msem.set_multicast(noc, peer_x, peer_y, peer_x, peer_y, 1);
            noc_async_write_barrier();
            noc_async_atomic_barrier();
        }
        store_u32(report_addr + 0, 1);
    }
#endif

    if constexpr (pattern == PATTERN_A) {
        if constexpr (active == role) {
            // Same sequence as the compute kernel; the host starts the word at initial_value = 5.
            store_u32(slot + 4, sem.value());  // 5
            sem.up(3);
            sem.wait(8);
            store_u32(slot + 8, sem.value());  // 8
            sem.down(2);
            sem.wait(6);
            sem.wait_min(4);                    // must return at 6: ">=", not "=="
            store_u32(slot + 12, sem.value());  // 6
            sem.set(0x12345);
            store_u32(slot + 16, sem.value());  // 0x12345
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
        // BRISC releases everyone at once, after the other three have checked in.
        bool ok = true;
        if constexpr (role == 0) {
            for (std::uint32_t r = 1; r < kNumRoles; ++r) {
                ok = poll_flag(report_addr, kReadyWord + r) && ok;
            }
            store_u32(report_addr + kGoWord * 4, 1);
        } else {
            store_u32(report_addr + (kReadyWord + role) * 4, 1);
            ok = poll_flag(report_addr, kGoWord);
        }

        if constexpr (plain_rmw) {
            // Test-only: the forbidden raw access, to prove the harness really creates contention.
            auto* word = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(get_semaphore(decltype(sem::sem)::id));
            for (std::uint32_t i = 0; i < num_iters; ++i) {
                invalidate_l1_cache();
                *word = *word + 1;
            }
        } else {
            for (std::uint32_t i = 0; i < num_iters; ++i) {
                sem.up(1);
            }
        }
        store_u32(slot + 4, sem.value());
        store_u32(slot + 0, 1);

        if constexpr (role == 0) {
            for (std::uint32_t r = 1; r < kNumRoles; ++r) {
                ok = poll_flag(report_addr, r * kSlotWords) && ok;
            }
            store_u32(report_addr + kFinalWord * 4, sem.value());
            store_u32(report_addr + kAddrWord * 4, get_semaphore(decltype(sem::sem)::id));
        }
        if (!ok) {
            store_u32(report_addr + kTimeoutWord * 4, 1);
        }
    }
}
