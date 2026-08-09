// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PROBE kernel: NoC 4-bit CAS spin-lock (0 = free, 1 = held) protecting the full
// check-then-decrement. This is the keystone-proven REFERENCE SHAPE the production Quasar-DM
// EXTERNAL down() (noc_semaphore.h) mirrors -- keep the two in lockstep. Gen1/TRISC keep the
// single-consumer spin+subtract path.
//
// Per lock-protected decrement:
//   acquire  : CAS(lock, cmp=0, swap=1); the response returns the PRE-OP word to this
//              hart's ret slot (R_SRC_ADDR is per-hart STICKY) -- acquired iff pre-op == 0
//   re-check : sem >= 1 via the uncached alias (else release the lock and retry)
//   decrement: noc_semaphore_inc(-1) -- the exact INCR_GET EXTERNAL down() emits
//   release  : CAS(lock, cmp=1, swap=0); pre-op must be 1 (we held it)
//
// Production down() additionally drains prior in-flight atomics at entry (a remote up() does not
// barrier); here every atomic is barriered or sentinel-consumed before the hart's next one, and a
// locking hart's first atomic is the acquire CAS, so no entry drain is needed.
//
// Modes (mode arg):
//   0 pure drain: every user hart does increment_times lock-protected decrements.
//   1 mixed: even user harts do increment_times PLAIN noc_semaphore_inc(+1) -- exactly
//     what up() emits, NO lock; odd user harts do increment_times lock-protected
//     decrements (blocking on >=1). Only the first `pairs` even/odd hart pairs act
//     (so totals balance); threads beyond the last full pair exit immediately.
//
// Raw CAS emit uses Quasar-only RoCC builtins; elsewhere the kernel is a no-op.

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

#if defined(ARCH_QUASAR)

// TL1 view of an L1 address: CAS responses and NoC atomics land at TL1, so reads and
// writes here must bypass the DM write-back cache.
inline volatile tt_l1_ptr uint32_t* uncached(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(static_cast<uintptr_t>(addr) + MEM_L1_UNCACHED_BASE);
}

void kernel_main() {
    const uint32_t sem_addr = get_arg(args::sem_addr);
    const uint32_t lock_addr = get_arg(args::lock_addr);
    const uint32_t ret_base = get_arg(args::ret_base);
    const uint32_t increment_times = get_arg(args::increment_times);
    const uint32_t mode = get_arg(args::mode);
    const uint32_t pairs = get_arg(args::pairs);

    uint64_t hart;
    asm volatile("csrr %0, mhartid" : "=r"(hart));
    const uint32_t user_idx = static_cast<uint32_t>(hart) - 2;  // Metal 2.0 reserves DM0/DM1; lowest user DM is 2

    // The lock pre-op is only ever 0 or 1, so this can never be a real response.
    constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
    // PRIVATE per-hart slot: R_SRC_ADDR is sticky, so every atomic this hart issues
    // (including the INCR_GET decrement) returns its pre-op word here.
    const uint32_t ret_slot = ret_base + static_cast<uint32_t>(hart) * 4;
    const uint64_t lock_noc_addr = get_noc_addr(lock_addr);  // loopback: this node's NIU
    const uint64_t sem_noc_addr = get_noc_addr(sem_addr);

    // One CAS on the lock word; returns its PRE-OP value (0 or 1).
    auto lock_cas = [&](uint32_t cmp, uint32_t swap) -> uint32_t {
        *uncached(ret_slot) = SENTINEL;
        noc_fast_atomic_cas4<DM_DEDICATED_NOC, true /*program_ret_addr*/>(
            noc_index,
            0 /*cmd_buf unused*/,
            lock_noc_addr,
            NOC_UNICAST_WRITE_VC,
            cmp,
            swap,
            false /*linked*/,
            false /*posted*/,
            ret_slot);
        noc_async_atomic_barrier();
        while (*uncached(ret_slot) == SENTINEL) {
        }
        return *uncached(ret_slot);
    };

    // Every INCR_GET this hart issues ALSO returns its pre-op word to the sticky ret_slot.
    // Consume it with the same sentinel discipline as the CASes, or a late-landing return
    // could overwrite the NEXT CAS's sentinel and corrupt the lock verdict. (Sem pre-op is a
    // small count plus 0x10000001-sized poisons -- never the sentinel, EXCEPT a double grant
    // spending the last credit, which wraps to 0xFFFFFFFF and hangs this poll: see the host
    // EXPECT message.)
    auto consumed_inc = [&](uint64_t noc_addr, uint32_t incr) {
        *uncached(ret_slot) = SENTINEL;
        noc_semaphore_inc(noc_addr, incr);
        noc_async_atomic_barrier();
        while (*uncached(ret_slot) == SENTINEL) {
        }
    };

    // release: pre-op must be 1 (we held the lock). Anything else means the lock was
    // lost or double-granted; poison the count so the host's exact-0 check fails.
    auto release_lock = [&]() {
        if (lock_cas(1 /*cmp*/, 0 /*swap*/) != 1) {
            consumed_inc(sem_noc_addr, 0x10000001u);  // not a divisor of 2^32: repeated poisons cannot sum to 0
        }
    };

    // One lock-protected conditional decrement: the production EXTERNAL down(1) body.
    auto locked_decrement = [&]() {
        while (true) {
            // Cheap wait for a credit before contending; re-checked under the lock.
            while (*uncached(sem_addr) < 1) {
            }
            // acquire: pre-op 0 => we took the lock; 1 => held elsewhere, retry.
            if (lock_cas(0 /*cmp*/, 1 /*swap*/) != 0) {
                continue;
            }
            if (*uncached(sem_addr) < 1) {
                // Credit vanished before we locked: released; back to the lock-free wait.
                release_lock();
                continue;
            }
            // The exact INCR_GET decrement EXTERNAL down() emits (wrap=31 => modular -1).
            consumed_inc(sem_noc_addr, (uint32_t)(-1));
            release_lock();
            return;
        }
    };

    if (mode == 0) {
        // Pure drain: every user hart competes for the lock.
        for (uint32_t i = 0; i < increment_times; i++) {
            locked_decrement();
        }
    } else {
        // Mixed: threads beyond the last full even/odd pair exit immediately.
        if (user_idx / 2 >= pairs) {
            return;
        }
        if ((user_idx & 1) == 0) {
            // Producer: plain UNLOCKED increments -- exactly what up() emits.
            for (uint32_t i = 0; i < increment_times; i++) {
                noc_semaphore_inc(sem_noc_addr, 1);
                noc_async_atomic_barrier();
            }
        } else {
            // Consumer: lock-protected decrements, blocking on >=1.
            for (uint32_t i = 0; i < increment_times; i++) {
                locked_decrement();
            }
        }
    }
}

#else

// Raw NoC CAS emit needs the Quasar RoCC builtins; nothing to probe on other archs.
void kernel_main() {}

#endif
