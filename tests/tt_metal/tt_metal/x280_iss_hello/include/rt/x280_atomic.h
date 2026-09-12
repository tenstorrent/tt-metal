/*
 * rt/x280_atomic.h -- RISC-V AMOs for the X280 baremetal firmware.
 *
 * Both helpers return the old value: a producer does
 * `my_slot = amoadd(&write_ptr, 1)` and the returned value is the slot claimed.
 * The volatile and the "memory" clobber keep the compiler from eliding the op,
 * hoisting it out of a loop, or dropping the result.
 *
 * amoadd works against LIM. LR/SC does not -- reservations are not tracked on
 * uncached LIM, so sc.d never succeeds and the retry loop live-locks, stranding
 * the hart, since an X280 cannot re-enter reset without a chip-level reset.
 * Hence no compare-and-swap helper; CAS semantics have to be expressed as a
 * fetch-and-add.
 */
#ifndef X280_RT_ATOMIC_H
#define X280_RT_ATOMIC_H

#include <stdint.h>

/* amoadd.w rd, rs2, (rs1): rd = *p (old, 32-bit), then *p += v. */
static inline uint32_t x280_amoadd_w(volatile uint32_t* p, uint32_t v) {
    uint32_t old;
    __asm__ volatile("amoadd.w %0, %2, (%1)" : "=r"(old) : "r"(p), "r"(v) : "memory");
    return old;
}

/* amoadd.d rd, rs2, (rs1): rd = *p (old, 64-bit), then *p += v. */
static inline uint64_t x280_amoadd_d(volatile uint64_t* p, uint64_t v) {
    uint64_t old;
    __asm__ volatile("amoadd.d %0, %2, (%1)" : "=r"(old) : "r"(p), "r"(v) : "memory");
    return old;
}

#endif /* X280_RT_ATOMIC_H */
