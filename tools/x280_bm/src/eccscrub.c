/* eccscrub.c - probe for a no-read line-zero primitive on the X280 (Zicboz /
 * cbo.zero). That primitive is the clean way to stamp valid ECC into an
 * UNPRIMED LIM line from a hart: it establishes a zeroed cache block WITHOUT
 * fetching the old (uninitialized) line, so no read-modify-write and no
 * double-bit-ECC fault. If the X280 has it, a hart booted from ECC-safe memory
 * can scrub LIM after a reset (no cache-controller WayEnable, no second reset).
 *
 * This probe boots from PRIMED LIM (safe on a working board), so it takes no
 * ECC risk. It seeds a SEPARATE primed line with a non-zero pattern, runs
 * cbo.zero on it, then reports one of three outcomes:
 *   - trap (mcause=2 illegal)  -> Zicboz NOT implemented
 *   - survived, line unchanged -> decoded as a no-op (not usable)
 *   - survived, line == 0      -> cbo.zero WORKS (the clean scrub primitive)
 *
 * Boots via boot/entry.S (calls main(hartid)); links with ld/x280-lim.ld.
 */
#include <stdint.h>

/* Status/result block in free SRAM at/after HB_COUNTER_ADDR (0x08010000). */
#define STATUS ((volatile uint64_t*)0x08010000ULL)   /* phase marker */
#define READBACK ((volatile uint64_t*)0x08010008ULL) /* target[0] after cbo.zero */
#define DONE ((volatile uint64_t*)0x08010010ULL)     /* 0xEC..FF when finished */
#define EFFECT ((volatile uint64_t*)0x08010018ULL)   /* 1 => line was zeroed */
/* 64 B-aligned line well above the FW's 64 KiB managed region, inside primed LIM. */
#define TARGET ((volatile uint64_t*)0x08030000ULL)

static inline void cbo_zero(volatile void* p) {
    register const void* a0 asm("a0") = p;
    /* cbo.zero (a0) == .word 0x0045200f (Zicboz). Raw-encoded so it assembles
     * under -march=rv64gc; if the core lacks Zicboz it traps illegal at run. */
    __asm__ volatile(".word 0x0045200f" ::"r"(a0) : "memory");
}

void main(uint64_t hartid) {
    if (hartid != 0) {
        for (;;) {
            __asm__ volatile("wfi");
        }
    }
    *STATUS = 0xEC00000001ULL; /* booted */
    for (int i = 0; i < 8; i++) {
        TARGET[i] = 0xAAAAAAAAAAAAAAAAULL; /* seed non-zero (primed LIM: safe) */
    }
    __asm__ volatile("fence iorw, iorw");
    *STATUS = 0xEC00000002ULL; /* about to run cbo.zero */
    cbo_zero((volatile void*)TARGET);
    __asm__ volatile("fence iorw, iorw");
    *STATUS = 0xEC00000003ULL; /* survived cbo.zero (no trap) */
    uint64_t w0 = TARGET[0];
    *READBACK = w0;
    *EFFECT = (w0 == 0) ? 1 : 0;
    *DONE = 0xEC000000FFULL;
    for (;;) {
        __asm__ volatile("wfi");
    }
}
