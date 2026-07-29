/* eccpoke.c - induce/confirm a boot-breaking ECC bad state from a hart.
 *
 * Boots from the PRIMED low LIM region (so the hart runs), then reads a single
 * target line whose address the host writes into a param slot before release.
 * If that line is uncorrectable (genuinely unprimed / defective), the read
 * raises a double-bit ECC fault -> the tile halts (halt_from_tile, §18.3) or
 * traps: STATUS stays at BOOTED and never reaches SURVIVED. If the line is
 * fine, STATUS advances to SURVIVED and RESULT holds the value read.
 *
 * This is the deterministic way to (a) find an address a hart actually halts
 * on, giving us the "no heartbeat" bad state, and (b) after an Option-A prime,
 * confirm the same read now SURVIVES.
 *
 * Boots via boot/entry.S (main(hartid)); links with ld/x280-lim.ld.
 */
#include <stdint.h>

#define STATUS ((volatile uint64_t*)0x08010000ULL) /* 1=booted, 2=survived the read */
#define RESULT ((volatile uint64_t*)0x08010010ULL) /* value read from the target line */
#define PARAM ((volatile uint64_t*)0x08010020ULL)  /* host writes target address here */

void main(uint64_t hartid) {
    if (hartid != 0) {
        for (;;) {
            __asm__ volatile("wfi");
        }
    }
    *STATUS = 0xEC00000001ULL; /* booted (primed region OK) */
    __asm__ volatile("fence iorw, iorw");
    uint64_t target = *PARAM;
    volatile uint64_t* p = (volatile uint64_t*)target;
    uint64_t v = *p; /* <-- if `target` is uncorrectable, this halts/traps the hart */
    *RESULT = v;
    *STATUS = 0xEC00000002ULL; /* survived: the read did NOT fault */
    for (;;) {
        __asm__ volatile("wfi");
    }
}
