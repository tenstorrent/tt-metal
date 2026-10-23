// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Test-only strand image for X280Runtime::revive.
 *
 * Each hart reads its own mode from the argument block and either returns
 * (mode 0, a conforming hart) or leaves the active-image contract in one of
 * the ways only an RNMI can recover: WFI with interrupts off, an interrupts-off
 * spin, an exception loop, an endless posted NoC write burst, or a trap. None
 * of them polls STOP, so a plain cleanup() times out on them.
 *
 * The argument block is one cache line the host writes into available LIM
 * before the enqueue; its magic and run value are its freshness marker.
 */
#include "x280_hart_protocol.h"
#include "x280_noc_coord.h"

#include "noc.h"
#include "rt/x280_hw.h"
#include "rt/x280_idle.h"

#define STRAND_ARGS_MAGIC 0x535452414E440001ULL

enum {
    STRAND_RETURN = 0,
    STRAND_WFI_IRQ_OFF = 1,
    STRAND_SPIN_IRQ_OFF = 2,
    STRAND_EXCEPTION_LOOP = 3,
    STRAND_NOC_BURST = 4,
    STRAND_TRAP = 5,
};

/* Reported through the result word before the hart strands, so the host can
 * tell a hart that reached its mode from one that never ran. */
#define STRAND_RESULT_BAD_ARGS 0xBAD0A265ULL
#define STRAND_RESULT_MODE(mode) (0x57A9D000ULL + (mode))

typedef struct {
    uint64_t magic;
    uint64_t run_value;
    uint32_t mode[X280_HART_COUNT];
    /* NoC burst target: a window to (noc_x, noc_y) at target_addr. */
    uint32_t noc_x;
    uint32_t noc_y;
    uint64_t target_addr;
    uint8_t reserved[16];
} strand_args_t;

_Static_assert(sizeof(strand_args_t) == 64, "strand arguments are one cache line");

/* Each hart bursts through its own firmware-owned window (outside the host
 * block), which a revived hart leaves programmed: the host test clears them. */
#define STRAND_NOC_WINDOW(hart) ((uint32_t)(hart))
#define STRAND_BURST_WORDS 1024u /* 4 KiB per hart per round */

static __attribute__((noreturn)) void noc_burst(const volatile strand_args_t* args, uint64_t hartid) {
    volatile void* window = 0;
    const x280_noc_coord_t at = {args->noc_x, args->noc_y};
    const uint64_t target = args->target_addr + hartid * STRAND_BURST_WORDS * sizeof(uint32_t);
    if (noc_open_window_2m(STRAND_NOC_WINDOW(hartid), at, target, NOC_ORDER_POSTED, &window) != NOC_WINDOW_OPEN) {
        x280_hart_set_result(hartid, STRAND_RESULT_BAD_ARGS);
        for (;;) {
        }
    }
    volatile uint32_t* dst = (volatile uint32_t*)window;
    for (uint64_t round = 1;; ++round) {
        for (uint32_t i = 0; i < STRAND_BURST_WORDS; ++i) {
            dst[i] = (uint32_t)round + i;
        }
        x280_fence_ow();
        x280_hart_set_result(hartid, STRAND_RESULT_MODE(STRAND_NOC_BURST) + (round << 32));
    }
}

int main(uint64_t hartid) {
    const uint64_t run_value = x280_hart_control(hartid)->run_value;
    const volatile strand_args_t* args = (const volatile strand_args_t*)(uintptr_t)x280_hart_args(hartid);
    if (args == 0 || x280_hart_args_bytes() < sizeof(strand_args_t) || args->magic != STRAND_ARGS_MAGIC ||
        args->run_value != run_value) {
        x280_hart_set_result(hartid, STRAND_RESULT_BAD_ARGS);
        return 0;
    }

    const uint32_t mode = args->mode[hartid];
    x280_hart_set_result(hartid, STRAND_RESULT_MODE(mode));
    x280_fence_ow();

    switch (mode) {
        case STRAND_WFI_IRQ_OFF:
            __asm__ volatile("csrw mie, zero\n csrci mstatus, 8\n1: wfi\n j 1b" ::: "memory");
            break;
        case STRAND_SPIN_IRQ_OFF: __asm__ volatile("csrw mie, zero\n csrci mstatus, 8\n1: j 1b" ::: "memory"); break;
        case STRAND_EXCEPTION_LOOP:
            /* mtvec aimed at an illegal instruction: every trap traps again. */
            __asm__ volatile(
                "la t0, 2f\n"
                " csrw mtvec, t0\n"
                " j 2f\n"
                " .balign 4\n"
                "2: .4byte 0\n" ::
                    : "t0", "memory");
            break;
        case STRAND_NOC_BURST: noc_burst(args, hartid);
        case STRAND_TRAP: __asm__ volatile("ebreak" ::: "memory"); break;
        default: return 0;
    }
    for (;;) {
    }
}
