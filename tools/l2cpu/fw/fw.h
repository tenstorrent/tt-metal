// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/* fw.h: internal declarations of the L2CPU firmware runtime (bare metal, M-mode, harts 0-3). */
#ifndef FW_H
#define FW_H

#include <stddef.h>
#include <stdint.h>

#include "l2cpu_boot.h"
#include "l2cpu_hw.h"

#ifdef L2CPU_FW_VERSION_OVERRIDE /* restart tests: a visibly different image at the same address */
#define FW_VERSION L2CPU_FW_VERSION_OVERRIDE
#else
#define FW_VERSION 0x00010000u /* 0.1.0 */
#endif

#ifndef L2CPU_TEST
#define L2CPU_TEST 0
#endif
#ifndef L2CPU_NOTIFY_POLL
#define L2CPU_NOTIFY_POLL 0
#endif
/* Idle wake period for heartbeats and mailbox polling (microseconds). */
#ifndef L2CPU_HB_PERIOD_US
#define L2CPU_HB_PERIOD_US 1000u
#endif

#define FW_BUILD_POLL L2CPU_BUILD_POLL
#define FW_BUILD_QEMU L2CPU_BUILD_QEMU
#define FW_BUILD_TEST L2CPU_BUILD_TEST

/* ---- CSR helpers ---------------------------------------------------------------------------- */
#define csr_read(csr)                                   \
    ({                                                  \
        uint64_t __v;                                   \
        __asm__ volatile("csrr %0, " #csr : "=r"(__v)); \
        __v;                                            \
    })
#define csr_write(csr, val) __asm__ volatile("csrw " #csr ", %0" ::"rK"(val))
#define csr_set(csr, val) __asm__ volatile("csrs " #csr ", %0" ::"rK"(val))
#define csr_clear(csr, val) __asm__ volatile("csrc " #csr ", %0" ::"rK"(val))

#define MIP_MSIP (1u << 3)
#define MIP_MTIP (1u << 7)
#define MIP_MEIP (1u << 11)

static inline void fence(void) { __asm__ volatile("fence iorw, iorw" ::: "memory"); }
static inline void compiler_barrier(void) { __asm__ volatile("" ::: "memory"); }
static inline void wfi(void) { __asm__ volatile("wfi" ::: "memory"); }
static inline uint64_t rdcycle64(void) { return csr_read(mcycle); }

/* ---- Volatile shared-memory access ------------------------------------------------------------ */
static inline uint32_t rd32(const void* p) { return *(const volatile uint32_t*)p; }
static inline uint64_t rd64(const void* p) { return *(const volatile uint64_t*)p; }
static inline void wr32(void* p, uint32_t v) { *(volatile uint32_t*)p = v; }
static inline void wr64(void* p, uint64_t v) { *(volatile uint64_t*)p = v; }
static inline uint32_t mmio_rd32(uint64_t a) { return *(volatile uint32_t*)(uintptr_t)a; }
static inline uint64_t mmio_rd64(uint64_t a) { return *(volatile uint64_t*)(uintptr_t)a; }
static inline void mmio_wr32(uint64_t a, uint32_t v) { *(volatile uint32_t*)(uintptr_t)a = v; }
static inline void mmio_wr64(uint64_t a, uint64_t v) { *(volatile uint64_t*)(uintptr_t)a = v; }

/* ---- Per-hart state (tp points at hl[mhartid]) ------------------------------------------------ */
#define FW_MAX_HARTS 5 /* 4 firmware harts + the QEMU test driver hart */

typedef struct hart_local {
    uint64_t recover_pc; /* MUST stay first (start.S safe_* helpers use 0(tp)) */
    uint32_t hartid;
    uint32_t last_work_seq; /* workers: last seq taken from work_seq[h] */
    uint64_t next_beat;     /* mtime of the next idle heartbeat */
    uint8_t* heap;          /* this hart's workspace in the region (L2CPU_OFF_HEAP) */
    uint64_t trap_depth;
    uint64_t _pad[3];
} hart_local_t;

extern hart_local_t hl[FW_MAX_HARTS];
static inline hart_local_t* self(void) {
    hart_local_t* p;
    __asm__ volatile("mv %0, tp" : "=r"(p));
    return p;
}

/* Arena pointer (set from the runtime image address by hart 0 before release). */
extern l2cpu_header_t* g_hdr; /* region base viewed as the header block */
extern uint8_t* g_region;     /* region base */
extern uint8_t* g_resident;   /* resident page (or 0 when the loader did not install one) */

/* ---- start.S: guarded accesses for the mailbox. err = 0 or the trapping mcause. ------------- */
typedef struct {
    uint64_t v;
    uint64_t err;
} safe_ret_t;
safe_ret_t safe_ld64(uint64_t addr);
safe_ret_t safe_lw32(uint64_t addr);
safe_ret_t safe_sd64(uint64_t addr, uint64_t v);
safe_ret_t safe_sw32(uint64_t addr, uint32_t v);
void fw_park(void) __attribute__((noreturn));
extern char _image_start[], _image_end[], _bss_start[], _bss_end[];

/* ---- log.c ------------------------------------------------------------------------------------ */
void log_init(void);
void fw_log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
int fw_vsnprintf(char* buf, int size, const char* fmt, __builtin_va_list ap);
int fw_snprintf(char* buf, int size, const char* fmt, ...) __attribute__((format(printf, 3, 4)));
int spin_trylock(volatile uint32_t* l, uint32_t tries);
void spin_unlock(volatile uint32_t* l);

/* ---- errors (main.c) -------------------------------------------------------------------------- */
void fw_set_error(uint32_t hart, uint32_t code, uint64_t arg, uint32_t seq);
void fw_error_park(uint32_t code, uint64_t arg, uint32_t seq) __attribute__((noreturn));
void heartbeat(uint32_t hart);

/* ---- mailbox.c -------------------------------------------------------------------------------- */
/* Hart 0: run one pending mailbox command if any. Returns 1 if one was served. */
int mailbox_poll(void);

/* ---- trap.c ----------------------------------------------------------------------------------- */
typedef struct {
    uint64_t x[32]; /* x[0] slot holds mepc on save */
    uint64_t mcause, mtval, mstatus;
} trap_frame_t;
void trap_handler(trap_frame_t* f);

/* ---- libc.c ----------------------------------------------------------------------------------- */
void* memcpy(void* d, const void* s, size_t n);
void* memset(void* d, int c, size_t n);
void* memmove(void* d, const void* s, size_t n);
int memcmp(const void* a, const void* b, size_t n);
size_t strlen(const char* s);

/* ---- main.c ----------------------------------------------------------------------------------- */
void fw_park_all(void) __attribute__((noreturn)); /* hart 0: park the workers, then itself (MB_PARK) */
uint64_t fw_mtime_ticks(uint64_t us);

/* ---- test driver (QEMU test flavour only) ----------------------------------------------------- */
void test_driver_main(void) __attribute__((noreturn));

#endif /* FW_H */
