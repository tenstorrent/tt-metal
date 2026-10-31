// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/*
 * test_driver.c: QEMU test flavour only. Runs on a fifth hart (mhartid 4) and plays the host: boot checks, the
 * mailbox (incl. guarded faults), the example application's echo work item (doorbell + worker dispatch), the trap
 * test, and the restart suite (L1 park + GO into another slot / the same slot, WARM and COLD, a work item published
 * while parked, 1000 restart cycles, restart after a worker trap). Exit code through the virt test finisher.
 * RNMI (L2) cannot be tested here: QEMU 8.2 has no Smrnmi.
 */
#include <stdarg.h>

#include "fw.h"
#include "platform.h"

#ifndef L2CPU_RESTART_CYCLES
#define L2CPU_RESTART_CYCLES 1000u
#endif

void console_write(const char* s);
extern char _load_end[];

/* the example application's echo protocol (examples/heartbeat/app.c) */
#define ECHO_REQ (L2CPU_OFF_APP_CTRL + 0x000)
#define ECHO_DONE (L2CPU_OFF_APP_CTRL + 0x040)
#define ECHO_VALUE (L2CPU_OFF_APP_CTRL + 0x080)
#define ECHO_RESULT (L2CPU_OFF_APP_CTRL + 0x100)
#define SLOT_C 0x240000u /* QEMU-only extra slot: the driver itself keeps running from slot A */

static uint8_t* R; /* region */
static l2cpu_header_t* H;
static uint32_t req;   /* echo req */
static uint32_t mbreq; /* mailbox req */

static void tprintf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void tprintf(const char* fmt, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    fw_vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    console_write(buf);
}

static uint64_t ticks_ms(uint64_t ms) { return (uint64_t)L2CPU_QEMU_MTIME_HZ / 1000u * ms; }

static void dump_log_tail(void) {
    uint64_t wr = rd64(R + L2CPU_OFF_LOG_WR);
    uint64_t n = wr < 3000 ? wr : 3000;
    char buf[2] = {0, 0};
    console_write("---- firmware log tail ----\n");
    for (uint64_t i = wr - n; i < wr; i++) {
        buf[0] = (char)R[L2CPU_OFF_LOG_DATA + i % L2CPU_LOG_DATA_SIZE];
        console_write(buf);
    }
    console_write("---- end of log ----\n");
}

#define FAIL(...)                                    \
    do {                                             \
        tprintf("FAIL %s:%d: ", __FILE__, __LINE__); \
        tprintf(__VA_ARGS__);                        \
        tprintf("\n");                               \
        dump_log_tail();                             \
        plat_finish(1);                              \
    } while (0)
#define CHECK(c, ...)          \
    do {                       \
        if (!(c))              \
            FAIL(__VA_ARGS__); \
    } while (0)

static uint32_t r32(uint32_t off) { return rd32(R + off); }

/* ---- mailbox ---------------------------------------------------------------------------------------------- */
static uint32_t mb(uint32_t cmd, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t* rep, int ring) {
    l2cpu_mailbox_t* m = (l2cpu_mailbox_t*)(R + L2CPU_OFF_MAILBOX);
    wr32(&m->cmd, cmd);
    wr64(&m->arg[0], a0);
    wr64(&m->arg[1], a1);
    wr64(&m->arg[2], a2);
    wr64(&m->arg[3], a3);
    fence();
    mbreq++;
    wr32(&m->req.v, mbreq);
    fence();
    if (ring) {
        plat_qemu_ring_doorbell(0x4D420000u | (mbreq & 0xFFFF));
    }
    uint64_t deadline = plat_mtime() + ticks_ms(2000);
    while (rd32(&m->ack.v) != mbreq) {
        CHECK(plat_mtime() < deadline, "mailbox cmd %u: no ack", cmd);
    }
    fence();
    for (int i = 0; i < 7 && rep; i++) {
        rep[i] = rd64(&m->reply[i]);
    }
    return rd32(&m->status);
}

static uint8_t* noc_ptr(uint8_t x, uint8_t y, uint64_t addr) {
    return (uint8_t*)(uintptr_t)(L2CPU_QEMU_FAKE_NOC_BASE +
                                 (((uint64_t)y * L2CPU_QEMU_FAKE_NOC_MAX_X + x) << L2CPU_QEMU_FAKE_NOC_SLOT_SHIFT) +
                                 addr);
}

static void mailbox_tests(void) {
    uint64_t r[7];
    uint8_t* S = R + L2CPU_OFF_SCRATCH;
    uint64_t scr = (uint64_t)(uintptr_t)S;
    CHECK(mb(L2CPU_MB_PING, 0, 0, 0, 0, r, 1) == L2CPU_MB_OK && r[0] == FW_VERSION && r[2] == 0 && r[4] == 1, "ping");
    CHECK(mb(L2CPU_MB_PING, 0, 0, 0, 0, r, 0) == L2CPU_MB_OK, "ping without doorbell (timer wake)");
    CHECK(mb(L2CPU_MB_POKE32, scr, 0x12345678u, 0, 0, r, 1) == L2CPU_MB_OK && rd32(S) == 0x12345678u, "poke32");
    CHECK(mb(L2CPU_MB_PEEK32, scr, 0, 0, 0, r, 1) == L2CPU_MB_OK && r[0] == 0x12345678u, "peek32");
    CHECK(mb(L2CPU_MB_POKE64, scr + 8, 0x1122334455667788ull, 0, 0, r, 1) == L2CPU_MB_OK, "poke64");
    CHECK(mb(L2CPU_MB_PEEK64, scr + 8, 0, 0, 0, r, 1) == L2CPU_MB_OK && r[0] == 0x1122334455667788ull, "peek64");
    CHECK(mb(L2CPU_MB_FILL32, scr + 0x100, 0xA5A5A5A5u, 256, 0, r, 1) == L2CPU_MB_OK, "fill");
    for (int i = 0; i < 64; i++) {
        CHECK(rd32(S + 0x100 + 4 * i) == 0xA5A5A5A5u, "fill content");
    }
    CHECK(mb(L2CPU_MB_COPY32, scr + 0x200, scr + 0x100, 256, 0, r, 1) == L2CPU_MB_OK, "copy");
    CHECK(mb(L2CPU_MB_MEMCMP32, scr + 0x200, scr + 0x100, 256, 0, r, 1) == L2CPU_MB_OK && r[0] == ~0ull, "memcmp eq");
    S[0x200 + 37] ^= 0x40;
    fence();
    CHECK(mb(L2CPU_MB_MEMCMP32, scr + 0x200, scr + 0x100, 256, 0, r, 1) == L2CPU_MB_OK && r[0] == 37, "memcmp diff");
    CHECK(mb(L2CPU_MB_CSR_READ, 0xf14, 0, 0, 0, r, 1) == L2CPU_MB_OK && r[0] == 0, "csr mhartid");
    CHECK(mb(L2CPU_MB_CSR_READ, 0x301, 0, 0, 0, r, 1) == L2CPU_MB_OK && (r[0] & (1u << 21)), "csr misa has V");
    uint32_t st = mb(L2CPU_MB_CSR_READ, 0x7c1, 0, 0, 0, r, 1);
    CHECK(st == L2CPU_MB_ERR_FAULT && r[1] == 2, "csr 0x7c1 on QEMU must fault (illegal instruction): st %u", st);
    st = mb(L2CPU_MB_PEEK32, 0x0, 0, 0, 0, r, 1);
    CHECK(st == L2CPU_MB_ERR_FAULT && r[1] == 5, "peek of an unmapped address must fault: st %u mcause %lu", st, r[1]);
    st = mb(L2CPU_MB_POKE32, 0x0, 1, 0, 0, r, 1);
    CHECK(st == L2CPU_MB_ERR_FAULT && r[1] == 7, "poke of an unmapped address must fault: st %u mcause %lu", st, r[1]);
    CHECK(mb(L2CPU_MB_NOC_WRITE32, 3, 1, 0x7F0000, 0xCAFEF00Du, r, 1) == L2CPU_MB_OK, "noc write");
    CHECK(*(volatile uint32_t*)noc_ptr(3, 1, 0x7F0000) == 0xCAFEF00Du, "noc write landed");
    CHECK(mb(L2CPU_MB_NOC_READ32, 3, 1, 0x7F0000, 0, r, 1) == L2CPU_MB_OK && r[0] == 0xCAFEF00Du, "noc read");
    CHECK(mb(L2CPU_MB_TIME, 0, 0, 0, 0, r, 1) == L2CPU_MB_OK && r[0] != 0, "time");
    CHECK(mb(L2CPU_MB_APP, 41, 0, 0, 0, r, 1) == L2CPU_MB_OK && r[0] == 42, "application mailbox command");
    CHECK(mb(999, 0, 0, 0, 0, r, 1) == L2CPU_MB_ERR_CMD, "unknown command");
    tprintf("PASS mailbox: ping, peek/poke, fill, copy, memcmp, csr, guarded faults, noc r/w, time, app command\n");
}

/* ---- echo work item (example application) ------------------------------------------------------------------ */
static void echo_issue(uint32_t value) {
    wr32(R + ECHO_VALUE, value);
    fence(); /* producer: data, then the sequence */
    req++;
    wr32(R + ECHO_REQ, req);
    fence();
    plat_qemu_ring_doorbell(req | 0x80000000u);
}
static int echo_wait(uint64_t ms) {
    uint64_t deadline = plat_mtime() + ticks_ms(ms);
    while (r32(ECHO_DONE) != req) {
        if (plat_mtime() > deadline) {
            return -1;
        }
    }
    fence();
    return 0;
}
static void echo_check(uint32_t value, const char* what) {
    for (uint32_t h = 0; h < 4; h++) {
        CHECK(
            r32(ECHO_RESULT + 64 * h) == value + h,
            "%s: result[%u] = %u expected %u",
            what,
            h,
            r32(ECHO_RESULT + 64 * h),
            value + h);
    }
}
static void echo(uint32_t value, const char* what) {
    echo_issue(value);
    CHECK(echo_wait(2000) == 0, "%s: echo %u not done (done %u)", what, req, r32(ECHO_DONE));
    echo_check(value, what);
}

/* ---- restart (L1 park + GO; the driver plays the host's l2cpu ctl) --------------------------------------------- */
static uint8_t* res(uint32_t off) { return R + L2CPU_OFF_RESIDENT + off; }
static uint32_t rec32(uint32_t h, uint32_t off) { return rd32(res(L2CPU_RES_REC + L2CPU_REC_SIZE * h + off)); }
static void copy_image(uint32_t slot) {
    memcpy(R + slot, _image_start, (size_t)(_load_end - _image_start));
    fence();
}
static void wait_parked(uint32_t mask, const char* what) {
    uint64_t deadline = plat_mtime() + ticks_ms(2000);
    for (uint32_t h = 0; h < 4; h++) {
        if (mask & (1u << h)) {
            while (rec32(h, L2CPU_REC_STATE) != L2CPU_STATE_PARKED) {
                CHECK(plat_mtime() < deadline, "%s: hart %u not parked (state %u)", what, h, rec32(h, L2CPU_REC_STATE));
            }
        }
    }
}
static uint64_t restart_ticks, restart_park_ticks;
static uint32_t restart(uint32_t slot, int warm, int already_parked) {
    uint64_t t0 = plat_mtime();
    uint64_t r[7];
    if (!already_parked) {
        CHECK(mb(L2CPU_MB_PARK, 0, 0, 0, 0, r, 1) == L2CPU_MB_OK, "MB_PARK");
    }
    wait_parked(0xF, "restart");
    uint64_t t_parked = plat_mtime();
    uint32_t ep = rd32(res(L2CPU_RES_BOOT_EPOCH)) + 1;
    wr32(R + L2CPU_OFF_FW_STATUS, 0);
    wr64(res(L2CPU_RES_ENTRY), (uint64_t)(uintptr_t)R + slot);
    wr32(res(L2CPU_RES_BOOT_MODE), warm ? L2CPU_BOOT_WARM : L2CPU_BOOT_COLD);
    wr32(res(L2CPU_RES_BOOT_EPOCH), ep);
    fence();
    wr32(res(L2CPU_RES_GO_EPOCH), rd32(res(L2CPU_RES_GO_EPOCH)) + 1); /* last */
    fence();
    uint64_t deadline = plat_mtime() + ticks_ms(3000);
    while (!(r32(L2CPU_OFF_FW_STATUS) == L2CPU_FW_STATUS_READY && r32(L2CPU_OFF_BOOT_EPOCH) == ep)) {
        CHECK(
            plat_mtime() < deadline,
            "restart epoch %u: not READY (status %u epoch %u)",
            ep,
            r32(L2CPU_OFF_FW_STATUS),
            r32(L2CPU_OFF_BOOT_EPOCH));
    }
    fence();
    restart_park_ticks += t_parked - t0;
    if (!warm) {
        req = 0;
        mbreq = 0;
    }
    restart_ticks += plat_mtime() - t0;
    return ep;
}

static void restart_suite(void) {
    uint64_t r[7];
    CHECK(H->ident.build_flags & L2CPU_BUILD_RESTART, "image does not advertise L2CPU_BUILD_RESTART");
    CHECK(H->ident.boot_epoch == 1 && H->ident.boot_mode == L2CPU_BOOT_COLD, "first boot epoch/mode");
    copy_image(L2CPU_SLOT_B);
    copy_image(SLOT_C);
    uint32_t done0 = r32(ECHO_DONE);
    uint64_t mbc0 = H->counters[0].mailbox_cmds;
    /* 1. WARM into a different address */
    uint32_t ep = restart(L2CPU_SLOT_B, 1, 0);
    CHECK(H->ident.image_base == (uint64_t)(uintptr_t)R + L2CPU_SLOT_B, "image base after restart");
    CHECK(
        r32(ECHO_DONE) == done0 && H->counters[0].mailbox_cmds >= mbc0 && H->ident.restart_count == 1,
        "warm: application window / counters not preserved");
    for (uint32_t h = 0; h < 4; h++) {
        CHECK(rec32(h, L2CPU_REC_KIND) == L2CPU_KIND_SOFT, "park kind hart %u", h);
    }
    echo(1000, "after warm restart");
    tprintf("PASS restart: WARM into slot B (different address), epoch %u, application window + counters kept\n", ep);
    /* 2. WARM at the same address, image rewritten while parked */
    CHECK(mb(L2CPU_MB_PARK, 0, 0, 0, 0, r, 1) == L2CPU_MB_OK, "MB_PARK");
    wait_parked(0xF, "same address");
    copy_image(L2CPU_SLOT_B);
    ep = restart(L2CPU_SLOT_B, 1, 1);
    echo(2000, "same address");
    tprintf("PASS restart: WARM at the same address (image rewritten while parked), epoch %u\n", ep);
    /* 3. a work item published while every hart is parked is served late after the restart */
    CHECK(mb(L2CPU_MB_PARK, 0, 0, 0, 0, r, 1) == L2CPU_MB_OK, "MB_PARK");
    wait_parked(0xF, "parked work");
    echo_issue(3000);
    CHECK(echo_wait(30) != 0, "work item served while every hart was parked");
    ep = restart(SLOT_C, 1, 1);
    CHECK(echo_wait(2000) == 0, "work item not served after the restart");
    echo_check(3000, "late work item");
    tprintf("PASS restart: work item published while parked, served late after WARM restart into slot C\n");
    /* 3b. a mailbox command posted while every hart is parked (what ctl.stop's L1 PARK to a hung hart 0 leaves
     * behind before the RNMI fallback) is acknowledged as stale by the WARM boot, not run: no re-park after READY */
    CHECK(mb(L2CPU_MB_PARK, 0, 0, 0, 0, r, 1) == L2CPU_MB_OK, "MB_PARK");
    wait_parked(0xF, "stale mailbox");
    {
        l2cpu_mailbox_t* m = (l2cpu_mailbox_t*)(R + L2CPU_OFF_MAILBOX);
        wr32(&m->cmd, L2CPU_MB_PARK);
        fence();
        mbreq++;
        wr32(&m->req.v, mbreq);
        fence();
        plat_qemu_ring_doorbell(0x4D420000u | (mbreq & 0xFFFF));
        ep = restart(L2CPU_SLOT_B, 1, 1);
        CHECK(
            rd32(&m->ack.v) == mbreq && rd32(&m->status) == L2CPU_MB_ERR_STALE,
            "stale command: ack %u req %u status %u",
            rd32(&m->ack.v),
            mbreq,
            rd32(&m->status));
        uint64_t t = plat_mtime() + ticks_ms(20);
        while (plat_mtime() < t) {
        }
        for (uint32_t h = 0; h < 4; h++) {
            CHECK(
                rec32(h, L2CPU_REC_STATE) != L2CPU_STATE_PARKED && H->hart_state[h].status == L2CPU_HART_IDLE,
                "hart %u parked again after the restart",
                h);
        }
        echo(3500, "after a stale mailbox command");
        CHECK(mb(L2CPU_MB_PING, 0, 0, 0, 0, r, 1) == L2CPU_MB_OK, "mailbox after a stale command");
    }
    tprintf(
        "PASS restart: mailbox PARK posted while parked is dropped as stale by the WARM boot (epoch %u), no re-park\n",
        ep);
    /* 3c. PLIC enable bits are not reset by a chip reset (random on the chip): poison every enable word of the
     * firmware harts' contexts while parked; the boot must leave only the doorbell source of hart 0's M context */
    CHECK(mb(L2CPU_MB_PARK, 0, 0, 0, 0, r, 1) == L2CPU_MB_OK, "MB_PARK");
    wait_parked(0xF, "plic poison");
    for (uint32_t ctx = 0; ctx < 2 * L2CPU_NHARTS; ctx++) {
        for (uint32_t k = 0; k < (L2CPU_QEMU_PLIC_NUM_SOURCES + 31) / 32; k++) {
            mmio_wr32(L2CPU_QEMU_PLIC_BASE + 0x2000ull + 0x80ull * ctx + 4ull * k, 0xFFFFFFFFu);
        }
    }
    fence();
    ep = restart(L2CPU_SLOT_B, 1, 1);
    for (uint32_t ctx = 0; ctx < 2 * L2CPU_NHARTS; ctx++) {
        for (uint32_t k = 0; k < (L2CPU_QEMU_PLIC_NUM_SOURCES + 31) / 32; k++) {
            uint32_t want = ctx == 0 && k == L2CPU_QEMU_DOORBELL_PLIC_SOURCE / 32
                                ? 1u << (L2CPU_QEMU_DOORBELL_PLIC_SOURCE % 32)
                                : 0u;
            uint32_t got = mmio_rd32(L2CPU_QEMU_PLIC_BASE + 0x2000ull + 0x80ull * ctx + 4ull * k);
            CHECK(got == want, "PLIC ctx %u enable word %u = 0x%x after boot (want 0x%x)", ctx, k, got, want);
        }
    }
    echo(3600, "after PLIC enable poisoning");
    tprintf(
        "PASS restart: stale PLIC enable bits of all 8 contexts cleared by the boot (epoch %u), doorbell only\n", ep);
    /* 4. COLD restart */
    ep = restart(L2CPU_SLOT_B, 0, 0);
    CHECK(
        r32(ECHO_REQ) == 0 && r32(ECHO_DONE) == 0 && H->ident.restart_count == 0 && r32(L2CPU_OFF_MB_ACK) == 0,
        "cold: region not reset");
    echo(4000, "after cold restart");
    tprintf("PASS restart: COLD (application window, mailbox, counters reset), epoch %u\n", ep);
    /* 5. restart cycles, alternating slots, WARM, one work item each */
    restart_ticks = restart_park_ticks = 0;
    for (uint32_t i = 0; i < L2CPU_RESTART_CYCLES; i++) {
        restart((i & 1) ? SLOT_C : L2CPU_SLOT_B, 1, 0);
        echo(10000 + i, "restart cycle");
    }
    tprintf(
        "PASS restart: %u WARM restart cycles (alternating slots B/C), echo bit-exact after each, avg restart %lu us "
        "(park %lu us)\n",
        L2CPU_RESTART_CYCLES,
        restart_ticks / L2CPU_RESTART_CYCLES / (L2CPU_QEMU_MTIME_HZ / 1000000u),
        restart_park_ticks / L2CPU_RESTART_CYCLES / (L2CPU_QEMU_MTIME_HZ / 1000000u));
    restart(L2CPU_SLOT_B, 1, 0);
}

static uint64_t hb(uint32_t h) { return rd64(&H->heartbeat[h].count); }

static void trap_and_restart_tests(void) {
    uint64_t r[7], before[4];
    CHECK(mb(L2CPU_MB_INJECT, 2, 0, 0, 0, r, 1) == L2CPU_MB_OK, "inject");
    uint64_t deadline = plat_mtime() + ticks_ms(2000);
    while (rd32(&H->hart_state[2].status) != L2CPU_HART_PARKED) {
        CHECK(plat_mtime() < deadline, "hart 2 not parked");
    }
    l2cpu_hart_state_t* hs = &H->hart_state[2];
    CHECK(hs->error == L2CPU_ERR_TRAP && hs->mcause == 2 && hs->mepc != 0 && hs->trap_count == 1, "trap record");
    CHECK(H->error.code == L2CPU_ERR_TRAP && H->error.hart == 2 && H->error.arg == 2, "global error");
    /* the firmware marks the hart PARKED before it jumps to the resident error park, which writes its record next */
    while (rec32(2, L2CPU_REC_STATE) != L2CPU_STATE_PARKED) {
        CHECK(plat_mtime() < deadline, "no resident record");
    }
    CHECK(rec32(2, L2CPU_REC_KIND) == L2CPU_KIND_ERROR, "hart 2 not in the resident error park");
    for (int h = 0; h < 4; h++) {
        before[h] = hb(h);
    }
    uint64_t t = plat_mtime() + ticks_ms(30);
    while (plat_mtime() < t) {
    }
    CHECK(hb(0) > before[0] && hb(1) > before[1] && hb(3) > before[3] && hb(2) == before[2], "heartbeats");
    tprintf("PASS trap: hart 2 illegal instruction -> error TRAP, mcause 2, resident error park; harts 0,1,3 alive\n");
    /* a work item needs hart 2: not published, WORKER_DEAD, hart 0 stays alive */
    echo_issue(5000);
    CHECK(echo_wait(300) != 0, "work item published with a parked worker");
    deadline = plat_mtime() + ticks_ms(1000);
    while (H->hart_state[0].error != L2CPU_ERR_WORKER_DEAD) {
        CHECK(plat_mtime() < deadline, "no WORKER_DEAD");
    }
    CHECK(mb(L2CPU_MB_PING, 0, 0, 0, 0, r, 1) == L2CPU_MB_OK, "hart 0 alive after WORKER_DEAD");
    tprintf("PASS worker dead: work item not published, hart 0 error WORKER_DEAD, mailbox alive\n");
    /* WARM restart revives hart 2; the error word survives; the next work item uses all 4 harts */
    uint32_t ep = restart(L2CPU_SLOT_B, 1, 0);
    for (uint32_t h = 0; h < 4; h++) {
        CHECK(H->hart_state[h].status == L2CPU_HART_IDLE, "hart %u not idle", h);
    }
    CHECK(H->error.code == L2CPU_ERR_TRAP, "error word must survive a WARM restart");
    echo(6000, "after trap + restart");
    tprintf("PASS restart after a worker trap: hart 2 back (epoch %u), all 4 harts serve\n", ep);
}

/* ---- forced bounds (README "Waits and their bounds") --------------------------------------------------------- */
static int mb_try(uint32_t cmd, uint64_t timeout_ms) { /* host-side mailbox bound */
    l2cpu_mailbox_t* m = (l2cpu_mailbox_t*)(R + L2CPU_OFF_MAILBOX);
    wr32(&m->cmd, cmd);
    fence();
    mbreq++;
    wr32(&m->req.v, mbreq);
    fence();
    plat_qemu_ring_doorbell(0x4D420000u | (mbreq & 0xFFFF));
    uint64_t deadline = plat_mtime() + ticks_ms(timeout_ms);
    while (rd32(&m->ack.v) != mbreq) {
        if (plat_mtime() > deadline) {
            return -1;
        }
    }
    return (int)rd32(&m->status);
}

static void hang_tests(void) {
    uint64_t r[7];
    /* stalled worker: hart 3 spins with interrupts off; a work item that needs it ends in WORK_TIMEOUT */
    CHECK(mb(L2CPU_MB_INJECT, 3, L2CPU_INJECT_SPIN, 0, 0, r, 1) == L2CPU_MB_OK, "inject spin");
    uint64_t t = plat_mtime() + ticks_ms(10);
    while (plat_mtime() < t) {
    }
    uint64_t t0 = plat_mtime();
    echo_issue(7000);
    uint64_t deadline = plat_mtime() + ticks_ms(L2CPU_WORK_TIMEOUT_US / 1000 + 2000);
    while (H->hart_state[0].error != L2CPU_ERR_WORK_TIMEOUT) {
        CHECK(plat_mtime() < deadline, "no WORK_TIMEOUT");
    }
    uint64_t waited_ms = (plat_mtime() - t0) / ticks_ms(1);
    CHECK(H->hart_state[0].error_arg == 3 && r32(ECHO_DONE) != req, "WORK_TIMEOUT must name hart 3, not publish");
    CHECK(mb(L2CPU_MB_PING, 0, 0, 0, 0, r, 1) == L2CPU_MB_OK, "hart 0 alive after WORK_TIMEOUT");
    tprintf(
        "PASS bound: stalled worker -> L2CPU_ERR_WORK_TIMEOUT(hart 3) after %lu ms (bound %u ms), hart 0 alive\n",
        waited_ms,
        L2CPU_WORK_TIMEOUT_US / 1000);
    /* a mailbox command that never completes: hart 0 itself spins with interrupts off after its reply */
    CHECK(mb(L2CPU_MB_INJECT, 0, L2CPU_INJECT_SPIN, 0, 0, r, 1) == L2CPU_MB_OK, "inject spin hart 0");
    uint64_t hb0 = hb(0);
    t0 = plat_mtime();
    int st = mb_try(L2CPU_MB_PING, 500);
    CHECK(st == -1, "mailbox must time out on the host side (status %d)", st);
    CHECK(hb(0) == hb0, "hart 0 heartbeat must stop");
    tprintf(
        "PASS bound: mailbox command never acknowledged -> host timeout after %lu ms, hart 0 heartbeat stopped "
        "(recovery on the chip: L2 RNMI park + restart)\n",
        (plat_mtime() - t0) / ticks_ms(1));
}

void test_driver_main(void) {
    R = g_region;
    H = g_hdr;
    tprintf(
        "l2cpu fw qemu-test (%s notify), driver on hart %u\n",
        L2CPU_NOTIFY_POLL ? "poll" : "irq",
        (unsigned)csr_read(mhartid));
    uint64_t deadline = plat_mtime() + ticks_ms(3000);
    while (r32(L2CPU_OFF_FW_STATUS) != L2CPU_FW_STATUS_READY) {
        CHECK(plat_mtime() < deadline, "firmware not READY");
    }
    fence();
    CHECK(
        H->ident.magic == L2CPU_MAGIC && H->ident.layout_version == L2CPU_LAYOUT_VERSION && H->ident.app_id == 1,
        "ident");
    CHECK(
        H->ident.region_base == (uint64_t)(uintptr_t)R && H->ident.image_base == (uint64_t)(uintptr_t)R + L2CPU_OFF_FW,
        "ident addresses");
    for (uint32_t h = 0; h < 4; h++) {
        CHECK(H->hart_state[h].status == L2CPU_HART_IDLE, "hart %u not idle", h);
    }
    CHECK(H->error.code == 0, "error at boot");
    req = r32(ECHO_REQ);
    mbreq = r32(L2CPU_OFF_MB_ACK);
    tprintf(
        "PASS boot: READY, magic, layout %u, region %p, 4 harts idle, garbage region cleared\n",
        L2CPU_LAYOUT_VERSION,
        R);
    mailbox_tests();
    for (uint32_t i = 0; i < 200; i++) {
        echo(i * 3 + 1, "echo");
    }
    tprintf("PASS echo: 200 work items over 4 harts (doorbell + dispatch)\n");
    restart_suite();
    trap_and_restart_tests();
    hang_tests();
    tprintf("ALL PASS (%s notify)\n", L2CPU_NOTIFY_POLL ? "poll" : "irq");
    plat_finish(0);
}
