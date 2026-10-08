// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/* main.c: boot (COLD / WARM by epoch), the per-hart idle loops, heartbeats, work dispatch, error handling, park. */
#include "app.h"
#include "fw.h"
#include "fw_asm.h"
#include "platform.h"

_Static_assert(sizeof(hart_local_t) == ASM_HL_SIZE, "hart_local size vs start.S");
_Static_assert(offsetof(hart_local_t, recover_pc) == 0, "recover_pc must be first");
_Static_assert(ASM_FW_MAX_HARTS == FW_MAX_HARTS, "max harts");
_Static_assert(FW_MAX_HARTS <= L2CPU_MAX_STACKS, "stacks");
_Static_assert(sizeof(trap_frame_t) <= ASM_FRAME_SIZE, "frame");
_Static_assert(ASM_QEMU_REGION_BASE == L2CPU_QEMU_REGION_BASE, "qemu region");

hart_local_t hl[FW_MAX_HARTS] __attribute__((aligned(64)));
l2cpu_header_t* g_hdr;
uint8_t* g_region;
uint8_t* g_resident;
static uint32_t g_boot_mode, g_boot_epoch;
static volatile uint32_t region_ready;  /* epoch-tagged: survives in .bss across a restart without reload */
static uint32_t stale_mb, stale_mb_req; /* WARM boot: a mailbox request dropped as stale (logged after log_init) */

uint64_t fw_mtime_ticks(uint64_t us) { return (uint64_t)plat_mtime_hz() / 1000000u * us; }

static uint64_t hb_period(void) {
    uint64_t t = fw_mtime_ticks(L2CPU_HB_PERIOD_US);
    return t ? t : 1;
}

void heartbeat(uint32_t hart) {
    l2cpu_heartbeat_t* hb = &g_hdr->heartbeat[hart];
    wr64(&hb->mtime, plat_mtime());
    wr64(&hb->count, rd64(&hb->count) + 1);
}

static void set_hart_status(uint32_t hart, uint32_t st) { wr32(&g_hdr->hart_state[hart].status, st); }

static void counter_add(uint32_t hart, uint32_t off, uint64_t v) {
    uint64_t* p = (uint64_t*)((uint8_t*)&g_hdr->counters[hart] + off);
    wr64(p, rd64(p) + v);
}

void fw_set_error(uint32_t hart, uint32_t code, uint64_t arg, uint32_t context) {
    l2cpu_header_t* h = g_hdr;
    if (hart < L2CPU_NHARTS) {
        wr32(&h->hart_state[hart].error, code);
        wr64(&h->hart_state[hart].error_arg, arg);
    }
    uint32_t expected = 0; /* first error wins */
    if (__atomic_compare_exchange_n(&h->error.code, &expected, code, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        wr32(&h->error.hart, hart);
        wr32(&h->error.context, context);
        wr64(&h->error.arg, arg);
    }
    fence();
}

void fw_error_park(uint32_t code, uint64_t arg, uint32_t context) {
    uint32_t hart = self()->hartid;
    fw_set_error(hart, code, arg, context);
    if (hart < L2CPU_NHARTS) {
        set_hart_status(hart, L2CPU_HART_PARKED);
    }
    fence();
    fw_log("PARK error=%u arg=0x%lx context=%u", code, arg, context);
    fw_park();
}

/* ---- inject (tests) and park ---------------------------------------------------------------------------- */
static void check_inject(uint32_t hart) {
    uint32_t k = rd32(&g_hdr->inject[hart].v);
    if (k == L2CPU_INJECT_NONE) {
        return;
    }
    wr32(&g_hdr->inject[hart].v, L2CPU_INJECT_NONE);
    fence();
    if (k == L2CPU_INJECT_ILLEGAL) {
        fw_log("inject: executing an illegal instruction");
        __asm__ volatile("unimp");
    } else if (k == L2CPU_INJECT_PARK && g_resident) {
        ((void (*)(void))(g_resident + L2CPU_RES_SOFT_PARK))();
    } else if (k == L2CPU_INJECT_SPIN) {
        fw_log("inject: spinning with interrupts off");
        __asm__ volatile("csrw mie, zero\n csrci mstatus, 8\n 1: j 1b" ::: "memory");
    } else if (k == L2CPU_INJECT_WFIPARK) {
        fw_log("inject: wfi park with interrupts off");
        __asm__ volatile("csrw mie, zero\n csrci mstatus, 8\n 1: wfi\n j 1b" ::: "memory");
    } else if ((k == L2CPU_INJECT_LOAD || k == L2CPU_INJECT_STORE || k == L2CPU_INJECT_JUMP) && fw_pmp_on()) {
        /* PMP fault tests: unguarded on purpose, the policy must turn them into a trap (error park) */
        uint64_t a = rd64((uint8_t*)&g_hdr->inject[hart] + L2CPU_INJECT_ADDR);
        fw_log("inject: %s 0x%lx", k == L2CPU_INJECT_LOAD ? "load" : k == L2CPU_INJECT_STORE ? "store" : "jump", a);
        if (k == L2CPU_INJECT_LOAD) {
            (void)*(volatile uint64_t*)(uintptr_t)a;
        } else if (k == L2CPU_INJECT_STORE) {
            *(volatile uint64_t*)(uintptr_t)a = 0x5A5A5A5A5A5A5A5Aull;
        } else {
            ((void (*)(void))(uintptr_t)a)();
        }
        fw_log("inject: access at 0x%lx did not trap", a);
    }
}

/* Hart 0, after replying to MB_PARK: park the workers (IPI), wait (bounded) until their records say PARKED, then
 * park itself. Workers that never answer (spinning, trapping) are left to the host's RNMI path. */
void fw_park_all(void) {
    for (uint32_t h = 1; h < L2CPU_NHARTS; h++) {
        wr32(&g_hdr->inject[h].v, L2CPU_INJECT_PARK);
        fence();
        plat_ipi_send(h);
    }
    uint64_t deadline = plat_mtime() + fw_mtime_ticks(L2CPU_PARK_WORKERS_US);
    for (uint32_t h = 1; h < L2CPU_NHARTS; h++) {
        while (rd32(g_resident + L2CPU_RES_REC + L2CPU_REC_SIZE * h + L2CPU_REC_STATE) != L2CPU_STATE_PARKED &&
               plat_mtime() < deadline) {
        }
    }
    fw_log("park: hart 0 parking (epoch %u)", g_boot_epoch);
    ((void (*)(void))(g_resident + L2CPU_RES_SOFT_PARK))();
    __builtin_unreachable();
}

/* ---- work dispatch (application service) ---------------------------------------------------------------- */
void fw_dispatch(uint32_t mask, uint32_t seq) {
    fence(); /* the application's job data before the sequence words */
    for (uint32_t h = 1; h < L2CPU_NHARTS; h++) {
        if (mask & (1u << h)) {
            wr32(&g_hdr->work_seq[h].v, seq);
            fence();
            plat_ipi_send(h);
        }
    }
}

int fw_wait_workers_us(uint32_t mask, uint32_t seq, uint64_t timeout_us) {
    uint64_t t0 = plat_mtime(), tmo = fw_mtime_ticks(timeout_us);
    for (uint32_t h = 1; h < L2CPU_NHARTS; h++) {
        if (!(mask & (1u << h))) {
            continue;
        }
        while (rd32(&g_hdr->work_done[h].v) != seq) {
            if (rd32(&g_hdr->hart_state[h].status) == L2CPU_HART_PARKED) {
                fw_log("work %u: worker %u is parked", seq, h);
                fw_set_error(0, L2CPU_ERR_WORKER_DEAD, h, seq);
                return -1;
            }
            if (plat_mtime() - t0 > tmo) {
                fw_log("work %u: worker %u did not finish within %lu us", seq, h, timeout_us);
                fw_set_error(0, L2CPU_ERR_WORK_TIMEOUT, h, seq);
                return -1;
            }
        }
    }
    fence();
    return 0;
}

int fw_wait_workers(uint32_t mask, uint32_t seq) { return fw_wait_workers_us(mask, seq, L2CPU_WORK_TIMEOUT_US); }

/* ---- idle loops ------------------------------------------------------------------------------------------ */
static void timer_rearm(hart_local_t* me) {
    me->next_beat = plat_mtime() + hb_period();
    plat_set_timer(me->hartid, me->next_beat);
}

static void worker_loop(uint32_t hart) {
    hart_local_t* me = self();
    me->last_work_seq = rd32(&g_hdr->work_seq[hart].v);
    csr_write(mie, MIP_MSIP | MIP_MTIP);
    timer_rearm(me);
    set_hart_status(hart, L2CPU_HART_IDLE);
    for (;;) {
        plat_ipi_clear(hart); /* clear first, then look: a later IPI stays pending and ends wfi */
        fence();
        if (csr_read(mip) & MIP_MTIP) {
            timer_rearm(me);
        }
        check_inject(hart);
        uint32_t ws = rd32(&g_hdr->work_seq[hart].v);
        if (ws != me->last_work_seq) {
            fence(); /* consumer: sequence first, then the job data */
            me->last_work_seq = ws;
            set_hart_status(hart, L2CPU_HART_BUSY);
            app_work(hart, ws);
            counter_add(hart, L2CPU_CNT_WORK, 1);
            wr32(&g_hdr->hart_state[hart].last_work, ws);
            fence(); /* producer: results, then the done word */
            wr32(&g_hdr->work_done[hart].v, ws);
            fence();
            set_hart_status(hart, L2CPU_HART_IDLE);
        }
        heartbeat(hart);
        wfi();
        counter_add(hart, L2CPU_CNT_WAKEUPS, 1);
        if ((csr_read(mip) & (MIP_MSIP | MIP_MTIP)) == 0) {
            counter_add(hart, L2CPU_CNT_SPURIOUS, 1);
        }
    }
}

static void hart0_pass(void) {
    if (mailbox_poll()) {
        counter_add(0, L2CPU_CNT_MAILBOX, 1);
    }
    check_inject(0);
    if (app_poll()) {
        heartbeat(0);
    }
}

static void hart0_loop(void) {
    hart_local_t* me = self();
#if L2CPU_NOTIFY_POLL
    me->next_beat = plat_mtime() + hb_period();
    for (;;) {
        hart0_pass();
        if (plat_mtime() >= me->next_beat) {
            me->next_beat = plat_mtime() + hb_period();
            heartbeat(0);
        }
    }
#else
    csr_write(mie, plat_doorbell_mie());
    timer_rearm(me);
    for (;;) {
        (void)plat_doorbell_ack(); /* acknowledge first, then look: a doorbell after this stays pending */
        if (csr_read(mip) & MIP_MTIP) {
            timer_rearm(me);
        }
        hart0_pass();
        heartbeat(0);
        wfi();
        counter_add(0, L2CPU_CNT_WAKEUPS, 1);
        if ((csr_read(mip) & plat_doorbell_mie()) == 0) {
            counter_add(0, L2CPU_CNT_SPURIOUS, 1);
        }
    }
#endif
}

/* ---- boot ------------------------------------------------------------------------------------------------- */
/* WARM boot (hart 0, before releasing the others): keep everything the application and the host rely on (the
 * application windows, mailbox counters, log, counters, heartbeats, error word); reset what describes the running
 * image and the dispatch generation. */
static void warm_reset(void) {
    l2cpu_header_t* h = g_hdr;
    wr32(&h->fw_status.v, L2CPU_FW_STATUS_BOOTING);
    wr32(&h->ident.magic, 0);
    for (uint32_t w = 0; w < L2CPU_NHARTS; w++) {
        wr32(&h->hart_state[w].status, L2CPU_HART_NONE);
        wr32(&h->hart_state[w].error, L2CPU_ERR_NONE);
        wr32(&h->inject[w].v, L2CPU_INJECT_NONE);
    }
    /* A mailbox command issued before the restart (e.g. the L1 PARK a host sent to a hung hart 0 before falling back
     * to RNMI) belongs to the previous image: acknowledge it as stale instead of running it (a stale PARK would park
     * the new image right after READY). */
    l2cpu_mailbox_t* mb = (l2cpu_mailbox_t*)(g_region + L2CPU_OFF_MAILBOX);
    uint32_t req = rd32(&mb->req.v);
    if (req != rd32(&mb->ack.v)) {
        wr32(&mb->status, L2CPU_MB_ERR_STALE);
        fence();
        wr32(&mb->ack.v, req);
        stale_mb = 1;
        stale_mb_req = req;
    }
    fence();
}

/* Every boot starts a new dispatch generation: any work item the application (re)dispatches is a change for
 * every worker, including the item a worker had finished just before a restart. */
static void reset_dispatch(void) {
    for (uint32_t w = 0; w < L2CPU_NHARTS; w++) {
        wr32(&g_hdr->work_seq[w].v, L2CPU_WORK_NONE);
        wr32(&g_hdr->work_done[w].v, L2CPU_WORK_NONE);
    }
    fence();
}

static void boot_hart0(uint8_t* region) {
    l2cpu_header_t* h = g_hdr;
    wr32(&h->fw_status.v, L2CPU_FW_STATUS_BOOTING);
    log_init();
    l2cpu_ident_t* id = &h->ident;
    wr32(&id->layout_version, L2CPU_LAYOUT_VERSION);
    wr32(&id->fw_version, FW_VERSION);
    wr32(
        &id->build_flags,
        plat_build_flags() | (L2CPU_NOTIFY_POLL ? FW_BUILD_POLL : 0) | (L2CPU_TEST ? FW_BUILD_TEST : 0) |
            (g_resident ? L2CPU_BUILD_RESTART : 0) | (app_build_flags() << L2CPU_BUILD_APP_SHIFT));
    wr32(&id->n_harts, L2CPU_NHARTS);
    wr32(&id->app_id, app_id());
    wr32(&id->app_layout_version, app_layout_version());
    wr32(&id->mtime_hz, plat_mtime_hz());
    wr64(&id->image_base, (uint64_t)(uintptr_t)_image_start);
    wr64(&id->region_base, (uint64_t)(uintptr_t)region);
    wr32(&id->boot_mtime_lo, (uint32_t)plat_mtime());
    wr32(&id->boot_epoch, g_resident ? g_boot_epoch : 0);
    wr32(&id->boot_mode, g_boot_mode);
    wr32(&id->restart_count, g_boot_mode == L2CPU_BOOT_WARM ? rd32(&id->restart_count) + 1 : 0);
    plat_boot_init();
    fw_log(
        "l2cpu fw %x layout %u app %u image %p region %p flags 0x%x",
        FW_VERSION,
        L2CPU_LAYOUT_VERSION,
        app_id(),
        _image_start,
        region,
        rd32(&id->build_flags));
    fw_log(
        "boot epoch %u %s restart %u, resident %p",
        g_boot_epoch,
        g_boot_mode == L2CPU_BOOT_WARM ? "WARM" : "COLD",
        rd32(&id->restart_count),
        g_resident);
    if (stale_mb) {
        fw_log("mailbox request %u from before the restart dropped (stale)", stale_mb_req);
    }
    app_init(g_boot_mode);

    /* Wait (bounded) for the workers to reach their loops so READY means "all harts serve". */
    uint64_t deadline = plat_mtime() + fw_mtime_ticks(L2CPU_BOOT_WORKERS_US);
    for (uint32_t w = 1; w < L2CPU_NHARTS; w++) {
        while (rd32(&h->hart_state[w].status) < L2CPU_HART_IDLE && plat_mtime() < deadline) {
        }
        if (rd32(&h->hart_state[w].status) < L2CPU_HART_IDLE) {
            fw_log("hart %u did not report in", w);
        }
    }
    set_hart_status(0, L2CPU_HART_IDLE);
    fence();
    wr32(&id->magic, L2CPU_MAGIC);
    fence();
    wr32(&h->fw_status.v, L2CPU_FW_STATUS_READY); /* last: the host waits for this */
    fence();
    fw_log("ready");
}

void fw_main(uint64_t hartid, uint8_t* region, uint64_t epoch) {
    uint32_t hart = (uint32_t)hartid;
    hart_local_t* me = self();
    me->hartid = hart;
    g_hdr = (l2cpu_header_t*)region; /* every hart computes the same values */
    g_region = region;
    {
        uint8_t* res = region + L2CPU_OFF_RESIDENT;
        int ok =
            rd32(res + L2CPU_RES_MAGIC) == L2CPU_RES_MAGIC_LO && rd32(res + L2CPU_RES_MAGIC + 4) == L2CPU_RES_MAGIC_HI;
        g_resident = ok ? res : 0;
        g_boot_epoch = (uint32_t)epoch;
        g_boot_mode = ok && rd32(res + L2CPU_RES_BOOT_MODE) == L2CPU_BOOT_WARM ? L2CPU_BOOT_WARM : L2CPU_BOOT_COLD;
    }
    /* PMP policy (boot record flag): before this hart touches anything outside the region. A bad table or a locked
     * set that differs from it parks the hart (resident record: pmp = 0x100 | entry). */
    if (fw_pmp_init(hart, region) < 0) {
        csr_write(mcause, 0);
        csr_write(mepc, 0);
        fw_park();
    }

    /* A COLD boot zeroes [region, region + 2 MiB): the region is a fresh DRAM buffer and the loader does not clear
     * it. Host rule: write nothing into the region before fw_status == READY. */
    if (hart == 0) {
        if (g_boot_mode == L2CPU_BOOT_WARM) {
            warm_reset();
        } else {
            memset(region, 0, L2CPU_COLD_ZERO_SIZE);
        }
        reset_dispatch();
        fence();
        region_ready = g_boot_epoch;
        fence();
    } else {
        uint64_t t0 = plat_mtime();
        while (region_ready != g_boot_epoch) { /* bounded */
            if (plat_mtime() - t0 > fw_mtime_ticks(L2CPU_REGION_GATE_US)) {
                fw_error_park(L2CPU_ERR_BOOT_TIMEOUT, hart, 0);
            }
        }
        fence();
    }

#if L2CPU_TEST
    if (hart == L2CPU_NHARTS) {
        test_driver_main();
    }
#endif
    if (hart >= L2CPU_NHARTS) {
        fw_park();
    }

    me->heap = region + L2CPU_OFF_HEAP + (uint64_t)hart * L2CPU_HEAP_PER_HART;
    plat_hart_init(hart);
    set_hart_status(hart, L2CPU_HART_BOOTING);
    heartbeat(hart);
    fence();
    if (hart == 0) {
        boot_hart0(region);
        hart0_loop();
    } else {
        worker_loop(hart);
    }
}
