// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/*
 * examples/heartbeat: the default image. It adds nothing to the runtime but a tiny "echo" work item so that the
 * doorbell, the worker dispatch and restart-in-the-middle-of-work can be exercised end to end:
 *
 *   APP_CTRL + 0x000  req   (host: u32 sequence, written after `value`)        own line
 *   APP_CTRL + 0x040  done  (firmware: = req when every result is written)       own line
 *   APP_CTRL + 0x080  value (host: u32)                                          own line
 *   APP_CTRL + 0x100 + 64*h  result[h] = value + h (hart h computed it)          own line each
 *
 * Host: write value, fence, req = req + 1, ring the doorbell (or let the 1 ms timer wake hart 0), poll done == req.
 * Mailbox command L2CPU_MB_APP (64): reply0 = arg0 + 1.
 */
#include "app.h"
#include "fw.h"

#define ECHO_REQ (L2CPU_OFF_APP_CTRL + 0x000)
#define ECHO_DONE (L2CPU_OFF_APP_CTRL + 0x040)
#define ECHO_VALUE (L2CPU_OFF_APP_CTRL + 0x080)
#define ECHO_RESULT (L2CPU_OFF_APP_CTRL + 0x100) /* + 64*h */
#define ECHO_WORKERS 0xEu                        /* harts 1..3 */

static uint32_t served;

uint32_t app_id(void) { return 1; }
uint32_t app_layout_version(void) { return 1; }
uint32_t app_build_flags(void) { return 0; }

void app_init(uint32_t boot_mode) {
    (void)boot_mode;
    served = rd32(g_region + ECHO_DONE); /* COLD: 0; WARM: continue where the last image stopped */
}

static void echo_result(uint32_t hart) {
    uint32_t v = rd32(g_region + ECHO_VALUE);
    wr32(g_region + ECHO_RESULT + 64 * hart, v + hart);
}

int app_poll(void) {
    uint32_t seq = rd32(g_region + ECHO_REQ);
    if (seq == served) {
        return 0;
    }
    fence(); /* consumer: sequence first, then the value */
    fw_dispatch(ECHO_WORKERS, seq);
    echo_result(0);
    if (fw_wait_workers(ECHO_WORKERS, seq)) {
        served = seq; /* a worker is parked: not published (L2CPU_ERR_WORKER_DEAD recorded), not retried */
        return 1;
    }
    fence(); /* producer: results, then done */
    wr32(g_region + ECHO_DONE, seq);
    fence();
    served = seq;
    return 1;
}

void app_work(uint32_t hart, uint32_t seq) {
    (void)seq;
    echo_result(hart);
}

uint32_t app_mailbox(uint32_t cmd, const uint64_t* arg, uint64_t* rep) {
    if (cmd != L2CPU_MB_APP) {
        return L2CPU_MB_ERR_CMD;
    }
    rep[0] = arg[0] + 1;
    return L2CPU_MB_OK;
}
