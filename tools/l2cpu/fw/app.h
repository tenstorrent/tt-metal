// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/*
 * app.h: the application hook of the L2CPU firmware runtime. An application (e.g. examples/heartbeat) provides
 * these functions; the runtime owns boot, the idle loops, heartbeats, the mailbox, traps, park and restart.
 *
 * Threading: app_init / app_poll / app_mailbox run on hart 0; app_work runs on worker hart h (1..3) after hart 0
 * dispatched a work item with fw_dispatch(). The application keeps its state in its region windows
 * (L2CPU_OFF_APP_CTRL, L2CPU_OFF_APP_LOW, L2CPU_OFF_APP_HIGH) and its .bss.
 */
#ifndef L2CPU_APP_H
#define L2CPU_APP_H

#include <stdint.h>

/* Constant application identity: written into ident.app_id / app_layout_version / build_flags (bits 8..). */
uint32_t app_id(void);
uint32_t app_layout_version(void);
uint32_t app_build_flags(void);

/* Hart 0, at every boot, after the runtime prepared the region and before READY. boot_mode: L2CPU_BOOT_COLD
 * (region [0, 2 MiB) zeroed) or L2CPU_BOOT_WARM (restart: the application windows are as the last image left them). */
void app_init(uint32_t boot_mode);

/* Hart 0, every pass of its loop (after a doorbell, an IPI-free timer wake, or continuously in poll builds).
 * Returns nonzero when it did work. */
int app_poll(void);

/* Worker h (1..3): run work item `seq` that hart 0 dispatched with fw_dispatch(). */
void app_work(uint32_t hart, uint32_t seq);

/* Hart 0: mailbox commands >= L2CPU_MB_APP. Fill rep[0..6]; return an L2CPU_MB_* status. */
uint32_t app_mailbox(uint32_t cmd, const uint64_t* arg, uint64_t* rep);

/* Runtime services for applications ------------------------------------------------------------------------- */
/* Hart 0: hand work item `seq` (any value but L2CPU_WORK_NONE) to the workers in `mask` (bits 1..3) and IPI them.
 * A seq equal to the previous one for that worker is ignored (use a fresh value per item; a new boot resets). */
void fw_dispatch(uint32_t mask, uint32_t seq);
/* Hart 0: wait until every worker in `mask` reported `seq`. Returns 0, or -1 if one of them is parked
 * (L2CPU_ERR_WORKER_DEAD, arg = that hart) or did not finish within the bound (L2CPU_ERR_WORK_TIMEOUT; default
 * L2CPU_WORK_TIMEOUT_US). Hart 0 stays alive in both cases. */
int fw_wait_workers(uint32_t mask, uint32_t seq);
int fw_wait_workers_us(uint32_t mask, uint32_t seq, uint64_t timeout_us);
/* Any hart: record an error (first error wins) without stopping; fw_error_park also parks the calling hart. */
void fw_set_error(uint32_t hart, uint32_t code, uint64_t arg, uint32_t context);
void fw_error_park(uint32_t code, uint64_t arg, uint32_t context) __attribute__((noreturn));
void fw_log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

#endif /* L2CPU_APP_H */
