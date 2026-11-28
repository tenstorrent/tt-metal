// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/* mailbox.c: host -> firmware debug commands, served by hart 0 (idle loop / poll loop).
 * Every raw memory access goes through the guarded safe_* helpers, so a bad address returns
 * L2CPU_MB_ERR_FAULT with mcause instead of parking hart 0. */
#include "fw.h"
#include "platform.h"
#include "app.h"
static int pending_park;
static uint32_t pending_self_kind;

/* Guarded CSR read: the trap handler resumes at 1: with a1 = mcause. */
#define SAFE_CSRR(csr, out, err)                                                                \
    do {                                                                                        \
        register uint64_t _a0 __asm__("a0");                                                    \
        register uint64_t _a1 __asm__("a1");                                                    \
        __asm__ volatile("lla t0, 1f\n\tsd t0, 0(tp)\n\tli a1, 0\n\tli a0, 0\n\tcsrr a0, " #csr \
                         "\n"                                                                   \
                         "1:\n\tsd zero, 0(tp)"                                                 \
                         : "=r"(_a0), "=r"(_a1)                                                 \
                         :                                                                      \
                         : "t0", "memory");                                                     \
        (out) = _a0;                                                                            \
        (err) = _a1;                                                                            \
    } while (0)

static int csr_read_any(uint32_t csr, uint64_t* v, uint64_t* err) {
    uint64_t o = 0, e = 0;
    switch (csr) {
        case 0x300: SAFE_CSRR(mstatus, o, e); break;
        case 0x301: SAFE_CSRR(misa, o, e); break;
        case 0x304: SAFE_CSRR(mie, o, e); break;
        case 0x305: SAFE_CSRR(mtvec, o, e); break;
        case 0x306: SAFE_CSRR(mcounteren, o, e); break;
        case 0x320: SAFE_CSRR(0x320, o, e); break; /* mcountinhibit */
        case 0x340: SAFE_CSRR(mscratch, o, e); break;
        case 0x341: SAFE_CSRR(mepc, o, e); break;
        case 0x342: SAFE_CSRR(mcause, o, e); break;
        case 0x343: SAFE_CSRR(mtval, o, e); break;
        case 0x344: SAFE_CSRR(mip, o, e); break;
        case 0x3a0: SAFE_CSRR(pmpcfg0, o, e); break;
        case 0x7c0: SAFE_CSRR(0x7c0, o, e); break; /* SiFive custom (feature enable/disable) */
        case 0x7c1: SAFE_CSRR(0x7c1, o, e); break;
        case 0x7c2: SAFE_CSRR(0x7c2, o, e); break;
        case 0xb00: SAFE_CSRR(mcycle, o, e); break;
        case 0xb02: SAFE_CSRR(minstret, o, e); break;
        case 0xf11: SAFE_CSRR(mvendorid, o, e); break;
        case 0xf12: SAFE_CSRR(marchid, o, e); break;
        case 0xf13: SAFE_CSRR(mimpid, o, e); break;
        case 0xf14: SAFE_CSRR(mhartid, o, e); break;
        case 0x003: SAFE_CSRR(fcsr, o, e); break;
        case 0xc22: SAFE_CSRR(0xc22, o, e); break; /* vlenb */
        case 0x350: SAFE_CSRR(0x350, o, e); break; /* x280 RNMI CSRs (ISA RNMIs.md): mnscratch */
        case 0x351: SAFE_CSRR(0x351, o, e); break; /* mnepc */
        case 0x352: SAFE_CSRR(0x352, o, e); break; /* mncause */
        case 0x353: SAFE_CSRR(0x353, o, e); break; /* mnstatus */
        case 0x3b0: SAFE_CSRR(pmpaddr0, o, e); break;
        default: return -1;
    }
    *v = o;
    *err = e;
    return 0;
}

int mailbox_poll(void) {
    l2cpu_mailbox_t* mb = (l2cpu_mailbox_t*)(g_region + L2CPU_OFF_MAILBOX);
    uint32_t req = rd32(&mb->req.v);
    if (req == rd32(&mb->ack.v)) {
        return 0;
    }
    fence(); /* consumer: req first, then cmd/args */
    uint32_t cmd = rd32(&mb->cmd);
    uint64_t arg[7];
    for (int i = 0; i < 7; i++) {
        arg[i] = rd64(&mb->arg[i]);
    }
    uint64_t rep[7] = {0, 0, 0, 0, 0, 0, 0};
    uint32_t st = L2CPU_MB_OK;
    safe_ret_t r;

    switch (cmd) {
        case L2CPU_MB_PING:
            rep[0] = FW_VERSION;
            rep[1] = L2CPU_LAYOUT_VERSION;
            rep[2] = csr_read(mhartid);
            rep[3] = (uint64_t)(uintptr_t)_image_start;
            rep[4] = app_id();
            break;
        case L2CPU_MB_PEEK32:
            r = safe_lw32(arg[0]);
            rep[0] = r.v;
            rep[1] = r.err;
            if (r.err) {
                st = L2CPU_MB_ERR_FAULT;
            }
            break;
        case L2CPU_MB_PEEK64:
            r = safe_ld64(arg[0]);
            rep[0] = r.v;
            rep[1] = r.err;
            if (r.err) {
                st = L2CPU_MB_ERR_FAULT;
            }
            break;
        case L2CPU_MB_POKE32:
            r = safe_sw32(arg[0], (uint32_t)arg[1]);
            rep[1] = r.err;
            if (r.err) {
                st = L2CPU_MB_ERR_FAULT;
            }
            break;
        case L2CPU_MB_POKE64:
            r = safe_sd64(arg[0], arg[1]);
            rep[1] = r.err;
            if (r.err) {
                st = L2CPU_MB_ERR_FAULT;
            }
            break;
        case L2CPU_MB_COPY32:
        case L2CPU_MB_FILL32:
        case L2CPU_MB_MEMCMP32: {
            uint64_t a0 = arg[0], a1 = arg[1], n = arg[2];
            rep[0] = ~0ull;
            if ((n & 3) || (a0 & 3) || (cmd != L2CPU_MB_FILL32 && (a1 & 3)) || n > L2CPU_MB_MAX_BYTES) {
                st = L2CPU_MB_ERR_ARG;
                break;
            }
            for (uint64_t i = 0; i < n; i += 4) {
                if (cmd == L2CPU_MB_FILL32) {
                    r = safe_sw32(a0 + i, (uint32_t)a1);
                } else {
                    r = safe_lw32(a1 + i);
                    if (r.err) {
                        break;
                    }
                    if (cmd == L2CPU_MB_COPY32) {
                        r = safe_sw32(a0 + i, (uint32_t)r.v);
                    } else {
                        uint32_t bv = (uint32_t)r.v;
                        r = safe_lw32(a0 + i);
                        if (!r.err && (uint32_t)r.v != bv) {
                            uint32_t x = (uint32_t)r.v ^ bv, k = 0;
                            while (!(x & 0xFFu)) {
                                x >>= 8;
                                k++;
                            }
                            rep[0] = i + k;
                            break;
                        }
                    }
                }
                if (r.err) {
                    break;
                }
            }
            rep[1] = r.err;
            if (r.err) {
                st = L2CPU_MB_ERR_FAULT;
            }
            fence();
            break;
        }
        case L2CPU_MB_CSR_READ: {
            uint64_t err = 0;
            if (csr_read_any((uint32_t)arg[0], &rep[0], &err)) {
                st = L2CPU_MB_ERR_ARG;
            } else if (err) {
                st = L2CPU_MB_ERR_FAULT;
            }
            rep[1] = err;
            break;
        }
        case L2CPU_MB_NOC_READ32:
        case L2CPU_MB_NOC_WRITE32: {
            void* p = plat_noc_map(plat_slot(0, PLAT_SLOT_MAILBOX), (uint8_t)arg[0], (uint8_t)arg[1], 0, arg[2], 4);
            if (!p) {
                st = L2CPU_MB_ERR_ARG;
                break;
            }
            rep[2] = (uint64_t)(uintptr_t)p;
            if (cmd == L2CPU_MB_NOC_READ32) {
                plat_noc_read_prepare(plat_slot(0, PLAT_SLOT_MAILBOX), p, 4);
                r = safe_lw32((uint64_t)(uintptr_t)p);
                rep[0] = r.v;
            } else {
                r = safe_sw32((uint64_t)(uintptr_t)p, (uint32_t)arg[3]);
                plat_noc_write_barrier(plat_slot(0, PLAT_SLOT_MAILBOX));
            }
            rep[1] = r.err;
            if (r.err) {
                st = L2CPU_MB_ERR_FAULT;
            }
            break;
        }
        case L2CPU_MB_INJECT:
            if (arg[0] >= L2CPU_NHARTS) {
                st = L2CPU_MB_ERR_ARG;
                break;
            }
            if (arg[1] != 0 && arg[1] != L2CPU_INJECT_SPIN && arg[1] != L2CPU_INJECT_WFIPARK) {
                st = L2CPU_MB_ERR_ARG;
                break;
            }
            if (arg[0] == 0) {
                pending_self_kind = arg[1] ? (uint32_t)arg[1] : L2CPU_INJECT_ILLEGAL; /* after the reply */
            } else {
                wr32(&g_hdr->inject[arg[0]].v, arg[1] ? (uint32_t)arg[1] : L2CPU_INJECT_ILLEGAL);
                fence();
                plat_ipi_send((uint32_t)arg[0]);
            }
            break;
        case L2CPU_MB_PARK:
            if (!g_resident) {
                st = L2CPU_MB_ERR_ARG; /* no resident page: nowhere to park */
                break;
            }
            pending_park = 1;
            break;
        case L2CPU_MB_TIME:
            rep[0] = plat_mtime();
            rep[1] = rdcycle64();
            break;
        default: st = cmd >= L2CPU_MB_APP ? app_mailbox(cmd, arg, rep) : L2CPU_MB_ERR_CMD;
    }

    for (int i = 0; i < 7; i++) {
        wr64(&mb->reply[i], rep[i]);
    }
    wr32(&mb->status, st);
    fence(); /* producer: reply, then ack */
    wr32(&mb->ack.v, req);
    fence();
    if (pending_self_kind) {
        wr32(&g_hdr->inject[0].v, pending_self_kind);
        pending_self_kind = 0;
        fence();
    }
    if (pending_park) {
        pending_park = 0;
        fw_park_all(); /* does not return */
    }
    return 1;
}
