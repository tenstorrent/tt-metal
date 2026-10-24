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

/* Guarded CSR read / write: the trap handler resumes at 1: with a1 = mcause. The CSR number is an assembler
 * immediate ("i"), so every allowed number is its own case below. */
#define SAFE_CSRR(csr, out, err)                                                  \
    do {                                                                          \
        register uint64_t _a0 __asm__("a0");                                      \
        register uint64_t _a1 __asm__("a1");                                      \
        __asm__ volatile(                                                         \
            "lla t0, 1f\n\tsd t0, 0(tp)\n\tli a1, 0\n\tli a0, 0\n\tcsrr a0, %2\n" \
            "1:\n\tsd zero, 0(tp)"                                                \
            : "=r"(_a0), "=r"(_a1)                                                \
            : "i"(csr)                                                            \
            : "t0", "memory");                                                    \
        (out) = _a0;                                                              \
        (err) = _a1;                                                              \
    } while (0)
#define SAFE_CSRW(csr, val, err)                                                   \
    do {                                                                           \
        register uint64_t _a1 __asm__("a1");                                       \
        __asm__ volatile(                                                          \
            "lla t0, 1f\n\tsd t0, 0(tp)\n\tli a1, 0\n\tcsrw %1, %2\n"              \
            "1:\n\tsd zero, 0(tp)"                                                 \
            : "=&r"(_a1) /* early clobber: val must not share a1 (zeroed first) */ \
            : "i"(csr), "r"(val)                                                   \
            : "t0", "memory");                                                     \
        (err) = _a1;                                                               \
    } while (0)
#define REP4(M, b) M((b) + 0) M((b) + 1) M((b) + 2) M((b) + 3)
#define REP16(M, b) REP4(M, b) REP4(M, (b) + 4) REP4(M, (b) + 8) REP4(M, (b) + 12)
#define REP64(M, b) REP16(M, b) REP16(M, (b) + 16) REP16(M, (b) + 32) REP16(M, (b) + 48)
#define CASE_RD(n) \
    case (n): SAFE_CSRR(n, o, e); break;
#define CASE_WR(n) \
    case (n): SAFE_CSRW(n, v, e); break;

static int csr_read_any(uint32_t csr, uint64_t* v, uint64_t* err) {
    uint64_t o = 0, e = 0;
    switch (csr) {
        CASE_RD(0x300)        /* mstatus */
        CASE_RD(0x301)        /* misa */
        CASE_RD(0x304)        /* mie */
        CASE_RD(0x305)        /* mtvec */
        CASE_RD(0x306)        /* mcounteren */
        CASE_RD(0x320)        /* mcountinhibit */
        CASE_RD(0x340)        /* mscratch */
        CASE_RD(0x341)        /* mepc */
        CASE_RD(0x342)        /* mcause */
        CASE_RD(0x343)        /* mtval */
        CASE_RD(0x344)        /* mip */
        REP16(CASE_RD, 0x3a0) /* pmpcfg0..15 (RV64: odd numbers are illegal) */
        REP64(CASE_RD, 0x3b0) /* pmpaddr0..63 */
        CASE_RD(0x747)        /* mseccfg (Smepmp) */
        CASE_RD(0x7c0)        /* SiFive custom (feature enable/disable) */
        CASE_RD(0x7c1)
        CASE_RD(0x7c2)
        CASE_RD(0xb00) /* mcycle */
        CASE_RD(0xb02) /* minstret */
        CASE_RD(0xf11) /* mvendorid */
        CASE_RD(0xf12) /* marchid */
        CASE_RD(0xf13) /* mimpid */
        CASE_RD(0xf14) /* mhartid */
        CASE_RD(0x003) /* fcsr */
        CASE_RD(0xc22) /* vlenb */
        CASE_RD(0x350) /* x280 RNMI CSRs (ISA RNMIs.md): mnscratch */
        CASE_RD(0x351) /* mnepc */
        CASE_RD(0x352) /* mncause */
        CASE_RD(0x353) /* mnstatus */
        default: return -1;
    }
    *v = o;
    *err = e;
    return 0;
}

/* CSR_WRITE: PMP CSRs only (hardware probe of entry count, granularity and lock behaviour). Locked entries ignore
 * writes (that is what the probe checks); with the PMP policy on every entry is locked. */
static int csr_write_pmp(uint32_t csr, uint64_t v, uint64_t* err) {
    uint64_t e = 0;
    switch (csr) {
        REP16(CASE_WR, 0x3a0)
        REP64(CASE_WR, 0x3b0)
        default: return -1;
    }
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
            if (!fw_pmp_allows(arg[0], 4, FW_PMP_R)) {
                st = L2CPU_MB_ERR_DENIED;
                break;
            }
            r = safe_lw32(arg[0]);
            rep[0] = r.v;
            rep[1] = r.err;
            if (r.err) {
                st = L2CPU_MB_ERR_FAULT;
            }
            break;
        case L2CPU_MB_PEEK64:
            if (!fw_pmp_allows(arg[0], 8, FW_PMP_R)) {
                st = L2CPU_MB_ERR_DENIED;
                break;
            }
            r = safe_ld64(arg[0]);
            rep[0] = r.v;
            rep[1] = r.err;
            if (r.err) {
                st = L2CPU_MB_ERR_FAULT;
            }
            break;
        case L2CPU_MB_POKE32:
            if (!fw_pmp_allows(arg[0], 4, FW_PMP_W)) {
                st = L2CPU_MB_ERR_DENIED;
                break;
            }
            r = safe_sw32(arg[0], (uint32_t)arg[1]);
            rep[1] = r.err;
            if (r.err) {
                st = L2CPU_MB_ERR_FAULT;
            }
            break;
        case L2CPU_MB_POKE64:
            if (!fw_pmp_allows(arg[0], 8, FW_PMP_W)) {
                st = L2CPU_MB_ERR_DENIED;
                break;
            }
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
            if (n && (!fw_pmp_allows(a0, n, cmd == L2CPU_MB_MEMCMP32 ? FW_PMP_R : FW_PMP_W) ||
                      (cmd != L2CPU_MB_FILL32 && !fw_pmp_allows(a1, n, FW_PMP_R)))) {
                st = L2CPU_MB_ERR_DENIED;
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
        case L2CPU_MB_CSR_WRITE: {
            uint64_t err = 0;
            if (csr_write_pmp((uint32_t)arg[0], arg[1], &err)) {
                st = L2CPU_MB_ERR_ARG;
            } else if (err) {
                st = L2CPU_MB_ERR_FAULT;
            } else {
                (void)csr_read_any((uint32_t)arg[0], &rep[0], &err); /* read-back (WARL) */
            }
            rep[1] = err;
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
            if (arg[1] == L2CPU_INJECT_LOAD || arg[1] == L2CPU_INJECT_STORE || arg[1] == L2CPU_INJECT_JUMP) {
                if (!fw_pmp_on()) {
                    st = L2CPU_MB_ERR_ARG; /* without the policy such an access may hang the chip */
                    break;
                }
                wr64((uint8_t*)&g_hdr->inject[arg[0]] + L2CPU_INJECT_ADDR, arg[2]);
                fence();
            } else if (arg[1] != 0 && arg[1] != L2CPU_INJECT_SPIN && arg[1] != L2CPU_INJECT_WFIPARK) {
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
