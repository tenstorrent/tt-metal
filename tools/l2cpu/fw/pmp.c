// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

/*
 * pmp.c: the PMP policy (README "PMP policy"). The host computes the entries (host/l2cpu/pmp.py), writes them into
 * the resident page (L2CPU_RES_PMP) and sets the boot record's PMP flag before the release. Every firmware hart, at
 * the top of fw_main and before it touches anything outside the region, writes the entries with the L bit (locked
 * entries also check M-mode and stay until the next chip reset), reads them back and compares with the table. A
 * restart in the same chip epoch finds them already locked: the writes are ignored and the read-back proves that
 * the table still describes the locked set.
 *
 * Flag off (or no flag word, as written by every earlier loader): nothing here touches a CSR and fw_pmp_allows()
 * accepts every address, so the firmware behaves as before.
 */
#include "fw.h"
#include "platform.h"

static uint32_t g_pmp_n; /* 0 = policy off */
static const uint64_t* g_pmp_addr;
static const uint8_t* g_pmp_cfg;

#define PMP_A_SHIFT 3
#define PMP_A_TOR 1u
#define PMP_A_NA4 2u
#define PMP_A_NAPOT 3u

#define REP4(M, b) M((b) + 0) M((b) + 1) M((b) + 2) M((b) + 3)
#define REP16(M, b) REP4(M, b) REP4(M, (b) + 4) REP4(M, (b) + 8) REP4(M, (b) + 12)
#define ADDR_WR(n) \
    case (n): __asm__ volatile("csrw %0, %1" ::"i"(0x3b0 + (n)), "r"(v)); break;
#define ADDR_RD(n) \
    case (n): __asm__ volatile("csrr %0, %1" : "=r"(v) : "i"(0x3b0 + (n))); break;

_Static_assert(L2CPU_PMP_MAX == 16, "pmpaddr0..15, pmpcfg0 + pmpcfg2 (RV64)");

static void pmpaddr_write(uint32_t i, uint64_t v) {
    switch (i) {
        REP16(ADDR_WR, 0)
        default: break;
    }
}

static uint64_t pmpaddr_read(uint32_t i) {
    uint64_t v = 0;
    switch (i) {
        REP16(ADDR_RD, 0)
        default: break;
    }
    return v;
}

static uint64_t cfg_word(const uint8_t* cfg, uint32_t n, uint32_t first) {
    uint64_t w = 0;
    for (uint32_t k = 0; k < 8 && first + k < n; k++) {
        w |= (uint64_t)cfg[first + k] << (8 * k);
    }
    return w;
}

static uint32_t* pmp_rec(uint8_t* region, uint32_t hart) {
    return (uint32_t*)(region + L2CPU_OFF_RESIDENT + L2CPU_RES_REC + L2CPU_REC_SIZE * hart + L2CPU_REC_PMP);
}

int fw_pmp_init(uint32_t hart, uint8_t* region) {
    if (hart >= L2CPU_NHARTS ||
        rd32((const void*)(uintptr_t)(plat_boot_record() + L2CPU_BOOT_PMP)) != L2CPU_PMP_MAGIC) {
        return L2CPU_PMP_STATE_OFF;
    }
    const uint8_t* t = region + L2CPU_OFF_RESIDENT;
    uint32_t n = rd32(t + L2CPU_RES_PMP_N);
    uint32_t* rec = pmp_rec(region, hart);
    if (rd32(t + L2CPU_RES_PMP) != L2CPU_PMP_MAGIC || n == 0 || n > L2CPU_PMP_MAX) {
        wr32(rec, 0x100u | 0xFFu);
        return -1;
    }
    const uint64_t* addr = (const uint64_t*)(t + L2CPU_RES_PMP_ADDR);
    const uint8_t* cfg = t + L2CPU_RES_PMP_CFG;
    uint64_t c0 = csr_read(pmpcfg0);
    int reused = (c0 & FW_PMP_L) != 0; /* entry 0 is always locked under the policy */
    if (!reused) {
        /* addresses first (a locked TOR entry also locks the address below it); then the configuration words in
         * ascending order: the table keeps its deny-all last, so the code that runs here stays allowed throughout */
        for (uint32_t i = 0; i < n; i++) {
            pmpaddr_write(i, addr[i]);
        }
        csr_write(pmpcfg0, cfg_word(cfg, n, 0));
        if (n > 8) {
            csr_write(pmpcfg2, cfg_word(cfg, n, 8));
        }
        __asm__ volatile("sfence.vma" ::: "memory"); /* PMP changes take effect for later accesses (priv spec) */
    }
    uint64_t got0 = csr_read(pmpcfg0), got2 = n > 8 ? csr_read(pmpcfg2) : 0;
    for (uint32_t i = 0; i < n; i++) {
        uint8_t c = (uint8_t)((i < 8 ? got0 >> (8 * i) : got2 >> (8 * (i - 8))) & 0xFF);
        if (c != cfg[i] || pmpaddr_read(i) != addr[i]) {
            wr32(rec, 0x100u | i);
            return -1;
        }
    }
    g_pmp_n = n;
    g_pmp_addr = addr;
    g_pmp_cfg = cfg;
    uint32_t st = reused ? L2CPU_PMP_STATE_REUSED : L2CPU_PMP_STATE_APPLIED;
    wr32(rec, st);
    fence();
    return (int)st;
}

int fw_pmp_on(void) { return g_pmp_n != 0; }

/* Same decision as the hardware for an M-mode access of `len` bytes at `a` (perm: FW_PMP_R / FW_PMP_W / FW_PMP_X): the
 * lowest-numbered entry that matches any byte decides, and it must cover every byte. Policy off: allowed. */
int fw_pmp_allows(uint64_t a, uint64_t len, uint32_t perm) {
    if (!g_pmp_n) {
        return 1;
    }
    if (len == 0 || a + len < a) {
        return 0;
    }
    uint64_t last = a + len - 1;
    for (uint32_t i = 0; i < g_pmp_n; i++) {
        uint32_t c = g_pmp_cfg[i], mode = (c >> PMP_A_SHIFT) & 3u;
        uint64_t lo, hi; /* [lo, hi] inclusive */
        if (mode == 0) {
            continue;
        }
        if (mode == PMP_A_TOR) {
            lo = i ? g_pmp_addr[i - 1] << 2 : 0;
            if ((g_pmp_addr[i] << 2) <= lo) {
                continue; /* empty range matches nothing */
            }
            hi = (g_pmp_addr[i] << 2) - 1;
        } else if (mode == PMP_A_NA4) {
            lo = g_pmp_addr[i] << 2;
            hi = lo + 3;
        } else {
            uint64_t v = g_pmp_addr[i];
            uint32_t ones = 0;
            while (ones < 64 && ((v >> ones) & 1)) {
                ones++;
            }
            if (ones >= 61) {
                lo = 0;
                hi = ~0ull;
            } else {
                lo = (v & ~((1ull << ones) - 1)) << 2;
                hi = lo + (1ull << (ones + 3)) - 1;
            }
        }
        int first_in = a >= lo && a <= hi, last_in = last >= lo && last <= hi;
        if (!first_in && !last_in && !(a < lo && last > hi)) {
            continue;
        }
        return first_in && last_in && (c & perm) == perm;
    }
    return 1; /* no entry matches: M-mode access succeeds */
}
