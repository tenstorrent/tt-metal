// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

// Tensix data-movement helpers for reaching the L2CPU tile (SiFive x280 cluster) of Blackhole.
//
// The L2CPU tile's local GDDR is coherent with the x280 caches only through its Memory Port alias, x280 physical
// address 0x4000_3000_0000 + offset, which is also the NoC address at the tile (passthrough). That is a 47-bit
// local address. tt-metal's NoC address helpers carry 36 local bits (NOC_ADDR_LOCAL_BITS) and the command-buffer
// writers mask NOC_*_ADDR_MID with NOC_PCIE_MASK, so get_noc_addr() + noc_async_write() cannot express it. These
// helpers program the command buffer directly with the full upper 32 bits in MID. The low aliases (System Port
// 0x3000_0000 + offset at the tile, or the DRAM tile itself) reach the same memory but are NOT coherent with the
// x280 caches; use them only for regions the x280 reads uncached.
#pragma once
#include "api/dataflow/dataflow_api.h"
#include "l2cpu_link.h"

// One write of len bytes (<= NOC_MAX_BURST_SIZE) from L1 to (x, y, addr), addr 64-bit. Counted like
// noc_async_write, so noc_async_write_barrier() / noc_async_writes_flushed() cover it.
static inline void l2cpu_noc_write(uint32_t noc, uint32_t src_l1, uint32_t x, uint32_t y, uint64_t addr, uint32_t len) {
    while (!noc_cmd_buf_ready(noc, write_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(
        noc,
        write_cmd_buf,
        NOC_CTRL,
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | NOC_CMD_RESP_MARKED);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_TARG_ADDR_LO, src_l1);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_LO, (uint32_t)addr);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(addr >> 32));
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_RET_ADDR_COORDINATE, NOC_XY_ENCODING(x, y));
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_AT_LEN_BE, len);
    NOC_CMD_BUF_WRITE_REG(noc, write_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += 1;
}

// One read of len bytes from (x, y, addr), addr 64-bit, into L1. Covered by noc_async_read_barrier().
static inline void l2cpu_noc_read(uint32_t noc, uint32_t dst_l1, uint32_t x, uint32_t y, uint64_t addr, uint32_t len) {
    while (!noc_cmd_buf_ready(noc, read_cmd_buf));
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_RET_ADDR_LO, dst_l1);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_LO, (uint32_t)addr);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_MID, (uint32_t)(addr >> 32));
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_TARG_ADDR_COORDINATE, NOC_XY_ENCODING(x, y));
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_AT_LEN_BE, len);
    NOC_CMD_BUF_WRITE_REG(noc, read_cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    noc_reads_num_issued[noc] += 1;
}

static inline void l2cpu_noc_write_bulk(
    uint32_t noc, uint32_t src_l1, uint32_t x, uint32_t y, uint64_t addr, uint32_t len) {
    constexpr uint32_t CHUNK = 8192;
    while (len) {
        uint32_t n = len < CHUNK ? len : CHUNK;
        l2cpu_noc_write(noc, src_l1, x, y, addr, n);
        src_l1 += n;
        addr += n;
        len -= n;
    }
}

// ---- channel operations (layout in l2cpu_link.h) ----
struct l2cpu_link {
    uint32_t noc, x, y;
    uint64_t base;  // channel base: NoC address at (x, y) == x280 physical address (Memory Port alias)
    uint32_t l1;    // >= 1 KiB of L1 scratch, 64-byte aligned
};

static inline uint32_t l2cpu_link_read32(const l2cpu_link& c, uint32_t off, uint32_t slot) {
    uint32_t dst = c.l1 + 64 * slot;
    l2cpu_noc_read(c.noc, dst, c.x, c.y, c.base + off, 64);
    noc_async_read_barrier();
    return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
}

static inline void l2cpu_link_write32(const l2cpu_link& c, uint32_t off, uint32_t v, uint32_t slot) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c.l1 + 64 * slot);
    for (uint32_t i = 0; i < 16; i++) {
        p[i] = 0;
    }
    p[0] = v;
    l2cpu_noc_write(c.noc, c.l1 + 64 * slot, c.x, c.y, c.base + off, 64);
    noc_async_write_barrier();
}

// Producer: everything this core wrote before is acked, then req_seq += 1, then the doorbell (a non-zero u32 into
// the MSI catcher FIFO; the catcher raises PLIC source 6 while non-empty). Returns the new request number.
static inline uint32_t l2cpu_link_notify(const l2cpu_link& c) {
    noc_async_write_barrier();
    uint32_t r = l2cpu_link_read32(c, L2CPU_LINK_OFF_REQ_SEQ, 0) + 1;
    l2cpu_link_write32(c, L2CPU_LINK_OFF_REQ_SEQ, r, 1);
    volatile tt_l1_ptr uint32_t* b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c.l1 + 64 * 2);
    b[0] = r ? r : 1;  // a 0 would read back as "FIFO empty" on the x280 side
    noc_async_write(c.l1 + 64 * 2, get_noc_addr(c.x, c.y, (uint32_t)L2CPU_BH_MSI_NOC_ADDR), 4);
    noc_async_write_barrier();
    return r;
}

// Wall-clock ticks per microsecond used to turn a timeout in us into a tick budget. The RISC-V wall clock runs at
// the AI clock (measured 1350 MHz on P300 under load); at a lower AI clock the real timeout is longer, never
// shorter. Override with -DL2CPU_WAIT_TICKS_PER_US=<n>.
#ifndef L2CPU_WAIT_TICKS_PER_US
#define L2CPU_WAIT_TICKS_PER_US 1350u
#endif
#define L2CPU_WAIT_STATUS_TIMEOUT 0xDEAD0000u

// Consumer: poll done_seq == req_seq (reads the sequence first; the caller reads data only afterwards), bounded by
// timeout_us of wall-clock time (one poll is one 64 B NoC read of the tile, ~1 us). Returns req_seq, or 0 at the
// bound, having written L2CPU_WAIT_STATUS_TIMEOUT | (req & 0xFFFF) into the wait status word. Never hangs.
static inline uint32_t l2cpu_link_wait(const l2cpu_link& c, uint32_t timeout_us) {
    uint32_t req = l2cpu_link_read32(c, L2CPU_LINK_OFF_REQ_SEQ, 0);
    uint32_t t0 = get_timestamp_32b();
    uint32_t budget = timeout_us * L2CPU_WAIT_TICKS_PER_US;  // timeout_us <= 3,000,000 (32-bit tick wrap)
    for (;;) {
        if (l2cpu_link_read32(c, L2CPU_LINK_OFF_DONE_SEQ, 1) == req) {
            return req;
        }
        if (get_timestamp_32b() - t0 > budget) {
            break;
        }
    }
    l2cpu_link_write32(c, L2CPU_LINK_OFF_WAIT_STATUS, L2CPU_WAIT_STATUS_TIMEOUT | (req & 0xFFFFu), 3);
    return 0;
}
