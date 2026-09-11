// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// x280 (L2CPU) fabric-worker firmware.
//
// Waits for a Tensix setup kernel to deliver the router connection parameters into
// the mailbox (fabric_mbox.h), opens a persistent connection to the local fabric
// router with l2cpu_fabric.h, then serves two kinds of traffic until the chip is
// reset:
//   * requests posted into FF_MBOX_REQ (by the host through a Tensix kernel):
//     "send N bytes from x280 address S to (x, y, addr) H hops away", optionally
//     followed by a 16 B inbox header so the receiver knows a message landed;
//   * messages that arrive in this tile's own inbox (FF_MBOX_INBOX) from a peer
//     L2CPU: if FF_CFLAG_AUTO_ECHO is set they are echoed back to the peer's inbox
//     over this tile's own connection — L2CPU-to-L2CPU traffic with no host or
//     Tensix in the loop.
//
// Runtime model (as the sibling l2cpu_noc_transfer/x280/fw.c): hart 0 runs fw_main,
// harts 1-3 park; .text runs from cached GDDR (dram.ld); the mailbox is in the
// uncached GDDR alias so Tensix/host NOC writes and x280 loads agree without cache
// maintenance. The hart release is one-shot per chip reset, so this firmware never
// wedges: every wait is bounded (or heartbeats while waiting), faults are reported
// in FF_MBOX_FAULT_CODE and the request loop keeps running.

#include <stdint.h>

#include "fabric_mbox.h"
#include "l2cpu_fabric.h"

#define REG64(a) L2F_REG64(a)
#define REG32(a) L2F_REG32(a)

// Deadlines in mcycle ticks (the hart runs at the 200 MHz PLL solution x280_boot
// programs): 2 s is far longer than any router round trip and well inside the host's
// default 20 s wait, so a stuck router surfaces as a fault, not a silent hang.
#define SPIN_CAP 400000000ull
#define CLOSE_SPIN_CAP 400000000ull

static l2f_conn_t g_conn;
static uint32_t g_peer_x, g_peer_y, g_peer_inbox, g_peer_hops, g_flags;
static uint64_t g_hb;
// 16 B inbox-header scratch in the UNCACHED mailbox alias (FF_MBOX_FW_SCRATCH), so the
// send path reads exactly what was written without any cache interaction.
#define INBOX_HDR_SCRATCH FF_MBOX_FW_SCRATCH

static inline uint64_t read_mhartid(void) {
    uint64_t v;
    __asm__ volatile("csrr %0, mhartid" : "=r"(v));
    return v;
}

static inline void heartbeat(void) {
    REG64(FF_MBOX_HEARTBEAT) = ++g_hb;
    l2f_fence();
}
static inline void set_state(uint64_t s) {
    REG64(FF_MBOX_FW_STATE) = s;
    l2f_fence();
}
static inline void set_fault(uint64_t f) {
    REG64(FF_MBOX_FAULT_CODE) = f;
    l2f_fence();
}
static inline void diag(uint32_t off, uint32_t v) {
    REG32(FF_MBOX_DIAG + off) = v;
    l2f_fence();
}
static inline uint32_t conn_u32(uint32_t off) { return REG32(FF_MBOX_CONN + off); }

// Read the connection block into g_conn. Returns 0 if a value is implausible.
static int load_conn(void) {
    g_conn.edm_x = conn_u32(FF_CONN_EDM_NOC_X);
    g_conn.edm_y = conn_u32(FF_CONN_EDM_NOC_Y);
    g_conn.buffer_base = conn_u32(FF_CONN_BUFFER_BASE);
    g_conn.num_buffers = conn_u32(FF_CONN_NUM_BUFFERS);
    g_conn.buffer_size = conn_u32(FF_CONN_BUFFER_SIZE);
    g_conn.handshake_addr = conn_u32(FF_CONN_HANDSHAKE);
    g_conn.worker_loc_info = conn_u32(FF_CONN_WORKER_LOC_INFO);
    g_conn.wr_counter_addr = conn_u32(FF_CONN_WR_COUNTER);
    g_conn.sreg_write_addr = conn_u32(FF_CONN_SREG_WRITE);
    g_conn.sreg_read_addr = conn_u32(FF_CONN_SREG_READ);
    g_conn.self_x = conn_u32(FF_CONN_SELF_NOC_X);
    g_conn.self_y = conn_u32(FF_CONN_SELF_NOC_Y);
    g_conn.hdr_size = conn_u32(FF_CONN_HDR_SIZE);
    g_conn.freeslots_sink = conn_u32(FF_CONN_FREESLOTS_SINK);
    g_conn.teardown_word = conn_u32(FF_CONN_TEARDOWN_WORD);
    g_conn.write_counter = 0;
    g_conn.write_index = 0;
    g_conn.packets_sent = 0;
    g_conn.is_open = 0;

    g_peer_x = conn_u32(FF_CONN_PEER_NOC_X);
    g_peer_y = conn_u32(FF_CONN_PEER_NOC_Y);
    g_peer_inbox = conn_u32(FF_CONN_PEER_INBOX);
    g_peer_hops = conn_u32(FF_CONN_PEER_HOPS);
    g_flags = conn_u32(FF_CONN_FLAGS);

    if (g_conn.num_buffers == 0 || g_conn.num_buffers > 64) {
        return 0;
    }
    if (g_conn.hdr_size != 48 && g_conn.hdr_size != 64) {
        return 0;
    }
    if (g_conn.buffer_size <= g_conn.hdr_size || g_conn.buffer_size > 0x10000) {
        return 0;
    }
    // Eth L1 addresses (Blackhole eth L1 < 2 MiB) and stream-register addresses.
    if (g_conn.buffer_base == 0 || g_conn.buffer_base >= 0x200000 || g_conn.handshake_addr == 0 ||
        g_conn.handshake_addr >= 0x200000 || g_conn.worker_loc_info == 0 || g_conn.worker_loc_info >= 0x200000 ||
        g_conn.wr_counter_addr == 0 || g_conn.wr_counter_addr >= 0x200000) {
        return 0;
    }
    if ((g_conn.sreg_write_addr & 0xFFFC0000u) != 0xFFB40000u || (g_conn.sreg_read_addr & 0xFFFC0000u) != 0xFFB40000u) {
        return 0;
    }
    if (g_conn.freeslots_sink == 0 || g_conn.teardown_word == 0) {
        return 0;
    }
    return 1;
}

// Decide which coordinates the TLB window needs for the EDM core. The setup kernel
// wrote FF_CONN_CURSOR_MAGIC into the router's cursor pad word (a word the router
// never touches after bring-up) using the coordinates a Tensix uses; we read that
// word back through the window with the same coordinates. A match proves the
// window addresses the same tile the same way. Measured 2026-09-11 on two p150b
// chips: the translated coords match, i.e. the L2CPU tile's NOC port translates
// outbound coordinates like every other tile. The NOC0-physical fallback is only
// tried when the host supplies it (FF_CONN_EDM_NOC0_X != 0xFFFFFFFF): with
// translation on, a physical eth coordinate is an unmapped translated coordinate
// and a load to it may never return.
static void probe_coords(void) {
    const uint32_t magic = conn_u32(FF_CONN_CURSOR_MAGIC);
    const uint32_t tx = conn_u32(FF_CONN_EDM_NOC_X), ty = conn_u32(FF_CONN_EDM_NOC_Y);
    const uint32_t px = conn_u32(FF_CONN_EDM_NOC0_X), py = conn_u32(FF_CONN_EDM_NOC0_Y);

    const uint32_t rt = l2f_rd32(tx, ty, g_conn.wr_counter_addr + L2F_SCPC_PAD0);
    diag(FF_DIAG_PROBE_TRANS, rt);
    if (rt == magic) {
        g_conn.edm_x = tx;
        g_conn.edm_y = ty;
        diag(FF_DIAG_PROBE_RESULT, FF_PROBE_TRANSLATED);
    } else if (px != 0xFFFFFFFFu) {
        const uint32_t rp = l2f_rd32(px, py, g_conn.wr_counter_addr + L2F_SCPC_PAD0);
        diag(FF_DIAG_PROBE_NOC0, rp);
        if (rp == magic) {
            g_conn.edm_x = px;
            g_conn.edm_y = py;
            diag(FF_DIAG_PROBE_RESULT, FF_PROBE_NOC0);
        } else {
            diag(FF_DIAG_PROBE_RESULT, FF_PROBE_NEITHER);
            set_fault(FF_FAULT_PROBE_FAILED);
        }
    } else {
        diag(FF_DIAG_PROBE_RESULT, FF_PROBE_NEITHER);
        set_fault(FF_FAULT_PROBE_FAILED);
    }
    diag(FF_DIAG_WINDOW_X, g_conn.edm_x);
    diag(FF_DIAG_WINDOW_Y, g_conn.edm_y);
}

static void do_open(void) {
    uint32_t ctr = 0, idx = 0, rdctr = 0;
    diag(FF_DIAG_SREG_OPEN, l2f_read_router_free_slots_reg(&g_conn));
    l2f_open(&g_conn, &ctr, &idx, &rdctr);
    diag(FF_DIAG_OPEN_CTR, ctr);
    diag(FF_DIAG_OPEN_IDX, idx);
    diag(FF_DIAG_OPEN_RDCTR, rdctr);
    diag(FF_DIAG_HANDSHAKE_RB, l2f_rd32(g_conn.edm_x, g_conn.edm_y, g_conn.handshake_addr));
    set_state(FF_STATE_OPENED);
}

static inline void resp_u32(uint32_t off, uint32_t v) { REG32(FF_MBOX_RESP + off) = v; }

// Send the 16 B inbox header {len, seq, tag, 0} to `flag_addr` on the destination.
static int send_inbox_header(
    uint32_t hops, uint32_t dx, uint32_t dy, uint32_t flag_addr, uint32_t len, uint32_t seq, uint32_t tag) {
    REG32(INBOX_HDR_SCRATCH + 0x0) = len;
    REG32(INBOX_HDR_SCRATCH + 0x4) = seq;
    REG32(INBOX_HDR_SCRATCH + 0x8) = tag;
    REG32(INBOX_HDR_SCRATCH + 0xc) = 0;
    l2f_fence();
    return l2f_send_packet(&g_conn, hops, l2f_noc_addr(dx, dy, flag_addr), (uintptr_t)INBOX_HDR_SCRATCH, 16, SPIN_CAP);
}

static void handle_request(uint32_t seq) {
    const uint32_t mode = REG32(FF_MBOX_REQ + FF_REQ_MODE);
    uint32_t status = FF_RSTATUS_OK;
    uint32_t packets = 0;
    uint32_t free_before = 0, free_after = 0, sreg_before = 0, sreg_after = 0;
    const uint64_t t0 = l2f_mcycle();

    if (mode == FF_MODE_CLOSE) {
        if (g_conn.is_open) {
            uint32_t ack_seen = 0;
            if (l2f_close(&g_conn, CLOSE_SPIN_CAP, &ack_seen)) {
                status = FF_RSTATUS_CLOSED;
            } else {
                status = FF_RSTATUS_CLOSE_TIMEOUT;
                set_fault(FF_FAULT_CLOSE_TIMEOUT);
            }
            diag(FF_DIAG_TEARDOWN_ACK, ack_seen);
        } else {
            status = FF_RSTATUS_CLOSED;
        }
        set_state(FF_STATE_CLOSED);
    } else if (mode == FF_MODE_REOPEN) {
        if (!g_conn.is_open) {
            do_open();
        }
    } else {  // FF_MODE_SEND
        const uint32_t src = REG32(FF_MBOX_REQ + FF_REQ_SRC_ADDR);
        const uint32_t size = REG32(FF_MBOX_REQ + FF_REQ_SIZE);
        const uint32_t dx = REG32(FF_MBOX_REQ + FF_REQ_DST_NOC_X);
        const uint32_t dy = REG32(FF_MBOX_REQ + FF_REQ_DST_NOC_Y);
        const uint32_t daddr = REG32(FF_MBOX_REQ + FF_REQ_DST_ADDR);
        const uint32_t hops = REG32(FF_MBOX_REQ + FF_REQ_NUM_HOPS);
        const uint32_t flag_addr = REG32(FF_MBOX_REQ + FF_REQ_FLAG_ADDR);
        if (!g_conn.is_open) {
            status = FF_RSTATUS_NOT_OPEN;
        } else if (size == 0 || hops == 0) {
            status = FF_RSTATUS_BAD_REQ;
        } else {
            free_before = l2f_free_slots(&g_conn);
            sreg_before = l2f_read_router_free_slots_reg(&g_conn);
            int ok = l2f_send(&g_conn, hops, dx, dy, daddr, (uintptr_t)src, size, SPIN_CAP, &packets);
            if (ok && flag_addr != 0) {
                ok = send_inbox_header(hops, dx, dy, flag_addr, size, seq, FF_TAG_ORIGINAL);
                if (ok) {
                    packets++;
                }
            }
            sreg_after = l2f_read_router_free_slots_reg(&g_conn);
            if (!ok) {
                status = FF_RSTATUS_SLOT_TIMEOUT;
                set_fault(FF_FAULT_SLOT_TIMEOUT);
            } else {
                set_state(FF_STATE_SENT);
            }
            l2f_wait_drained(&g_conn, SPIN_CAP);
            free_after = l2f_free_slots(&g_conn);
        }
    }

    const uint64_t dt = l2f_mcycle() - t0;
    resp_u32(FF_RESP_STATUS, status);
    resp_u32(FF_RESP_PACKETS, packets);
    resp_u32(FF_RESP_FREE_BEFORE, free_before);
    resp_u32(FF_RESP_FREE_AFTER, free_after);
    resp_u32(FF_RESP_SREG_BEFORE, sreg_before);
    resp_u32(FF_RESP_SREG_AFTER, sreg_after);
    resp_u32(FF_RESP_CYCLES_LO, (uint32_t)dt);
    resp_u32(FF_RESP_CYCLES_HI, (uint32_t)(dt >> 32));
    l2f_fence();
    resp_u32(FF_RESP_SEQ, seq);  // publish last
    l2f_fence();
}

static uint32_t g_inbox_seen, g_echoes;
static uint32_t g_config_gen;  // FF_CONN_VALID value the current connection was configured from

// (Re)configure from the connection block: load params, probe coordinates, open.
// Called at boot and whenever the host publishes a new FF_CONN_VALID value (e.g. a
// new host run after the fabric routers were torn down and brought up again).
static void configure(uint32_t gen) {
    g_config_gen = gen;
    set_fault(FF_FAULT_NONE);
    g_conn.is_open = 0;
    if (!load_conn()) {
        set_fault(FF_FAULT_BAD_PARAMS);
        return;  // keep serving the loop; requests answer NOT_OPEN
    }
    set_state(FF_STATE_PARAMS_READY);
    probe_coords();
    set_state(FF_STATE_PROBED);
    do_open();
    diag(FF_DIAG_CONFIG_GEN, gen);
}

static void handle_inbox(uint32_t iseq) {
    uint32_t seen = ++g_inbox_seen;
    uint32_t echoes = g_echoes;
    diag(FF_DIAG_INBOX_SEEN, seen);
    const uint32_t tag = REG32(FF_MBOX_INBOX + FF_INBOX_TAG);
    uint32_t len = REG32(FF_MBOX_INBOX + FF_INBOX_LEN);
    if (len > FF_INBOX_DATA_MAX) {
        len = FF_INBOX_DATA_MAX;
    }
    if ((g_flags & FF_CFLAG_AUTO_ECHO) && tag == FF_TAG_ORIGINAL && g_conn.is_open && g_peer_x != 0xFFFFFFFFu) {
        uint32_t packets = 0;
        int ok = l2f_send(
            &g_conn,
            g_peer_hops,
            g_peer_x,
            g_peer_y,
            g_peer_inbox + (FF_MBOX_INBOX_DATA - FF_MBOX_INBOX),
            (uintptr_t)FF_MBOX_INBOX_DATA,
            len,
            SPIN_CAP,
            &packets);
        if (ok) {
            ok = send_inbox_header(g_peer_hops, g_peer_x, g_peer_y, g_peer_inbox, len, iseq, FF_TAG_ECHO);
        }
        if (ok) {
            g_echoes = ++echoes;
            diag(FF_DIAG_ECHOES, echoes);
        } else {
            set_fault(FF_FAULT_SLOT_TIMEOUT);
        }
    }
}

void fw_main(void) {
    g_hb = 0;
    g_inbox_seen = 0;
    g_echoes = 0;
    REG64(FF_MBOX_HARTID) = read_mhartid();
    set_fault(FF_FAULT_NONE);
    set_state(FF_STATE_ALIVE);

    // 1) Wait for the first connection block. Unbounded, but heartbeating — the host
    //    can see the hart is alive while fabric bring-up (seconds) completes.
    uint32_t gen = 0;
    while ((gen = conn_u32(FF_CONN_VALID)) == 0) {
        heartbeat();
    }
    // 2)+3) load params, probe the window coordinates, open the connection.
    configure(gen);

    // 4) Serve requests and inbox messages forever.
    uint32_t last_seq = 0;
    uint32_t last_inbox = REG32(FF_MBOX_INBOX + FF_INBOX_SEQ);
    for (;;) {
        heartbeat();
        if (g_conn.is_open) {
            diag(FF_DIAG_FREE_NOW, l2f_free_slots(&g_conn));
        }
        // New connection block published by the host? Reconfigure in place.
        gen = conn_u32(FF_CONN_VALID);
        if (gen != 0 && gen != g_config_gen) {
            configure(gen);
            last_inbox = REG32(FF_MBOX_INBOX + FF_INBOX_SEQ);
        }
        const uint32_t seq = REG32(FF_MBOX_REQ + FF_REQ_SEQ);
        if (seq != 0 && seq != last_seq) {
            handle_request(seq);
            last_seq = seq;
        }
        const uint32_t iseq = REG32(FF_MBOX_INBOX + FF_INBOX_SEQ);
        if (iseq != 0 && iseq != last_inbox) {
            handle_inbox(iseq);
            last_inbox = iseq;
        }
    }
}
