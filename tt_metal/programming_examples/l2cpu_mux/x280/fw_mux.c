// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// x280 (L2CPU) fabric MUX firmware.
//
// The L2CPU tile plays the role a Tensix mux core plays for tt_fabric_mux.cpp: worker
// kernels open per-channel connections to it, write packets into its slot rings and
// commit them; this firmware forwards each committed slot into its own router
// connection (l2cpu_fabric.h) and returns credits by writing its read counter into the
// worker's L1 through the TLB window. Handshake, location info, cursor and teardown
// follow the V1 mux protocol byte for byte. The single substitution is the commit
// signal: workers write their write counter into LM_WRITE_COUNTER(ch) (plain NOC write)
// because this tile has no stream registers.
//
// Bring-up mirrors fw_fabric.c: wait for the router connection block in the fabric
// mailbox (fabric_mbox.h), probe window coordinates, open the router connection, then
// serve channels until a termination signal. After TERMINATED it re-arms when the host
// clears LM_TERMINATION, so repeated host runs need no chip reset.

#include <stdint.h>

#include "fabric_mbox.h"
#include "l2cpu_fabric.h"
#include "l2cpu_mux_layout.h"

#define REG64(a) L2F_REG64(a)
#define REG32(a) L2F_REG32(a)
#define REG16(a) L2F_REG16(a)

#define SPIN_CAP 400000000ull  // 2 s of mcycle at 200 MHz

static l2f_conn_t g_conn;
static uint64_t g_hb;
static uint32_t g_config_gen;

// ---------------------------------------------------------------------------
// Fabric mailbox plumbing (same as fw_fabric.c)
// ---------------------------------------------------------------------------
static inline uint64_t read_mhartid(void) {
    uint64_t v;
    __asm__ volatile("csrr %0, mhartid" : "=r"(v));
    return v;
}
static inline void heartbeat(void) {
    REG64(FF_MBOX_HEARTBEAT) = ++g_hb;
    REG32(LM_HEARTBEAT) = (uint32_t)g_hb;
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
    if (g_conn.num_buffers == 0 || g_conn.num_buffers > 64 || (g_conn.hdr_size != 48 && g_conn.hdr_size != 64) ||
        g_conn.buffer_size <= g_conn.hdr_size || g_conn.buffer_size > 0x10000) {
        return 0;
    }
    if (g_conn.buffer_base == 0 || g_conn.buffer_base >= 0x200000 || g_conn.handshake_addr == 0 ||
        g_conn.handshake_addr >= 0x200000 || g_conn.worker_loc_info == 0 || g_conn.worker_loc_info >= 0x200000 ||
        g_conn.wr_counter_addr == 0 || g_conn.wr_counter_addr >= 0x200000) {
        return 0;
    }
    if ((g_conn.sreg_write_addr & 0xFFFC0000u) != 0xFFB40000u || (g_conn.sreg_read_addr & 0xFFFC0000u) != 0xFFB40000u) {
        return 0;
    }
    return g_conn.freeslots_sink != 0 && g_conn.teardown_word != 0;
}

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

static void configure(uint32_t gen) {
    g_config_gen = gen;
    set_fault(FF_FAULT_NONE);
    g_conn.is_open = 0;
    if (!load_conn()) {
        set_fault(FF_FAULT_BAD_PARAMS);
        return;
    }
    set_state(FF_STATE_PARAMS_READY);
    probe_coords();
    set_state(FF_STATE_PROBED);
    do_open();
    diag(FF_DIAG_CONFIG_GEN, gen);
    REG32(LM_CONFIG_GEN) = gen;
    l2f_fence();
}

// ---------------------------------------------------------------------------
// Mux channels
// ---------------------------------------------------------------------------
typedef struct {
    uint32_t read_counter;  // packets forwarded (free-running, matches the worker's write counter)
    uint32_t read_index;    // slot index of the next packet to forward
    uint32_t worker_x, worker_y, worker_flow_addr, worker_teardown_addr;
    int established;
} lm_channel_t;

static lm_channel_t g_ch[LM_MAX_CHANNELS];

static inline void stat_inc(uint32_t ch, uint32_t off) { REG32(LM_STATS(ch) + off) = REG32(LM_STATS(ch) + off) + 1; }

static void mux_reset_channels(void) {
    for (uint32_t ch = 0; ch < LM_MAX_CHANNELS; ++ch) {
        g_ch[ch].read_counter = 0;
        g_ch[ch].read_index = 0;
        g_ch[ch].established = 0;
        REG32(LM_HANDSHAKE(ch)) = 0;
        REG32(LM_WRITE_COUNTER(ch)) = 0;
        for (uint32_t w = 0; w < 16; w += 4) {
            REG32(LM_CURSOR(ch) + w) = 0;
        }
        for (uint32_t w = 0; w < 64; w += 4) {
            REG32(LM_CONN_INFO(ch) + w) = 0;
        }
        for (uint32_t w = 0; w < 16; w += 4) {
            REG32(LM_STATS(ch) + w) = 0;
        }
    }
    REG32(LM_NUM_CHANNELS) = LM_MAX_CHANNELS;
    REG32(LM_SLOT_BYTES) = LM_SLOT_STRIDE(g_conn.buffer_size);
    l2f_fence();
}

// Publish this channel's read counter: into the location-info block (so a re-opening
// producer seeds correctly, like the router's copy_read_counter_to_worker_location_info)
// and, when a worker is connected, into the worker's L1 flow-control word.
static inline void mux_publish_read_counter(uint32_t ch) {
    lm_channel_t* c = &g_ch[ch];
    REG32(LM_CONN_INFO(ch) + L2F_WLI_EDM_READ_COUNTER) = c->read_counter;
    if (c->established) {
        l2f_wr32(c->worker_x, c->worker_y, c->worker_flow_addr, c->read_counter);
    }
}

static inline void mux_cache_worker(lm_channel_t* c, uint32_t ch) {
    c->worker_flow_addr = REG32(LM_CONN_INFO(ch) + L2F_WLI_WORKER_SEMAPHORE_ADDRESS);
    c->worker_teardown_addr = REG32(LM_CONN_INFO(ch) + L2F_WLI_WORKER_TEARDOWN_SEMAPHORE_ADDRESS);
    const uint32_t xy = REG32(LM_CONN_INFO(ch) + L2F_WLI_WORKER_XY);
    c->worker_x = xy & 0xFFFFu;
    c->worker_y = xy >> 16;
}

static void mux_service_channel(uint32_t ch) {
    lm_channel_t* c = &g_ch[ch];
    const uint32_t hs = REG32(LM_HANDSHAKE(ch));
    if (hs == L2F_OPEN_CONNECTION_VALUE && !c->established) {
        // Worker published its location info before writing the handshake.
        mux_cache_worker(c, ch);
        c->established = 1;
        stat_inc(ch, LM_STAT_CONNECTS);
        mux_publish_read_counter(ch);
    } else if (hs == L2F_CLOSE_CONNECTION_REQUEST_VALUE) {
        // A fast producer can open, send and request close between two of our polls, so
        // the open value may never have been observed (the Tensix mux treats "close" as
        // connect-then-teardown for the same reason). Always take the location info from
        // the block and ack, whether or not we saw the open.
        mux_cache_worker(c, ch);
        if (!c->established) {
            stat_inc(ch, LM_STAT_CONNECTS);
        }
        REG32(LM_HANDSHAKE(ch)) = 0;
        l2f_fence();
        l2f_wr32(c->worker_x, c->worker_y, c->worker_teardown_addr, 1);
        c->established = 0;
        stat_inc(ch, LM_STAT_TEARDOWNS);
    }

    // Anything committed but not yet forwarded?
    const uint32_t wc = REG32(LM_WRITE_COUNTER(ch));
    if (wc != c->read_counter && g_conn.is_open) {
        const uint32_t slot =
            LM_CHANNEL_BASE(ch, g_conn.buffer_size) + c->read_index * LM_SLOT_STRIDE(g_conn.buffer_size);
        const uint32_t total = g_conn.hdr_size + (uint32_t)REG16(slot + L2F_PH_PAYLOAD_SIZE);
        if (!l2f_forward_packet(&g_conn, (uintptr_t)slot, total, SPIN_CAP)) {
            set_fault(FF_FAULT_SLOT_TIMEOUT);
            return;  // leave the packet pending; retry next pass
        }
        REG32(LM_STATS(ch) + LM_STAT_LAST_BYTES) = total;
        stat_inc(ch, LM_STAT_FORWARDED);
        c->read_counter++;
        if (++c->read_index >= LM_NUM_BUFFERS) {
            c->read_index = 0;
        }
        mux_publish_read_counter(ch);
    }
}

static int mux_all_drained(void) {
    for (uint32_t ch = 0; ch < LM_MAX_CHANNELS; ++ch) {
        if (REG32(LM_WRITE_COUNTER(ch)) != g_ch[ch].read_counter) {
            return 0;
        }
    }
    return 1;
}

void fw_main(void) {
    g_hb = 0;
    g_config_gen = 0;
    REG64(FF_MBOX_HARTID) = read_mhartid();
    set_fault(FF_FAULT_NONE);
    set_state(FF_STATE_ALIVE);
    REG32(LM_STATUS) = LM_STATUS_STARTED;
    REG32(LM_TERMINATION) = LM_TERM_KEEP_RUNNING;
    REG32(LM_CONFIG_GEN) = 0;

    // Router connection first (the mux is a worker of the router).
    uint32_t gen = 0;
    while ((gen = conn_u32(FF_CONN_VALID)) == 0) {
        heartbeat();
    }
    configure(gen);

    for (;;) {
        // ---- one mux "run": init channels, serve until terminated ----
        mux_reset_channels();
        REG32(LM_STATUS) = g_conn.is_open ? LM_STATUS_READY_FOR_TRAFFIC : LM_STATUS_STARTED;
        l2f_fence();

        for (;;) {
            heartbeat();
            gen = conn_u32(FF_CONN_VALID);
            if (gen != 0 && gen != g_config_gen) {
                REG32(LM_STATUS) = LM_STATUS_STARTED;
                configure(gen);  // host re-ran fabric bring-up: new router, new connection
                mux_reset_channels();
                REG32(LM_STATUS) = g_conn.is_open ? LM_STATUS_READY_FOR_TRAFFIC : LM_STATUS_STARTED;
                l2f_fence();
            }
            const uint32_t term = REG32(LM_TERMINATION);
            if (term == LM_TERM_IMMEDIATE || (term == LM_TERM_GRACEFUL && mux_all_drained())) {
                break;
            }
            for (uint32_t ch = 0; ch < LM_MAX_CHANNELS; ++ch) {
                mux_service_channel(ch);
            }
        }
        l2f_wait_drained(&g_conn, SPIN_CAP);
        REG32(LM_STATUS) = LM_STATUS_TERMINATED;
        l2f_fence();

        // Re-arm: the next host run clears the termination word before it starts.
        while (REG32(LM_TERMINATION) != LM_TERM_KEEP_RUNNING) {
            heartbeat();
            gen = conn_u32(FF_CONN_VALID);
            if (gen != 0 && gen != g_config_gen) {
                configure(gen);
            }
        }
    }
}
