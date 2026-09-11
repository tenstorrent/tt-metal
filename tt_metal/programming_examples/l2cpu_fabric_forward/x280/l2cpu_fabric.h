// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// l2cpu_fabric.h — a small, freestanding fabric-worker library for the Blackhole
// L2CPU (x280). It lets firmware on the L2CPU open a connection to the on-chip
// fabric router (EDM) on an ethernet core and push packets to any core on any chip
// the fabric reaches, exactly like a Tensix worker does with WorkerToFabricEdmSender.
//
//   l2f_conn_t c = { ...params from the connection block... };
//   l2f_open(&c);
//   l2f_send(&c, hops, dst_x, dst_y, dst_addr, src_addr, len, max_cycles, &packets);
//   l2f_close(&c, max_cycles, &ack);  // optional; the connection can stay open for
//                                      // the lifetime of the firmware
//
// The x280 has no NOC command interface. Its only NOC egress is a TLB window: point
// the window at a tile (x, y) + address, then plain loads/stores through the aperture
// become NOC reads/writes. Everything here is built from that primitive:
//   * payload + header:   64/32-bit stores into the EDM's channel buffer slot (eth L1)
//   * worker->EDM credit: one 32-bit store of the packed value -1 to the router's
//                         free-slots stream register (the same register a Tensix
//                         worker hits with noc_inline_dw_write<InlineWriteDst::REG>)
//   * EDM->worker credit: the router writes its read counter into a local word this
//                         library hands it at open (plain NOC write into the tile)
//
// Protocol and layout constants are reproduced from tt-metal (file:line cited) —
// this header is compiled for RV64 and must not include tt-metal C++ headers.
//   tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp   (open/send/close)
//   tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp  (handshake values)
//   tt_metal/api/tt-metalium/experimental/fabric/fabric_edm_types.hpp  (worker location info, cursor)
//   tt_metal/fabric/fabric_edm_packet_header.hpp                        (LowLatencyPacketHeaderT)
//   tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h            (1D routing encoding)
//   tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_overlay_parameters.h (stream reg packing)

#ifndef L2CPU_FABRIC_H
#define L2CPU_FABRIC_H

#include <stdint.h>

#define L2F_REG64(a) (*(volatile uint64_t*)(uintptr_t)(a))
#define L2F_REG32(a) (*(volatile uint32_t*)(uintptr_t)(a))
#define L2F_REG16(a) (*(volatile uint16_t*)(uintptr_t)(a))
#define L2F_REG8(a) (*(volatile uint8_t*)(uintptr_t)(a))

// ---------------------------------------------------------------------------
// TLB window 0 (config registers + 2 MiB uncached aperture). Same programming as
// the echo firmware in ../../l2cpu_noc_transfer/x280/fw.c.
// ---------------------------------------------------------------------------
#define L2F_TLB_CFG_BASE 0x20000000UL
#define L2F_TLB_APERTURE 0x0430000000UL
#define L2F_WINDOW_MASK 0x1FFFFFUL

static inline void l2f_fence(void) { __asm__ volatile("fence rw,rw" ::: "memory"); }

static inline uint64_t l2f_mcycle(void) {
    uint64_t v;
    __asm__ volatile("csrr %0, mcycle" : "=r"(v));
    return v;
}

typedef struct {
    uint32_t x, y, page;
    int valid;
} l2f_window_t;

static l2f_window_t l2f_win = {0, 0, 0, 0};

// Aim window 0 at NOC tile (x, y) so that aperture + (addr & mask) reaches addr.
// Re-programs the window only when the tile or the 2 MiB page changes.
static inline void l2f_aim(uint32_t x, uint32_t y, uint32_t addr) {
    const uint32_t page = addr >> 21;
    if (l2f_win.valid && l2f_win.x == x && l2f_win.y == y && l2f_win.page == page) {
        return;
    }
    volatile uint64_t* cfg64 = (volatile uint64_t*)(uintptr_t)L2F_TLB_CFG_BASE;
    volatile uint32_t* cfg32 = (volatile uint32_t*)(uintptr_t)L2F_TLB_CFG_BASE;
    l2f_fence();
    cfg64[0] = (uint64_t)page;                  // local_offset (address bits above the aperture)
    cfg32[2] = (x & 0x3f) | ((y & 0x3f) << 6);  // x_end[5:0], y_end[11:6]; unicast, NOC0
    cfg32[3] = 0;
    l2f_fence();
    l2f_win.x = x;
    l2f_win.y = y;
    l2f_win.page = page;
    l2f_win.valid = 1;
}

static inline uint32_t l2f_rd32(uint32_t x, uint32_t y, uint32_t addr) {
    l2f_aim(x, y, addr);
    return L2F_REG32(L2F_TLB_APERTURE + (addr & L2F_WINDOW_MASK));
}
static inline void l2f_wr32(uint32_t x, uint32_t y, uint32_t addr, uint32_t v) {
    l2f_aim(x, y, addr);
    L2F_REG32(L2F_TLB_APERTURE + (addr & L2F_WINDOW_MASK)) = v;
}
static inline void l2f_wr64(uint32_t x, uint32_t y, uint32_t addr, uint64_t v) {
    l2f_aim(x, y, addr);
    L2F_REG64(L2F_TLB_APERTURE + (addr & L2F_WINDOW_MASK)) = v;
}

// ---------------------------------------------------------------------------
// Protocol constants (reproduced; see file header for sources).
// ---------------------------------------------------------------------------
// fabric_connection_interface.hpp:14-16
#define L2F_OPEN_CONNECTION_VALUE 1u
#define L2F_CLOSE_CONNECTION_REQUEST_VALUE 2u

// EDMChannelWorkerLocationInfo (fabric_edm_types.hpp): 16 B-strided fields, 64 B total
#define L2F_WLI_WORKER_SEMAPHORE_ADDRESS 0x00
#define L2F_WLI_WORKER_TEARDOWN_SEMAPHORE_ADDRESS 0x10
#define L2F_WLI_WORKER_XY 0x20  // (y << 16) | x
#define L2F_WLI_EDM_READ_COUNTER 0x30

// SenderChannelProducerCursor (fabric_edm_types.hpp): {write_counter, write_index, pad0, pad1}
#define L2F_SCPC_WRITE_COUNTER 0x00
#define L2F_SCPC_WRITE_INDEX 0x04
#define L2F_SCPC_PAD0 0x08  // unused by the router after bring-up; used here as a coordinate probe

// pack_value_for_inc_on_write_stream_reg_write(-1) = -1 << REMOTE_DEST_BUF_WORDS_FREE_INC (6 on Blackhole)
#define L2F_CREDIT_MINUS_ONE 0xFFFFFFC0u

// LowLatencyPacketHeaderT<N> (fabric_edm_packet_header.hpp). Offsets identical for the
// 48 B (N=0) and 64 B (N>=1) instantiations; the size is a connection parameter.
#define L2F_PH_NOC_ADDRESS 0x00    // u64 NocUnicastCommandHeader::noc_address
#define L2F_PH_PAYLOAD_SIZE 0x28   // u16
#define L2F_PH_NOC_SEND_TYPE 0x2A  // u8, NOC_UNICAST_WRITE = 0
#define L2F_PH_SRC_CH_ID 0x2B      // u8, reserved (router-owned)
#define L2F_PH_ROUTING 0x2C        // u32 LowLatencyRoutingFieldsT::value
#define L2F_NOC_SEND_TYPE_UNICAST_WRITE 0u

// Blackhole NOC address: y at bit 42, x at bit 36 (NOC_ADDR_LOCAL_BITS=36, NODE_ID_BITS=6).
// On Blackhole NOC0/NOC1 use the same (translated) coordinates — noc_nonblocking_api.h
// NOC_0_X()/NOC_0_Y() are identity — so the header carries the destination's translated
// coords unchanged, whichever NOC the receiving router writes on.
static inline uint64_t l2f_noc_addr(uint32_t x, uint32_t y, uint32_t addr) {
    return (((uint64_t)y) << 42) | (((uint64_t)x) << 36) | (uint64_t)addr;
}

// 1D unicast routing word for `hops` hops (fabric_common.h encode_1d_unicast):
// 2 bits per hop consumed LSB-first, FORWARD_ONLY (0b10) for transit, WRITE_ONLY
// (0b01) at the destination. Only the first routing word (<=16 hops) is filled; the
// extension word stays 0.
static inline uint32_t l2f_routing_1d(uint32_t hops) {
    if (hops == 0) {
        return 0;
    }
    const uint32_t write_bit_pos = ((hops - 1) % 16u) * 2u;
    const uint32_t forward_mask = (write_bit_pos == 0) ? 0u : ((1u << write_bit_pos) - 1u);
    return (0xAAAAAAAAu & forward_mask) | (0x1u << write_bit_pos);
}

// ---------------------------------------------------------------------------
// Connection descriptor. The caller fills the parameter fields (normally from the
// connection block a Tensix setup kernel resolved via WorkerToFabricEdmSender::
// build_from_args) and l2f_open() takes over the state fields.
// ---------------------------------------------------------------------------
typedef struct {
    // --- parameters ---
    uint32_t edm_x, edm_y;  // EDM eth core, in the coordinate system the window uses
    uint32_t buffer_base;   // edm_buffer_base_addr (eth L1)
    uint32_t num_buffers;   // slots in the sender channel
    uint32_t buffer_size;   // bytes per slot, header included
    uint32_t handshake_addr;
    uint32_t worker_loc_info;
    uint32_t wr_counter_addr;  // SenderChannelProducerCursor in eth L1
    uint32_t sreg_write_addr;  // router free-slots stream reg, inc-on-write address
    uint32_t sreg_read_addr;   // same stream reg, read address (diagnostics)
    uint32_t self_x, self_y;   // this tile's NOC coords (where the router pushes credits)
    uint32_t hdr_size;         // 48 or 64
    uint32_t freeslots_sink;   // local UNCACHED word the router writes its read counter to
    uint32_t teardown_word;    // local UNCACHED word the router acks teardown to
    // --- state ---
    uint32_t write_counter;  // free-running, mirrors SenderChannelProducerCursor.write_counter
    uint32_t write_index;    // 0..num_buffers-1
    uint32_t packets_sent;
    int is_open;
} l2f_conn_t;

// Number of slots the router can accept right now (adapter get_num_free_write_slots()).
static inline uint32_t l2f_free_slots(const l2f_conn_t* c) {
    const uint32_t edm_read_counter = L2F_REG32(c->freeslots_sink);
    const uint32_t used = c->write_counter - edm_read_counter;  // wrap-safe
    return used >= c->num_buffers ? 0 : c->num_buffers - used;
}

// All waits below are bounded by a deadline in mcycle ticks (`max_cycles`), not
// iterations, so the bound is independent of how slow each poll is.
static inline int l2f_wait_slot(const l2f_conn_t* c, uint64_t max_cycles) {
    const uint64_t t0 = l2f_mcycle();
    do {
        if (l2f_free_slots(c) != 0) {
            return 1;
        }
    } while (l2f_mcycle() - t0 < max_cycles);
    return 0;
}

// Router-side free-slot counter (low bits of the stream register). Diagnostics only.
static inline uint32_t l2f_read_router_free_slots_reg(const l2f_conn_t* c) {
    return l2f_rd32(c->edm_x, c->edm_y, c->sreg_read_addr);
}

// open(): port of WorkerToFabricEdmSender::open_start()/open_finish() (worker flavour).
// Optional out params return what was adopted from the router for diagnostics.
static inline void l2f_open(l2f_conn_t* c, uint32_t* out_ctr, uint32_t* out_idx, uint32_t* out_rdctr) {
    // Adopt the previous producer's cursor verbatim (adapter open_start/open_finish).
    const uint32_t ctr = l2f_rd32(c->edm_x, c->edm_y, c->wr_counter_addr + L2F_SCPC_WRITE_COUNTER);
    uint32_t idx = l2f_rd32(c->edm_x, c->edm_y, c->wr_counter_addr + L2F_SCPC_WRITE_INDEX);
    // Seed the local credit sink with the router's read counter.
    const uint32_t rdctr = l2f_rd32(c->edm_x, c->edm_y, c->worker_loc_info + L2F_WLI_EDM_READ_COUNTER);
    L2F_REG32(c->freeslots_sink) = rdctr;
    L2F_REG32(c->teardown_word) = 0;
    l2f_fence();
    // Tell the router where we live and where to push credits / the teardown ack.
    l2f_wr32(c->edm_x, c->edm_y, c->worker_loc_info + L2F_WLI_WORKER_SEMAPHORE_ADDRESS, c->freeslots_sink);
    l2f_wr32(c->edm_x, c->edm_y, c->worker_loc_info + L2F_WLI_WORKER_TEARDOWN_SEMAPHORE_ADDRESS, c->teardown_word);
    l2f_wr32(c->edm_x, c->edm_y, c->worker_loc_info + L2F_WLI_WORKER_XY, (c->self_y << 16) | (c->self_x & 0xFFFF));
    l2f_fence();
    if (idx >= c->num_buffers) {
        idx = 0;
    }
    c->write_counter = ctr;
    c->write_index = idx;
    // Publish the connection (must come after the location info is written).
    l2f_wr32(c->edm_x, c->edm_y, c->handshake_addr, L2F_OPEN_CONNECTION_VALUE);
    l2f_fence();
    c->is_open = 1;
    if (out_ctr) {
        *out_ctr = ctr;
    }
    if (out_idx) {
        *out_idx = idx;
    }
    if (out_rdctr) {
        *out_rdctr = rdctr;
    }
}

// Header scratch (cached GDDR .bss); built locally, then streamed into the slot.
static uint8_t l2f_hdr[64] __attribute__((aligned(16)));

// Send ONE packet: `len` bytes from local address `src` to `dst_noc_addr` on the chip
// `hops` hops away. len <= buffer_size - hdr_size. Port of
// send_current_slot_non_blocking(): payload, then header, then the credit.
//
// `src` is a full x280 address (uintptr_t). Firmware .data/.bss live in the CACHED
// GDDR alias above 4 GiB (0x4000_3000_0000+); truncating such a pointer to 32 bits
// silently redirects the read to the UNCACHED alias of the same DRAM and returns
// whatever is physically in DRAM, not what this core wrote through its cache.
static inline int l2f_send_packet(
    l2f_conn_t* c, uint32_t hops, uint64_t dst_noc_addr, uintptr_t src, uint32_t len, uint64_t max_cycles) {
    if (!c->is_open || len > c->buffer_size - c->hdr_size) {
        return 0;
    }
    if (!l2f_wait_slot(c, max_cycles)) {
        return 0;
    }
    const uint32_t slot = c->buffer_base + c->write_index * c->buffer_size;
    const uint32_t pay = slot + c->hdr_size;

    // 1) payload: 64-bit stores when both sides are 8 B aligned, else 32-bit, tail packed.
    uint32_t off = 0;
    if (((src | pay) & 7u) == 0) {
        while (off + 8 <= len) {
            l2f_wr64(c->edm_x, c->edm_y, pay + off, L2F_REG64(src + off));
            off += 8;
        }
    }
    while (off + 4 <= len) {
        l2f_wr32(c->edm_x, c->edm_y, pay + off, L2F_REG32(src + off));
        off += 4;
    }
    if (off < len) {
        uint32_t w = 0;
        for (uint32_t k = 0; off + k < len; ++k) {
            w |= ((uint32_t)L2F_REG8(src + off + k)) << (8 * k);
        }
        l2f_wr32(c->edm_x, c->edm_y, pay + off, w);  // extra bytes land in slot padding, never forwarded
    }

    // 2) header
    for (uint32_t b = 0; b < c->hdr_size; ++b) {
        l2f_hdr[b] = 0;
    }
    *(uint64_t*)(l2f_hdr + L2F_PH_NOC_ADDRESS) = dst_noc_addr;
    *(uint16_t*)(l2f_hdr + L2F_PH_PAYLOAD_SIZE) = (uint16_t)len;
    l2f_hdr[L2F_PH_NOC_SEND_TYPE] = (uint8_t)L2F_NOC_SEND_TYPE_UNICAST_WRITE;
    l2f_hdr[L2F_PH_SRC_CH_ID] = 0;
    *(uint32_t*)(l2f_hdr + L2F_PH_ROUTING) = l2f_routing_1d(hops);
    for (uint32_t b = 0; b < c->hdr_size; b += 8) {
        l2f_wr64(c->edm_x, c->edm_y, slot + b, *(uint64_t*)(l2f_hdr + b));
    }
    l2f_fence();

    // 3) credit: one slot consumed (adapter update_edm_buffer_free_slots, non-stateful path)
    l2f_wr32(c->edm_x, c->edm_y, c->sreg_write_addr, L2F_CREDIT_MINUS_ONE);
    l2f_fence();

    // 4) advance (adapter advance_buffer_slot_write_index)
    c->write_counter++;
    if (++c->write_index >= c->num_buffers) {
        c->write_index = 0;
    }
    c->packets_sent++;
    return 1;
}

// Send `len` bytes from local `src` to (dst_x, dst_y, dst_addr) `hops` away, chunked
// into as many packets as the slot payload size requires. Packets on one connection
// are delivered in order.
static inline int l2f_send(
    l2f_conn_t* c,
    uint32_t hops,
    uint32_t dst_x,
    uint32_t dst_y,
    uint32_t dst_addr,
    uintptr_t src,
    uint32_t len,
    uint64_t max_cycles,
    uint32_t* packets) {
    const uint32_t max_payload = c->buffer_size - c->hdr_size;
    while (len) {
        const uint32_t n = len < max_payload ? len : max_payload;
        if (!l2f_send_packet(c, hops, l2f_noc_addr(dst_x, dst_y, dst_addr), src, n, max_cycles)) {
            return 0;
        }
        dst_addr += n;
        src += n;
        len -= n;
        if (packets) {
            (*packets)++;
        }
    }
    return 1;
}

// Forward a COMPLETE packet — header already built by the producer at `src`, payload
// right after it, `total_bytes` = header + payload — into the router's current slot.
// This is what a mux does with a slot a worker committed to it: no header building,
// just copy and credit. The router reads the header only after the credit, so the
// raw copy order is safe.
static inline int l2f_forward_packet(l2f_conn_t* c, uintptr_t src, uint32_t total_bytes, uint64_t max_cycles) {
    if (!c->is_open || total_bytes < c->hdr_size || total_bytes > c->buffer_size) {
        return 0;
    }
    if (!l2f_wait_slot(c, max_cycles)) {
        return 0;
    }
    const uint32_t slot = c->buffer_base + c->write_index * c->buffer_size;
    uint32_t off = 0;
    if (((src | slot) & 7u) == 0) {
        while (off + 8 <= total_bytes) {
            l2f_wr64(c->edm_x, c->edm_y, slot + off, L2F_REG64(src + off));
            off += 8;
        }
    }
    while (off + 4 <= total_bytes) {
        l2f_wr32(c->edm_x, c->edm_y, slot + off, L2F_REG32(src + off));
        off += 4;
    }
    if (off < total_bytes) {
        uint32_t w = 0;
        for (uint32_t k = 0; off + k < total_bytes; ++k) {
            w |= ((uint32_t)L2F_REG8(src + off + k)) << (8 * k);
        }
        l2f_wr32(c->edm_x, c->edm_y, slot + off, w);
    }
    l2f_fence();
    l2f_wr32(c->edm_x, c->edm_y, c->sreg_write_addr, L2F_CREDIT_MINUS_ONE);
    l2f_fence();
    c->write_counter++;
    if (++c->write_index >= c->num_buffers) {
        c->write_index = 0;
    }
    c->packets_sent++;
    return 1;
}

// Bounded wait until the router has drained everything we pushed (all slots free).
static inline int l2f_wait_drained(const l2f_conn_t* c, uint64_t max_cycles) {
    const uint64_t t0 = l2f_mcycle();
    do {
        if (l2f_free_slots(c) == c->num_buffers) {
            return 1;
        }
    } while (l2f_mcycle() - t0 < max_cycles);
    return 0;
}

// close(): port of close_start()/close_finish(). Persists the cursor for the next
// producer and requests teardown. The router acknowledges by (a) setting the
// connection handshake word back to unused (0) and (b) a noc_semaphore_inc of 1 into
// teardown_word. (a) is observable through the window and is what this waits for;
// whether (b) landed is reported via *teardown_ack_seen (0/1) so the caller can log
// it without depending on it. Returns 1 once the router released the connection.
static inline int l2f_close(l2f_conn_t* c, uint64_t max_cycles, uint32_t* teardown_ack_seen) {
    l2f_wr32(c->edm_x, c->edm_y, c->wr_counter_addr + L2F_SCPC_WRITE_COUNTER, c->write_counter);
    l2f_wr32(c->edm_x, c->edm_y, c->wr_counter_addr + L2F_SCPC_WRITE_INDEX, c->write_index);
    l2f_fence();
    l2f_wr32(c->edm_x, c->edm_y, c->handshake_addr, L2F_CLOSE_CONNECTION_REQUEST_VALUE);
    l2f_fence();
    c->is_open = 0;
    int released = 0;
    const uint64_t t0 = l2f_mcycle();
    do {
        if (l2f_rd32(c->edm_x, c->edm_y, c->handshake_addr) == 0) {
            released = 1;
            break;
        }
    } while (l2f_mcycle() - t0 < max_cycles);
    if (teardown_ack_seen) {
        *teardown_ack_seen = (L2F_REG32(c->teardown_word) == 1) ? 1u : 0u;
    }
    L2F_REG32(c->teardown_word) = 0;
    l2f_fence();
    return released;
}

#endif  // L2CPU_FABRIC_H
