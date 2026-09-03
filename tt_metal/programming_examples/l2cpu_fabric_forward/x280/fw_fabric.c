// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// x280 (L2CPU) fabric-worker firmware.
// ============================================================================
// A Tensix producer on chip A stages a payload into the L2CPU tile's LIM and
// pokes this firmware's mailbox. This firmware acts as a *from-scratch* fabric
// worker: it re-implements the `WorkerToFabricEdmSender` open/send/close
// protocol (tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp)
// using ONLY the x280 TLB window (plain NOC loads/stores + one stream-register
// write) so the payload is forwarded over one ethernet link, through the
// standard unmodified EDM router, to a receiver Tensix on chip B.
//
// This file is deliberately FREESTANDING and includes NO tt-metal headers
// (they are Tensix/host C++). Every EDM constant, struct offset, stream-register
// macro and packet-header field below is REPRODUCED locally with a comment that
// cites the exact source `file:line` it was copied from. Anything that could not
// be resolved to a single concrete value from the headers is tagged
//
//     // CONFIRM ON HARDWARE
//
// so the hardware-equipped follow-up agent knows exactly what to check. This
// firmware has NEVER been run on hardware — treat every "// CONFIRM ON HARDWARE"
// as an open question, not a verified fact.
//
// Tasks 5 (open) and 6 (send) of
// docs/superpowers/plans/2026-09-03-x280-fabric-worker.md are implemented here,
// plus a best-effort close() (Task 7 Step 1). The single genuinely unproven
// primitive is the x280 -> EDM *stream-register credit write* (see
// `credit_write_minus_one()`); it is commented prominently.
//
// Runtime model (mirrors the sibling l2cpu_noc_transfer/x280/fw.c):
//   * Hart 0 runs fw_main; harts 1-3 park in wfi (start.S).
//   * The x280 has NO NOC command interface. TLB window 0 (config regs at
//     0x2000_0000, uncached aperture at 0x0430_0000_00) is the ONLY NOC egress:
//     point the window at a NOC (x, y) + address, then plain loads/stores
//     through the aperture become NOC reads/writes. Reused VERBATIM from fw.c.
//   * The mailbox lives in the tile's UNCACHED GDDR alias (0x3010_0000) so
//     firmware stores are instantly NOC-visible and host/Tensix reads see them
//     without a cache flush. Firmware .text runs from the CACHED GDDR alias
//     0x4000_3000_0000 (dram.ld).
//   * The hart release is ONE-SHOT per chip reset: this firmware must NEVER
//     wedge. Every spin loop is bounded; on deadline it writes an FF_FAULT_*
//     code and parks with a live heartbeat so the host can diagnose.
// ============================================================================

#include <stdint.h>
#include "fabric_mbox.h"  // FF_MBOX_*, FF_REQ_*, FF_CONN_*, FF_STATUS_*, FF_STATE_*, FF_FAULT_*

// ----------------------------------------------------------------------------
// Register access helpers + TLB window (copied verbatim from
// tt_metal/programming_examples/l2cpu_noc_transfer/x280/fw.c:35-71).
// ----------------------------------------------------------------------------
#define REG64(a) (*(volatile uint64_t*)(uintptr_t)(a))
#define REG32(a) (*(volatile uint32_t*)(uintptr_t)(a))

// Small TLB window 0: config registers + uncached access aperture.
// (fw.c:40-42)
#define TLB_CFG_BASE 0x20000000UL
#define TLB_APERTURE 0x0430000000UL
#define WINDOW_MASK 0x1FFFFFUL  // 2 MiB aperture; the low 21 bits index within it.

static inline void fence(void) { __asm__ volatile("fence rw,rw" ::: "memory"); }
static inline void cbo_flush(uintptr_t a) { __asm__ volatile("cbo.flush (%0)" ::"r"(a) : "memory"); }
static inline void cbo_inval(uintptr_t a) { __asm__ volatile("cbo.inval (%0)" ::"r"(a) : "memory"); }

static inline uint64_t read_mhartid(void) {
    uint64_t v;
    __asm__ volatile("csrr %0, mhartid" : "=r"(v));
    return v;
}

// Point small window 0 at NOC0 tile (x, y). The low 21 bits of the target come
// from the offset within the 2 MiB aperture; the rest from `addr >> 21`.
// noc_properties_lo: x_end[5:0], y_end[11:6]; NOC0, no multicast. (fw.c:63-71)
static void set_window(uint32_t x, uint32_t y, uint32_t addr) {
    volatile uint64_t* cfg64 = (volatile uint64_t*)(uintptr_t)TLB_CFG_BASE;
    volatile uint32_t* cfg32 = (volatile uint32_t*)(uintptr_t)TLB_CFG_BASE;
    fence();
    cfg64[0] = (uint64_t)addr >> 21;
    cfg32[2] = (x & 0x3f) | ((y & 0x3f) << 6);
    cfg32[3] = 0;
    fence();
}

// ============================================================================
// EDM / fabric constants reproduced locally (NO tt-metal headers allowed).
// Each block cites the exact source file:line it was copied from.
// ============================================================================

// --- Connection state values ------------------------------------------------
// tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp:15-16
#define EDM_OPEN_CONNECTION_VALUE 1u
#define EDM_CLOSE_CONNECTION_REQUEST_VALUE 2u
// tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_interface.hpp:24
// VC2 (infrastructure / runtime-arg) worker sender free-slots stream id. The
// host normally passes this via FF_CONN_CREDITS_STREAM_ID; this is the fallback
// default the adapter itself uses for VC2 (build_from_args sets
// worker_free_slots_stream_id = STREAM_ID, adapter L147 + L70-72).
#define EDM_VC2_SENDER_FREE_SLOTS_STREAM_ID 30u

// --- EDMChannelWorkerLocationInfo field offsets -----------------------------
// tt_metal/api/tt-metalium/experimental/fabric/fabric_edm_types.hpp:69-91
// struct is 64B; every field is 16B-strided (padded for safe NOC block reads).
#define WLI_WORKER_SEMAPHORE_ADDRESS 0x00           // uint32_t worker_semaphore_address
#define WLI_WORKER_TEARDOWN_SEMAPHORE_ADDRESS 0x10  // uint32_t worker_teardown_semaphore_address
#define WLI_WORKER_XY 0x20                          // WorkerXY worker_xy (packed (y<<16)|x)
#define WLI_EDM_READ_COUNTER 0x30                   // uint32_t edm_read_counter
#define SIZEOF_EDM_CHANNEL_WORKER_LOCATION_INFO 64

// --- SenderChannelProducerCursor field offsets ------------------------------
// tt_metal/api/tt-metalium/experimental/fabric/fabric_edm_types.hpp:100-107 (16B)
#define SCPC_WRITE_COUNTER 0x00  // uint32_t write_counter (free-running, wraps at 2^32)
#define SCPC_WRITE_INDEX 0x04    // uint32_t write_index   (0 .. num_buffers-1)

// --- WorkerXY packing -------------------------------------------------------
// tt_metal/api/tt-metalium/experimental/fabric/fabric_edm_types.hpp:29
#define WORKER_XY_TO_UINT32(x, y) ((((uint32_t)(y)) << 16) | ((uint32_t)(x) & 0xFFFF))

// --- Blackhole NOC address encoding -----------------------------------------
// tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h
//   NOC_ADDR_LOCAL_BITS  = 36        (L221)
//   NOC_ADDR_NODE_ID_BITS = 6        (L222)
//   NOC_XY_ADDR(x,y,addr) = ((uint64_t)y << (36+6)) | ((uint64_t)x << 36) | addr   (L336-338)
// This is the 64-bit destination NOC address the EDM router uses to write on
// chip B. It is built with the *NOC0* coordinate scheme (the request supplies
// chip-B receiver NOC0 x/y). No coordinate translation is applied here.
// CONFIRM ON HARDWARE: to_noc_unicast_write() in fabric_edm_packet_header.hpp:476-495
// passes the coords through safe_get_noc_addr(..., edm_to_local_chip_noc), which
// may apply NOC virtualization / DYNAMIC_NOC translation. For NOC0 that mapping
// is identity, but if the fabric was built to use translated/virtual coordinates
// the (x,y) here must be the *translated* coords, not raw NOC0.
#define NOC_ADDR_LOCAL_BITS 36
#define NOC_ADDR_NODE_ID_BITS 6
static inline uint64_t noc_xy_addr(uint32_t x, uint32_t y, uint32_t addr) {
    return (((uint64_t)y) << (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) | (((uint64_t)x) << NOC_ADDR_LOCAL_BITS) |
           ((uint64_t)addr);
}

// --- Stream register (the CRITICAL unproven credit primitive) ---------------
// tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_overlay_parameters.h
//   NOC_OVERLAY_START_ADDR     = 0xFFB40000   (L43)
//   NOC_STREAM_REG_SPACE_SIZE  = 0x1000       (L44)
//   STREAM_REG_ADDR(sid, rid)  = 0xFFB40000 + sid*0x1000 + (rid << 2)   (L46-47)
//   STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX = 270       (L779)  [Blackhole]
//   REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM        = 0          (L780)
//   REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM_WIDTH  = 6          (L781)
//   REMOTE_DEST_BUF_WORDS_FREE_INC = 0 + 6 = 6                          (L782-783)
// tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp:83-90 (get_stream_reg_write_addr)
// tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp:92-94 (pack_value_for_inc_on_write_stream_reg_write)
//
// !!! CONFIRM ON HARDWARE (highest-risk item in the whole design) !!!
// The register index 270 and inc-shift 6 are the *Blackhole* values (Wormhole
// differs: index 34, but same shift derivation). More importantly, whether an
// x280 TLB-window store even REACHES a NOC stream register with the hardware's
// inc-on-write (atomic add) semantics is UNPROVEN. If this does not work, the
// receiver never gets data — isolate with the standalone stream-reg probe
// (plan Task 8) before deeper debugging.
#define NOC_OVERLAY_START_ADDR 0xFFB40000u
#define NOC_STREAM_REG_SPACE_SIZE 0x1000u
#define STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX 270u  // Blackhole
#define REMOTE_DEST_BUF_WORDS_FREE_INC 6u
static inline uint32_t stream_reg_write_addr(uint32_t stream_id) {
    return NOC_OVERLAY_START_ADDR + stream_id * NOC_STREAM_REG_SPACE_SIZE +
           (STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX << 2);
}
// pack_value_for_inc_on_write_stream_reg_write(-1) == (-1) << 6 == 0xFFFFFFC0.
// The reference shifts a signed int32; we shift the equivalent unsigned bit
// pattern (0xFFFFFFFF << 6) to get the identical 0xFFFFFFC0 without relying on
// signed-shift UB.
#define CREDIT_MINUS_ONE_PACKED ((uint32_t)0xFFFFFFFFu << REMOTE_DEST_BUF_WORDS_FREE_INC)

// --- Packet header layout (PACKET_HEADER_TYPE for FABRIC_1D low-latency) -----
// tt_metal/fabric/fabric_edm_packet_header.hpp
//   Default FABRIC_1D header is LowLatencyPacketHeaderT<ExtensionWords>.
//   Layout of PacketHeaderBase (L407-423) + LowLatencyRoutingFieldsT (L1088-1122):
//     0x00  command_fields (NocCommandFields union, 40B). For a unicast write
//           the first 8B are NocUnicastCommandHeader::noc_address (uint64).
//           (union def L357-369; NocUnicastCommandHeader L173-175)
//     0x28  payload_size_bytes (uint16)
//     0x2A  noc_send_type      (uint8)  -> NOC_UNICAST_WRITE = 0 (enum L97-98)
//     0x2B  src_ch_id          (uint8)  (reserved for worker; leave 0)
//     0x2C  routing_fields.value (uint32) -> 1-hop unicast encoding (see below)
//     0x30  route_buffer[0]    (uint32, only present when ExtensionWords>=1)
//     0x34  padding0[...]
//   sizeof:
//     LowLatencyPacketHeaderT<0> == 48B, <1> == 64B (L1194-1197).
//
// !!! CONFIRM ON HARDWARE: header size / ExtensionWords !!!
// The host-side default is LowLatencyPacketHeader = LowLatencyPacketHeaderT<1>
// (64B) — fabric_edm_packet_header.hpp:1208. The DEVICE build selects the
// instantiation via the injected FABRIC_1D_PKT_HDR_EXTENSION_WORDS define
// (L1210-1213), which mirrors LOW_LATENCY_EXTENSION_WORDS
// (hostdevcommon/api/hostdevcommon/fabric_common.h:150). The EDM router on the
// eth core is compiled with THAT same value, and the slot layout (payload sits
// at slot_base + sizeof(header), adapter L363) must byte-match it. If the fabric
// on this box was built with EXTENSION_WORDS=0 the header is 48B, not 64B —
// change PKT_HEADER_SIZE below to 48. Fields at 0x00..0x2C are identical in both.
#define PKT_HEADER_SIZE 64u  // CONFIRM ON HARDWARE: 64 (ExtensionWords=1) or 48 (ExtensionWords=0)
#define PH_OFF_NOC_ADDRESS 0x00u
#define PH_OFF_PAYLOAD_SIZE 0x28u
#define PH_OFF_NOC_SEND_TYPE 0x2Au
#define PH_OFF_SRC_CH_ID 0x2Bu
#define PH_OFF_ROUTING_VALUE 0x2Cu
#define NOC_SEND_TYPE_UNICAST_WRITE 0u  // NocSendType::NOC_UNICAST_WRITE (L98)

// 1D unicast routing_fields.value for a given hop count.
// tt_metal/hostdevcommon/api/hostdevcommon/fabric_common.h:239-267 (encode_1d_unicast)
//   Each hop is 2 bits, consumed LSB-first: FORWARD_ONLY (0b10) for transit hops,
//   WRITE_ONLY (0b01) for the final hop.
//   For num_hops == 1: write_hop_index=0, forward_mask=0 -> value = WRITE_ONLY = 0b01 = 1.
//   route_buffer[0] (the extension word) stays 0 for a single hop.
// CONFIRM ON HARDWARE: only num_hops==1 (adjacent chip) is exercised here; the
// general encoder is reproduced in routing_value_for_hops() for >1 hops.
static inline uint32_t routing_value_for_hops(uint32_t num_hops) {
    // Reproduces encode_1d_unicast for buffer[0] (the active routing word).
    // FWD_ONLY_FIELD = 0xAAAAAAAA (all 0b10); WRITE_ONLY = 0b01. (fabric_common.h:179-180,175)
    if (num_hops == 0) {
        return 0;  // self-route
    }
    const uint32_t write_hop_index = num_hops - 1;
    // BASE_HOPS = 16 hops per 32-bit word (fabric_common.h:178). >16 hops spill
    // into route_buffer[] extension words, which this single-hop path never uses.
    const uint32_t write_bit_pos = (write_hop_index % 16u) * 2u;  // FIELD_WIDTH = 2
    const uint32_t forward_mask = (write_bit_pos == 0) ? 0u : ((1u << write_bit_pos) - 1u);
    const uint32_t WRITE_ONLY = 0b01u;
    const uint32_t FWD_ONLY_FIELD = 0xAAAAAAAAu;
    return (FWD_ONLY_FIELD & forward_mask) | (WRITE_ONLY << write_bit_pos);
}

// ----------------------------------------------------------------------------
// x280 SELF NOC coordinates (the L2CPU tile hosting this firmware).
// ----------------------------------------------------------------------------
// !!! CONFIRM ON HARDWARE / KNOWN GAP !!!
// open_start() writes the worker's own NOC (x,y) into the EDM's
// EDMChannelWorkerLocationInfo.worker_xy (adapter L502-506, using
// WorkerXY(my_x[0], my_y[0])). The EDM combines worker_xy with
// worker_semaphore_address to know WHERE to push free-slot credits back to.
// A Tensix worker reads my_x[]/my_y[] from its own NIU; the x280 has no such
// globals and the shared mailbox (fabric_mbox.h) currently carries NO field for
// the L2CPU tile's own NOC coordinates. Until the host delivers them, these are
// placeholders. The follow-up agent MUST set these to the L2CPU tile's NOC0
// (x,y), OR extend the mailbox with two words and read them here.
// NOTE: for the FIRST single packet the credit sink is *seeded* during open()
// from the EDM's edm_read_counter (see open handshake), so one packet can be
// sent even before the EDM ever pushes an update here — but correct multi-packet
// flow control REQUIRES a correct worker_xy.
#define X280_SELF_NOC_X 0u  // CONFIRM ON HARDWARE: L2CPU tile NOC0 x
#define X280_SELF_NOC_Y 0u  // CONFIRM ON HARDWARE: L2CPU tile NOC0 y

// LIM scratch addresses (UNCACHED GDDR alias) the firmware owns for its own
// worker-side state that the EDM reaches over NOC. These live just past the
// status block (FF_MBOX_STATUS + 0x20 = FF_MBOX + 0x1A0) and before any host
// region. The host-chosen FF_CONN_WORKER_FREESLOTS_L1 is the credit sink; this
// teardown word is picked here (the host does not currently supply one).
// CONFIRM ON HARDWARE: these must be NOC-addressable local addresses on the
// L2CPU tile. The uncached alias base 0x3010_0000 is the same the mailbox uses.
#define FF_WORKER_TEARDOWN_L1 (FF_MBOX + 0x1C0)  // u32, EDM writes 1 here to ack teardown

// ----------------------------------------------------------------------------
// Bounded-spin cap. Sized to be "large" without being effectively infinite; at
// ~1 GHz a bare load loop of 1e8 iterations is well under a second, plenty for
// an EDM handshake, and guarantees the one-shot hart never wedges.
// CONFIRM ON HARDWARE: tune against real EDM open/drain latencies.
// ----------------------------------------------------------------------------
#define SPIN_CAP 100000000ull

// ----------------------------------------------------------------------------
// Mailbox helpers.
// ----------------------------------------------------------------------------
static inline void set_state(uint64_t s) {
    REG64(FF_MBOX_FW_STATE) = s;
    REG32(FF_MBOX_STATUS + FF_STATUS_STATE) = (uint32_t)s;
    fence();
}
static inline void fault_and_park(uint64_t code) {
    REG64(FF_MBOX_FAULT_CODE) = code;
    fence();
    // Keep heartbeating so the host can read the fault without a hang. The hart
    // release is one-shot per chip reset; never return / never wedge silently.
    uint64_t hb = REG64(FF_MBOX_HEARTBEAT);
    for (;;) {
        REG64(FF_MBOX_HEARTBEAT) = ++hb;
        fence();
    }
}

// ----------------------------------------------------------------------------
// Window-access helpers. Two aiming contexts on the SAME eth (EDM) tile:
//   aim_edm_l1():  window base 0  -> aperture+off reaches any eth-L1 addr < 2MB.
//                  (BH eth-core L1 is < 2 MiB, so one aim covers the whole tile.)
//   aim_edm_reg(): window base for the 0xFFB4_xxxx stream-register page.
// Re-aim whenever you switch between L1 and the stream-register page.
// ----------------------------------------------------------------------------
static uint32_t g_edm_x, g_edm_y;

static inline void aim_edm_l1(void) { set_window(g_edm_x, g_edm_y, 0); }

static inline uint32_t edm_l1_rd32(uint32_t off) { return REG32(TLB_APERTURE + (off & WINDOW_MASK)); }
static inline void edm_l1_wr32(uint32_t off, uint32_t v) { REG32(TLB_APERTURE + (off & WINDOW_MASK)) = v; }

// !!! THE critical unproven primitive: x280 -> EDM stream-register credit write.
// Writes packed -1 to the sender-channel free-slots UPDATE register on the EDM
// tile; the stream hardware is supposed to perform an inc-on-write (atomic add)
// so the router learns one more slot is occupied. Mirrors adapter L686-689
// (update_edm_buffer_free_slots, non-stateful worker path) and
// fabric_stream_regs.hpp:88-94.
static inline void credit_write_minus_one(uint32_t stream_id) {
    const uint32_t addr = stream_reg_write_addr(stream_id);
    set_window(g_edm_x, g_edm_y, addr);  // aim at the 0xFFB4_xxxx overlay page
    REG32(TLB_APERTURE + (addr & WINDOW_MASK)) = CREDIT_MINUS_ONE_PACKED;
    fence();
}

// ----------------------------------------------------------------------------
// Connection parameters snapshot (read once from FF_MBOX_CONN).
// ----------------------------------------------------------------------------
typedef struct {
    uint32_t edm_noc_x;
    uint32_t edm_noc_y;
    uint32_t edm_buffer_base;
    uint32_t num_buffers;
    uint32_t buffer_size;
    uint32_t handshake_addr;       // edm_connection_handshake_l1_addr (eth L1)
    uint32_t worker_loc_info;      // edm_worker_location_info_addr    (eth L1)
    uint32_t wr_counter_addr;      // edm_copy_of_wr_counter_addr / SenderChannelProducerCursor (eth L1)
    uint32_t credits_stream_id;    // sender_channel_credits_stream_id
    uint32_t worker_freeslots_l1;  // LIM addr the EDM pushes free-slot credits into (x280 polls)
    uint32_t num_hops;             // hops to chip B (1 for adjacent)
} conn_params_t;

static conn_params_t g_cp;

// Worker-local flow-control state (mirrors buffer_slot_write_counter + index).
static uint32_t g_write_counter;  // free-running, matches SenderChannelProducerCursor.write_counter
static uint32_t g_write_index;    // 0 .. num_buffers-1

// Read the ten connection params. Returns 1 if plausible, 0 if a sanity check
// fails (-> FF_FAULT_BAD_PARAMS).
static int read_and_check_conn_params(void) {
    // The whole FF_MBOX_CONN block is in the uncached alias; invalidate to be
    // safe in case an earlier cached read is lingering, then read.
    cbo_inval(FF_MBOX_CONN);
    fence();
    g_cp.edm_noc_x = REG32(FF_MBOX_CONN + FF_CONN_EDM_NOC_X);
    g_cp.edm_noc_y = REG32(FF_MBOX_CONN + FF_CONN_EDM_NOC_Y);
    g_cp.edm_buffer_base = REG32(FF_MBOX_CONN + FF_CONN_EDM_BUFFER_BASE);
    g_cp.num_buffers = REG32(FF_MBOX_CONN + FF_CONN_NUM_BUFFERS);
    g_cp.buffer_size = REG32(FF_MBOX_CONN + FF_CONN_BUFFER_SIZE);
    g_cp.handshake_addr = REG32(FF_MBOX_CONN + FF_CONN_HANDSHAKE_ADDR);
    g_cp.worker_loc_info = REG32(FF_MBOX_CONN + FF_CONN_WORKER_LOC_INFO);
    g_cp.wr_counter_addr = REG32(FF_MBOX_CONN + FF_CONN_WR_COUNTER_ADDR);
    g_cp.credits_stream_id = REG32(FF_MBOX_CONN + FF_CONN_CREDITS_STREAM_ID);
    g_cp.worker_freeslots_l1 = REG32(FF_MBOX_CONN + FF_CONN_WORKER_FREESLOTS_L1);
    g_cp.num_hops = REG32(FF_MBOX_CONN + FF_CONN_NUM_HOPS);

    g_edm_x = g_cp.edm_noc_x;
    g_edm_y = g_cp.edm_noc_y;

    // Sanity checks. Values are conservative; tighten once real ranges known.
    // CONFIRM ON HARDWARE: eth-L1 address ceiling (using 2 MiB here) and the
    // valid NOC coordinate range for this box's harvesting.
    if (g_cp.num_buffers == 0 || g_cp.num_buffers > 64) {
        return 0;
    }
    if (g_cp.buffer_size == 0 || g_cp.buffer_size > 0x200000) {
        return 0;
    }
    if (g_cp.edm_buffer_base == 0 || g_cp.edm_buffer_base >= 0x200000) {
        return 0;
    }
    if (g_cp.worker_loc_info == 0 || g_cp.worker_loc_info >= 0x200000) {
        return 0;
    }
    if (g_cp.handshake_addr == 0 || g_cp.handshake_addr >= 0x200000) {
        return 0;
    }
    if (g_cp.wr_counter_addr == 0 || g_cp.wr_counter_addr >= 0x200000) {
        return 0;
    }
    if (g_cp.worker_freeslots_l1 == 0) {
        return 0;
    }
    if (g_cp.credits_stream_id == 0 || g_cp.credits_stream_id > 63) {
        return 0;
    }
    if (g_cp.num_hops == 0 || g_cp.num_hops > 64) {
        return 0;
    }
    // edm_noc_x/y left unchecked beyond the 6-bit window field; a 0,0 EDM core is
    // implausible but harvesting-dependent, so we do not hard-fail on it.
    return 1;
}

// ----------------------------------------------------------------------------
// OPEN: port of open_start()/open_finish() to window MMIO.
//   adapter L445-561 (open_start / open_finish / open).
// ----------------------------------------------------------------------------
// Returns 1 on success, 0 on timeout.
static int edm_open(void) {
    aim_edm_l1();

    // --- open_start ---------------------------------------------------------
    // 1) Read the SenderChannelProducerCursor block left by the previous
    //    producer on this channel (adapter L457-461). Fields taken verbatim.
    const uint32_t cursor_write_counter = edm_l1_rd32(g_cp.wr_counter_addr + SCPC_WRITE_COUNTER);
    uint32_t cursor_write_index = edm_l1_rd32(g_cp.wr_counter_addr + SCPC_WRITE_INDEX);

    // 2) Read the EDM's edm_read_counter and SEED the worker's free-slots sink
    //    (adapter L463-472: the read lands in edm_buffer_local_free_slots_read_ptr,
    //    which for the worker IS the local L1 credit sink). We store it into the
    //    x280's LIM sink so get_num_free_write_slots() starts consistent.
    const uint32_t edm_read_counter = edm_l1_rd32(g_cp.worker_loc_info + WLI_EDM_READ_COUNTER);
    REG32(g_cp.worker_freeslots_l1) = edm_read_counter;  // LIM (uncached) store
    fence();

    // 3) Write the worker location info the EDM needs:
    //    worker_semaphore_address = the LIM addr where EDM pushes free-slot
    //    credits (adapter L474-492 writes edm_buffer_local_free_slots_update_ptr).
    edm_l1_wr32(g_cp.worker_loc_info + WLI_WORKER_SEMAPHORE_ADDRESS, g_cp.worker_freeslots_l1);
    //    worker_teardown_semaphore_address (adapter L493-501).
    edm_l1_wr32(g_cp.worker_loc_info + WLI_WORKER_TEARDOWN_SEMAPHORE_ADDRESS, FF_WORKER_TEARDOWN_L1);
    //    worker_xy = our own NOC coords (adapter L502-506). See X280_SELF_NOC_* gap.
    edm_l1_wr32(g_cp.worker_loc_info + WLI_WORKER_XY, WORKER_XY_TO_UINT32(X280_SELF_NOC_X, X280_SELF_NOC_Y));
    fence();

    // Clear our teardown ack word before signaling open (adapter L545:
    // *worker_teardown_addr = 0).
    REG32(FF_WORKER_TEARDOWN_L1) = 0;
    fence();

    // --- open_finish --------------------------------------------------------
    // Adopt the previous producer's cursor verbatim (adapter L521-535). The
    // write_index must be < num_buffers (adapter ASSERT L531); if the channel
    // was never initialized this reads back as 0, which is valid.
    if (cursor_write_index >= g_cp.num_buffers) {
        // Defensive: an out-of-range index means the channel's cursor was not
        // initialized as this path expects. Fall back to 0 rather than faulting;
        // CONFIRM ON HARDWARE whether a fresh channel guarantees a zeroed cursor.
        cursor_write_index = 0;
    }
    g_write_counter = cursor_write_counter;
    g_write_index = cursor_write_index;

    // Signal "connected": write open_connection_value to the handshake addr
    // (adapter L540-544).
    edm_l1_wr32(g_cp.handshake_addr, EDM_OPEN_CONNECTION_VALUE);
    fence();

    // The adapter's open_finish does a read barrier, not an explicit ack spin:
    // the EDM's acceptance is observed later when it starts draining slots.
    // For robustness we do a bounded read-back of the handshake word to confirm
    // the store landed (window store completion), not a protocol ack.
    // CONFIRM ON HARDWARE: whether the EDM clears/echoes the handshake word, and
    // whether any positive open-ack is observable to the worker. If not, this
    // read-back is only a store-visibility check and OPEN "succeeds" optimistically.
    for (uint64_t i = 0; i < SPIN_CAP; ++i) {
        uint32_t hs = edm_l1_rd32(g_cp.handshake_addr);
        // Accept either "still shows our open value" or "EDM consumed it" (0 or 1).
        // We cannot distinguish a real ack without the EDM's semantics, so we
        // simply require the store to be readable and return.
        (void)hs;
        return 1;
    }
    return 0;  // unreachable with the current optimistic policy; kept for shape.
}

// ----------------------------------------------------------------------------
// Free-slot accounting (worker / non-stream-reg-receive path).
//   adapter get_num_free_write_slots() L288-301:
//     used = write_counter - *free_slots_read_ptr
//     free = used >= num_buffers ? 0 : num_buffers - used
// The EDM pushes its read counter into the LIM sink (plain inbound write —
// stage-1 proven). We poll that LIM word.
// ----------------------------------------------------------------------------
static uint32_t get_num_free_slots(void) {
    // Invalidate the LIM line so we observe the EDM's latest pushed value even if
    // the CPU cached it (mirrors adapter's invalidate_l1_cache(), L294).
    cbo_inval(g_cp.worker_freeslots_l1);
    fence();
    uint32_t edm_read_counter = REG32(g_cp.worker_freeslots_l1);
    uint32_t used = g_write_counter - edm_read_counter;  // wrap-safe unsigned diff
    if (used >= g_cp.num_buffers) {
        return 0;
    }
    return g_cp.num_buffers - used;
}

// ----------------------------------------------------------------------------
// SEND one packet: port of fabric_unicast_noc_unicast_write (linear/api.h:81-95)
// + send_current_slot_non_blocking (adapter L352-367), translated to window MMIO.
// ----------------------------------------------------------------------------
// Header scratch in CACHED GDDR (.bss). The CPU builds it here coherently, then
// streams it word-by-word into the EDM slot through the window.
static uint8_t g_hdr[PKT_HEADER_SIZE] __attribute__((aligned(16)));

// Returns 1 on success, 0 on free-slot timeout.
static int edm_send(uint32_t payload_lim_addr, uint32_t size, uint32_t dest_x, uint32_t dest_y, uint32_t dest_l1) {
    // 1) Wait for a free slot (bounded). adapter wait_for_empty_write_slot L309-313.
    uint32_t free_slots = 0;
    uint64_t i = 0;
    for (; i < SPIN_CAP; ++i) {
        free_slots = get_num_free_slots();
        if (free_slots > 0) {
            break;
        }
    }
    REG32(FF_MBOX_STATUS + FF_STATUS_LAST_FREESLOTS) = free_slots;
    REG32(FF_MBOX_STATUS + FF_STATUS_SLOTS_SEEN) = free_slots;
    fence();
    if (free_slots == 0) {
        return 0;
    }

    // 2) Build the packet header in LIM/GDDR scratch.
    //    to_chip_unicast(num_hops) + to_noc_unicast_write(dest, size).
    for (uint32_t b = 0; b < PKT_HEADER_SIZE; ++b) {
        g_hdr[b] = 0;
    }
    const uint64_t dst_noc_addr = noc_xy_addr(dest_x, dest_y, dest_l1);
    *(volatile uint64_t*)(g_hdr + PH_OFF_NOC_ADDRESS) = dst_noc_addr;     // command_fields.unicast_write.noc_address
    *(volatile uint16_t*)(g_hdr + PH_OFF_PAYLOAD_SIZE) = (uint16_t)size;  // payload_size_bytes
    g_hdr[PH_OFF_NOC_SEND_TYPE] = (uint8_t)NOC_SEND_TYPE_UNICAST_WRITE;   // noc_send_type
    g_hdr[PH_OFF_SRC_CH_ID] = 0;                                          // src_ch_id (reserved)
    *(volatile uint32_t*)(g_hdr + PH_OFF_ROUTING_VALUE) =
        routing_value_for_hops(g_cp.num_hops);  // routing_fields.value
    // route_buffer[0] and padding remain 0 (single-hop => no extension word set).

    // 3) Compute the EDM slot base and stream payload THEN header (payload before
    //    header — the header write publishes the slot). adapter L361-366.
    const uint32_t slot_base = g_cp.edm_buffer_base + g_write_index * g_cp.buffer_size;
    aim_edm_l1();  // window base 0 covers all eth-L1 addresses < 2 MiB.

    // 3a) Payload: LIM (source) -> slot_base + sizeof(header). 32-bit chunks;
    //     tail bytes handled with a masked read-modify is avoided by requiring
    //     4-byte-aligned size for the fast path and copying the remainder byte
    //     by byte through a temporary word.
    const uint32_t pay_dst = slot_base + PKT_HEADER_SIZE;
    uint32_t n = size;
    uint32_t off = 0;
    // Invalidate the whole payload region in LIM first (cache-line granular).
    for (uint32_t a = payload_lim_addr & ~63u; a < payload_lim_addr + size; a += 64) {
        cbo_inval(a);
    }
    fence();
    while (n >= 4) {
        uint32_t w = REG32(payload_lim_addr + off);
        edm_l1_wr32(pay_dst + off, w);
        off += 4;
        n -= 4;
    }
    if (n) {
        // Tail (<4 bytes): assemble a partial word. The EDM copies exactly
        // payload_size_bytes downstream, so writing a few extra bytes into the
        // slot's payload region is harmless (they are not forwarded).
        uint32_t w = 0;
        for (uint32_t k = 0; k < n; ++k) {
            w |= ((uint32_t)(*(volatile uint8_t*)(uintptr_t)(payload_lim_addr + off + k))) << (8 * k);
        }
        edm_l1_wr32(pay_dst + off, w);
    }
    fence();

    // 3b) Header: scratch -> slot_base. This publishes the slot.
    for (uint32_t b = 0; b < PKT_HEADER_SIZE; b += 4) {
        edm_l1_wr32(slot_base + b, *(volatile uint32_t*)(g_hdr + b));
    }
    fence();

    // 4) Credit: window-store packed -1 to the sender-channel free-slots UPDATE
    //    stream register on the EDM tile. THE unproven primitive.
    credit_write_minus_one(g_cp.credits_stream_id);
    REG32(FF_MBOX_STATUS + FF_STATUS_CREDIT_WRITES) = REG32(FF_MBOX_STATUS + FF_STATUS_CREDIT_WRITES) + 1;
    fence();

    // 5) Advance local write pointers. adapter advance_buffer_slot_write_index L699-718.
    g_write_counter++;
    g_write_index++;
    if (g_write_index >= g_cp.num_buffers) {
        g_write_index = 0;
    }

    // Re-aim back to L1 base for any subsequent poll/read.
    aim_edm_l1();
    return 1;
}

// ----------------------------------------------------------------------------
// CLOSE: best-effort port of close_start()/close_finish() (adapter L568-620).
//   Persist the producer cursor for the next connection, request teardown, and
//   bounded-wait for the EDM's ack. Never wedge.
// ----------------------------------------------------------------------------
static void edm_close(void) {
    aim_edm_l1();
    // Persist cursor (adapter L576-587): write_counter then write_index.
    edm_l1_wr32(g_cp.wr_counter_addr + SCPC_WRITE_COUNTER, g_write_counter);
    edm_l1_wr32(g_cp.wr_counter_addr + SCPC_WRITE_INDEX, g_write_index);
    fence();
    // Request close (adapter L594-599).
    edm_l1_wr32(g_cp.handshake_addr, EDM_CLOSE_CONNECTION_REQUEST_VALUE);
    fence();
    // Wait for teardown ack: EDM writes 1 into our teardown word
    // (adapter close_finish L615, *worker_teardown_addr != 1). Bounded.
    for (uint64_t i = 0; i < SPIN_CAP; ++i) {
        cbo_inval(FF_WORKER_TEARDOWN_L1);
        fence();
        if (REG32(FF_WORKER_TEARDOWN_L1) == 1) {
            REG32(FF_WORKER_TEARDOWN_L1) = 0;
            fence();
            set_state(FF_STATE_CLOSED);
            return;
        }
    }
    // Teardown never acked. Record the fault but do NOT park — the packet was
    // already sent; keep heartbeating from the caller.
    REG64(FF_MBOX_FAULT_CODE) = FF_FAULT_CLOSE_TIMEOUT;
    fence();
}

// ----------------------------------------------------------------------------
// fw_main
// ----------------------------------------------------------------------------
void fw_main(void) {
    uint64_t hb = 0;

    REG64(FF_MBOX_HARTID) = read_mhartid();
    REG64(FF_MBOX_FAULT_CODE) = FF_FAULT_NONE;
    set_state(FF_STATE_ALIVE);

    // 1) Wait for the host to deliver connection params. We treat a nonzero
    //    credits-stream-id as the "params written" marker (the host writes the
    //    whole block, then this field last is not guaranteed — so we also
    //    require the buffer base to be nonzero). Bounded.
    {
        uint64_t i = 0;
        for (; i < SPIN_CAP; ++i) {
            REG64(FF_MBOX_HEARTBEAT) = ++hb;
            fence();
            cbo_inval(FF_MBOX_CONN + FF_CONN_EDM_BUFFER_BASE);
            fence();
            uint32_t marker = REG32(FF_MBOX_CONN + FF_CONN_EDM_BUFFER_BASE);
            if (marker != 0) {
                break;
            }
        }
        if (i >= SPIN_CAP) {
            fault_and_park(FF_FAULT_BAD_PARAMS);
        }
    }

    if (!read_and_check_conn_params()) {
        fault_and_park(FF_FAULT_BAD_PARAMS);
    }
    set_state(FF_STATE_PARAMS_READY);

    // 2) OPEN the EDM connection.
    if (!edm_open()) {
        fault_and_park(FF_FAULT_OPEN_TIMEOUT);
    }
    set_state(FF_STATE_OPENED);

    // 3) Main request loop. A stale request at boot is ignored.
    uint32_t last_seq = REG32(FF_MBOX_REQ + FF_REQ_SEQ);

    for (;;) {
        REG64(FF_MBOX_HEARTBEAT) = ++hb;
        fence();

        cbo_inval(FF_MBOX_REQ);
        fence();
        uint32_t seq = REG32(FF_MBOX_REQ + FF_REQ_SEQ);
        if (seq != 0 && seq != last_seq) {
            // Re-read after a short settle to avoid a torn request block.
            cbo_inval(FF_MBOX_REQ);
            fence();
            if (REG32(FF_MBOX_REQ + FF_REQ_SEQ) != seq) {
                continue;  // still landing; retry next pass
            }
            uint32_t payload_lim = REG32(FF_MBOX_REQ + FF_REQ_PAYLOAD_LIM);
            uint32_t size = REG32(FF_MBOX_REQ + FF_REQ_SIZE);
            uint32_t dest_x = REG32(FF_MBOX_REQ + FF_REQ_DEST_NOC_X);
            uint32_t dest_y = REG32(FF_MBOX_REQ + FF_REQ_DEST_NOC_Y);
            uint32_t dest_l1 = REG32(FF_MBOX_REQ + FF_REQ_DEST_L1);

            // Reject oversize payloads (single-packet first cut: <= one slot).
            if (size == 0 || size > g_cp.buffer_size) {
                REG64(FF_MBOX_FAULT_CODE) = FF_FAULT_BAD_PARAMS;
                fence();
                last_seq = seq;
                continue;
            }

            if (!edm_send(payload_lim, size, dest_x, dest_y, dest_l1)) {
                fault_and_park(FF_FAULT_SLOT_TIMEOUT);
            }
            set_state(FF_STATE_SENT);
            last_seq = seq;

            // CLOSE after the (single) packet. For a multi-packet future this
            // would move out of the loop; here we tear down cleanly so a re-run
            // doesn't leave the connection dangling.
            edm_close();
        }
    }
}
