// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Resident idle-eth clock tracker that PUSHES ITS OWN RING. Runs on one idle ethernet core per chip for the
// life of the profiling session: it samples this chip AICLK wall clock against the eth tile free-running 50 MHz
// counter into its own SPSC lane as PP_CLOCK, then ships that lane to the host over its OWN D2H socket in the
// relay wire format (SPSC span frames). Nothing is fitted here; all fitting is on the host.
//
// WHY THE ETH CORE PUSHES ITSELF: the DRAM relay is unrolled for exactly five Tensix RISCs per core, and an
// eth core has two. Putting eth in the relay roster meant a heterogeneous frame format through the relay,
// the frame sizing and the decoder lane indexing. Instead the eth core, which is idle between refclk strides,
// runs the egress itself: one socket per idle-eth core, the relay never sees an eth core, and the host
// enumerates this core as a standard 5-lane core whose sibling lanes are simply always empty (the decoder
// skips a lane whose extent is 0 exactly as it does an idle TRISC).
//
// WIRE: identical to the relay (hostdev/streaming_profiler_common.h): w0 | payload_words | 5 heads | xy |
// control-vector words 16..31 | per-lane runs preceded by spsc_span_pack_pad. Same three run shapes as the
// relay (flat, near-full wrap image, two-piece wrap) chosen by the SHARED spsc_span_wrap_image predicate,
// so the decoder linearises exactly what was shipped.
//
// EGRESS ORDERING: every data write is flushed before bytes_sent is announced (socket_notify_receiver), the
// same rule the relay enforces -- the PCIe tile keeps no order between packets, and a 4 B notify has been
// seen landing ahead of the data it announces.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t kStrideTicks = get_compile_time_arg_val(0);       // refclk ticks between samples (50/us)
constexpr uint32_t kSocketConfigAddr = get_compile_time_arg_val(1);  // D2HSocket config in this core L1
constexpr uint32_t kStageAddr = get_compile_time_arg_val(2);         // one frame slot in this core L1
constexpr uint32_t kCtrlAddr = get_compile_time_arg_val(3);          // done at +0, heartbeat at +4, stop at +64
constexpr uint32_t kMyXy = get_compile_time_arg_val(4);              // y << 16 | x, the frame identity word

constexpr uint32_t kWallClockL = 0xFFB121F0;
constexpr uint32_t kWallClockH = 0xFFB121F8;
constexpr uint32_t kRefclkLoAddr = 0xFFB98850;
constexpr uint32_t kRefclkHiAddr = 0xFFB98854;

namespace kp = kernel_profiler;

// Frame geometry: the standard 5-slot core the host enumerated this eth core as.
constexpr uint32_t kNumRisc = kp::PROFILER_SPSC_TENSIX_RISC;
constexpr uint32_t kNumEthRisc = 2;  // DM0, DM1: the lanes that physically exist on this core
static_assert(kNumEthRisc <= kNumRisc, "eth lanes must fit the standard slot count");
constexpr uint32_t kRingWords = kp::PROFILER_L1_VECTOR_SIZE;
constexpr uint32_t kPrefix = kp::SPSC_SPAN_PREFIX_WORDS;
constexpr uint32_t kWireCtrl = kp::SPSC_SPAN_WIRE_CTRL_WORDS;
constexpr uint32_t kPageWords = kp::SPSC_SPAN_PAGE_WORDS;
constexpr uint32_t kPageBytes = kPageWords * 4u;
constexpr uint32_t kLenWord = 1;
// Ship once the live lane holds this many words, or after this many strides regardless, so a trickle still
// reaches the host within ~1 ms at a 3 us stride.
constexpr uint32_t kShipWords = kRingWords / 4u;
constexpr uint32_t kMaxDeferStrides = 333;

// Reading L latches H, so L must be read first; both halves then belong to the same instant.
inline __attribute__((always_inline)) void read_wall(uint32_t& hi, uint32_t& lo) {
    lo = *reinterpret_cast<volatile uint32_t*>(kWallClockL);
    hi = *reinterpret_cast<volatile uint32_t*>(kWallClockH);
}
// The refclk pair has no latch: hi/lo/hi guards a splice across a 2^32 boundary.
inline __attribute__((always_inline)) void read_refclk(uint32_t& hi, uint32_t& lo) {
    volatile uint32_t* lop = reinterpret_cast<volatile uint32_t*>(kRefclkLoAddr);
    volatile uint32_t* hip = reinterpret_cast<volatile uint32_t*>(kRefclkHiAddr);
    const uint32_t h1 = *hip;
    uint32_t l = *lop;
    const uint32_t h2 = *hip;
    if (h1 != h2) {
        l = *lop;
    }
    hi = h2;
    lo = l;
}
inline __attribute__((always_inline)) uint64_t refclk64() {
    uint32_t hi, lo;
    read_refclk(hi, lo);
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

#if defined(PROFILE_KERNEL)

constexpr uint32_t kEmitWords = 1 + 2;  // optional sticky timer + w0 + wall_lo

// RESERVE-OR-SKIP, never the blocking reserve: a producer on an eth core must not stall. The pairs are
// absolute, so a gap is still measurable as a straight segment between its neighbours.
inline __attribute__((always_inline)) bool ring_has_room(uint32_t nwords) {
    invalidate_l1_cache();
    const uint32_t head = kp::profiler_control_buffer[kp::HEAD_INDEX];
    return (kp::wIndex - head) <= (kp::RING_USABLE - nwords);
}

// The wall clock is the packet own timestamp (low half here, high half from the sticky timer); the refclk
// low 24 bits ride in low27. The refclk high half is read for the splice guard and never sent: it moves once
// per 85.9 s and the host reconstructs it by unwrapping.
inline __attribute__((always_inline)) void emit_pair() {
    uint32_t whi, wlo, rhi, rlo;
    read_wall(whi, wlo);
    read_refclk(rhi, rlo);
    (void)rhi;
    kp::ring_write_sticky_timer(whi);
    kp::ring_write_word(kp::ppfmt::clock_w0(kp::ppfmt::CLOCK_LOCAL_REFCLK, rlo));
    kp::ring_write_word(wlo);
    kp::publish_tail();
}

// ---- egress -------------------------------------------------------------------------------------------------

inline void write_to_host(const SocketSenderInterface& s, uint32_t src_l1, uint64_t dst_pcie, uint32_t size) {
    noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
        NOC_INDEX, src_l1, s.d2h.pcie_xy_enc, dst_pcie, size, 1);
}

// A frame that crosses the FIFO wrap splits into two writes; socket_push_pages only wraps the pointer.
inline void push_fifo(const SocketSenderInterface& s, uint32_t src, uint32_t dst, uint32_t len) {
    const uint32_t fifo_size = s.downstream_fifo_curr_size;
    if (dst >= fifo_size) {
        dst -= fifo_size;
    }
    const uint64_t base = (static_cast<uint64_t>(s.d2h.data_addr_hi) << 32) | s.downstream_fifo_addr;
    const uint32_t first = (dst + len > fifo_size) ? fifo_size - dst : len;
    write_to_host(s, src, base + dst, first);
    if (first < len) {
        write_to_host(s, src + first, base, len - first);
    }
}

inline void copy_words(volatile tt_l1_ptr uint32_t* dst, const volatile tt_l1_ptr uint32_t* src, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// Packs this core live lanes into one frame at kStageAddr and returns its byte length rounded to a page, or 0
// when nothing is live. Advances this core own heads: the consumer of these rings is this very RISC.
inline uint32_t pack_own_frame() {
    volatile tt_l1_ptr uint32_t* cv = kp::profiler_control_buffer;
    volatile tt_l1_ptr uint32_t* frame = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageAddr);
    uint32_t off = kPrefix + kWireCtrl;
    bool live = false;
    invalidate_l1_cache();
    // Control block: CV words 16..31 exactly as the relay one 64 B read lands them (state slots, then tails).
    copy_words(frame + kPrefix, cv + kp::SPSC_WIRE_CV_BASE, kWireCtrl);
    for (uint32_t r = 0; r < kNumRisc; r++) {
        const uint32_t start = cv[kp::SPSC_RING_HEAD_0 + r];
        const uint32_t tail = r < kNumEthRisc ? cv[kp::SPSC_RING_TAIL_0 + r] : start;
        const uint32_t take = tail - start;
        frame[kp::SPSC_PREFIX_HEAD_0 + r] = start;
        if (take == 0) {
            continue;
        }
        live = true;
        const volatile tt_l1_ptr uint32_t* ring = kp::profiler_data_buffer[r].data;
        const uint32_t hm = start & (kRingWords - 1u);
        if (hm + take <= kRingWords) {
            off += kp::spsc_span_pack_pad(start, off);
            copy_words(frame + off, ring + hm, take);
            off += take;
        } else if (kp::spsc_span_wrap_image(start, take, kRingWords)) {
            off += kp::spsc_span_pack_pad(0u, off);
            copy_words(frame + off, ring, kRingWords);
            off += kRingWords;
        } else {
            off += kp::spsc_span_pack_pad(start, off);
            const uint32_t first = kRingWords - hm;
            copy_words(frame + off, ring + hm, first);
            copy_words(frame + off + first, ring, take - first);
            off += take;
        }
        cv[kp::SPSC_RING_HEAD_0 + r] = tail;  // consumed: frees the ring for the producer on this core
    }
    if (!live) {
        return 0;
    }
    frame[0] = kp::spsc_span_w0();
    frame[kp::SPSC_PREFIX_XY] = kMyXy;
    frame[kLenWord] = off - kPrefix;
    const uint32_t bytes = off * 4u;
    return (bytes + kPageBytes - 1u) & ~(kPageBytes - 1u);
}

inline void ship(SocketSenderInterface& s, uint32_t bytes) {
    const uint32_t pages = bytes / kPageBytes;
    socket_reserve_pages(s, pages);
    push_fifo(s, kStageAddr, s.write_ptr, bytes);
    socket_push_pages(s, pages);
    // Data lands before its announcement: flush, then bytes_sent.
    noc_async_writes_flushed();
    socket_notify_receiver(s);
}

#endif

void kernel_main() {
#if defined(PROFILE_KERNEL)
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCtrlAddr);
    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCtrlAddr + 4);
    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCtrlAddr + 64);
    *done = 0;
    *hb = 0;
    *stop = 0;

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    set_sender_socket_page_size(sender, kPageBytes);
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);

    bool armed = false;  // PROFILER_TERMINATE observed clear at least once (a stale set from a past session)
    uint32_t strides = 0;
    uint64_t target = refclk64() + kStrideTicks;
    while (true) {
        // Pace in REFCLK, not wall clock: a cadence DVFS cannot stretch.
        uint64_t rc = refclk64();
        while (rc < target) {
            rc = refclk64();
        }
        target = rc + kStrideTicks;
        (*hb)++;

        if (ring_has_room(kEmitWords)) {
            emit_pair();
        }

        // Ship on fill or on time.
        invalidate_l1_cache();
        const uint32_t fill = kp::profiler_control_buffer[kp::TAIL_INDEX] - kp::profiler_control_buffer[kp::HEAD_INDEX];
        if (fill >= kShipWords || ++strides >= kMaxDeferStrides) {
            strides = 0;
            const uint32_t bytes = pack_own_frame();
            if (bytes != 0) {
                ship(sender, bytes);
            }
        }

        // Teardown: the relay stop word, or the profiler own terminate flag (armed on a clear read first).
        invalidate_l1_cache();
        const uint32_t term = kp::profiler_control_buffer[kp::PROFILER_TERMINATE];
        if (!armed) {
            armed = (term == 0u);
        }
        if (*stop != 0u || (armed && term != 0u)) {
            break;
        }
    }

    // Final sweep, then the relay done protocol: Drained once the last page is pushed, Done once every byte acked.
    const uint32_t bytes = pack_own_frame();
    if (bytes != 0) {
        ship(sender, bytes);
    }
    *done = kp::kRelayDrainedWord;
    socket_barrier(sender);
    noc_async_writes_flushed();
    update_socket_config(sender);
    *done = kp::kRelayDoneWord;
#endif
}
