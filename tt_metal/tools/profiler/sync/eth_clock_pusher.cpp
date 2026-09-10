// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Resident idle-eth clock tracker that PUSHES ITS OWN RING, and drains its chip's active eth cores too.
//
// Runs on one idle ethernet core per chip for the life of the profiling session. It samples this chip's AICLK
// wall clock against the eth tile's free-running 50 MHz counter into its own SPSC lane as PP_CLOCK, then ships
// that lane to the host over its OWN D2H socket in the relay wire format (SPSC span frames). Between its refclk
// strides it also acts as a mini-relay for the chip's ACTIVE eth cores: those run the fabric router and can spend
// no cycles on egress, so this core NoC-reads their control vectors and rings, packs a frame stamped with THEIR
// coordinate, pushes it on the same socket, and writes their heads back -- the decoder resolves a frame's core by
// its XY, so one socket carries every core this pusher serves, exactly as one relay socket carries its cores.
// Nothing is fitted here; all fitting is on the host.
//
// WHY THE ETH CORE PUSHES: the DRAM relay is unrolled for exactly five Tensix RISCs per core and an eth core has
// two. Putting eth in the relay roster meant a heterogeneous frame format through the relay, the frame sizing and
// the decoder lane indexing. Instead the host enumerates every eth core here as a standard 5-lane core whose
// sibling lanes are simply always empty (the decoder skips a lane whose extent is 0 exactly as it does an idle
// TRISC), and the relay never sees an eth core.
//
// WIRE: identical to the relay (hostdev/streaming_profiler_common.h): w0 | payload_words | 5 heads | xy |
// control-vector words 16..31 | per-lane runs preceded by spsc_span_pack_pad. The same three run shapes as the
// relay (flat, near-full wrap image, two-piece wrap), chosen by the SHARED spsc_span_wrap_image predicate, so the
// decoder linearises exactly what was shipped.
//
// EGRESS ORDERING: every data write is flushed before bytes_sent is announced (socket_notify_receiver), the rule
// the relay enforces: the PCIe tile keeps no order between packets, and a 4 B notify has been seen landing ahead
// of the data it announces.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t kStrideTicks = get_compile_time_arg_val(0);       // refclk ticks between samples (50/us)
constexpr uint32_t kSocketConfigAddr = get_compile_time_arg_val(1);  // D2HSocket config in this core's L1
constexpr uint32_t kStageAddr = get_compile_time_arg_val(2);         // one frame slot in this core's L1
constexpr uint32_t kCtrlAddr = get_compile_time_arg_val(3);          // done at +0, heartbeat at +4, stop at +64
constexpr uint32_t kMyXy = get_compile_time_arg_val(4);              // y << 16 | x, this core's frame identity
// Scratch for a linked core: its control vector (256 B) at +0, then its two ring images (2 KiB each) -- 4608 B,
// separate from the frame slot, which is sized for the packed payload alone.
constexpr uint32_t kScratchAddr = get_compile_time_arg_val(5);

constexpr uint32_t kWallClockL = 0xFFB121F0;
constexpr uint32_t kWallClockH = 0xFFB121F8;
constexpr uint32_t kRefclkLoAddr = 0xFFB98850;
constexpr uint32_t kRefclkHiAddr = 0xFFB98854;

namespace kp = kernel_profiler;

// Frame geometry: the standard 5-slot core the host enumerated every eth core as.
constexpr uint32_t kNumRisc = kp::PROFILER_SPSC_TENSIX_RISC;
constexpr uint32_t kNumEthRisc = 2;  // DM0, DM1: the lanes that physically exist on an eth core
static_assert(kNumEthRisc <= kNumRisc, "eth lanes must fit the standard slot count");
constexpr uint32_t kRingWords = kp::PROFILER_L1_VECTOR_SIZE;
constexpr uint32_t kRingBytes = kRingWords * 4u;
constexpr uint32_t kCvWords = kp::PROFILER_L1_CONTROL_VECTOR_SIZE;
constexpr uint32_t kPrefix = kp::SPSC_SPAN_PREFIX_WORDS;
constexpr uint32_t kWireCtrl = kp::SPSC_SPAN_WIRE_CTRL_WORDS;
constexpr uint32_t kPageWords = kp::SPSC_SPAN_PAGE_WORDS;
constexpr uint32_t kPageBytes = kPageWords * 4u;
constexpr uint32_t kLenWord = 1;
// Ship once the live lane holds this many words, or after this many strides regardless, so a trickle still
// reaches the host within ~1 ms at a 3 us stride. The linked cores are swept on the same cadence.
constexpr uint32_t kShipWords = kRingWords / 4u;
constexpr uint32_t kMaxDeferStrides = 333;
constexpr uint32_t kMaxLinked = 16;  // BH has 14 eth cores

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

// RESERVE-OR-SKIP, never the blocking reserve: a producer on an eth core must not stall. The pairs are absolute,
// so a gap is still measurable as a straight segment between its neighbours.
inline __attribute__((always_inline)) bool ring_has_room(uint32_t nwords) {
    invalidate_l1_cache();
    const uint32_t head = kp::profiler_control_buffer[kp::HEAD_INDEX];
    return (kp::wIndex - head) <= (kp::RING_USABLE - nwords);
}

// The wall clock is the packet's own timestamp (low half here, high half from the sticky timer); the refclk low
// 24 bits ride in low27. The refclk high half is read for the splice guard and never sent: it moves once per
// 85.9 s and the host reconstructs it by unwrapping.
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

// ---- egress ------------------------------------------------------------------------------------------------

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

inline void ship(SocketSenderInterface& s, uint32_t bytes) {
    const uint32_t pages = bytes / kPageBytes;
    socket_reserve_pages(s, pages);
    push_fifo(s, kStageAddr, s.write_ptr, bytes);
    socket_push_pages(s, pages);
    // Data lands before its announcement: flush, then bytes_sent.
    noc_async_writes_flushed();
    socket_notify_receiver(s);
}

// ---- packing ----------------------------------------------------------------------------------------------

// Places one lane's run [start, start+take) from `ring` (its L1 image, already local) into the frame at `off`,
// in the shape the decoder expects for (start, take). Returns the new offset.
inline uint32_t place_run(
    volatile tt_l1_ptr uint32_t* frame,
    const volatile tt_l1_ptr uint32_t* ring,
    uint32_t start,
    uint32_t take,
    uint32_t off) {
    const uint32_t hm = start & (kRingWords - 1u);
    if (hm + take <= kRingWords) {
        off += kp::spsc_span_pack_pad(start, off);
        copy_words(frame + off, ring + hm, take);
        return off + take;
    }
    if (kp::spsc_span_wrap_image(start, take, kRingWords)) {
        off += kp::spsc_span_pack_pad(0u, off);
        copy_words(frame + off, ring, kRingWords);
        return off + kRingWords;
    }
    off += kp::spsc_span_pack_pad(start, off);
    const uint32_t first = kRingWords - hm;
    copy_words(frame + off, ring + hm, first);
    copy_words(frame + off + first, ring, take - first);
    return off + take;
}

inline uint32_t finish_frame(volatile tt_l1_ptr uint32_t* frame, uint32_t xy, uint32_t off) {
    frame[0] = kp::spsc_span_w0();
    frame[kp::SPSC_PREFIX_XY] = xy;
    frame[kLenWord] = off - kPrefix;
    const uint32_t bytes = off * 4u;
    return (bytes + kPageBytes - 1u) & ~(kPageBytes - 1u);
}

// This core: rings and control vector are local. Advances its own heads -- this RISC is their consumer.
inline uint32_t pack_own_frame() {
    volatile tt_l1_ptr uint32_t* cv = kp::profiler_control_buffer;
    volatile tt_l1_ptr uint32_t* frame = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageAddr);
    uint32_t off = kPrefix + kWireCtrl;
    bool live = false;
    invalidate_l1_cache();
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
        off = place_run(frame, kp::profiler_data_buffer[r].data, start, take, off);
        cv[kp::SPSC_RING_HEAD_0 + r] = tail;
    }
    return live ? finish_frame(frame, kMyXy, off) : 0u;
}

// A linked (active) eth core: its control vector and live rings are NoC-read into the scratch, the frame is
// packed from those images, and its heads are written back over the NoC once the frame is staged.
inline uint32_t pack_linked_frame(uint32_t xy, uint32_t prof_l1) {
    constexpr uint32_t kCvScratch = kScratchAddr;
    constexpr uint32_t img_base = kScratchAddr + kp::PROFILER_L1_CONTROL_BUFFER_SIZE;
    volatile tt_l1_ptr uint32_t* cv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCvScratch);
    volatile tt_l1_ptr uint32_t* frame = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageAddr);
    const uint32_t x = xy & 0xFFFFu;
    const uint32_t y = xy >> 16;
    // Control vector first: a tail observed here bounds the ring words read after it.
    noc_async_read(get_noc_addr(x, y, prof_l1), kCvScratch, kCvWords * 4u);
    noc_async_read_barrier();
    bool live = false;
    uint32_t takes[kNumEthRisc];
    for (uint32_t r = 0; r < kNumEthRisc; r++) {
        takes[r] = cv[kp::SPSC_RING_TAIL_0 + r] - cv[kp::SPSC_RING_HEAD_0 + r];
        live = live || takes[r] != 0;
    }
    if (!live) {
        return 0;
    }
    // Whole-ring images of the live lanes.
    for (uint32_t r = 0; r < kNumEthRisc; r++) {
        if (takes[r] != 0) {
            const uint32_t ring_l1 = prof_l1 + kp::PROFILER_L1_CONTROL_BUFFER_SIZE + r * kRingBytes;
            noc_async_read(get_noc_addr(x, y, ring_l1), img_base + r * kRingBytes, kRingBytes);
        }
    }
    noc_async_read_barrier();
    uint32_t off = kPrefix + kWireCtrl;
    copy_words(frame + kPrefix, cv + kp::SPSC_WIRE_CV_BASE, kWireCtrl);
    for (uint32_t r = 0; r < kNumRisc; r++) {
        const uint32_t start = cv[kp::SPSC_RING_HEAD_0 + r];
        frame[kp::SPSC_PREFIX_HEAD_0 + r] = start;
        if (r >= kNumEthRisc || takes[r] == 0) {
            continue;
        }
        const volatile tt_l1_ptr uint32_t* img =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(img_base + r * kRingBytes);
        off = place_run(frame, img, start, takes[r], off);
        // Head write-back: the producer on that core sees its ring drain, as it would from a relay.
        noc_inline_dw_write(
            get_noc_addr(x, y, prof_l1 + (kp::SPSC_RING_HEAD_0 + r) * 4u),
            cv[kp::SPSC_RING_HEAD_0 + r] + takes[r],
            0xF);
    }
    return finish_frame(frame, xy, off);
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

    // Linked cores: rt args [0] = n, then (xy, profiler L1 base) per core.
    const uint32_t n_linked_arg = get_arg_val<uint32_t>(0);
    const uint32_t n_linked = n_linked_arg < kMaxLinked ? n_linked_arg : kMaxLinked;
    uint32_t linked_xy[kMaxLinked];
    uint32_t linked_l1[kMaxLinked];
    for (uint32_t i = 0; i < n_linked; i++) {
        linked_xy[i] = get_arg_val<uint32_t>(1 + 2 * i);
        linked_l1[i] = get_arg_val<uint32_t>(2 + 2 * i);
    }

    SocketSenderInterface sender = create_sender_socket_interface(kSocketConfigAddr);
    set_sender_socket_page_size(sender, kPageBytes);
    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);

    const auto sweep = [&]() {
        uint32_t bytes = pack_own_frame();
        if (bytes != 0) {
            ship(sender, bytes);
        }
        for (uint32_t i = 0; i < n_linked; i++) {
            bytes = pack_linked_frame(linked_xy[i], linked_l1[i]);
            if (bytes != 0) {
                ship(sender, bytes);
            }
        }
    };

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

        // Sweep on fill or on time.
        invalidate_l1_cache();
        const uint32_t fill = kp::profiler_control_buffer[kp::TAIL_INDEX] - kp::profiler_control_buffer[kp::HEAD_INDEX];
        if (fill >= kShipWords || ++strides >= kMaxDeferStrides) {
            strides = 0;
            sweep();
        }

        // Teardown: the relay stop word, written by the host at quiesce. The streaming control layout has no
        // terminate slot; this word is the only stop signal a resident eth kernel gets.
        invalidate_l1_cache();
        if (*stop != 0u) {
            break;
        }
    }

    // Final sweep, then the relay done protocol: Drained once the last page is pushed, Done once every byte acked.
    sweep();
    *done = kp::kRelayDrainedWord;
    socket_barrier(sender);
    noc_async_writes_flushed();
    update_socket_config(sender);
    *done = kp::kRelayDoneWord;
#endif
}
