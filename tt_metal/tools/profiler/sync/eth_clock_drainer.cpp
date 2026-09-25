// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Resident idle-eth drainer: ships everything its chip's eth cores produce, so the clock pusher never leaves its
// sampling loop.
//
// Runs on a second idle ethernet core per chip, the NoC-nearest to the pusher, for the life of the profiling
// session. Over the NoC it reads the pusher's sync ring (its clock model's instants) and ships them as sync frames
// stamped with the PUSHER's coordinate on the SYNC socket, writing the consumed count back into the pusher's control
// block; and it is the mini-relay for the chip's eth cores -- the pusher's own firmware markers, the active eth
// cores' rings and their link ends' stamp rings -- exactly as the pusher was before it (eth_clock_pusher.cpp has the
// history; hostdev/streaming_profiler_common.h the wire format). Every transfer here ends in a PCIe write flush of
// several microseconds, which on the pusher was a hole in its samples that a glide bent across.
//
// EGRESS ORDERING: every data write is flushed before bytes_sent is announced (socket_notify_receiver), the rule
// the relay enforces: the PCIe tile keeps no order between packets, and a 4 B notify has been seen landing ahead
// of the data it announces.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "internal/ethernet/eth_ptp_clock.hpp"

#if defined(PROFILE_KERNEL)
constexpr uint32_t kSocketConfigAddr = get_compile_time_arg_val(0);  // PROFILER socket: the eth cores' frames
constexpr uint32_t kSyncCfgAddr = get_compile_time_arg_val(1);       // SYNC socket: the pusher's and link ends' records
constexpr uint32_t kStageAddr = get_compile_time_arg_val(2);         // one frame slot in this core's L1
constexpr uint32_t kCtrlAddr = get_compile_time_arg_val(3);          // done +0, heartbeat +4, go +8, stop +64
constexpr uint32_t kScratchAddr = get_compile_time_arg_val(4);       // images read over the NoC
constexpr uint32_t kPusherXy = get_compile_time_arg_val(5);          // y << 16 | x of the pusher
constexpr uint32_t kPusherCtrl = get_compile_time_arg_val(6);  // the pusher's control block: sync tail +12, head +16
constexpr uint32_t kPusherSyncRing = get_compile_time_arg_val(7);  // the pusher's sync ring, kSyncRingRecords records
constexpr uint32_t kLinkRingAddr =
    get_compile_time_arg_val(8);  // the link ends' sync ring, one address on every active eth core; 0 = none

namespace kp = kernel_profiler;
namespace eth_ptp = tt::tt_metal::eth_ptp;
constexpr uint32_t kNumRisc = kp::PROFILER_SPSC_TENSIX_RISC;
constexpr uint32_t kNumEthRisc = 2;  // DM0, DM1: the lanes that physically exist on an eth core
constexpr uint32_t kRingWords = kp::PROFILER_L1_VECTOR_SIZE;
constexpr uint32_t kRingBytes = kRingWords * 4u;
constexpr uint32_t kCvWords = kp::PROFILER_L1_CONTROL_VECTOR_SIZE;
constexpr uint32_t kPrefix = kp::SPSC_SPAN_PREFIX_WORDS;
constexpr uint32_t kWireCtrl = kp::SPSC_SPAN_WIRE_CTRL_WORDS;
constexpr uint32_t kPageWords = kp::SPSC_SPAN_PAGE_WORDS;
constexpr uint32_t kPageBytes = kPageWords * 4u;
constexpr uint32_t kLenWord = 1;
constexpr uint32_t kMaxLinked = 16;  // BH has 14 eth cores
constexpr uint32_t kRecordBytes = kp::kSyncRecordWords * 4u;
constexpr uint32_t kSweepCycles = 1u << 20;  // ~1 ms of this core's wall clock between frame sweeps
constexpr uint32_t kPollCycles = 4096;       // ~3 us between reads of the pusher's tail
// Scratch: a control vector image, two ring images, a link end's record ring image, a slice of the pusher's sync
// ring, then a 64 B block of the pusher's control words.
constexpr uint32_t kCvScratch = kScratchAddr;
constexpr uint32_t kImgBase = kScratchAddr + kp::PROFILER_L1_CONTROL_BUFFER_SIZE;
constexpr uint32_t kLinkScratch = kImgBase + kNumEthRisc * kRingBytes;
constexpr uint32_t kLinkRingBytes = kp::kLinkSyncRingRecords * kRecordBytes;
constexpr uint32_t kSliceScratch = kLinkScratch + kLinkRingBytes;
constexpr uint32_t kSliceBytes = kp::kSyncFrameRecords * kRecordBytes;
constexpr uint32_t kPusherCtrlScratch = (kSliceScratch + kSliceBytes + 63u) & ~63u;
constexpr uint32_t kAnchorRecords = 8;
constexpr uint32_t kAnchorScratch = kPusherCtrlScratch + 64;
// The anchor audit: anchors waiting for the pusher's next point, then the histogram's bins.
constexpr uint32_t kPendingAnchors = 512;
constexpr uint32_t kPendingScratch = kAnchorScratch + kAnchorRecords * kRecordBytes;
constexpr uint32_t kBinScratch = kPendingScratch + kPendingAnchors * 16u;
static_assert(kBinScratch + kp::kSyncAnchorHistBins * 4u <= kScratchAddr + 16384, "the host carves 16384 B of scratch");

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

__attribute__((noinline)) void ship(SocketSenderInterface& s, uint32_t bytes) {
    const uint32_t pages = bytes / kPageBytes;
    socket_reserve_pages(s, pages);
    push_fifo(s, kStageAddr, s.write_ptr, bytes);
    socket_push_pages(s, pages);
    // Data lands before its announcement: flush, then bytes_sent.
    noc_async_writes_flushed();
    socket_notify_receiver(s);
}

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

// A sync frame for core `xy`: records [first, first + n) of the ring of `ring_records` at `recs`.
inline uint32_t pack_sync_frame(
    uint32_t xy, const volatile tt_l1_ptr uint32_t* recs, uint32_t first, uint32_t n, uint32_t ring_records) {
    volatile tt_l1_ptr uint32_t* frame = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageAddr);
    for (uint32_t r = 0; r < kNumRisc; r++) {
        frame[kp::SPSC_PREFIX_HEAD_0 + r] = 0;
    }
    frame[kp::SPSC_PREFIX_HEAD_0] = n;
    uint32_t off = kPrefix;
    for (uint32_t i = 0; i < n; i++) {
        copy_words(frame + off, recs + ((first + i) % ring_records) * kp::kSyncRecordWords, kp::kSyncRecordWords);
        off += kp::kSyncRecordWords;
    }
    while (off < kPrefix + kWireCtrl) {
        frame[off++] = 0;
    }
    return finish_frame(frame, xy, off);
}

// An eth core's frame: its control vector and live rings are NoC-read into the scratch, the frame is packed from
// those images, and its heads are written back over the NoC once the frame is staged.
inline uint32_t pack_linked_frame(uint32_t xy, uint32_t prof_l1) {
    volatile tt_l1_ptr uint32_t* cv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCvScratch);
    volatile tt_l1_ptr uint32_t* frame = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kStageAddr);
    const uint32_t x = xy & 0xFFFFu;
    const uint32_t y = xy >> 16;
    // Control vector first: a tail observed here bounds the ring words read after it.
    noc_async_read(get_noc_addr(x, y, prof_l1), kCvScratch, kCvWords * 4u);
    noc_async_read_barrier();
    bool live = false;
    uint32_t starts[kNumEthRisc];
    uint32_t takes[kNumEthRisc];
    for (uint32_t r = 0; r < kNumEthRisc; r++) {
        const uint32_t tail = cv[kp::SPSC_RING_TAIL_0 + r];
        starts[r] = cv[kp::SPSC_RING_HEAD_0 + r];
        takes[r] = tail - starts[r];
        if (takes[r] > kRingWords) {
            // Lapped: a producer wrote past its consumer. Only the last ring image is still intact; ship that and
            // never index past it.
            starts[r] = tail - kRingWords;
            takes[r] = kRingWords;
        }
        live = live || takes[r] != 0;
    }
    if (!live) {
        return 0;
    }
    for (uint32_t r = 0; r < kNumEthRisc; r++) {
        if (takes[r] == 0) {
            continue;
        }
        const uint32_t ring_l1 = prof_l1 + kp::PROFILER_L1_CONTROL_BUFFER_SIZE + r * kRingBytes;
        const uint32_t img = kImgBase + r * kRingBytes;
        const uint32_t hm = starts[r] & (kRingWords - 1u);
        if (hm + takes[r] <= kRingWords) {
            noc_async_read(get_noc_addr(x, y, ring_l1 + hm * 4u), img + hm * 4u, takes[r] * 4u);
        } else if (kp::spsc_span_wrap_image(starts[r], takes[r], kRingWords)) {
            noc_async_read(get_noc_addr(x, y, ring_l1), img, kRingBytes);
        } else {
            const uint32_t first = kRingWords - hm;
            noc_async_read(get_noc_addr(x, y, ring_l1 + hm * 4u), img + hm * 4u, first * 4u);
            noc_async_read(get_noc_addr(x, y, ring_l1), img, (takes[r] - first) * 4u);
        }
    }
    noc_async_read_barrier();
    uint32_t off = kPrefix + kWireCtrl;
    copy_words(frame + kPrefix, cv + kp::SPSC_WIRE_CV_BASE, kWireCtrl);
    for (uint32_t r = 0; r < kNumRisc; r++) {
        const uint32_t start = r < kNumEthRisc ? starts[r] : cv[kp::SPSC_RING_HEAD_0 + r];
        frame[kp::SPSC_PREFIX_HEAD_0 + r] = start;
        if (r >= kNumEthRisc || takes[r] == 0) {
            continue;
        }
        const volatile tt_l1_ptr uint32_t* img =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kImgBase + r * kRingBytes);
        off = place_run(frame, img, start, takes[r], off);
        // Head write-back (to the tail observed above): the producer on that core sees its ring drain, as it
        // would from a relay.
        noc_inline_dw_write(get_noc_addr(x, y, prof_l1 + (kp::SPSC_RING_HEAD_0 + r) * 4u), start + takes[r], 0xF);
    }
    return finish_frame(frame, xy, off);
}

// Records of this core's own, eight to a frame: anchors the host checks itself, and the audit at stop.
struct Outbox {
    uint32_t n = 0;
    SocketSenderInterface* sync = nullptr;
    __attribute__((noinline)) void add(uint32_t meta, uint32_t rnd, uint64_t v, uint64_t w, uint64_t ref) {
        volatile tt_l1_ptr uint32_t* r =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kAnchorScratch + n * kRecordBytes);
        r[kp::SYNC_META] = meta;
        r[kp::SYNC_ROUND] = rnd;
        r[kp::SYNC_VALUE_LO] = static_cast<uint32_t>(v);
        r[kp::SYNC_VALUE_HI] = static_cast<uint32_t>(v >> 32);
        r[kp::SYNC_WALL_LO] = static_cast<uint32_t>(w);
        r[kp::SYNC_WALL_HI] = static_cast<uint32_t>(w >> 32);
        r[kp::SYNC_REF_LO] = static_cast<uint32_t>(ref);
        r[kp::SYNC_REF_HI] = static_cast<uint32_t>(ref >> 32);
        if (++n == kAnchorRecords) {
            flush();
        }
    }
    void flush() {
        if (n != 0) {
            ship(
                *sync,
                pack_sync_frame(
                    kPusherXy, reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kAnchorScratch), 0, n, kAnchorRecords));
            n = 0;
        }
    }
};

// Anchors: this core's own (wall, refclk) pair at a refclk update, once per poll, independent of the pusher's samples.
// Each waits for the pusher's next point; between two centroids at one FBDIV its distance from their chord (the host's
// line there too) goes into the histogram in 1/16 ns, the wall moved into the pusher's wall domain by this core's tile
// offset. Any other anchor, and one off by more than the histogram's range, goes to the host as it is; an anchor whose
// read found no refclk update is not one.
struct Audit {
    struct Pending {
        uint32_t r_lo, r_hi, w_lo, w_hi;
    };
    Outbox* outbox = nullptr;
    int32_t tile_offset = 0;
    uint32_t head = 0, tail = 0, unbracketed = 0;
    uint64_t worst_r = 0;
    int32_t worst = 0;
    kp::SyncLocalPoint prev{};
    bool have_prev = false;

    static volatile tt_l1_ptr Pending& pending(uint32_t i) {
        return reinterpret_cast<volatile tt_l1_ptr Pending*>(kPendingScratch)[i % kPendingAnchors];
    }
    static volatile tt_l1_ptr uint32_t* bins() { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kBinScratch); }

    void start(Outbox& o, int32_t offset) {
        outbox = &o;
        tile_offset = offset;
        for (uint32_t b = 0; b < kp::kSyncAnchorHistBins; b++) {
            bins()[b] = 0;
        }
    }
    void raw(const volatile tt_l1_ptr Pending& a) {
        outbox->add(
            kp::kSyncKindAnchor << 8,
            0,
            0,
            (static_cast<uint64_t>(a.w_hi) << 32) | a.w_lo,
            (static_cast<uint64_t>(a.r_hi) << 32) | a.r_lo);
    }
    __attribute__((noinline)) void anchor() {
        const eth_ptp::Instant t = eth_ptp::read_bracketed();
        if (t.spins == 0) {
            unbracketed++;
            return;
        }
        if (tail - head == kPendingAnchors) {
            raw(pending(head++));
        }
        volatile tt_l1_ptr Pending& a = pending(tail++);
        a.r_lo = static_cast<uint32_t>(t.refclk);
        a.r_hi = static_cast<uint32_t>(t.refclk >> 32);
        a.w_lo = t.wall_lo;
        a.w_hi = t.wall_hi;
    }
    // The chord's residue term is a 32-bit product and quotient: consecutive points at one FBDIV are under 2^16 refclk
    // ticks apart and their residue from the slope is a few eighths; anything else ships raw.
    __attribute__((noinline)) void point(const kp::SyncLocalPoint& p) {
        while (head != tail) {
            const volatile tt_l1_ptr Pending& a = pending(head);
            const uint64_t ar = (static_cast<uint64_t>(a.r_hi) << 32) | a.r_lo;
            if (ar > p.r) {
                break;
            }
            head++;
            if (!have_prev || ar < prev.r || prev.k8 == 0 || prev.k8 != p.k8) {
                raw(a);
                continue;
            }
            const uint32_t dr = static_cast<uint32_t>(p.r - prev.r);
            const uint32_t x = static_cast<uint32_t>(ar - prev.r);
            const int32_t e =
                static_cast<int32_t>(static_cast<uint32_t>(p.w8) - static_cast<uint32_t>(prev.w8) - p.k8 * dr);
            if (dr == 0 || dr >= (1u << 16) || e >= (1 << 15) || e <= -(1 << 15)) {
                raw(a);
                continue;
            }
            const uint32_t line = static_cast<uint32_t>(prev.w8) + p.k8 * x +
                                  static_cast<uint32_t>(e * static_cast<int32_t>(x) / static_cast<int32_t>(dr));
            const int32_t err8 = static_cast<int32_t>((a.w_lo + static_cast<uint32_t>(tile_offset)) * 8u - line);
            const int32_t q = err8 * 320;
            const int32_t k = static_cast<int32_t>(p.k8);
            const int32_t ns16 = q >= 0 ? q / k : -((-q + k - 1) / k);
            const int32_t b = ns16 + static_cast<int32_t>(kp::kSyncAnchorHistBins / 2);
            if (b < 0 || b >= static_cast<int32_t>(kp::kSyncAnchorHistBins)) {
                raw(a);
                continue;
            }
            bins()[b]++;
            if ((ns16 < 0 ? -ns16 : ns16) > (worst < 0 ? -worst : worst)) {
                worst = ns16;
                worst_r = ar;
            }
        }
        prev = p;
        have_prev = true;
    }
    // To the host at stop: every nonzero run of six bins, then the worst with its refclk.
    void send() {
        while (head != tail) {
            raw(pending(head++));
        }
        for (uint32_t b = 0; b < kp::kSyncAnchorHistBins; b += 6) {
            uint32_t c[6] = {};
            uint32_t any = 0;
            for (uint32_t j = 0; j < 6 && b + j < kp::kSyncAnchorHistBins; j++) {
                c[j] = bins()[b + j];
                any |= c[j];
            }
            if (any != 0) {
                outbox->add(
                    kp::kSyncKindAnchorHist << 8,
                    b,
                    (static_cast<uint64_t>(c[1]) << 32) | c[0],
                    (static_cast<uint64_t>(c[3]) << 32) | c[2],
                    (static_cast<uint64_t>(c[5]) << 32) | c[4]);
            }
        }
        outbox->add(
            kp::kSyncKindAnchorHist << 8, kp::kSyncAnchorHistWorst, worst_r, unbracketed, static_cast<uint32_t>(worst));
        outbox->flush();
    }
};

// The chip's eth cores this core ships: rt args [0] = n, then (xy, profiler L1 base) per core, the pusher first, then
// the active eth cores (whose link ends' records ship too), then this core's tile offset.
struct Drainer {
    SocketSenderInterface sender, sync_sender;
    uint32_t n_linked = 0;
    uint32_t linked_xy[kMaxLinked] = {}, linked_l1[kMaxLinked] = {}, link_cursor[kMaxLinked] = {};
    uint32_t head = 0;  // the pusher's records consumed
    Outbox outbox;
    Audit audit;

    void start() {
        const uint32_t n_linked_arg = get_arg_val<uint32_t>(0);
        n_linked = n_linked_arg < kMaxLinked ? n_linked_arg : kMaxLinked;
        for (uint32_t i = 0; i < n_linked; i++) {
            linked_xy[i] = get_arg_val<uint32_t>(1 + 2 * i);
            linked_l1[i] = get_arg_val<uint32_t>(2 + 2 * i);
        }
        sender = create_sender_socket_interface(kSocketConfigAddr);
        set_sender_socket_page_size(sender, kPageBytes);
        sync_sender = create_sender_socket_interface(kSyncCfgAddr);
        set_sender_socket_page_size(sync_sender, kPageBytes);
        noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
        outbox.sync = &sync_sender;
        audit.start(outbox, static_cast<int32_t>(get_arg_val<uint32_t>(1 + 2 * n_linked_arg)));
    }

    // The pusher's records [head, tail): the tail from its control block, the records from its ring, the consumed
    // count written back so it knows how far it may overwrite. Full frames only, or everything with `all` (at each
    // sweep and at stop): a frame pads to 24 words, so one of a single record is four times its size.
    __attribute__((noinline)) void drain_pusher(bool all) {
        const uint32_t px = kPusherXy & 0xFFFFu, py = kPusherXy >> 16;
        volatile tt_l1_ptr uint32_t* pctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kPusherCtrlScratch);
        for (;;) {
            noc_async_read(get_noc_addr(px, py, kPusherCtrl), kPusherCtrlScratch, 64);
            noc_async_read_barrier();
            const uint32_t tail = pctl[3];
            if (tail == head || (!all && tail - head < kp::kSyncFrameRecords)) {
                return;
            }
            const uint32_t left = tail - head;
            const uint32_t n = left < kp::kSyncFrameRecords ? left : kp::kSyncFrameRecords;
            const uint32_t h = head % kp::kSyncRingRecords;
            const uint32_t to_end = kp::kSyncRingRecords - h;
            if (n <= to_end) {
                noc_async_read(
                    get_noc_addr(px, py, kPusherSyncRing + h * kRecordBytes), kSliceScratch, n * kRecordBytes);
            } else {
                noc_async_read(
                    get_noc_addr(px, py, kPusherSyncRing + h * kRecordBytes), kSliceScratch, to_end * kRecordBytes);
                noc_async_read(
                    get_noc_addr(px, py, kPusherSyncRing),
                    kSliceScratch + to_end * kRecordBytes,
                    (n - to_end) * kRecordBytes);
            }
            noc_async_read_barrier();
            for (uint32_t i = 0; i < n; i++) {
                const volatile tt_l1_ptr uint32_t* rec =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kSliceScratch + i * kRecordBytes);
                if (((rec[kp::SYNC_META] >> 8) & 0xFFu) == kp::kSyncKindLocal) {
                    kp::SyncLocalPoint pts[kp::kSyncLocalPoints];
                    const uint32_t np = kp::sync_local_unpack(rec, pts);
                    for (uint32_t j = 0; j < np; j++) {
                        audit.point(pts[j]);
                    }
                }
            }
            ship(
                sync_sender,
                pack_sync_frame(
                    kPusherXy,
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kSliceScratch),
                    0,
                    n,
                    kp::kSyncFrameRecords));
            head += n;
            noc_inline_dw_write(get_noc_addr(px, py, kPusherCtrl + 16), head, 0xF);
        }
    }

    // Linked core i's link-end records [cursor, tail), the tail from the control vector pack_linked_frame just read.
    // The end never waits for this core, so a tail more than a ring ahead means the oldest are gone. A round's two
    // records land back to back, so a run reaching into the ring's last two slots re-reads the tail after the image
    // and drops what the end may have overwritten meanwhile.
    void ship_link_sync(uint32_t i) {
        constexpr uint32_t kTailBlock = (kp::SPSC_LINK_SYNC_TAIL * 4u) & ~63u;
        if (kLinkRingAddr == 0) {
            return;
        }
        volatile tt_l1_ptr uint32_t* cv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCvScratch);
        const uint32_t tail = cv[kp::SPSC_LINK_SYNC_TAIL];
        if (tail == link_cursor[i]) {
            return;
        }
        const uint32_t x = linked_xy[i] & 0xFFFFu, y = linked_xy[i] >> 16;
        uint32_t first =
            tail - link_cursor[i] > kp::kLinkSyncRingRecords ? tail - kp::kLinkSyncRingRecords : link_cursor[i];
        noc_async_read(get_noc_addr(x, y, kLinkRingAddr), kLinkScratch, kLinkRingBytes);
        noc_async_read_barrier();
        if (tail - first > kp::kLinkSyncRingRecords - 2) {
            noc_async_read(get_noc_addr(x, y, linked_l1[i] + kTailBlock), kCvScratch + kTailBlock, 64);
            noc_async_read_barrier();
            const uint32_t now = cv[kp::SPSC_LINK_SYNC_TAIL];
            if (now - first > kp::kLinkSyncRingRecords) {
                first = now - kp::kLinkSyncRingRecords;
            }
        }
        if (first < tail) {
            ship(
                sync_sender,
                pack_sync_frame(
                    linked_xy[i],
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kLinkScratch),
                    first,
                    tail - first,
                    kp::kLinkSyncRingRecords));
        }
        link_cursor[i] = tail;
    }

    __attribute__((noinline)) void sweep() {
        for (uint32_t i = 0; i < n_linked; i++) {
            const uint32_t bytes = pack_linked_frame(linked_xy[i], linked_l1[i]);
            if (bytes != 0) {
                ship(sender, bytes);
            }
            if (i != 0) {  // the pusher carries no link end
                ship_link_sync(i);
            }
        }
    }
};
#endif

void kernel_main() {
#if defined(PROFILE_KERNEL)
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCtrlAddr);
    volatile tt_l1_ptr uint32_t* hb = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCtrlAddr + 4);
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCtrlAddr + 8);
    volatile tt_l1_ptr uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kCtrlAddr + 64);
    *done = 0;
    *hb = 0;
    *go = 0;
    *stop = 0;

    Drainer d;
    d.start();
    // Shipping waits for the host's go word, written once the receiver's ingest threads drain these sockets.
    while (*go == 0u && *stop == 0u) {
        (*hb)++;
        invalidate_l1_cache();
    }
    uint32_t last_sweep = eth_ptp::rd(eth_ptp::kWallClockLo);
    while (true) {
        (*hb)++;
        invalidate_l1_cache();
        d.drain_pusher(false);
        const uint32_t now = eth_ptp::rd(eth_ptp::kWallClockLo);
        if (now - last_sweep >= kSweepCycles) {
            last_sweep = now;
            d.drain_pusher(true);
            d.sweep();
        }
        // Teardown: the relay stop word, written by the host at quiesce once the pusher has stopped, so its tail
        // is final.
        if (*stop != 0u) {
            break;
        }
        d.audit.anchor();
        for (uint32_t d = 0; d < kPollCycles; d++) {
            asm volatile("nop");
        }
    }
    d.drain_pusher(true);
    d.audit.send();
    d.sweep();
    *done = kp::kRelayDrainedWord;
    socket_barrier(d.sender);
    socket_barrier(d.sync_sender);
    noc_async_writes_flushed();
    update_socket_config(d.sender);
    update_socket_config(d.sync_sender);
    *done = kp::kRelayDoneWord;
#endif
}
