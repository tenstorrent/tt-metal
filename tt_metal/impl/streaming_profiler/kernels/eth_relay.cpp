// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Ships everything the chip's eth cores produce, so the clock tracker never leaves its sampling loop for the
// multi-microsecond PCIe flush each transfer ends in.

#include <algorithm>
#include <array>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/socket_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "tt_metal/impl/streaming_profiler/kernels/relay_common.hpp"

constexpr uint32_t kFramesCfgAddr = get_named_compile_time_arg_val("frames_cfg");
constexpr uint32_t kSyncCfgAddr = get_named_compile_time_arg_val("sync_cfg");
constexpr uint32_t kStageAddr = get_named_compile_time_arg_val("stage");
constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl");
constexpr uint32_t kControlVectorScratch = get_named_compile_time_arg_val("scratch");
constexpr uint32_t kTrackerXy = get_named_compile_time_arg_val("tracker_xy");
constexpr uint32_t kTrackerControlVectorL1 = get_named_compile_time_arg_val("tracker_prof_l1");
constexpr uint32_t kSyncRingAddr = get_named_compile_time_arg_val("sync_ring");
constexpr uint32_t kLinkRingAddr = get_named_compile_time_arg_val("link_ring");
#if defined(PROFILE_STREAMING_SYNC_CHECK)
constexpr uint32_t kRulerXy = get_named_compile_time_arg_val("ruler_xy");
#endif
// Only eth zones put profiler frames on the eth cores' rings; the link records and the sync rings ship either way.
#if defined(PROFILE_STREAMING_ETH)
constexpr bool kEthZones = true;
#else
constexpr bool kEthZones = false;
#endif

constexpr uint32_t kNumEthRisc = 2;  // DM0, DM1: the lanes that physically exist on an eth core
constexpr uint32_t kRecordBytes = sizeof(kernel_profiler::SyncRecord);
constexpr uint32_t kLinkRecords = kernel_profiler::kLinkSyncRingRecords;
constexpr uint32_t kBlockBytes = 64;
constexpr uint32_t kCtrlScratch = kControlVectorScratch + kBlockBytes;
static_assert(kCtrlScratch + kBlockBytes <= kControlVectorScratch + kernel_profiler::kEthSyncScratchBytes);

FORCE_INLINE volatile tt_l1_ptr uint32_t* l1_words(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}

FORCE_INLINE uint64_t noc_at(uint32_t xy, uint32_t addr) {
    const auto coord = kernel_profiler::word_as<kernel_profiler::NocXy>(xy);
    return get_noc_addr(coord.x, coord.y, addr);
}

FORCE_INLINE void write_word(uint32_t xy, uint32_t addr, uint32_t value) {
    noc_inline_dw_write(noc_at(xy, addr), value, 0xF);
}

inline volatile tt_l1_ptr uint32_t* read_block(uint32_t xy, uint32_t src, uint32_t dst, uint32_t bytes) {
    noc_async_read(noc_at(xy, src), dst, bytes);
    noc_async_read_barrier();
    return l1_words(dst);
}

inline void read_records(uint32_t xy, uint32_t ring, uint32_t capacity, uint32_t first, uint32_t count) {
    constexpr uint32_t kDst = kStageAddr + kPrefix * 4u;
    const uint32_t first_slot = first & (capacity - 1);
    const uint32_t before_wrap = std::min(count, capacity - first_slot);
    noc_async_read(noc_at(xy, ring + first_slot * kRecordBytes), kDst, before_wrap * kRecordBytes);
    if (before_wrap < count) {
        noc_async_read(noc_at(xy, ring), kDst + before_wrap * kRecordBytes, (count - before_wrap) * kRecordBytes);
    }
    noc_async_read_barrier();
}

__attribute__((noinline)) void ship(SocketSenderInterface& sender, uint32_t xy, uint32_t frame_words) {
    l1_words(kStageAddr)[kernel_profiler::SPSC_PREFIX_XY] = xy;
    l1_words(kStageAddr)[kernel_profiler::SPSC_PREFIX_PAYLOAD_WORDS] = frame_words - kPrefix;
    const uint32_t bytes = page_round(frame_words * 4u);
    const uint32_t pages = bytes / kPageBytes;
    socket_reserve_pages(sender, pages);
    staged_store_fence();
    push_fifo(sender, kStageAddr, sender.write_ptr, bytes);
    socket_push_pages(sender, pages);
    notify_bytes_sent(sender);
}

// A ring of sync records on `xy`, whose frames name it.
struct SyncRing {
    uint32_t xy;
    uint32_t head_addr, addr, capacity;
};
// Whether a ring's last, partial frame goes out now. Holding it saves the padding: a frame is at least 24 words, two
// socket pages on the wire, so a lone record's frame is four times the record.
enum class PartialFrame { Hold, Send };

struct EthRelay {
    // The relay is the only writer of the ring heads, so it keeps its own copy and reads only the tails' block of the
    // control vector.
    struct Core {
        uint32_t xy, control_vector_l1;
        std::array<uint32_t, kernel_profiler::PROFILER_SPSC_TENSIX_RISC> heads;
        uint32_t link_head = 0;
    };
    enum Socket : uint32_t { kSync, kFrames };
    static constexpr uint32_t kSockets = kEthZones ? 2 : 1;
    std::array<SocketSenderInterface, 2> sockets;
    uint32_t core_count = 0;
    std::array<Core, 1 + kernel_profiler::kEthRelayMaxDrained> cores;
    uint32_t tracker_head = 0;
#if defined(PROFILE_STREAMING_SYNC_CHECK)
    uint32_t ruler_head = 0;
#endif

    void start() {
        cores[0] = {.xy = kTrackerXy, .control_vector_l1 = kTrackerControlVectorL1};
        core_count = 1 + get_arg_val<uint32_t>(0);
        for (uint32_t i = 1; i < core_count; i++) {
            cores[i].xy = get_arg_val<uint32_t>(2 * i - 1);
            cores[i].control_vector_l1 = get_arg_val<uint32_t>(2 * i);
        }
        sockets[kSync] = create_sender_socket_interface(kSyncCfgAddr);
        if constexpr (kEthZones) {
            for (uint32_t i = 0; i < core_count; i++) {
                const volatile tt_l1_ptr uint32_t* control_vector =
                    read_block(cores[i].xy, cores[i].control_vector_l1, kControlVectorScratch, kBlockBytes);
                for (uint32_t risc = 0; risc < kernel_profiler::PROFILER_SPSC_TENSIX_RISC; risc++) {
                    cores[i].heads[risc] = control_vector[kernel_profiler::SPSC_RING_HEAD_0 + risc];
                }
            }
            sockets[kFrames] = create_sender_socket_interface(kFramesCfgAddr);
        }
        for (uint32_t i = 0; i < kSockets; i++) {
            set_sender_socket_page_size(sockets[i], kPageBytes);
        }
        noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
        l1_words(kStageAddr)[0] = kernel_profiler::spsc_span_w0();
    }

    void ship_sync(uint32_t xy, uint32_t count) {
        volatile tt_l1_ptr uint32_t* frame = l1_words(kStageAddr);
        frame[kernel_profiler::SPSC_PREFIX_SYNC_RECORD_COUNT] = count;
        ship(sockets[kSync], xy, std::max(kPrefix + count * kernel_profiler::kSyncRecordWords, kPrefix + kWireCtrl));
    }

    __attribute__((noinline)) bool drain_ring(
        const SyncRing& ring, uint32_t tail, uint32_t& consumed, PartialFrame partial) {
        const uint32_t first = consumed;
        for (uint32_t left; (left = tail - consumed) != 0 &&
                            (partial == PartialFrame::Send || left >= kernel_profiler::kSyncFrameRecords);) {
            const uint32_t batch = std::min(left, kernel_profiler::kSyncFrameRecords);
            read_records(ring.xy, ring.addr, ring.capacity, consumed, batch);
            ship_sync(ring.xy, batch);
            consumed += batch;
            write_word(ring.xy, ring.head_addr, consumed);
        }
        return consumed != first;
    }

    bool drain_sync_ring(uint32_t xy, uint32_t& consumed, PartialFrame partial) {
        constexpr uint32_t kTail = offsetof(kernel_profiler::ResidentCtrl, sync_tail);
        static_assert(kTail + 4u <= kBlockBytes);
        const uint32_t tail = read_block(xy, kCtrlAddr, kCtrlScratch, kBlockBytes)[kTail / 4u];
        const SyncRing ring{
            .xy = xy,
            .head_addr = kCtrlAddr + offsetof(kernel_profiler::ResidentCtrl, sync_head),
            .addr = kSyncRingAddr,
            .capacity = kernel_profiler::kSyncRingRecords};
        return drain_ring(ring, tail, consumed, partial);
    }

    bool drain(PartialFrame partial) {
        bool shipped = drain_sync_ring(kTrackerXy, tracker_head, partial);
#if defined(PROFILE_STREAMING_SYNC_CHECK)
        shipped = drain_sync_ring(kRulerXy, ruler_head, partial) || shipped;
#endif
        return shipped;
    }

    // Read the tails first: a tail seen there bounds the ring words read after it.
    bool ship_core(Core& core) {
        static_assert(kWireCtrl * 4u == kBlockBytes);
        volatile tt_l1_ptr uint32_t* block = read_block(
            core.xy,
            core.control_vector_l1 + kernel_profiler::SPSC_WIRE_CV_BASE * 4u,
            kControlVectorScratch,
            kBlockBytes);
        const SyncRing ring{
            .xy = core.xy,
            .head_addr = core.control_vector_l1 + kernel_profiler::SPSC_LINK_SYNC_HEAD * 4u,
            .addr = kLinkRingAddr,
            .capacity = kLinkRecords};
        const bool linked =
            drain_ring(ring, block[kernel_profiler::SPSC_WIRE_LINK_SYNC_TAIL], core.link_head, PartialFrame::Send);
        if constexpr (!kEthZones) {
            return linked;
        }
        std::array<uint32_t, kNumEthRisc> takes;
        bool live = false;
        for (uint32_t risc = 0; risc < kNumEthRisc; risc++) {
            takes[risc] = block[kernel_profiler::SPSC_WIRE_TAIL_0 + risc] - core.heads[risc];
            live = live || takes[risc] != 0;
        }
        if (!live) {
            return linked;
        }
        volatile tt_l1_ptr uint32_t* frame = l1_words(kStageAddr);
        for (uint32_t i = 0; i < kWireCtrl; i++) {
            frame[kPrefix + i] = block[i];
        }
        for (uint32_t risc = 0; risc < kernel_profiler::PROFILER_SPSC_TENSIX_RISC; risc++) {
            frame[kernel_profiler::SPSC_PREFIX_HEAD_0 + risc] = core.heads[risc];
        }
        uint32_t frame_words = kPrefix + kWireCtrl;
        for (uint32_t risc = 0; risc < kNumEthRisc; risc++) {
            if (takes[risc] != 0) {
                const uint32_t ring_l1 =
                    core.control_vector_l1 + kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE + risc * kRingWords * 4u;
                frame_words = place_run(
                    core.heads[risc],
                    takes[risc],
                    frame_words,
                    kStageAddr,
                    [&](uint32_t src, uint32_t dst, uint32_t bytes, bool) {
                        noc_async_read(noc_at(core.xy, ring_l1 + src), dst, bytes);
                    });
            }
        }
        noc_async_read_barrier();
        for (uint32_t risc = 0; risc < kNumEthRisc; risc++) {
            if (takes[risc] != 0) {
                core.heads[risc] += takes[risc];
                write_word(
                    core.xy,
                    core.control_vector_l1 + (kernel_profiler::SPSC_RING_HEAD_0 + risc) * 4u,
                    core.heads[risc]);
            }
        }
        ship(sockets[kFrames], core.xy, frame_words);
        return true;
    }

    bool sweep() {
        bool live = false;
        for (uint32_t i = 0; i < core_count; i++) {
            live = ship_core(cores[i]) || live;
        }
        return live;
    }
};

void kernel_main() {
    volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::ResidentCtrl*>(kCtrlAddr);
    EthRelay relay;
    relay.start();
    while (ctrl->go == 0u && ctrl->stop == 0u) {
        ctrl->heartbeat++;
        invalidate_l1_cache();
    }
    uint32_t gap = 0, deferred = 0;
    for (bool stop = false; !stop;) {
        invalidate_l1_cache();
        // The host writes stop after the ruler and tracker have stopped, so their tails are final by then.
        stop = ctrl->stop != 0u;
        const bool busy = relay.sweep();
        // A sync ring's partial frame waits the same way a DRISC lane below its ship gate does.
        const bool send_partial = stop || !busy || ++deferred >= kMaxDeferSweeps;
        deferred = send_partial ? 0u : deferred;
        idle_wait(gap, relay.drain(send_partial ? PartialFrame::Send : PartialFrame::Hold) || busy, [] {});
    }
    ctrl->done = kernel_profiler::kResidentAwaitingAcksWord;
    for (uint32_t i = 0; i < EthRelay::kSockets; i++) {
        socket_barrier(relay.sockets[i]);
    }
    noc_async_writes_flushed();
    for (uint32_t i = 0; i < EthRelay::kSockets; i++) {
        update_socket_config(relay.sockets[i]);
    }
    ctrl->done = kernel_profiler::kResidentDoneWord;
}
