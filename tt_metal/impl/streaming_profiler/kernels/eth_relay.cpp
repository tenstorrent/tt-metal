// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ships everything the chip's eth cores produce to the host. It runs on its own core so the wall-clock core never has
// to leave its sampling loop for the multi-microsecond PCIe flush that ends each transfer.

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
constexpr uint32_t kWallClockXy = get_named_compile_time_arg_val("wall_clock_xy");
constexpr uint32_t kWallClockControlVectorL1 = get_named_compile_time_arg_val("wall_clock_control_vector_l1");
constexpr uint32_t kSyncRingAddr = get_named_compile_time_arg_val("sync_ring");
constexpr uint32_t kLinkRingAddr = get_named_compile_time_arg_val("link_ring");
constexpr bool kSyncCheck = get_named_compile_time_arg_val("sync_check") != 0;
constexpr uint32_t kCheckXy = get_named_compile_time_arg_val("check_xy");
// The external socket's config; 0 means no external process.
constexpr uint32_t kExternalConfigAddr = get_named_compile_time_arg_val("external_socket_config_addr");
constexpr bool kExternal = kExternalConfigAddr != 0;

constexpr uint32_t kNumEthRisc = 2;  // ERISC0 and ERISC1, the only lanes an eth core has
constexpr uint32_t kSyncRecordBytes = sizeof(kernel_profiler::SyncRecord);
constexpr uint32_t kBlockBytes = kWireControlWords * 4u;
// Where the staged frame's words after its prefix start. They hold a sync frame's records or a profiler frame's control
// block. Every 64 B control read lands here, so a profiler frame ships the tails' block as it was read.
constexpr uint32_t kStageBody = kStageAddr + kPrefixWords * 4u;

FORCE_INLINE volatile tt_l1_ptr uint32_t* l1_words(uint32_t addr) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
}

FORCE_INLINE uint64_t noc_at(uint32_t xy, uint32_t addr) {
    const auto coord = kernel_profiler::word_as<kernel_profiler::NocXy>(xy);
    return get_noc_addr(coord.x, coord.y, addr);
}

FORCE_INLINE void write_word(uint32_t xy, uint32_t addr, uint32_t value) {
    noc_inline_dw_write(noc_at(xy, addr), value);
}

inline volatile tt_l1_ptr uint32_t* read_block(uint32_t xy, uint32_t src) {
    noc_async_read(noc_at(xy, src), kStageBody, kBlockBytes);
    noc_async_read_barrier();
    return l1_words(kStageBody);
}

__attribute__((noinline)) void ship(SocketSenderInterface& sender, uint32_t xy, uint32_t frame_words) {
    l1_words(kStageAddr)[kernel_profiler::SPSC_PREFIX_XY] = xy;
    l1_words(kStageAddr)[kernel_profiler::SPSC_PREFIX_PAYLOAD_WORDS] = frame_words - kPrefixWords;
    const uint32_t bytes = page_round(frame_words * 4u);
    const uint32_t pages = bytes / kPageBytes;
    socket_reserve_pages(sender, pages);
    staged_store_fence();
    push_fifo(sender, kStageAddr, sender.write_ptr, bytes);
    socket_push_pages(sender, pages);
    notify_bytes_sent(sender);
}

struct SyncRing {
    uint32_t xy;
    uint32_t head_addr, addr, capacity;
};
// Whether a ring's last, partial frame is sent now. Holding it lets later records go out in the same frame.
enum class PartialFrame { Hold, Send };

struct EthRelay {
    // An eth core the relay drains. The relay is the only writer of the ring heads, so it keeps its own copy of them
    // and only reads the tails' block of the control vector.
    struct Core {
        uint32_t xy, control_vector_l1;
        std::array<uint32_t, kernel_profiler::PROFILER_SPSC_TENSIX_RISC> heads;
        uint32_t link_head = 0;
        // The ERISCs' routed halfwords as the relay last marked them.
        uint32_t routed_last = 0;
    };
    enum Socket : uint32_t { kSync, kFrames, kSocketCount };
    std::array<SocketSenderInterface, kSocketCount> sockets;
    uint32_t core_count = 0;
    std::array<Core, 1 + kernel_profiler::kEthRelayMaxDrained> cores;
    uint32_t wall_clock_head = 0;
    uint32_t check_head = 0;
    // The external socket, which follows ResidentCtrl::external.attach.
    SocketSenderInterface external{};
    uint32_t external_attach = 0;
    uint32_t external_copied = 0;

    EthRelay() {
        cores[0] = {.xy = kWallClockXy, .control_vector_l1 = kWallClockControlVectorL1};
        core_count = 1 + get_arg_val<uint32_t>(0);
        for (uint32_t i = 1; i < core_count; i++) {
            cores[i].xy = get_arg_val<uint32_t>(2 * i - 1);
            cores[i].control_vector_l1 = get_arg_val<uint32_t>(2 * i);
        }
        sockets[kSync] = create_sender_socket_interface(kSyncCfgAddr);
        sockets[kFrames] = create_sender_socket_interface(kFramesCfgAddr);
        // Heads start at the current tails and are written back, because a fabric router's rings are never reset.
        for (uint32_t i = 0; i < core_count; i++) {
            Core& core = cores[i];
            const volatile tt_l1_ptr uint32_t* block =
                read_block(core.xy, core.control_vector_l1 + kernel_profiler::SPSC_WIRE_CV_BASE * 4u);
            core.heads = {};
            for (uint32_t risc = 0; risc < kNumEthRisc; risc++) {
                core.heads[risc] = block[kernel_profiler::SPSC_WIRE_TAIL_0 + risc];
                write_word(
                    core.xy,
                    core.control_vector_l1 + (kernel_profiler::SPSC_RING_HEAD_0 + risc) * 4u,
                    core.heads[risc]);
            }
            core.link_head = block[kernel_profiler::SPSC_WIRE_LINK_SYNC_TAIL];
            // Routed positions the cores show at launch predate the capture.
            core.routed_last = block[kernel_profiler::SPSC_WIRE_ROUTED_AT_0];
            write_word(core.xy, core.control_vector_l1 + kernel_profiler::SPSC_LINK_SYNC_HEAD * 4u, core.link_head);
        }
        for (SocketSenderInterface& socket : sockets) {
            set_sender_socket_page_size(socket, kPageBytes);
        }
        noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
        l1_words(kStageAddr)[0] = kernel_profiler::spsc_span_w0();
    }

    __attribute__((noinline)) bool drain_ring(
        const SyncRing& ring, uint32_t tail, uint32_t& head, PartialFrame partial) {
        const uint32_t first = head;
        while (head != tail) {
            const uint32_t left = tail - head;
            if (partial == PartialFrame::Hold && left < kernel_profiler::kSyncFrameRecords) {
                break;
            }
            // A batch stops at the ring's end, so it is one read.
            const uint32_t slot = head & (ring.capacity - 1);
            const uint32_t batch = std::min({left, kernel_profiler::kSyncFrameRecords, ring.capacity - slot});
            noc_async_read(noc_at(ring.xy, ring.addr + slot * kSyncRecordBytes), kStageBody, batch * kSyncRecordBytes);
            noc_async_read_barrier();
            ship(sockets[kSync], ring.xy, kPrefixWords + batch * kernel_profiler::kSyncRecordWords);
            head += batch;
            write_word(ring.xy, ring.head_addr, head);
        }
        return head != first;
    }

    // Ships the new records in the sync ring of the wall-clock or check core at `xy`, and returns whether it shipped
    // any. Both cores keep their ctrl block and sync ring at this core's addresses.
    bool drain_sync_ring(uint32_t xy, uint32_t& head, PartialFrame partial) {
        constexpr uint32_t kTail = offsetof(kernel_profiler::ResidentCtrl, sync.tail);
        static_assert(kTail + 4u <= kBlockBytes);
        const uint32_t tail = read_block(xy, kCtrlAddr)[kTail / 4u];
        const SyncRing ring{
            .xy = xy,
            .head_addr = kCtrlAddr + offsetof(kernel_profiler::ResidentCtrl, sync.head),
            .addr = kSyncRingAddr,
            .capacity = kernel_profiler::kSyncRingRecords};
        return drain_ring(ring, tail, head, partial);
    }

    bool drain(PartialFrame partial) {
        bool shipped = drain_sync_ring(kWallClockXy, wall_clock_head, partial);
        if constexpr (kSyncCheck) {
            shipped = drain_sync_ring(kCheckXy, check_head, partial) || shipped;
        }
        return shipped;
    }

    // Ships an eth core's new profiler words and link sync records, and returns whether it shipped any. It reads the
    // tails first, so every ring word read afterwards is behind a tail already seen. The profiler frame ships before
    // the link records, which reuse the staged frame.
    bool ship_core(Core& core) {
        const volatile tt_l1_ptr uint32_t* block =
            read_block(core.xy, core.control_vector_l1 + kernel_profiler::SPSC_WIRE_CV_BASE * 4u);
        const uint32_t link_tail = block[kernel_profiler::SPSC_WIRE_LINK_SYNC_TAIL];
        const bool framed = ship_frame(core, block);
        const SyncRing ring{
            .xy = core.xy,
            .head_addr = core.control_vector_l1 + kernel_profiler::SPSC_LINK_SYNC_HEAD * 4u,
            .addr = kLinkRingAddr,
            .capacity = kernel_profiler::kLinkSyncRingRecords};
        return drain_ring(ring, link_tail, core.link_head, PartialFrame::Send) || framed;
    }

    // If either lane has new words, ships a frame built around `block`, which holds the core's tails.
    bool ship_frame(Core& core, const volatile tt_l1_ptr uint32_t* block) {
        std::array<uint32_t, kNumEthRisc> takes;
        bool live = false;
        for (uint32_t risc = 0; risc < kNumEthRisc; risc++) {
            takes[risc] = block[kernel_profiler::SPSC_WIRE_TAIL_0 + risc] - core.heads[risc];
            live = live || takes[risc] != 0;
        }
        if (!live) {
            return false;
        }
        volatile tt_l1_ptr uint32_t* frame = l1_words(kStageAddr);
        for (uint32_t risc = 0; risc < kernel_profiler::PROFILER_SPSC_TENSIX_RISC; risc++) {
            frame[kernel_profiler::SPSC_PREFIX_HEAD_0 + risc] = core.heads[risc];
        }
        uint32_t frame_words = kPrefixWords + kWireControlWords;
        std::array<uint32_t, kNumEthRisc> lane_start{}, lane_end{};
        for (uint32_t risc = 0; risc < kNumEthRisc; risc++) {
            lane_start[risc] = frame_words;
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
            lane_end[risc] = frame_words;
        }
        noc_async_read_barrier();
        uint32_t mark = 0;
        if constexpr (kExternal) {
            mark = mark_routed(core, block, takes, lane_start, lane_end);
        }
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
        if (mark != 0) {
            copy_marked(mark);
        }
        return true;
    }

    // The copied form of the staged frame's word 0 when an ERISC's routed halfword moved: the pages of that lane's run,
    // up to the end of its newest routed record when the run is in ring order. A moved halfword whose record is in the
    // frame is marked; one at or past its lane's tail is marked with the core's next frame. 0 for no copy.
    uint32_t mark_routed(
        Core& core,
        const volatile tt_l1_ptr uint32_t* block,
        const std::array<uint32_t, kNumEthRisc>& takes,
        const std::array<uint32_t, kNumEthRisc>& lane_start,
        const std::array<uint32_t, kNumEthRisc>& lane_end) {
        const uint32_t now = block[kernel_profiler::SPSC_WIRE_ROUTED_AT_0];
        if (now == core.routed_last) {
            return 0;
        }
        if (external_attach == 0) {
            core.routed_last = now;
            return 0;
        }
        // The lanes' reads wrote the staged frame behind the cache.
        invalidate_l1_cache();
        const volatile tt_l1_ptr uint32_t* frame = l1_words(kStageAddr);
        uint32_t first = ~0u, end = 0;
        for (uint32_t risc = 0; risc < kNumEthRisc; risc++) {
            const uint32_t shift = 16u * risc;
            const auto newest = static_cast<uint16_t>(now >> shift);
            if (newest == static_cast<uint16_t>(core.routed_last >> shift)) {
                continue;
            }
            const uint32_t tail = block[kernel_profiler::SPSC_WIRE_TAIL_0 + risc];
            if (static_cast<int16_t>(newest - static_cast<uint16_t>(tail)) < 0) {
                core.routed_last = (core.routed_last & ~(0xFFFFu << shift)) | uint32_t{newest} << shift;
            }
            const uint32_t take = takes[risc];
            if (take == 0) {
                continue;
            }
            const uint32_t start = core.heads[risc];
            uint32_t hi = lane_end[risc];
            const bool image = (start & (kRingWords - 1u)) + take > kRingWords && wrap_image(take);
            if (!image) {
                const int32_t ahead = static_cast<int16_t>(newest - static_cast<uint16_t>(start));
                if (ahead < 0) {
                    continue;  // its routed records are all in earlier frames
                }
                if (static_cast<uint32_t>(ahead) < take) {
                    const uint32_t data = lane_start[risc] + kernel_profiler::spsc_span_pack_pad(start, lane_start[risc]);
                    const uint32_t w0 = frame[data + ahead];
                    const uint32_t body_words =
                        (w0 >> kernel_profiler::ppfmt::LENGTH_SHIFT) & kernel_profiler::ppfmt::LENGTH_MASK;
                    const uint32_t record_end = data + ahead + 1u + body_words;
                    hi = record_end < hi ? record_end : hi;
                }
            }
            first = std::min(first, lane_start[risc] * 4u);
            end = std::max(end, hi * 4u);
        }
        if (end == 0) {
            return 0;
        }
        return kernel_profiler::spsc_span_w0() | kernel_profiler::SPSC_SPAN_TO_EXTERNAL |
               (first / kPageBytes) << kernel_profiler::SPSC_SPAN_ROUTED_FIRST_SHIFT |
               (page_round(end) / kPageBytes) << kernel_profiler::SPSC_SPAN_ROUTED_END_SHIFT;
    }

    // After the frame's own push, which ship() has flushed: its first pages, up to its second page or its routed runs'
    // first page, then those runs' pages, with the copied form of word 0.
    void copy_marked(uint32_t mark) {
        const uint32_t first =
            ((mark >> kernel_profiler::SPSC_SPAN_ROUTED_FIRST_SHIFT) & kernel_profiler::SPSC_SPAN_ROUTED_PAGE_MASK) *
            kPageBytes;
        const uint32_t end =
            ((mark >> kernel_profiler::SPSC_SPAN_ROUTED_END_SHIFT) & kernel_profiler::SPSC_SPAN_ROUTED_PAGE_MASK) *
            kPageBytes;
        const uint32_t head = std::min(first, 2u * kPageBytes);
        l1_words(kStageAddr)[0] = mark;
        staged_store_fence();
        push_fifo(external, kStageAddr, external.write_ptr, head);
        push_fifo(external, kStageAddr + first, external.write_ptr + head, end - first);
        socket_push_pages(external, (head + end - first) / kPageBytes);
        notify_bytes_sent(external);
        l1_words(kStageAddr)[0] = kernel_profiler::spsc_span_w0();
        external_copied++;
    }

    // Points the external socket at the external process's buffer, or turns its copies off, as the host asks.
    void follow_attach(volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl, uint32_t attach) {
        // Every copy has landed before the external process may free the buffer it went to.
        noc_async_writes_flushed();
        if (attach != 0) {
            external = create_sender_socket_interface(kExternalConfigAddr);
            set_sender_socket_page_size(external, kPageBytes);
        }
        external_attach = attach;
        ctrl->external.attached = attach;
    }

    bool sweep() {
        bool live = false;
        for (uint32_t i = 0; i < core_count; i++) {
            live = ship_core(cores[i]) || live;
        }
        return live;
    }

    void finish(volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl) {
        if constexpr (kExternal) {
            ctrl->external.copied = external_copied;
        }
        ctrl->done = kernel_profiler::kResidentAwaitingAcksWord;
        for (SocketSenderInterface& socket : sockets) {
            socket_barrier(socket);
        }
        noc_async_writes_flushed();
        for (SocketSenderInterface& socket : sockets) {
            update_socket_config(socket);
        }
        // write_to_host leaves NOC_RET_ADDR_MID routed to host, and the idle eth firmware does not reset it before the
        // next kernel on this core.
        noc_async_write_clear_pcie_state(NOC_INDEX, write_cmd_buf);
        ctrl->done = kernel_profiler::kResidentDoneWord;
    }
};

void kernel_main() {
    volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::ResidentCtrl*>(kCtrlAddr);
    if constexpr (kExternal) {
        ctrl->external.attached = 0;
    }
    EthRelay relay;
    while (ctrl->go == 0u && ctrl->stop == 0u) {
        ctrl->heartbeat++;
        invalidate_l1_cache();
    }
    uint32_t gap = 0;
    for (bool stop = false; !stop;) {
        invalidate_l1_cache();
        // By the time stop is set, the check and wall-clock cores have stopped, so their tails are final.
        stop = ctrl->stop != 0u;
        if constexpr (kExternal) {
            const uint32_t attach = ctrl->external.attach;
            if (attach != relay.external_attach) {
                relay.follow_attach(ctrl, attach);
            }
        }
        const bool busy = relay.sweep();
        const bool drained = relay.drain(stop || !busy ? PartialFrame::Send : PartialFrame::Hold);
        idle_wait(gap, drained || busy, [] {});
    }
    relay.finish(ctrl);
}
