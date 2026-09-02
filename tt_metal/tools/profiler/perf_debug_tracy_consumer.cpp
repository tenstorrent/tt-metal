// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/perf_debug_tracy_consumer.hpp"

#include <tt-logger/tt-logger.hpp>

#include "hostdevcommon/profiler_common.h"
#include "tools/profiler/perf_debug_profiler_packets.hpp"
#include "tools/profiler/perf_debug_profiler_tracy_handler.hpp"

namespace tt::tt_metal::perf_debug {

PerfDebugTracyConsumer::PerfDebugTracyConsumer(PerfDebugTracyHandler* handler) : handler_(handler) {}

PerfDebugTracyConsumer::~PerfDebugTracyConsumer() { log_unnamed_ids("tracy", names_); }

void PerfDebugTracyConsumer::note_ts(uint32_t dev, uint64_t ts) {
    uint64_t& base = ts_base_[dev];
    if (base == 0 || ts < base) {
        base = ts;
    }
}

void PerfDebugTracyConsumer::flush_event(const PerfDebugCaptureContext& ctx) {
    if (!pend_.active) {
        return;
    }
    pend_.active = false;
    const auto& dev = ctx.devices[pend_.dev];
    const auto& li = dev.lanes[pend_.lane];
    WorkerEventPacket pkt;
    pkt.chip_id = li.chip_id;
    pkt.core_virtual_x = li.virtual_x;
    pkt.core_virtual_y = li.virtual_y;
    pkt.core_noc0_x = li.noc0_x;
    pkt.core_noc0_y = li.noc0_y;
    pkt.risc = li.risc;
    pkt.id = pend_.id;
    // BOTH point-marker types carry a compile-time structural id, so both resolve exactly like a zone.
    // (PP_EVENT used to carry a RUNTIME value here, which was the one id on this wire that could not be
    // named; a runtime value now ships as ordinary PP_DATA payload instead.)
    pkt.name = names_.lookup(pend_.id);
    const uint64_t base = dev.clock_synced ? 0 : ts_base_[pend_.dev];
    pkt.timestamp = pend_.ts >= base ? pend_.ts - base : 0;
    pkt.runtime_host_id = pend_.prog;
    pkt.values = pend_.vals;
    pkt.num_values = pend_.got;
    handler_->HandleWorkerEvent(pkt);
}

void PerfDebugTracyConsumer::operator()(const PerfDebugRecordBatch& batch) {
    const PerfDebugCaptureContext& ctx = *batch.context;
    if (ts_base_.size() < ctx.devices.size()) {
        ts_base_.resize(ctx.devices.size(), 0);
        clock_synced_.resize(ctx.devices.size(), 0);
        for (size_t d = 0; d < ctx.devices.size(); d++) {
            clock_synced_[d] = ctx.devices[d].clock_synced ? 1 : 0;
        }
    }
    // Mirror any zone names registered since the last batch. Names arrive per-ELF as binaries load, so
    // the table GROWS throughout a model run -- a one-shot snapshot would be taken when it holds a
    // fraction of its final size.
    names_.refresh();
    for (const PerfDebugRec& r : batch.records) {
        const auto type = r.meta.type;
        if (type == PerfDebugRecType::Cont) {
            if (pend_.active && pend_.got < kMaxEventValues) {
                pend_.vals[pend_.got++] = r.data.payload;
            }
            if (pend_.active && pend_.got >= pend_.want) {
                flush_event(ctx);
            }
            continue;
        }
        if (type == PerfDebugRecType::Ext) {
            if (pend_.active) {
                pend_.want = (static_cast<uint32_t>(r.data.ext) + 1) / 2;
                if (pend_.want == 0) {
                    flush_event(ctx);
                }
            }
            continue;
        }
        flush_event(ctx);  // any non-continuation record terminates a truncated predecessor
        if (type == PerfDebugRecType::Data || type == PerfDebugRecType::Event) {
            note_ts(r.meta.dev, r.data.ts);
            pend_ = PendingEvent{};
            pend_.active = true;
            pend_.dev = r.meta.dev;
            pend_.lane = r.meta.lane;
            pend_.ts = r.data.ts;
            pend_.id = r.id;
            pend_.prog = r.prog;
            continue;
        }
        // Zone: forward WHOLE, immediately. The paired stream's per-lane arrival order IS zone
        // completion order, which is exactly the ordering contract TracyTTPushZone requires, so
        // there is nothing to buffer and nothing to reorder -- see the header.
        note_ts(r.meta.dev, r.data.zone.start);
        const PerfDebugLaneInfo& li = ctx.devices[r.meta.dev].lanes[r.meta.lane];
        WorkerZonePacket pkt;
        pkt.chip_id = li.chip_id;
        pkt.core_virtual_x = li.virtual_x;
        pkt.core_virtual_y = li.virtual_y;
        pkt.core_noc0_x = li.noc0_x;
        pkt.core_noc0_y = li.noc0_y;
        pkt.risc = li.risc;
        pkt.timer_id = r.id;
        pkt.name = names_.lookup(r.id);
        const uint64_t base = clock_synced_[r.meta.dev] ? 0 : ts_base_[r.meta.dev];
        const uint64_t start = r.data.zone.start;
        const uint64_t end = r.data.zone.start + r.data.zone.duration;
        pkt.start = start >= base ? start - base : 0;
        pkt.end = end >= base ? end - base : 0;
        handler_->HandleWorkerZone(pkt);
    }
}

}  // namespace tt::tt_metal::perf_debug
