// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/perf_debug_tracy_consumer.hpp"

#include <tt-logger/tt-logger.hpp>

#include "hostdevcommon/profiler_common.h"
#include "tools/profiler/perf_debug_profiler_packets.hpp"
#include "tools/profiler/perf_debug_profiler_tracy_handler.hpp"

#include <atomic>

#include <array>

namespace tt::tt_metal::perf_debug {

// The correction store and its application both moved: the store to perf_debug_consumer.cpp
// (keyed by CHIP id), the application to the receiver's pairing stage -- records arrive here
// already on the unified timeline, so this consumer forwards timestamps as-is.

PerfDebugTracyConsumer::PerfDebugTracyConsumer(PerfDebugTracyHandler* handler) : handler_(handler) {
    // The SWEEP/PACE alternation is what a drainer row is read by, so those two must contrast; PACE is
    // deliberate idleness and gets a recessive grey. Mover rows use their own hues because the two roles'
    // same-named phases have different meanings and scales (a filler's CREDIT-WAIT is DRAM ring room, a
    // mover's is host FIFO credit). Keys are the zone NAMES the drain kernel declares (TT_ZONE_DEFINE_ID
    // in drisc_profiler_drain.cpp) -- names are the only stable handle on a structural zone id.
    zone_colors_["DRISC-SWEEP"] = 0x2E86C1;
    zone_colors_["DRISC-PACE"] = 0x707B7C;
    zone_colors_["DRISC-READ"] = 0x27AE60;
    zone_colors_["DRISC-READ-WAIT"] = 0x196F3D;
    zone_colors_["DRISC-PROC"] = 0x8E44AD;
    zone_colors_["DRISC-CREDIT-WAIT"] = 0xC0392B;
    zone_colors_["DRISC-WRITE"] = 0xD35400;
    zone_colors_["DRISC-WR-BARRIER"] = 0xF1C40F;
    // White, and the same on both roles: the sync marker is a fiducial, not a phase.
    zone_colors_["DRISC-SYNC"] = 0xFFFFFF;
    zone_colors_mover_["DRISC-SYNC"] = 0xFFFFFF;
    zone_colors_mover_["DRISC-SWEEP"] = 0x16A085;
    zone_colors_mover_["DRISC-READ"] = 0x52BE80;
    zone_colors_mover_["DRISC-CREDIT-WAIT"] = 0xE74C3C;
    zone_colors_mover_["DRISC-WRITE"] = 0xE67E22;
    zone_colors_mover_["DRISC-WR-BARRIER"] = 0xF7DC6F;
}

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
        {
            // Colour by zone NAME and by role -- see zone_colors_ in the header.
            const auto& tbl = li.role == PerfDebugLaneRole::Mover ? zone_colors_mover_ : zone_colors_;
            if (auto cit = tbl.find(pkt.name); cit != tbl.end()) {
                pkt.color = cit->second;
            } else if (auto cit2 = zone_colors_.find(pkt.name); cit2 != zone_colors_.end()) {
                pkt.color = cit2->second;  // mover table has no override for this zone
            }
        }
        const uint64_t base = clock_synced_[r.meta.dev] ? 0 : ts_base_[r.meta.dev];
        // Live drift correction is already baked in by the receiver's pairing stage (one unified
        // timeline for every consumer); both zone ends carry it, so durations are untouched.
        const uint64_t cs = r.data.zone.start;
        const uint64_t ce = r.data.zone.start + r.data.zone.duration;
        pkt.start = cs >= base ? cs - base : 0;
        pkt.end = ce >= base ? ce - base : 0;
        handler_->HandleWorkerZone(pkt);
    }
}

}  // namespace tt::tt_metal::perf_debug
