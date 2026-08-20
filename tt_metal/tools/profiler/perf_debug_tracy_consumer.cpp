// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/perf_debug_tracy_consumer.hpp"

#include <algorithm>

#include <tt-logger/tt-logger.hpp>

#include "hostdevcommon/profiler_common.h"
#include "tools/profiler/perf_debug_profiler_packets.hpp"
#include "tools/profiler/perf_debug_profiler_tracy_handler.hpp"

namespace tt::tt_metal::perf_debug {

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

PerfDebugTracyConsumer::~PerfDebugTracyConsumer() {
    flush_zones();
    log_unnamed_ids("tracy", names_);
}

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
    // named; DeviceRuntimeEvent now ships that value as PP_DATA payload instead.)
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
        if (type == PerfDebugRecType::ZoneTotal) {
            continue;
        }
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
        // Zone: buffer for the teardown flush -- see the header for why nothing pushes here.
        note_ts(r.meta.dev, r.data.zone.start);
        auto [it, inserted] = lanes_.try_emplace((r.meta.dev << 10) | r.meta.lane);
        Lane& lane = it->second;
        if (inserted) {
            lane.info = ctx.devices[r.meta.dev].lanes[r.meta.lane];
            lane.dev = r.meta.dev;
        }
        lane.zones.push_back(BufZone{r.data.zone.start, r.data.zone.start + r.data.zone.duration, r.id, r.prog});
        zones_buffered_++;
    }
}

void PerfDebugTracyConsumer::flush_zones() {
    if (lanes_.empty()) {
        return;
    }
    names_.refresh();
    // Deterministic lane order (cosmetic: fixes Tracy context-creation order across runs).
    std::vector<uint32_t> keys;
    keys.reserve(lanes_.size());
    for (const auto& [k, v] : lanes_) {
        keys.push_back(k);
    }
    std::sort(keys.begin(), keys.end());
    uint64_t pushes = 0;
    for (const uint32_t key : keys) {
        Lane& lane = lanes_[key];
        const auto& li = lane.info;
        // Arrival is per-lane END order = post-order over the zone forest. Pre-order is (start asc,
        // end desc); reversing first makes the stable sort break exact [start,end] ties toward the
        // LATER-arrived zone, which under stack discipline is the outer one.
        std::reverse(lane.zones.begin(), lane.zones.end());
        std::stable_sort(lane.zones.begin(), lane.zones.end(), [](const BufZone& a, const BufZone& b) {
            return a.start != b.start ? a.start < b.start : a.end > b.end;
        });
        const uint64_t base = clock_synced_[lane.dev] ? 0 : ts_base_[lane.dev];
        const auto push = [&](const BufZone& z, bool is_start) {
            WorkerZonePacket pkt;
            pkt.chip_id = li.chip_id;
            pkt.core_virtual_x = li.virtual_x;
            pkt.core_virtual_y = li.virtual_y;
            pkt.core_noc0_x = li.noc0_x;
            pkt.core_noc0_y = li.noc0_y;
            pkt.risc = li.risc;
            pkt.timer_id = z.id;
            pkt.name = names_.lookup(z.id);
            {
                // Colour by zone NAME and by role -- see zone_colors_ in the header.
                const auto& tbl = li.role == PerfDebugLaneRole::Mover ? zone_colors_mover_ : zone_colors_;
                if (auto it = tbl.find(pkt.name); it != tbl.end()) {
                    pkt.color = it->second;
                } else if (auto it2 = zone_colors_.find(pkt.name); it2 != zone_colors_.end()) {
                    pkt.color = it2->second;  // mover table has no override for this zone
                }
            }
            const uint64_t ts = is_start ? z.start : z.end;
            pkt.timestamp = ts >= base ? ts - base : 0;
            pkt.is_start = is_start;
            handler_->HandleWorkerZone(pkt);
            pushes++;
        };
        // Emit the bracket sequence: begins in pre-order, each open zone's end as soon as the next
        // zone starts at or past it. `<=` closes an exactly-abutting zone before its successor opens,
        // matching device program order (the RAII end write precedes the next scope's start).
        std::vector<BufZone> open;
        for (const BufZone& z : lane.zones) {
            while (!open.empty() && open.back().end <= z.start) {
                push(open.back(), false);
                open.pop_back();
            }
            push(z, true);
            open.push_back(z);
        }
        while (!open.empty()) {
            push(open.back(), false);
            open.pop_back();
        }
    }
    log_info(
        tt::LogMetal,
        "[perf-debug tracy] deferred flush: {} zones buffered across {} lanes -> {} pushes",
        zones_buffered_,
        lanes_.size(),
        pushes);
    lanes_.clear();
}

}  // namespace tt::tt_metal::perf_debug
