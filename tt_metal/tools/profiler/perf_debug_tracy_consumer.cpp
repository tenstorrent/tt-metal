// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tools/profiler/perf_debug_tracy_consumer.hpp"

#include "hostdevcommon/profiler_common.h"
#include "tools/profiler/perf_debug_profiler_packets.hpp"
#include "tools/profiler/perf_debug_profiler_tracy_handler.hpp"

namespace tt::tt_metal::perf_debug {

PerfDebugTracyConsumer::PerfDebugTracyConsumer(PerfDebugTracyHandler* handler) : handler_(handler) {
    // The SWEEP/PACE alternation is what a drainer row is read by, so those two must contrast; PACE is
    // deliberate idleness and gets a recessive grey. Mover rows use their own hues because the two roles'
    // same-named phases have different meanings and scales (a filler's CREDIT-WAIT is DRAM ring room, a
    // mover's is host FIFO credit). Keys are the zone NAMES the drain kernel declares (TT_ZONE_DEFINE_ID
    // in drisc_drain_common.hpp) -- names are the only stable handle on a structural zone id.
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

void PerfDebugTracyConsumer::operator()(const PerfDebugRawRecordBatch& batch) {
    const PerfDebugCaptureContext& ctx = *batch.context;
    if (ts_base_.size() < ctx.devices.size()) {
        ts_base_.resize(ctx.devices.size(), 0);
    }
    // Mirror any zone names registered since the last batch. Names arrive per-ELF as binaries load, so
    // the table GROWS throughout a model run -- a one-shot snapshot would be taken when it holds a
    // fraction of its final size.
    names_.refresh();
    for (const PerfDebugRawRec& r : batch.records) {
        const auto type = r.meta.type;
        if (type == PerfDebugRawRecType::Cont) {
            if (pend_.active && pend_.got < kMaxEventValues) {
                pend_.vals[pend_.got++] = r.ts;
            }
            if (pend_.active && pend_.got >= pend_.want) {
                flush_event(ctx);
            }
            continue;
        }
        if (type == PerfDebugRawRecType::Ext) {
            if (pend_.active) {
                const uint32_t n = r.id;
                pend_.want = (n + 1) / 2;
                if (n != 0 && pend_.got < kMaxEventValues) {
                    pend_.vals[pend_.got++] = r.ts;
                }
                if (pend_.got >= pend_.want) {
                    flush_event(ctx);
                }
            }
            continue;
        }
        flush_event(ctx);  // any non-continuation record terminates a truncated predecessor
        if (type == PerfDebugRawRecType::ZoneTotal) {
            continue;
        }
        if (ts_base_[r.meta.dev] == 0) {
            ts_base_[r.meta.dev] = r.ts;
        }
        if (type == PerfDebugRawRecType::Data || type == PerfDebugRawRecType::Event) {
            pend_ = PendingEvent{};
            pend_.active = true;
            pend_.dev = r.meta.dev;
            pend_.lane = r.meta.lane;
            pend_.ts = r.ts;
            pend_.id = r.id;
            pend_.prog = r.prog;
            if (type == PerfDebugRawRecType::Event) {
                flush_event(ctx);  // an Event is payload-less and arrives whole -- no Ext follows it
            }
            continue;
        }
        const auto& dev = ctx.devices[r.meta.dev];
        const auto& li = dev.lanes[r.meta.lane];
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
            if (auto it = tbl.find(pkt.name); it != tbl.end()) {
                pkt.color = it->second;
            } else if (auto it2 = zone_colors_.find(pkt.name); it2 != zone_colors_.end()) {
                pkt.color = it2->second;  // mover table has no override for this zone
            }
        }
        const uint64_t base = dev.clock_synced ? 0 : ts_base_[r.meta.dev];
        if (type == PerfDebugRawRecType::Zone) {
            // A COMPLETE zone: the device's atomic-zone path reports the END in ts with the length in
            // `duration`, so there is no start record to pair with. Tracy's timeline is built from
            // begin/end pushes, so expand it here -- without this the zone arrives as an unpaired end and
            // renders as nothing, which is why an atomic-zone workload showed only the START/END pairs that
            // PRODUCER-STALL happens to emit.
            const uint64_t end = r.ts >= base ? r.ts - base : 0;
            pkt.timestamp = end >= r.duration ? end - r.duration : 0;
            pkt.is_start = true;
            handler_->HandleWorkerZone(pkt);
            pkt.timestamp = end;
            pkt.is_start = false;
            handler_->HandleWorkerZone(pkt);
            continue;
        }
        pkt.timestamp = r.ts >= base ? r.ts - base : 0;
        pkt.is_start = type == PerfDebugRawRecType::ZoneStart;
        // A complete zone (PP_ZONE_ATOMIC) reports the END with the length in `duration` and has no start
        // record to pair with; the handler pushes it as one GpuZone item rather than synthesising a pair.
        pkt.duration = type == PerfDebugRawRecType::Zone ? r.duration : 0;
        handler_->HandleWorkerZone(pkt);
    }
}

}  // namespace tt::tt_metal::perf_debug
