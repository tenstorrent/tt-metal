// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "perf_debug_profiler_tracy_handler.hpp"

#include <cmath>
#include <cstring>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include <string>
#include <utility>

#include <hostdevcommon/profiler_common.h>

#if defined(TRACY_ENABLE)
#include <common/TracyTTDeviceData.hpp>
#include <tracy/Tracy.hpp>
#include <client/TracyProfiler.hpp>
#endif

namespace tt::tt_metal {

#if defined(TRACY_ENABLE)
namespace {

// Bytes per NoC word on this part. The NIU counters count WORDS, and the device reports the authoritative
// scale in out[129] (NOC_WORD_BYTES straight out of noc_parameters.h) -- which the per-run log block already
// uses. This constant exists only because noc_parameters.h is a device header and cannot be included here;
// if the two ever disagree the log block is right and this is wrong.
constexpr double kNocWordBytes = 64.0;

// Plot names must outlive the capture: the server queries the CLIENT to dereference the pointer, so a
// std::string that goes out of scope leaves the server resolving freed memory. These are interned once per
// (core, series) and deliberately never freed -- there are at most 6 drainers x 4 series + 1 aggregate.
const char* intern_plot_name(uint64_t key, const std::string& text) {
    static std::mutex mu;
    static std::unordered_map<uint64_t, const char*> names;
    std::lock_guard<std::mutex> g(mu);
    auto it = names.find(key);
    if (it != names.end()) {
        return it->second;
    }
    char* p = new char[text.size() + 1];
    std::memcpy(p, text.c_str(), text.size() + 1);
    names.emplace(key, p);
    return p;
}

}  // namespace
#endif

PerfDebugTracyHandler::PerfDebugTracyHandler() = default;

PerfDebugTracyHandler::~PerfDebugTracyHandler() {
    std::lock_guard<std::mutex> lock(mutex_);
#if defined(TRACY_ENABLE)
    for (auto& entry : tracy_contexts_) {
        TracyTTDestroy(entry.second);
    }
    tracy_contexts_.clear();
#endif
}

void PerfDebugTracyHandler::AddDevice(
    [[maybe_unused]] uint32_t chip_id,
    [[maybe_unused]] int64_t host_start,
    [[maybe_unused]] double first_timestamp,
    [[maybe_unused]] double frequency) {
#if defined(TRACY_ENABLE)
    std::lock_guard<std::mutex> lock(mutex_);
    // Record the chip anchor; per-core contexts are created (Populated with this) in PreCreateContexts
    // / GetOrCreateContext. A single per-chip context would collapse every core's RISCs into one row.
    chip_anchors_[chip_id] = ChipAnchor{host_start, first_timestamp, frequency};
#endif
}

void PerfDebugTracyHandler::AddCore(
    [[maybe_unused]] uint32_t chip_id,
    [[maybe_unused]] uint32_t noc0_x,
    [[maybe_unused]] uint32_t noc0_y,
    [[maybe_unused]] int64_t host_start,
    [[maybe_unused]] double first_timestamp,
    [[maybe_unused]] double frequency) {
#if defined(TRACY_ENABLE)
    std::lock_guard<std::mutex> lock(mutex_);
    core_anchors_[ContextKey(chip_id, noc0_x, noc0_y)] = ChipAnchor{host_start, first_timestamp, frequency};
#endif
}

bool PerfDebugTracyHandler::LookupAnchorLocked(
    [[maybe_unused]] uint32_t chip_id,
    [[maybe_unused]] uint32_t core_x,
    [[maybe_unused]] uint32_t core_y,
    [[maybe_unused]] ChipAnchor& out) {
#if defined(TRACY_ENABLE)
    // Per-core FIRST: a DRAM core's entry exists precisely because its clock origin differs from the chip's.
    if (auto cit = core_anchors_.find(ContextKey(chip_id, core_x, core_y)); cit != core_anchors_.end()) {
        out = cit->second;
        return true;
    }
    if (auto ait = chip_anchors_.find(chip_id); ait != chip_anchors_.end()) {
        out = ait->second;
        return true;
    }
#endif
    return false;
}

bool PerfDebugTracyHandler::LookupAnchor(uint32_t chip_id, uint32_t core_x, uint32_t core_y, ChipAnchor& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    return LookupAnchorLocked(chip_id, core_x, core_y, out);
}

TracyTTCtx PerfDebugTracyHandler::GetOrCreateContext(
    [[maybe_unused]] uint32_t chip_id,
    [[maybe_unused]] uint32_t core_x,
    [[maybe_unused]] uint32_t core_y,
    [[maybe_unused]] const std::string& name) {
#if defined(TRACY_ENABLE)
    std::lock_guard<std::mutex> lock(mutex_);
    const uint64_t key = ContextKey(chip_id, core_x, core_y);
    if (auto it = tracy_contexts_.find(key); it != tracy_contexts_.end()) {
        return it->second;
    }
    // Rename ETH rows so they are findable: without this they are formatted exactly like a Tensix core
    // and the sync lanes disappear into a list of otherwise-identical "Physical (x,y)" rows.
    const bool is_eth = eth_cores_.count(key) != 0;
    const std::string ctx_name = is_eth ? fmt::format("Device: {} ETH ({},{})", chip_id, core_x, core_y) : name;
    ChipAnchor a{};
    if (!LookupAnchorLocked(chip_id, core_x, core_y, a)) {
        return nullptr;  // device was never AddDevice'd
    }
    ZoneScopedNC("ctx-create", 0xD35400);  // orange: creating a Tracy GPU context (GpuNewContext+Populate+name)
                                           // -- one per core, all lazily on the first batch -> startup spike
    TracyTTCtx ctx = TracyTTContext();
    // Calibrated variant: marks the context calibrated (calibrationMod=1.0, no calibration events) so the
    // Tracy GUI does NOT show a per-context "Drift (ns/s)/Auto" control under every core. Timestamps are
    // host-rebased, so the anchor mapping is exact and no drift correction is wanted.
    TracyTTContextPopulateCalibrated(ctx, a.host_start, a.first_timestamp, a.frequency);
    TracyTTContextName(ctx, ctx_name.c_str(), ctx_name.size());
    tracy_contexts_[key] = ctx;
    return ctx;
#else
    return nullptr;
#endif
}

bool PerfDebugTracyHandler::IsEthCore(uint32_t chip_id, uint32_t noc0_x, uint32_t noc0_y) {
    std::lock_guard<std::mutex> lock(mutex_);
    return eth_cores_.count(ContextKey(chip_id, noc0_x, noc0_y)) != 0;
}

void PerfDebugTracyHandler::RegisterEthCore(uint32_t chip_id, uint32_t noc0_x, uint32_t noc0_y) {
    std::lock_guard<std::mutex> lock(mutex_);
    eth_cores_.insert(ContextKey(chip_id, noc0_x, noc0_y));
}

void PerfDebugTracyHandler::SetDriscRole(uint32_t chip_id, uint32_t noc0_x, uint32_t noc0_y, const char* role) {
    drisc_roles_[ContextKey(chip_id, noc0_x, noc0_y)] = role;
}

void PerfDebugTracyHandler::PreCreateContexts(
    [[maybe_unused]] uint32_t chip_id, [[maybe_unused]] const std::vector<std::pair<uint32_t, uint32_t>>& worker_noc0) {
#if defined(TRACY_ENABLE)
    for (const auto& [cx, cy] : worker_noc0) {
        GetOrCreateContext(chip_id, cx, cy, fmt::format("Device: {} Physical ({},{})", chip_id, cx, cy));
    }
    log_info(
        tt::LogMetal,
        "[perf-debug profiler] Device {}: pre-created {} per-core Tracy contexts (off the drain hot path)",
        chip_id,
        worker_noc0.size());
#endif
}

void PerfDebugTracyHandler::HandleWorkerZone([[maybe_unused]] const perf_debug::WorkerZonePacket& zone) {
#if defined(TRACY_ENABLE)
    TracyTTCtx ctx = GetOrCreateContext(
        zone.chip_id,
        zone.core_noc0_x,
        zone.core_noc0_y,
        fmt::format("Device: {} Physical ({},{})", zone.chip_id, zone.core_noc0_x, zone.core_noc0_y));
    if (!ctx) {
        return;
    }

    static constexpr tracy::RiscType kRisc[5] = {
        tracy::RiscType::BRISC,
        tracy::RiscType::NCRISC,
        tracy::RiscType::TRISC_0,
        tracy::RiscType::TRISC_1,
        tracy::RiscType::TRISC_2};

    // Same thread-id packing the marker path uses (TTDeviceMarker::get_thread_id), so zones and point
    // markers land on the same per-RISC row of the core's context.
    tracy::TTDeviceMarker tm;
    tm.chip_id = zone.chip_id;
    tm.core_x = zone.core_noc0_x;
    tm.core_y = zone.core_noc0_y;
    // An ETH core has no Tensix RISCs: kRisc would label its row BRISC, naming hardware the tile does
    // not have. ERISC is the lane these samples actually ran on.
    tm.risc = IsEthCore(zone.chip_id, zone.core_noc0_x, zone.core_noc0_y) ? tracy::RiscType::ERISC
                                                                         : kRisc[zone.risc % 5];
    const uint32_t thread = tm.get_thread_id();

    // Colour resolution, matching the legacy wire's getMarkerColor exactly (TracyTTDevice.hpp): an
    // explicit colour wins (the DRISC role tables), then PROFILER-keyword names go Tomato3, then the
    // per-RISC palette -- BRISC Orange2, NCRISC SeaGreen3, TRISC_0/1/2 SkyBlue3/Turquoise2/CadetBlue1.
    // Without this every worker zone shipped colour 0 and the GUI fell back to its own palette -- the
    // "colors are all wrong" regression of the zones-at-arrival rework.
    uint32_t color = zone.color;
    if (color == 0) {
        static constexpr uint32_t kRiscColor[5] = {0xEE9A00u, 0x43CD80u, 0x6CA6CDu, 0x00E5EEu, 0x98F5FFu};
        color = zone.name.find("PROFILER") != std::string_view::npos ? 0xCD4F39u : kRiscColor[zone.risc % 5];
    }

    // Intern the srcloc: QueueGpuZone ships a raw pointer that the SERVER dereferences by querying this
    // process later, so the SourceLocationData and its name string must outlive the capture -- allocated
    // once per (zone id, colour), never freed. Bounded by distinct zone names x RISCs, not zone count
    // (the same name on two RISCs carries two colours, hence two entries -- the colour is in the key).
    const tracy::SourceLocationData* srcloc = nullptr;
    {
        const uint64_t key = (static_cast<uint64_t>(zone.timer_id) << 32) | color;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = zone_srclocs_.find(key);
        if (it == zone_srclocs_.end()) {
            const std::string nm = zone.name.empty() ? fmt::format("Zone_{}", zone.timer_id) : std::string(zone.name);
            char* nm_copy = new char[nm.size() + 1];
            std::memcpy(nm_copy, nm.c_str(), nm.size() + 1);
            auto* sl = new tracy::SourceLocationData{nm_copy, "kernel_profiler", "kernel_profiler", 0, color};
            it = zone_srclocs_.emplace(key, sl).first;
        }
        srcloc = static_cast<const tracy::SourceLocationData*>(it->second);
    }

    // One complete zone, one SERIAL queue item. No begin/end split and no per-lane depth mirror (there
    // is nothing to orphan: the server never pops a stack). Serial rather than lock-free, deliberately:
    // context creation and point markers ride the serial queue, and the client drains the lock-free
    // queues BEFORE the serial one each pass, so a lock-free zone could overtake the GpuNewContext it
    // references -- an intermittent tracy-capture segfault, reproduced 2026-08-26. All-serial is totally
    // ordered by construction, and one serial item per zone is still cheaper than the legacy pair (two
    // serial items plus an alloc'd srcloc each).
    TracyTTPushZoneSerial(ctx, srcloc, thread, zone.start, zone.end);
#endif
}

void PerfDebugTracyHandler::HandleWorkerEvent([[maybe_unused]] const perf_debug::WorkerEventPacket& event) {
#if defined(TRACY_ENABLE)
    TracyTTCtx ctx = GetOrCreateContext(
        event.chip_id,
        event.core_noc0_x,
        event.core_noc0_y,
        fmt::format("Device: {} Physical ({},{})", event.chip_id, event.core_noc0_x, event.core_noc0_y));
    if (!ctx) {
        return;
    }

    static constexpr tracy::RiscType kRisc[5] = {
        tracy::RiscType::BRISC,
        tracy::RiscType::NCRISC,
        tracy::RiscType::TRISC_0,
        tracy::RiscType::TRISC_1,
        tracy::RiscType::TRISC_2};

    tracy::TTDeviceMarker marker;
    marker.chip_id = event.chip_id;
    marker.core_x = event.core_noc0_x;
    marker.core_y = event.core_noc0_y;
    marker.risc = IsEthCore(event.chip_id, event.core_noc0_x, event.core_noc0_y) ? tracy::RiscType::ERISC
                                                                                : kRisc[event.risc % 5];
    marker.timestamp = event.timestamp;
    marker.runtime_host_id = event.runtime_host_id;
    // Two kinds, by payload -- every id on this wire is a compile-time structural id, so there is no
    // "runtime id" kind (a runtime value rides PP_DATA payload):
    //   DATA: compile-time tag + payload (DeviceData)
    //   FLAG: compile-time tag, no payload (DeviceFlag)
    marker.marker_type = (event.num_values != 0) ? tracy::TTDeviceMarkerType::DATA : tracy::TTDeviceMarkerType::FLAG;
    // Every id resolves from its kernel's ELF; an empty name here is the bug the teardown summary counts.
    marker.marker_name = event.name.empty() ? fmt::format("Event_{}", event.id) : std::string(event.name);
    marker.file = "kernel_profiler";
    marker.line = 0;

    // The first two uint64s ride the marker's dedicated fields (the Tracy tooltip prints them as
    // Data / Data high); any beyond that go into the metadata map so nothing is silently dropped.
    if (event.num_values > 0) {
        marker.data = event.values[0];
    }
    if (event.num_values > 1) {
        marker.data_high = event.values[1];
    }
#ifdef TRACY_TT_HAS_FULL_DEPS
    for (uint32_t i = 2; i < event.num_values; i++) {
        marker.meta_data[fmt::format("value{}", i)] = event.values[i];
    }
#endif

    TracyTTPushMarker(ctx, marker);

    // ---- NoC-FOOTPRINT per-sweep sample -> Tracy PLOTS, on the DEVICE timebase ------------------------
    //
    // The whole point is the timestamp. A plot stamped at decode time would land milliseconds to the RIGHT of
    // the drainer zones it explains (the mover's ring tail alone trails the last worker zone by 2.5-2.9 ms),
    // so PlotDataAt takes the instant explicitly and we convert the device tick into the TSC domain the
    // server will map. That conversion is the exact inverse of the device-ZONE mapping, which is why a sample
    // lands on the same pixel column as the DRISC-SWEEP zone it came from:
    //
    //   a zone displays at   ConvertGpuTime(tgpu) = round(ts/freq) - anchor_ns + TscTime(host_start)
    //   a plot displays at   TscTime(tsc) = (tsc - baseTime) * timerMul
    //   equate them  =>      tsc = (round(ts/freq) - anchor_ns) / timerMul + host_start
    //
    // and the server-private baseTime cancels, which is what makes this computable client-side at all.
    // Discriminated BY NAME, never by id value: the sample's id is an ordinary structural source-location
    // id now (it used to be the fixed kernel_profiler::SPSC_DATA_ID_NOCFP, 0x7FF0, in a reserved band), and a
    // structural id legitimately moves whenever a source line in the drain kernel does. The name comes from
    // the drain kernel's own ELF, so it is the stable handle.
    if (event.name == "DRISC-NOC-FOOTPRINT" && event.num_values >= 2) {
        // PER-CORE anchor, same as the zone path. These samples come off a DRAM core, whose wall clock has its
        // own origin, so resolving the CHIP anchor here would land every plot point the same reset->open gap to
        // the right of the workload -- off the visible timeline entirely, not merely misplaced.
        ChipAnchor a{};
        if (!LookupAnchor(event.chip_id, event.core_noc0_x, event.core_noc0_y, a) || a.frequency == 0.0) {
            return;  // no anchor yet: a sample with no mapping is worse than no sample
        }
        const double timer_mul = tracy::Profiler::GetTimerMul();
        if (timer_mul == 0.0) {
            return;
        }
        const double dev_ns = static_cast<double>(event.timestamp) / a.frequency;
        const double anchor_ns = a.first_timestamp / a.frequency;
        const int64_t tsc = static_cast<int64_t>((dev_ns - anchor_ns) / timer_mul) + static_cast<int64_t>(a.host_start);

        const uint64_t ck = ContextKey(event.chip_id, event.core_noc0_x, event.core_noc0_y);
        // Payload order is the shared contract in profiler_common.h (SpscNocFpWord). The consumer packed each
        // pair hi-word first, so values[0] = rd_words<<32 | rd_txns and values[1] = wr_words<<32 | wr_txns.
        const uint64_t rd_words = event.values[0] >> 32, rd_txns = event.values[0] & 0xFFFFFFFFu;
        const uint64_t wr_words = event.values[1] >> 32, wr_txns = event.values[1] & 0xFFFFFFFFu;
        // RATE, not per-sweep volume. The denominator is the SAMPLING INTERVAL -- exactly the window this
        // counter delta accumulated over -- so it is the honest divisor. Per-sweep volume was not comparable
        // across roles: a filler's ~307 KB lands over a ~17 us sweep and a mover's ~74 KB over ~1.4 us, so the
        // row heights INVERTED the real bandwidths (~18 GB/s vs ~53 GB/s).
        //
        // bytes/ns IS GB/s with GB = 10^9 B, so no scale factor is needed. Decimal GB, not GiB.
        double dt_ns = 0.0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto pit = nocfp_last_ns_.find(ck);
            if (pit != nocfp_last_ns_.end()) {
                dt_ns = dev_ns - pit->second;
            }
            nocfp_last_ns_[ck] = dev_ns;
        }
        if (dt_ns <= 0.0) {
            return;  // first sample for this drainer (or a non-advancing clock): a rate needs an interval
        }
        // Two decimals. Tracy's ConfigurePlot exposes type/step/fill/colour but NO precision, so the only
        // lever is the value itself -- quantise here and the GUI has nothing longer to print.
        //
        // The cost, stated because it is a floor-to-zero and this code has been bitten by silent zeros: any
        // rate below 0.005 GB/s (5 MB/s) renders as 0.00 and becomes indistinguishable from "no traffic". The
        // smallest real value here is a mover's idle sweep -- 8 B of head polling over ~1.3 us = 0.006 GB/s --
        // which survives, but only just. If a future sampler runs at a longer interval, check this again.
        const auto r2 = [](double v) { return std::round(v * 100.0) / 100.0; };
        const double rd_gbps = r2(static_cast<double>(rd_words) * kNocWordBytes / dt_ns);
        const double wr_gbps = r2(static_cast<double>(wr_words) * kNocWordBytes / dt_ns);
        const double rd_mtxns = r2(static_cast<double>(rd_txns) * 1000.0 / dt_ns);  // txns/ns * 1e9 / 1e6
        const double wr_mtxns = r2(static_cast<double>(wr_txns) * 1000.0 / dt_ns);

        // Role in the label, because the two roles share plot NAMES but not meanings: a filler's rd is a
        // 10,496 B span out of a worker's L1, a mover's is a frame run out of DRAM, and their write sides are
        // DRAM versus the PCIe tile. Reading one row's scale onto the other is the mistake this prevents --
        // the same reason the zone colours are per-role.
        const auto rit = drisc_roles_.find(ck);
        const std::string who = rit != drisc_roles_.end()
                                    ? fmt::format("DRISC {}-{} {}", event.core_noc0_x, event.core_noc0_y, rit->second)
                                    : fmt::format("DRISC {}-{}", event.core_noc0_x, event.core_noc0_y);
        tracy::Profiler::PlotDataAt(intern_plot_name(ck * 8 + 0, who + " NoC rd GB/s"), rd_gbps, tsc);
        tracy::Profiler::PlotDataAt(intern_plot_name(ck * 8 + 1, who + " NoC wr GB/s"), wr_gbps, tsc);
        tracy::Profiler::PlotDataAt(intern_plot_name(ck * 8 + 2, who + " NoC rd Mtxn/s"), rd_mtxns, tsc);
        tracy::Profiler::PlotDataAt(intern_plot_name(ck * 8 + 3, who + " NoC wr Mtxn/s"), wr_mtxns, tsc);

        // DERIVED AGGREGATE, SPLIT BY ROLE. Sum over the drainers of one role of each one's most recent
        // per-sweep bytes. Still not a measurement -- drainers sweep independently, so no instant exists at
        // which all of them sampled together -- but split it is at least a sum of LIKE quantities.
        //
        // The single all-drainer total this replaces was actively misleading. It added ~307 kB/sweep from a
        // filler to ~0.6 kB from an idle mover, so the row swung between roughly 2,000 and 124 depending on
        // WHICH drainer's sample happened to arrive, and since it was emitted on every arrival from any of the
        // six it produced pairs of points at one x with wildly different values. Same lesson as the per-role
        // zone colours and the role in these plot names: the two roles share units but not meaning, and adding
        // them together produces a number that describes neither.
        //
        // NOTE what this does NOT fix: two drainers of the SAME role can still quantise onto one TSC tick
        // (the conversion divides by timer_mul and truncates), so a role row can carry two points at one x.
        // Within a role those values are comparable, so it reads as sample noise rather than a 16x cliff.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            nocfp_last_gbps_[ck] = rd_gbps + wr_gbps;
            // Bucket by role rather than summing everything. A drainer whose role was never registered lands
            // in its own "UNLABELLED" bucket instead of being folded into one of the real ones.
            const char* my_role = rit != drisc_roles_.end() ? rit->second : "UNLABELLED";
            double role_total = 0.0;
            for (const auto& [k, v] : nocfp_last_gbps_) {
                const auto krit = drisc_roles_.find(k);
                const char* kr = krit != drisc_roles_.end() ? krit->second : "UNLABELLED";
                if (std::strcmp(kr, my_role) == 0) {
                    role_total += v;
                }
            }
            // Core count for the LABEL comes from the registered roles, not from how many have sampled so far.
            // nocfp_last_kb_ fills in as samples arrive, so counting it would put "(1 cores)" in the name that
            // gets interned first and keep it forever -- the name must be stable from the first emission.
            uint32_t role_n = 0;
            for (const auto& [k, r] : drisc_roles_) {
                if (std::strcmp(r, my_role) == 0) {
                    role_n++;
                }
            }
            // Intern keys for DERIVED rows live in a reserved high range, because the per-drainer keys are
            // `ck * 8 + series` and ContextKey(chip 0, x 0, y 0) is ZERO -- so mover 0-0's four plots occupy
            // keys 0..3. Small integers for derived rows therefore ALIAS them, and intern_plot_name keeps the
            // first name seen for a key, so a row vanishes with no error. That is exactly what happened: the
            // single all-drainer aggregate used key 1 and silently swallowed "DRISC 0-0 MOVER NoC wr KB/sweep",
            // which read as an absent row rather than a fault.
            constexpr uint64_t kDerivedKeyBase = 1ull << 60;
            const uint64_t role_key = kDerivedKeyBase + (std::strcmp(my_role, "FILLER") == 0  ? 0u
                                                         : std::strcmp(my_role, "MOVER") == 0 ? 1u
                                                                                              : 2u);
            role_total = std::round(role_total * 100.0) / 100.0;  // a sum of 2dp values can still carry fp dust
            tracy::Profiler::PlotDataAt(
                intern_plot_name(
                    role_key, fmt::format("DRISC {} total NoC GB/s ({} cores, sum of latest)", my_role, role_n)),
                role_total,
                tsc);
        }
    }
#endif
}

}  // namespace tt::tt_metal
