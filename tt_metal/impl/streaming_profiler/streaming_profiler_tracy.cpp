// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_tracy.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <map>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#if defined(TRACY_ENABLE)
#include <common/TracyTTDeviceData.hpp>
#include <tracy/Tracy.hpp>
#include <client/TracyProfiler.hpp>
#endif

#include "impl/streaming_profiler/spsc_packet.h"
#include "impl/streaming_profiler/streaming_profiler_d2d_sync.hpp"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "impl/streaming_profiler/streaming_profiler_decode.hpp"
#include "impl/streaming_profiler/streaming_profiler_placement_map.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace api = experimental::streaming_profiler;

namespace {

constexpr uint32_t kStallColor = 0xCD4F39u;
constexpr size_t kSrclocTableInitial = 1024;

uint64_t lane_key(const api::Core& core) {
    return (static_cast<uint64_t>(core.chip_id) << 32) | ((static_cast<uint64_t>(core.logical.x) & 0xFFFu) << 20) |
           ((static_cast<uint64_t>(core.logical.y) & 0xFFFu) << 8) | (static_cast<uint64_t>(core.risc) & 0xFFu);
}

uint64_t srcloc_key(std::string_view name, uint32_t color, uint32_t risc) {
    return (static_cast<uint64_t>(name.size()) << 32) | (static_cast<uint64_t>(color) << 3) | risc;
}

size_t srcloc_hash(const char* name, uint64_t key) {
    return (((reinterpret_cast<uintptr_t>(name) >> 3) ^ key) * 0x9E3779B97F4A7C15ull) >> 32;
}

#if defined(TRACY_ENABLE)
constexpr tracy::RiscType kRisc[5] = {
    tracy::RiscType::BRISC,
    tracy::RiscType::NCRISC,
    tracy::RiscType::TRISC_0,
    tracy::RiscType::TRISC_1,
    tracy::RiscType::TRISC_2};
constexpr uint32_t kRiscColor[5] = {0xEE9A00u, 0x43CD80u, 0x6CA6CDu, 0x00E5EEu, 0x98F5FFu};
#endif

// Tracy's timer at construction; every context's cpuTime is expressed against it. Without Tracy there is no
// timer and no consumer for the value, so the anchor is simply zero.
int64_t tracy_anchor_now() {
#if defined(TRACY_ENABLE)
    return tracy::Profiler::GetTime();
#else
    return 0;
#endif
}

// The GPU contexts' origin: two seconds before the anchor, or as much of it as lies after the profiler's own epoch.
int64_t origin_margin_ns([[maybe_unused]] int64_t anchor_tracy) {
#if defined(TRACY_ENABLE)
    const int64_t anchor_ns =
        static_cast<int64_t>(static_cast<double>(anchor_tracy - TracyGetBaseTime()) * TracyGetTimerMul());
    return std::max<int64_t>(0, std::min<int64_t>(2'000'000'000, anchor_ns - 1'000'000));
#else
    return 0;
#endif
}

}  // namespace

TracySink::TracySink(Service& service) :
    service_(service),
    anchor_tracy_(tracy_anchor_now()),
    origin_margin_ns_(origin_margin_ns(anchor_tracy_)),
    srcloc_table_(kSrclocTableInitial) {
    handle_ = service_.add_consumer(
        "tracy",
        [this](const Batch& b, uint64_t) { on_batch(b); },
        ConsumerHooks{
            .on_attach =
                [this](const CaptureContext& ctx) {
                    chips_.clear();
                    for (const auto& d : ctx.devices) {
                        chips_.push_back(d.chip_id);
                    }
                    root_dev_ = ctx.root_dev;
                },
            .clock_sink =
                [this](const ClockSample& cs) {
                    plot_samples_.push_back(
                        PlotSample{cs.dev, cs.kind, cs.lane / profiler::kSpscNRiscDecode, cs.ts, cs.value});
                },
            .on_capture_end =
                [this](const CaptureContext&) {
                    SyncPlots::wait_complete(std::chrono::seconds(10));
                    emit_plots();
                }});
}

TracySink::~TracySink() {
    service_.remove_consumer(handle_);
#if defined(TRACY_ENABLE)
    for (auto& [key, core] : cores_) {
        TracyTTDestroy(core.ctx);
    }
#endif
}

void TracySink::on_batch(const Batch& batch) {
    if (plots_only_) {
        return;
    }
    for (const api::Zone& z : batch.zones()) {
        emit_zone(z);
    }
    for (const api::TimestampedData& d : batch.timestamped_data()) {
        emit_data(d);
    }
    for (const api::Event& e : batch.events()) {
        emit_event(e);
    }
}

void TracySink::emit_zone(const api::Zone& z) {
    push_zone(
        z.core(),
        z.site().name,
        api::host_clock::tsc(z.start_time()),
        api::host_clock::tsc(z.end_time()),
        z.site().name == api::STALL_ZONE_NAME ? kStallColor : 0);
}

void TracySink::emit_data(const api::TimestampedData& d) {
    push_marker(d.core(), d.site().name, api::host_clock::tsc(d.time()), d.runtime_id(), d.payload());
}

void TracySink::emit_event(const api::Event& e) {
    push_marker(e.core(), e.site().name, api::host_clock::tsc(e.time()), e.runtime_id(), {});
}

// The contexts are populated with cpuTime = anchor_tracy_ and gpuTime = origin_margin_ns_, so a timestamp of
// origin_margin_ns_ lands at the anchor and the tick's distance from it is scaled by Tracy's own multiplier: the
// record sits exactly where a host zone stamped at that tick would.
int64_t TracySink::timeline_ns(int64_t tsc) const {
#if defined(TRACY_ENABLE)
    return static_cast<int64_t>(static_cast<double>(tsc - anchor_tracy_) * TracyGetTimerMul()) + origin_margin_ns_;
#else
    return tsc;
#endif
}

TracySink::Lane TracySink::lane(const Core& core) {
    const uint64_t key = lane_key(core);
    if (key == lane_key_) {
        return lane_hit_;
    }
    Lane ln;
    ln.risc = static_cast<uint32_t>(core.risc);
#if defined(TRACY_ENABLE)
    auto [it, fresh] = cores_.try_emplace(key & ~uint64_t{0xFF});
    CoreEntry& ce = it->second;
    if (fresh) {
        ce.ctx = TracyTTContext();
        // Timestamps arrive already on the host timeline in nanoseconds from anchor_tracy_: identity mapping, and
        // calibrated so the GUI offers no drift control. Everything the sink emits goes through this thread's
        // lock-free queue, whose FIFO order is what keeps the context ahead of the zones that reference it.
        TracyTTContextPopulateCalibratedLockfree(ce.ctx, anchor_tracy_, static_cast<double>(origin_margin_ns_), 1.0);
        const std::string name = fmt::format(
            "Device: {}, Logical ({},{}) Physical ({},{})",
            core.chip_id,
            core.logical.x,
            core.logical.y,
            core.physical.x,
            core.physical.y);
        TracyTTContextNameLockfree(ce.ctx, name.c_str(), name.size());
    }
    if ((ce.named & (1u << ln.risc)) == 0) {
        // Same thread-id packing as the marker path (TTDeviceMarker::get_thread_id), so zones and markers share the
        // per-RISC row.
        tracy::TTDeviceMarker tm;
        tm.chip_id = core.chip_id;
        tm.core_x = core.logical.x;
        tm.core_y = core.logical.y;
        tm.risc = kRisc[ln.risc];
        ce.thread[ln.risc] = tm.get_thread_id();
        tracy::SetThreadName(ce.thread[ln.risc], kRiscNames[ln.risc]);
        ce.named |= static_cast<uint8_t>(1u << ln.risc);
    }
    ln.ctx = ce.ctx;
    ln.thread = ce.thread[ln.risc];
#endif
    lane_key_ = key;
    lane_hit_ = ln;
    return ln;
}

const void* TracySink::srcloc(std::string_view name, uint32_t color, uint32_t risc) {
    static constexpr char kUnnamed[] = "";
    if (name.data() == nullptr) {
        name = kUnnamed;  // a null address would read as an empty table slot
    }
    const uint64_t key = srcloc_key(name, color, risc);
    const size_t mask = srcloc_table_.size() - 1;
    // Pointer identity, not text: the table is keyed by the name's address (a literal that never moves), so
    // the null-termination this check warns about does not apply.
    const char* const name_ptr = name.data();  // NOLINT(bugprone-suspicious-stringview-data-usage)
    for (size_t i = srcloc_hash(name_ptr, key) & mask;; i = (i + 1) & mask) {
        const SrclocEntry& e = srcloc_table_[i];
        if (e.name == name_ptr && e.key == key) {
            return e.srcloc;
        }
        if (e.name == nullptr) {
            return srcloc_slow(name, color, risc);
        }
    }
}

const void* TracySink::srcloc_slow(
    [[maybe_unused]] std::string_view name, [[maybe_unused]] uint32_t color, [[maybe_unused]] uint32_t risc) {
#if defined(TRACY_ENABLE)
    // Pointer identity again -- see srcloc() above.
    const char* const name_ptr = name.data();  // NOLINT(bugprone-suspicious-stringview-data-usage)
    const SrclocEntry entry{name_ptr, srcloc_key(name, color, risc), nullptr};
    // Matches getMarkerColor (TracyTTDevice.hpp): explicit colour, then Tomato3 for PROFILER-keyword names, then
    // the per-RISC palette; colour 0 makes the GUI fall back to its own.
    if (color == 0) {
        color = name.find("PROFILER") != std::string_view::npos ? 0xCD4F39u : kRiscColor[risc];
    }
    const std::string skey = fmt::format("{}#{:08x}", name, color);
    auto it = srclocs_.find(skey);
    if (it == srclocs_.end()) {
        char* nm_copy = new char[name.size() + 1];
        std::memcpy(nm_copy, name.data(), name.size());
        nm_copy[name.size()] = 0;
        auto* sl = new tracy::SourceLocationData{nm_copy, "kernel_profiler", "kernel_profiler", 0, color};
        it = srclocs_.emplace(skey, sl).first;
    }
    auto insert = [](std::vector<SrclocEntry>& table, const SrclocEntry& e) {
        const size_t mask = table.size() - 1;
        size_t i = srcloc_hash(e.name, e.key) & mask;
        while (table[i].name != nullptr) {
            i = (i + 1) & mask;
        }
        table[i] = e;
    };
    if ((srcloc_count_ + 1) * 2 > srcloc_table_.size()) {
        std::vector<SrclocEntry> old = std::move(srcloc_table_);
        srcloc_table_.assign(old.size() * 2, {});
        for (const SrclocEntry& e : old) {
            if (e.name != nullptr) {
                insert(srcloc_table_, e);
            }
        }
    }
    insert(srcloc_table_, {entry.name, entry.key, it->second});
    srcloc_count_++;
    return it->second;
#else
    return nullptr;
#endif
}

void TracySink::push_zone(
    [[maybe_unused]] const Core& core,
    [[maybe_unused]] std::string_view name,
    [[maybe_unused]] int64_t start_tsc,
    [[maybe_unused]] int64_t end_tsc,
    [[maybe_unused]] uint32_t color) {
#if defined(TRACY_ENABLE)
    int64_t s = timeline_ns(start_tsc), e = timeline_ns(end_tsc);
    if (s < 0) {
        clamped_zones_++;
        s = 0;
    }
    if (e < s) {
        clamped_zones_++;
        e = s;
    }
    const Lane ln = lane(core);
    TracyTTPushZone(
        ln.ctx,
        static_cast<const tracy::SourceLocationData*>(srcloc(name, color, ln.risc)),
        ln.thread,
        static_cast<uint64_t>(s),
        static_cast<uint64_t>(e));
#endif
}

void TracySink::push_marker(
    [[maybe_unused]] const Core& core,
    [[maybe_unused]] std::string_view name,
    [[maybe_unused]] int64_t tsc,
    [[maybe_unused]] uint32_t runtime_id,
    [[maybe_unused]] std::span<const uint64_t> values) {
#if defined(TRACY_ENABLE)
    int64_t ts = timeline_ns(tsc);
    if (ts < 0) {
        clamped_markers_++;
        ts = 0;
    }
    TracyTTCtx ctx = lane(core).ctx;
    tracy::TTDeviceMarker marker;
    marker.chip_id = core.chip_id;
    marker.core_x = core.logical.x;
    marker.core_y = core.logical.y;
    marker.risc = kRisc[static_cast<uint32_t>(core.risc)];
    marker.timestamp = static_cast<uint64_t>(ts);
    marker.runtime_host_id = runtime_id;
    marker.marker_type = values.empty() ? tracy::TTDeviceMarkerType::EVENT : tracy::TTDeviceMarkerType::DATA;
    marker.marker_name = std::string(name);
    marker.file = "kernel_profiler";
    marker.line = 0;
    if (!values.empty()) {
        marker.data = values[0];
    }
    if (values.size() > 1) {
        marker.data_high = values[1];
    }
#ifdef TRACY_TT_HAS_FULL_DEPS
    for (size_t i = 2; i < values.size(); i++) {
        marker.meta_data[fmt::format("value{}", i)] = values[i];
    }
#endif
    TracyTTPushMarkerLockfree(ctx, marker);
#endif
}

// PlotDataAt takes an absolute timer stamp, which a TSC tick already is; a zone at the same tick lands at the same
// place through its context.
void TracySink::plot_point([[maybe_unused]] const char* name, [[maybe_unused]] double value, int64_t tsc) {
#if defined(TRACY_ENABLE)
    if (tsc <= 0) {
        clamped_plot_points_++;
        return;
    }
    tracy::Profiler::PlotDataAt(name, value, tsc);
#endif
}

void TracySink::emit_plots() {
    // Each stream (device, kind, eth core) is one refclk counter, differenced on its own.
    std::sort(plot_samples_.begin(), plot_samples_.end(), [](const PlotSample& a, const PlotSample& b) {
        if (a.dev != b.dev) {
            return a.dev < b.dev;
        }
        if (a.kind != b.kind) {
            return a.kind < b.kind;
        }
        if (a.core != b.core) {
            return a.core < b.core;
        }
        return a.ts < b.ts;
    });
    struct Series {
        uint32_t dev, kind;
        std::vector<FreqPoint> pts;
    };
    std::vector<Series> series;
    for (size_t i = 0; i < plot_samples_.size();) {
        size_t j = i + 1;
        while (j < plot_samples_.size() && plot_samples_[j].dev == plot_samples_[i].dev &&
               plot_samples_[j].kind == plot_samples_[i].kind && plot_samples_[j].core == plot_samples_[i].core) {
            j++;
        }
        series.push_back(Series{plot_samples_[i].dev, plot_samples_[i].kind, compute_frequency(i, j)});
        i = j;
    }
    // The root's scale is exactly 1. Every other chip is plotted as its AICLK over the root's at the same host
    // instant, per sync kind: the k_B/k_A of wall_B = (k_B*s/k_A)*wall_A, the factor that scales that chip's
    // wall-clock rate onto the root's. (The refclk ratio s, ~ppm, is the separate scale-convergence plot from the d2d
    // consumer.)
    const uint32_t root_dev = root_dev_;
    std::map<uint32_t, std::vector<FreqPoint>> root_ref;  // kind -> the root's series, sorted by host tick
    for (const Series& s : series) {
        if (s.dev == root_dev) {
            auto& r = root_ref[s.kind];
            r.insert(r.end(), s.pts.begin(), s.pts.end());
        }
    }
    for (auto& [kind, r] : root_ref) {
        std::sort(r.begin(), r.end(), [](const FreqPoint& a, const FreqPoint& b) { return a.tsc < b.tsc; });
    }
#if defined(TRACY_ENABLE)
    for (const Series& s : series) {
        if (s.pts.empty() || s.dev >= chips_.size() || root_dev >= chips_.size()) {
            continue;
        }
        const char* name = intern_name(fmt::format("d2d freq scale chip{}/chip{}", chips_[s.dev], chips_[root_dev]));
        if (s.dev == root_dev) {
            for (const FreqPoint& p : s.pts) {
                plot_point(name, 1.0, p.tsc);
            }
            continue;
        }
        const auto rit = root_ref.find(s.kind);
        if (rit == root_ref.end() || rit->second.empty()) {
            continue;
        }
        const std::vector<FreqPoint>& ref = rit->second;
        for (const FreqPoint& p : s.pts) {
            // The root's estimate at or just before this instant (<= one sample stale: 3 us local, 1 ms link).
            auto it = std::upper_bound(
                ref.begin(), ref.end(), p.tsc, [](int64_t h, const FreqPoint& q) { return h < q.tsc; });
            if (it == ref.begin()) {
                continue;
            }
            --it;
            if (it->ghz <= 0.0) {
                continue;
            }
            plot_point(name, p.ghz / it->ghz, p.tsc);
        }
    }
    // Series the d2d consumer computed at capture end (the cross-chip refclk scale regression and the sync error),
    // placed the same way; it runs on its own thread, so wait for its final publish.
    SyncPlots::wait_complete(std::chrono::seconds(10));
    for (auto& [name, pts] : SyncPlots::drain()) {
        const char* nm = intern_name(name);
        for (const SyncPlotPoint& p : pts) {
            plot_point(nm, p.value, p.tsc);
        }
    }
#else
    SyncPlots::wait_complete(std::chrono::seconds(10));
    SyncPlots::drain();
#endif
    plot_samples_.clear();
    plot_samples_.shrink_to_fit();
    if (clamped_zones_ != 0 || clamped_markers_ != 0 || clamped_plot_points_ != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] Tracy sink clamped {} zones, {} markers and {} plot points to the timeline origin: "
            "records placed before the contexts' origin",
            clamped_zones_,
            clamped_markers_,
            clamped_plot_points_);
    }
}

// PlotDataAt keys a plot by its name POINTER, so every name lives for the sink's lifetime.
const char* TracySink::intern_name(const std::string& name) { return plot_names_.insert(name).first->c_str(); }

// One stream of PP_CLOCK samples [begin, end) -> that chip's applied AICLK in GHz at every sample: the sliding
// dwall/drefclk over the trailing window (the refclk is a fixed 50 MHz, so wall ticks per refclk tick x 50 MHz is
// the AICLK), placed on the host as the chip's records are. The window is set in REFCLK ticks so a dropped sample
// only widens it: >= 1 ms (2e-5 quantisation) for both the 1 ms link stamps and the tracker's 1 ms keepalives, whose
// change bursts a few us apart the window steps over, as it does the link sender's two stamps per round.
std::vector<TracySink::FreqPoint> TracySink::compute_frequency(size_t begin, size_t end) const {
    std::vector<FreqPoint> out;
    if (end <= begin) {
        return out;
    }
    const PlotSample& s0 = plot_samples_[begin];
    if (s0.dev >= chips_.size()) {
        return out;
    }
    // A link stamp's value is the frame event's time in quarter-ns, recorded at a later wall instant; it is not a
    // clock reading paired with its own wall tick, so only the local refclk kind carries a rate.
    if (s0.kind != PP_CLOCK_LOCAL_REFCLK) {
        return out;
    }
    constexpr double kRefclkHz = 50.0e6;
    const uint64_t window = 50'000;  // refclk ticks: 1 ms
    std::vector<uint64_t> refclk(end - begin);
    for (size_t i = begin; i < end; i++) {
        refclk[i - begin] = plot_samples_[i].value;
    }
    out.reserve(end - begin);
    size_t j = 0;  // trailing edge: the LATEST sample still >= window behind, for the tightest window over target
    for (size_t i = 0; i < end - begin; i++) {
        while (j + 1 < i && refclk[i] - refclk[j + 1] >= window) {
            j++;
        }
        if (i == j || refclk[i] - refclk[j] < window) {
            continue;
        }
        const double dr = static_cast<double>(refclk[i] - refclk[j]);
        const double dw =
            static_cast<double>(plot_samples_[begin + i].ts) - static_cast<double>(plot_samples_[begin + j].ts);
        const int64_t tsc =
            service_.sync().map().lookup_tsc(chips_[s0.dev], static_cast<int64_t>(plot_samples_[begin + i].ts));
        if (tsc == 0) {
            continue;
        }
        out.push_back(FreqPoint{tsc, (dw / dr) * kRefclkHz * 1e-9});
    }
    return out;
}

}  // namespace tt::tt_metal::streaming_profiler
