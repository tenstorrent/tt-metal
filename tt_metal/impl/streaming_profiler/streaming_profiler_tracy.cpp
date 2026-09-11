// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_tracy.hpp"

#include <algorithm>
#include <map>
#include <cstring>
#include <limits>

#include <umd/device/driver_atomics.hpp>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#if defined(TRACY_ENABLE)
#include <common/TracyTTDeviceData.hpp>
#include <tracy/Tracy.hpp>
#include <client/TracyProfiler.hpp>
#endif

#include "impl/streaming_profiler/spsc_packet.h"
#include "impl/streaming_profiler/streaming_profiler_service.hpp"
#include "impl/streaming_profiler/streaming_profiler_decode.hpp"
#include "impl/streaming_profiler/streaming_profiler_sync_correction.hpp"
#include "impl/streaming_profiler/spsc_packet.h"

namespace tt::tt_metal::streaming_profiler {

namespace api = experimental::streaming_profiler;

namespace {

constexpr uint32_t kStallColor = 0xCD4F39u;
constexpr size_t kSrclocTableInitial = 1024;
// A probe pair is good to ~35 ns, so the ratio is only worth measuring over a baseline well above that; the map
// then takes a new segment this often, and each one sees a longer baseline than the last.
constexpr int64_t kMinSlopeBaselineNs = 200'000'000;
constexpr int64_t kRefineEveryNs = 1'000'000'000;

int64_t ns_since_epoch(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

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
    {
        // Two seconds, or as much of it as lies after the profiler's own epoch.
        const double mul = TracyGetTimerMul() > 0.0 ? TracyGetTimerMul() : 1.0;
        const int64_t anchor_ns = static_cast<int64_t>(static_cast<double>(anchor_tracy_) * mul);
        origin_margin_ns_ = std::max<int64_t>(0, std::min<int64_t>(2'000'000'000, anchor_ns - 1'000'000));
    }
#endif
}

}  // namespace

TracySink::TracySink(Service& service) :
    service_(service), anchor_tracy_(tracy_anchor_now()), srcloc_table_(kSrclocTableInitial) {
    base_ = probe();
    handle_ = service_.add_consumer(
        "tracy",
        [this](const Batch& b, uint64_t capture) { on_batch(b, capture); },
        ConsumerHooks{
            .on_attach =
                [this](const CaptureContext& ctx) {
                    clocks_.clear();
                    eth_clocks_.clear();
                    for (const auto& d : ctx.devices) {
                        clocks_.push_back(d.clock);
                        eth_clocks_.push_back(d.eth_clock);
                    }
                },
            .clock_sink =
                [this](const ClockSample& cs) {
                    plot_samples_.push_back(
                        PlotSample{cs.dev, cs.kind, cs.lane / profiler::kSpscNRiscDecode, cs.ts, cs.value});
                },
            .on_capture_end =
                [this](const CaptureContext&) {
                    SyncPlots::wait_complete(std::chrono::seconds(10));
                    flush_held(true);
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

void TracySink::on_batch(const Batch& batch, uint64_t capture) {
    // A map may only start fresh between captures: re-measuring an offset mid-capture moved every later zone by the
    // read pair's jitter, and a zone spanning the change no longer contained its children.
    if (capture != capture_) {
        start_capture_map();
        capture_ = capture;
    } else if (ns_since_epoch(std::chrono::steady_clock::now()) >= next_refine_ns_) {
        refine_map();
    }
    if (plots_only_) {
        return;
    }
    for (const api::Zone& z : batch.zones()) {
        held_[z.core().chip_id].zones.push_back(z);
    }
    for (const api::TimestampedData& d : batch.timestamped_data()) {
        std::vector<std::byte> bytes(d.size_bytes());
        std::memcpy(bytes.data(), &d, d.size_bytes());
        held_[d.core().chip_id].data.push_back(std::move(bytes));
    }
    for (const api::Event& e : batch.events()) {
        held_[e.core().chip_id].events.push_back(e);
    }
    flush_held(false);
}

void TracySink::flush_held(bool all) {
    for (auto& [chip, held] : held_) {
        const int64_t until = all ? std::numeric_limits<int64_t>::max() : SyncCorrections::published_until_ns(chip);
        while (!held.zones.empty() && held.zones.front().base_ns(held.zones.front().end_timestamp()) <= until) {
            emit_zone(held.zones.front());
            held.zones.pop_front();
        }
        while (!held.data.empty()) {
            const auto& d = *reinterpret_cast<const api::TimestampedData*>(held.data.front().data());
            if (d.base_ns(d.timestamp()) > until) {
                break;
            }
            emit_data(d);
            held.data.pop_front();
        }
        while (!held.events.empty() && held.events.front().base_ns(held.events.front().timestamp()) <= until) {
            emit_event(held.events.front());
            held.events.pop_front();
        }
    }
}

void TracySink::emit_zone(const api::Zone& z) {
    const auto [start, end] = z.host_span();
    push_zone(
        z.core(),
        z.site().name,
        ns_since_epoch(start),
        ns_since_epoch(end),
        z.site().name == api::kStallZoneName ? kStallColor : 0);
}

void TracySink::emit_data(const api::TimestampedData& d) {
    push_marker(d.core(), d.site().name, ns_since_epoch(d.time()), d.runtime_id(), d.payload());
}

void TracySink::emit_event(const api::Event& e) {
    push_marker(e.core(), e.site().name, ns_since_epoch(e.time()), e.runtime_id(), {});
}

TracySink::Probe TracySink::probe() const {
    Probe best{0, 0};
#if defined(TRACY_ENABLE)
    // The steady_clock read sits between two Tracy reads: the cheaper read outside makes the tightest bracket, the
    // fences keep Tracy's bare rdtsc from retiring across it, and the tightest of several pairs excludes a preemption.
    int64_t best_gap = std::numeric_limits<int64_t>::max();
    for (int i = 0; i < 16; i++) {
        const int64_t t0 = tracy::Profiler::GetTime();
        tt_driver_atomics::lfence();
        const int64_t s = ns_since_epoch(std::chrono::steady_clock::now());
        tt_driver_atomics::lfence();
        const int64_t t1 = tracy::Profiler::GetTime();
        if (t1 - t0 < best_gap) {
            best_gap = t1 - t0;
            best.steady_ns = s;
            best.tracy_ns = static_cast<int64_t>(
                ((static_cast<double>(t0) + static_cast<double>(t1)) / 2.0 - static_cast<double>(anchor_tracy_)) *
                TracyGetTimerMul());
        }
    }
#endif
    return best;
}

// Both clocks scale the TSC, Tracy with a multiplier it calibrated once over 200 ms at startup and steady_clock
// with the kernel's, so their ratio is a constant (1 - 2.5e-7 on the reference box) that a long baseline measures.
double TracySink::slope_since_base(const Probe& p) const {
    const int64_t baseline = p.steady_ns - base_.steady_ns;
    if (baseline < kMinSlopeBaselineNs) {
        return 1.0;
    }
    return static_cast<double>(p.tracy_ns - base_.tracy_ns) / static_cast<double>(baseline);
}

void TracySink::start_capture_map() {
    const Probe p = probe();
    segments_.clear();
    segments_.push_back({p.steady_ns, p.tracy_ns, slope_since_base(p)});
    next_refine_ns_ = p.steady_ns + kRefineEveryNs;
}

// The new segment starts where the old one would have placed this instant, so the map stays continuous and
// monotonic; only its slope improves.
void TracySink::refine_map() {
    const Probe p = probe();
    segments_.push_back({p.steady_ns, to_timeline(p.steady_ns), slope_since_base(p)});
    next_refine_ns_ = p.steady_ns + kRefineEveryNs;
}

int64_t TracySink::to_timeline(int64_t steady_ns) const {
    if (segments_.empty()) {
        return steady_ns;
    }
    const Segment* seg = &segments_.back();
    while (seg != segments_.data() && steady_ns < seg->steady_ns) {
        seg--;
    }
    const int64_t ns =
        seg->tracy_ns + static_cast<int64_t>(static_cast<double>(steady_ns - seg->steady_ns) * seg->slope);
    return ns;  // may be negative: a corrected time before the capture anchor; callers drop, never clamp
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
    [[maybe_unused]] int64_t start_ns,
    [[maybe_unused]] int64_t end_ns,
    [[maybe_unused]] uint32_t color) {
#if defined(TRACY_ENABLE)
    int64_t s = to_timeline(start_ns) + origin_margin_ns_, e = to_timeline(end_ns) + origin_margin_ns_;
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
    [[maybe_unused]] int64_t timestamp_ns,
    [[maybe_unused]] uint32_t runtime_id,
    [[maybe_unused]] std::span<const uint64_t> values) {
#if defined(TRACY_ENABLE)
    int64_t ts = to_timeline(timestamp_ns) + origin_margin_ns_;
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

// The stamp PlotDataAt needs for a point at host time host_ns. The server displays (tsc - baseTime) * m_timerMul --
// an absolute timer-tick stamp like GetTime() -- while to_timeline() yields ns since anchor_tracy_; so
// anchor_tracy_ + ns / mul, which is exactly where the device zones land through their GPU context.
void TracySink::plot_point(const char* name, double value, int64_t host_ns) {
#if defined(TRACY_ENABLE)
    int64_t stamp = plot_stamp(host_ns);
    if (stamp < 0) {
        clamped_plot_points_++;
        stamp = 0;
    }
    tracy::Profiler::PlotDataAt(name, value, stamp);
#endif
}

int64_t TracySink::plot_stamp(int64_t host_ns) const {
#if defined(TRACY_ENABLE)
    const double timer_mul = TracyGetTimerMul() > 0.0 ? TracyGetTimerMul() : 1.0;
    return anchor_tracy_ + static_cast<int64_t>(static_cast<double>(to_timeline(host_ns)) / timer_mul);
#else
    return to_timeline(host_ns);
#endif
}

void TracySink::emit_plots() {
    // Samples arrive in decode order; group them into streams (device, kind, eth core) in time order. Each stream is
    // one refclk counter, unwrapped and differenced on its own.
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
    // Every stream's frequency series first: (host ns, applied AICLK GHz) at each sample.
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
    // The ROOT is the lowest device index with samples -- the d2d consumer's rule; its correction is the identity,
    // so its scale is exactly 1. Every other chip is plotted as its AICLK over the root's at the same host instant,
    // per sync kind: the k_B/k_A of wall_B = (k_B*s/k_A)*wall_A, the factor that scales that chip's wall-clock rate
    // onto the root's. (The refclk ratio s, ~ppm, is the separate scale-convergence plot from the d2d consumer.)
    uint32_t root_dev = std::numeric_limits<uint32_t>::max();
    for (const Series& s : series) {
        root_dev = std::min(root_dev, s.dev);
    }
    std::map<uint32_t, std::vector<FreqPoint>> root_ref;  // kind -> the root's series, sorted by host ns
    for (const Series& s : series) {
        if (s.dev == root_dev) {
            auto& r = root_ref[s.kind];
            r.insert(r.end(), s.pts.begin(), s.pts.end());
        }
    }
    for (auto& [kind, r] : root_ref) {
        std::sort(r.begin(), r.end(), [](const FreqPoint& a, const FreqPoint& b) { return a.host_ns < b.host_ns; });
    }
#if defined(TRACY_ENABLE)
    const auto chip_of = [this](uint32_t dev) {
        return (dev < eth_clocks_.size() && eth_clocks_[dev].frequency_ghz > 0.0) ? eth_clocks_[dev].chip_id
                                                                                  : clocks_[dev].chip_id;
    };
    for (const Series& s : series) {
        if (s.pts.empty() || s.dev >= clocks_.size() || root_dev >= clocks_.size()) {
            continue;
        }
        const bool link = s.kind == PP_CLOCK_LINK_REFCLK || s.kind == PP_CLOCK_LINK_PTP;
        const char* name = intern_name(fmt::format(
            "d2d freq scale chip{}/chip{} {}", chip_of(s.dev), chip_of(root_dev), link ? "link 1ms" : "local 1ms"));
        if (s.dev == root_dev) {
            for (const FreqPoint& p : s.pts) {
                plot_point(name, 1.0, p.host_ns);
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
                ref.begin(), ref.end(), p.host_ns, [](int64_t h, const FreqPoint& q) { return h < q.host_ns; });
            if (it == ref.begin()) {
                continue;
            }
            --it;
            if (it->ghz <= 0.0) {
                continue;
            }
            plot_point(name, p.ghz / it->ghz, p.host_ns);
        }
    }
    // Series the d2d consumer computed at capture end (the cross-chip refclk scale regression and the sync error),
    // placed the same way; it runs on its own thread, so wait for its final publish.
    SyncPlots::wait_complete(std::chrono::seconds(10));
    for (auto& [name, pts] : SyncPlots::drain()) {
        const char* nm = intern_name(name);
        for (const SyncPlotPoint& p : pts) {
            plot_point(nm, p.value, p.host_ns);
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
            "[streaming profiler] Tracy sink clamped {} zones, {} markers and {} plot points to the timeline origin: a "
            "d2d correction exceeded its bound",
            clamped_zones_,
            clamped_markers_,
            clamped_plot_points_);
    }
}

// PlotDataAt keys a plot by its name POINTER, so every name lives for the sink's lifetime.
const char* TracySink::intern_name(const std::string& name) { return plot_names_.insert(name).first->c_str(); }

// One stream of PP_CLOCK samples [begin, end) -> that chip's applied AICLK in GHz at every sample: the sliding
// dwall/drefclk over the trailing window (the refclk is a fixed 50 MHz, so wall ticks per refclk tick x 50 MHz is
// the AICLK), at the sample's host time via the eth clock. The window is set in REFCLK ticks so a dropped sample
// only widens it: >= 1 ms (2e-5 quantisation) for both the 1 ms link stamps and the tracker's 1 ms keepalives, whose
// change bursts a few us apart the window steps over, as it does the link sender's two stamps per round.
std::vector<TracySink::FreqPoint> TracySink::compute_frequency(size_t begin, size_t end) const {
    std::vector<FreqPoint> out;
    if (end <= begin) {
        return out;
    }
    const PlotSample& s0 = plot_samples_[begin];
    if (s0.dev >= clocks_.size()) {
        return out;
    }
    const DeviceClock& eclk = (s0.dev < eth_clocks_.size() && eth_clocks_[s0.dev].frequency_ghz > 0.0)
                                  ? eth_clocks_[s0.dev]
                                  : clocks_[s0.dev];
    if (eclk.frequency_ghz <= 0.0) {
        return out;
    }
    const bool link = s0.kind == PP_CLOCK_LINK_REFCLK || s0.kind == PP_CLOCK_LINK_PTP;
    if (!link && s0.kind != PP_CLOCK_LOCAL_REFCLK) {
        return out;
    }
    constexpr double kRefclkHz = 50.0e6;
    const uint64_t window = 50'000;  // refclk ticks: 1 ms, for the 1 ms link stamps and the tracker's 1 ms keepalives
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
        const int64_t host_ns =
            eclk.anchor_host_ns +
            static_cast<int64_t>(
                (static_cast<double>(plot_samples_[begin + i].ts) - static_cast<double>(eclk.anchor_ticks)) /
                eclk.frequency_ghz);
        out.push_back(FreqPoint{host_ns, (dw / dr) * kRefclkHz * 1e-9});
    }
    return out;
}

}  // namespace tt::tt_metal::streaming_profiler
