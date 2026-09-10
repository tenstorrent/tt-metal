// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/streaming_profiler_tracy.hpp"

#include <cstring>
#include <limits>

#include <umd/device/driver_atomics.hpp>

#include <fmt/format.h>
#if defined(TRACY_ENABLE)
#include <common/TracyTTDeviceData.hpp>
#include <tracy/Tracy.hpp>
#include <client/TracyProfiler.hpp>
#endif

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

}  // namespace

TracySink::TracySink(Service& service) : service_(service), srcloc_table_(kSrclocTableInitial) {
#if defined(TRACY_ENABLE)
    anchor_tracy_ = tracy::Profiler::GetTime();
#endif
    base_ = probe();
    handle_ = service_.add_consumer(
        "tracy",
        [this](const Batch& b, uint64_t capture) { on_batch(b, capture); },
        ConsumerHooks{
            .on_attach =
                [this](const CaptureContext& ctx) {
                    clocks_.clear();
                    for (const auto& d : ctx.devices) {
                        clocks_.push_back(d.clock);
                    }
                },
            .clock_sink = [this](const ClockSample& cs) { plot_clock(cs.dev, cs.kind, cs.ts); }});
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
    for (const api::Zone& z : batch.zones()) {
        push_zone(
            z.core(),
            z.site().name,
            ns_since_epoch(z.start_time()),
            ns_since_epoch(z.end_time()),
            z.site().name == api::kStallZoneName ? kStallColor : 0);
    }
    for (const api::TimestampedData& d : batch.timestamped_data()) {
        push_marker(d.core(), d.site().name, ns_since_epoch(d.time()), d.runtime_id(), d.payload());
    }
    for (const api::Event& e : batch.events()) {
        push_marker(e.core(), e.site().name, ns_since_epoch(e.time()), e.runtime_id(), {});
    }
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
            best.tracy_ns =
                static_cast<int64_t>(static_cast<double>(t0 + (t1 - t0) / 2 - anchor_tracy_) * TracyGetTimerMul());
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
        return steady_ns < 0 ? 0 : steady_ns;
    }
    const Segment* seg = &segments_.back();
    while (seg != segments_.data() && steady_ns < seg->steady_ns) {
        seg--;
    }
    const int64_t ns =
        seg->tracy_ns + static_cast<int64_t>(static_cast<double>(steady_ns - seg->steady_ns) * seg->slope);
    return ns < 0 ? 0 : ns;  // a record cannot predate the capture; clamp rather than wrap
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
        TracyTTContextPopulateCalibratedLockfree(ce.ctx, anchor_tracy_, 0.0, 1.0);
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
    for (size_t i = srcloc_hash(name.data(), key) & mask;; i = (i + 1) & mask) {
        const SrclocEntry& e = srcloc_table_[i];
        if (e.name == name.data() && e.key == key) {
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
    const SrclocEntry entry{name.data(), srcloc_key(name, color, risc), nullptr};
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
    const Lane ln = lane(core);
    TracyTTPushZone(
        ln.ctx,
        static_cast<const tracy::SourceLocationData*>(srcloc(name, color, ln.risc)),
        ln.thread,
        static_cast<uint64_t>(to_timeline(start_ns)),
        static_cast<uint64_t>(to_timeline(end_ns)));
#endif
}

void TracySink::push_marker(
    [[maybe_unused]] const Core& core,
    [[maybe_unused]] std::string_view name,
    [[maybe_unused]] int64_t timestamp_ns,
    [[maybe_unused]] uint32_t runtime_id,
    [[maybe_unused]] std::span<const uint64_t> values) {
#if defined(TRACY_ENABLE)
    TracyTTCtx ctx = lane(core).ctx;
    tracy::TTDeviceMarker marker;
    marker.chip_id = core.chip_id;
    marker.core_x = core.logical.x;
    marker.core_y = core.logical.y;
    marker.risc = kRisc[static_cast<uint32_t>(core.risc)];
    marker.timestamp = static_cast<uint64_t>(to_timeline(timestamp_ns));
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

const char* TracySink::plot_name(uint32_t chip, bool linked) {
    auto& m = linked ? plot_linked_ : plot_local_;
    auto it = m.find(chip);
    if (it == m.end()) {
        it = m.emplace(chip, fmt::format("d2d {} ns chip{}", linked ? "linked" : "local", chip)).first;
    }
    return it->second.c_str();
}

// A PP_CLOCK sample's correction, plotted at the sample's device time on the chip's timeline. The device tick is
// mapped to the same steady-clock ns a zone at that tick would use, then through to_timeline onto Tracy's timeline
// (PlotDataAt carries the time on the wire -- plain PlotData/TracyPlot would stamp decode-time and pile every point
// at one instant). LOCAL samples feed the local-only plot, LINK samples the linked plot.
void TracySink::plot_clock(uint32_t dev, uint32_t kind, uint64_t device_ticks) {
#if defined(TRACY_ENABLE)
    if (dev >= clocks_.size()) {
        return;
    }
    const DeviceClock& clk = clocks_[dev];
    if (clk.frequency_ghz <= 0.0) {
        return;
    }
    const uint32_t chip = clk.chip_id;
    const double hz = clk.frequency_ghz * 1e9;
    const int64_t base_ns =
        clk.anchor_host_ns +
        static_cast<int64_t>(
            static_cast<double>(static_cast<int64_t>(device_ticks) - static_cast<int64_t>(clk.anchor_ticks)) * 1e9 /
            hz);
    const int64_t tsc = to_timeline(base_ns);
    if (kind == PP_CLOCK_LOCAL_REFCLK) {
        tracy::Profiler::PlotDataAt(plot_name(chip, false), SyncCorrections::lookup_local_ns(chip, device_ticks), tsc);
    } else if (kind == PP_CLOCK_LINK_REFCLK) {
        tracy::Profiler::PlotDataAt(plot_name(chip, true), SyncCorrections::lookup_ns(chip, device_ticks), tsc);
    }
#else
    (void)dev;
    (void)kind;
    (void)device_ticks;
#endif
}

}  // namespace tt::tt_metal::streaming_profiler
