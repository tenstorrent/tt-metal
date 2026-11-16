// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "build_cache_telemetry.hpp"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include <fmt/ranges.h>
#include <tt-logger/tt-logger.hpp>

#include "env_lib.hpp"
#include "common/executor.hpp"

namespace tt::tt_metal {

// --- TelemetryToken ---

TelemetryToken::TelemetryToken(std::string name, std::string unit) : name_(std::move(name)), unit_(std::move(unit)) {}

void TelemetryToken::set_recording_enabled(bool enabled) {
    recording_enabled_.store(enabled, std::memory_order_release);
}

void TelemetryToken::record(double value) {
    if (!recording_enabled_.load(std::memory_order_relaxed)) {
        return;
    }
    std::lock_guard lk(data_mutex_);
    ++data_.count;
    data_.total += value;
    data_.min_val = std::min(data_.min_val, value);
    data_.max_val = std::max(data_.max_val, value);
}

TelemetryTokenData TelemetryToken::snapshot() const {
    std::lock_guard lk(data_mutex_);
    return data_;
}

void TelemetryToken::set_single_sample(double value) {
    if (!recording_enabled_.load(std::memory_order_relaxed)) {
        return;
    }
    std::lock_guard lk(data_mutex_);
    data_ = {.count = 1, .total = value, .min_val = value, .max_val = value};
}

// --- BuildCacheTelemetry ---

// One busy clock (see BuildCacheTelemetry::note_busy_start).
struct BusyClock {
    TelemetryToken* token{nullptr};
    uint32_t in_flight{0};
    // Most intervals open at once.
    uint32_t peak{0};
    // Intervals opened: for the program clocks, programs built or compiled.
    uint32_t intervals{0};
    // Interval durations added up, so overlapping intervals each count.
    std::chrono::steady_clock::duration summed_time{};
    // Earliest start and latest end among the current busy stretch's closed intervals.
    std::chrono::steady_clock::time_point stretch_start;
    std::chrono::steady_clock::time_point stretch_end;
};

// The program with the least kernel config buffer headroom on one core type.
struct TightestProgram {
    uint32_t headroom{0};
    uint32_t limit{0};
    SummaryRows breakdown;
};

struct BuildCacheTelemetryImpl {
    std::atomic<uint64_t> srcs_and_compiled{0};
    std::atomic<uint32_t> cached_hit_count{0};
    std::atomic<uint32_t> merged_artifacts{0};
    std::atomic<uint32_t> merged_genfiles{0};
    std::atomic<uint32_t> jit_once_dedup_count{0};

    std::mutex token_registry_mutex;
    std::vector<TelemetryToken*> registered_tokens;

    // Endpoints of the JIT build window, as steady_clock nanoseconds. Sentinels are ordered so
    // that last < first means "no build activity recorded", which dump_metrics() treats as
    // nothing to report.
    std::atomic<int64_t> build_window_first_ns{std::numeric_limits<int64_t>::max()};
    std::atomic<int64_t> build_window_last_ns{std::numeric_limits<int64_t>::min()};

    // Guards busy_clocks and tightest_programs.
    std::mutex summary_mutex;
    // By clock name.
    std::map<std::string, BusyClock, std::less<>> busy_clocks;
    // By core type.
    std::map<std::string, TightestProgram, std::less<>> tightest_programs;
};

BuildCacheTelemetry::BuildCacheTelemetry() {
    enable();
    build_window_token_ = &get_or_register_metric("jit_build_window");
    build_peak_in_flight_token_ = &get_or_register_metric("jit_build_peak_in_flight", "builds");
}

BuildCacheTelemetry::~BuildCacheTelemetry() {
    // Dump metrics here rather than via std::atexit. The atexit handler is registered inside
    // the singleton constructor, so glibc queues it *before* the constructor completes.
    // Because atexit/static-dtor ordering is LIFO over a shared queue, the singleton destructor
    // would fire *before* the atexit handler, leaving impl_ already reset when dump_metrics()
    // ran — causing a silent no-op.
    //
    // The destructor body is the correct place: impl_ is still valid here (member destructors
    // run after the body), so dump_metrics() can read all counters and registered tokens.
    //
    // LoggerRegistry::instance() is a leaky singleton (allocated with `new`, never freed),
    // so its loggers and sinks are alive for the entire process lifetime — including here.
    dump_metrics();
}

BuildCacheTelemetry& BuildCacheTelemetry::inst() {
    static BuildCacheTelemetry instance;
    return instance;
}

TelemetryToken& per_target_telemetry_token(
    std::string_view metric_name, std::string_view target_name, std::string_view unit) {
    std::string key(metric_name);
    key += '.';
    key += target_name;
    return BuildCacheTelemetry::inst().get_or_register_metric(key, unit);
}

void BuildCacheTelemetry::enable() {
    if (impl_) {
        return;
    }
    impl_ = std::make_unique<BuildCacheTelemetryImpl>();
    std::lock_guard owned_lk(owned_tokens_mutex_);
    for (auto& token : owned_tokens_) {
        token->set_recording_enabled(true);
    }
    std::lock_guard reg_lk(impl_->token_registry_mutex);
    for (auto& token : owned_tokens_) {
        impl_->registered_tokens.push_back(token.get());
    }
}

void BuildCacheTelemetry::disable() {
    {
        std::lock_guard owned_lk(owned_tokens_mutex_);
        for (auto& token : owned_tokens_) {
            token->set_recording_enabled(false);
        }
    }
    impl_.reset();
}

void BuildCacheTelemetry::record_compile(uint32_t num_srcs, uint32_t num_compiled) {
    if (!impl_) {
        return;
    }
    impl_->srcs_and_compiled.fetch_add(
        (static_cast<uint64_t>(num_srcs) << 32) | static_cast<uint64_t>(num_compiled), std::memory_order_release);
}

void BuildCacheTelemetry::record_cache_hit() {
    if (!impl_) {
        return;
    }
    impl_->cached_hit_count.fetch_add(1, std::memory_order_release);
}

void BuildCacheTelemetry::record_merge(uint32_t count) {
    if (!impl_) {
        return;
    }
    impl_->merged_artifacts.fetch_add(count, std::memory_order_release);
}

void BuildCacheTelemetry::record_genfile_merge(uint32_t count) {
    if (!impl_) {
        return;
    }
    impl_->merged_genfiles.fetch_add(count, std::memory_order_release);
}

void BuildCacheTelemetry::record_jit_once_dedup() {
    if (!impl_) {
        return;
    }
    impl_->jit_once_dedup_count.fetch_add(1, std::memory_order_release);
}

uint32_t BuildCacheTelemetry::get_srcs_count() const {
    if (!impl_) {
        return 0;
    }
    return impl_->srcs_and_compiled.load(std::memory_order_acquire) >> 32;
}

uint32_t BuildCacheTelemetry::get_compile_count() const {
    if (!impl_) {
        return 0;
    }
    return impl_->srcs_and_compiled.load(std::memory_order_acquire) & 0xFFFF'FFFFu;
}

uint32_t BuildCacheTelemetry::get_cache_hit_count() const {
    if (!impl_) {
        return 0;
    }
    return impl_->cached_hit_count.load(std::memory_order_acquire);
}

uint32_t BuildCacheTelemetry::get_merge_count() const {
    if (!impl_) {
        return 0;
    }
    return impl_->merged_artifacts.load(std::memory_order_acquire);
}

uint32_t BuildCacheTelemetry::get_genfile_merge_count() const {
    if (!impl_) {
        return 0;
    }
    return impl_->merged_genfiles.load(std::memory_order_acquire);
}

uint32_t BuildCacheTelemetry::get_jit_once_dedup_count() const {
    if (!impl_) {
        return 0;
    }
    return impl_->jit_once_dedup_count.load(std::memory_order_acquire);
}

void BuildCacheTelemetry::note_build_window(
    std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    if (!impl_) {
        return;
    }
    const auto to_ns = [](std::chrono::steady_clock::time_point tp) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
    };
    const int64_t start_ns = to_ns(start);
    const int64_t end_ns = to_ns(end);

    // Concurrent builds each widen their own end of the window, so both endpoints need a
    // read-modify-write rather than a plain store.
    int64_t first = impl_->build_window_first_ns.load(std::memory_order_relaxed);
    while (start_ns < first &&
           !impl_->build_window_first_ns.compare_exchange_weak(first, start_ns, std::memory_order_relaxed)) {
    }
    int64_t last = impl_->build_window_last_ns.load(std::memory_order_relaxed);
    while (end_ns > last &&
           !impl_->build_window_last_ns.compare_exchange_weak(last, end_ns, std::memory_order_relaxed)) {
    }
}

void BuildCacheTelemetry::note_busy_start(std::string_view clock) {
    if (!impl_) {
        return;
    }
    // Registered before taking summary_mutex: dump_metrics() takes summary_mutex while holding the
    // registry lock.
    auto& token = get_or_register_metric(std::string(clock));
    std::lock_guard lk(impl_->summary_mutex);
    auto& c = impl_->busy_clocks.try_emplace(std::string(clock), BusyClock{.token = &token}).first->second;
    if (c.in_flight++ == 0) {
        c.stretch_start = std::chrono::steady_clock::time_point::max();
        c.stretch_end = std::chrono::steady_clock::time_point::min();
    }
    c.peak = std::max(c.peak, c.in_flight);
    ++c.intervals;
}

void BuildCacheTelemetry::note_busy_end(
    std::string_view clock, std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    if (!impl_) {
        return;
    }
    std::lock_guard lk(impl_->summary_mutex);
    auto it = impl_->busy_clocks.find(clock);
    // An interval that started before enable() rebuilt impl_ has no matching start here.
    if (it == impl_->busy_clocks.end() || it->second.in_flight == 0) {
        return;
    }
    auto& c = it->second;
    c.summed_time += end - start;
    // Callers take their timestamps before summary_mutex, so intervals can close out of time order: the
    // stretch spans its intervals' earliest start and latest end, whichever interval closes it.
    c.stretch_start = std::min(c.stretch_start, start);
    c.stretch_end = std::max(c.stretch_end, end);
    if (--c.in_flight == 0) {
        c.token->record(std::chrono::duration<double, std::milli>(c.stretch_end - c.stretch_start).count());
    }
}

void BuildCacheTelemetry::note_program_headroom(
    std::string_view core_type, uint32_t headroom, uint32_t limit, const std::function<SummaryRows()>& breakdown) {
    if (!impl_) {
        return;
    }
    std::lock_guard lk(impl_->summary_mutex);
    auto it = impl_->tightest_programs.find(core_type);
    if (it == impl_->tightest_programs.end() || headroom < it->second.headroom) {
        // Built before the map changes, so a throwing breakdown() leaves the last minimum in place.
        impl_->tightest_programs.insert_or_assign(
            std::string(core_type), TightestProgram{headroom, limit, breakdown()});
    }
}

ScopedBuildWindow::ScopedBuildWindow() : start_(std::chrono::steady_clock::now()) {}

ScopedBuildWindow::~ScopedBuildWindow() {
    BuildCacheTelemetry::inst().note_build_window(start_, std::chrono::steady_clock::now());
}

ScopedBusyClock::ScopedBusyClock(std::string_view clock) : clock_(clock), start_(std::chrono::steady_clock::now()) {
    BuildCacheTelemetry::inst().note_busy_start(clock_);
}

ScopedBusyClock::~ScopedBusyClock() {
    BuildCacheTelemetry::inst().note_busy_end(clock_, start_, std::chrono::steady_clock::now());
}

void BuildCacheTelemetry::log_compile_summary() const {
    if (!impl_) {
        return;
    }
    const uint64_t packed = impl_->srcs_and_compiled.load(std::memory_order_acquire);
    const uint32_t total = packed >> 32;
    const uint32_t compiled = packed & 0xFFFF'FFFFu;

    if (total == 0) {
        log_info(tt::LogBuildKernels, "JIT cache stats: no build info recorded");
        return;
    }

    const uint32_t hits = total - compiled;
    const uint32_t cached = impl_->cached_hit_count.load(std::memory_order_acquire);
    const uint32_t artifacts = impl_->merged_artifacts.load(std::memory_order_acquire);
    const uint32_t genfiles = impl_->merged_genfiles.load(std::memory_order_acquire);
    const uint32_t dedup = impl_->jit_once_dedup_count.load(std::memory_order_acquire);
    log_info(
        tt::LogBuildKernels,
        "JIT cache stats: {}/{} hits ({:.1f}%) [{} cached, {} build-once dedup, "
        "{} merged artifacts, {} merged genfiles]",
        hits,
        total,
        100.0 * static_cast<double>(hits) / static_cast<double>(total),
        cached,
        dedup,
        artifacts,
        genfiles);
}

TelemetryToken& BuildCacheTelemetry::get_or_register_metric(
    const std::string& name, std::optional<std::string_view> unit) {
    std::lock_guard lk(owned_tokens_mutex_);
    auto [it, inserted] = tokens_by_name_.try_emplace(name, nullptr);
    if (!inserted) {
        // A mismatch means two call sites are feeding one stream values of different kinds, which
        // mislabels the dump. That is a bug worth shouting about, but telemetry is a diagnostic
        // subsystem: aborting the process over a metric label would take down every JIT build for
        // a mistake that costs nothing but a wrong unit in one log line. Keep the unit the stream
        // was registered with and warn once per name, so a build loop cannot flood the log.
        if (unit.has_value() && it->second->unit() != *unit) {
            if (unit_conflict_warned_.insert(name).second) {
                log_warning(
                    tt::LogBuildKernels,
                    "Telemetry metric '{}' is registered with unit '{}' but was requested with unit '{}'; keeping "
                    "'{}'. Values from these call sites are being aggregated into one stream.",
                    name,
                    it->second->unit(),
                    *unit,
                    it->second->unit());
            }
        }
        return *it->second;
    }
    owned_tokens_.push_back(
        unit.has_value() ? std::make_unique<TelemetryToken>(name, std::string(*unit))
                         : std::make_unique<TelemetryToken>(name));
    auto* token = owned_tokens_.back().get();
    it->second = token;
    token->set_recording_enabled(impl_ != nullptr);
    if (impl_) {
        std::lock_guard reg_lk(impl_->token_registry_mutex);
        impl_->registered_tokens.push_back(token);
    }
    return *token;
}

namespace {

// Headline rows of the logged summary. The caller holds impl.token_registry_mutex and
// impl.summary_mutex.
SummaryRows summarize(const BuildCacheTelemetryImpl& impl) {
    // Total of the metric named `name`, in seconds: the busy and compile metrics record milliseconds.
    const auto seconds_of = [&](std::string_view name) {
        for (const auto* token : impl.registered_tokens) {
            if (token->name() == name) {
                return token->snapshot().total / 1e3;
            }
        }
        return 0.0;
    };
    SummaryRows rows;
    for (const auto& [clock, label] :
         {std::pair{kProgramBuildBusy, "program build from descriptor"},
          std::pair{kProgramCompileBusy, "program compile (host waiting)"}}) {
        if (auto it = impl.busy_clocks.find(clock); it != impl.busy_clocks.end() && it->second.intervals > 0) {
            rows.emplace_back(label, fmt::format("{:.1f} s, {} programs", seconds_of(clock), it->second.intervals));
        }
    }
    // The JIT build clock has one interval per JitBuildState::build call, closed however the call ends,
    // including early returns and exceptions. busy_s is the wall time with at least one build running.
    // summed_build_s adds up every build's duration, so overlapping builds each count.
    const auto jit = impl.busy_clocks.find(kJitBuildBusy);
    const double busy_s = seconds_of(kJitBuildBusy);
    if (jit == impl.busy_clocks.end() || busy_s == 0) {
        return rows;
    }
    const BusyClock& builds = jit->second;
    const double summed_build_s = std::chrono::duration<double>(builds.summed_time).count();

    // build.cpp records one {kernel,fw}_link_time.<target> sample per linked binary; cache hits link nothing.
    std::map<std::string, uint32_t, std::less<>> kernels_by_target;
    uint32_t firmware = 0;
    double link_s = 0;
    for (const auto* token : impl.registered_tokens) {
        const std::string& name = token->name();
        const bool is_kernel = name.starts_with("kernel_link_time.");
        if (!is_kernel && !name.starts_with("fw_link_time.")) {
            continue;
        }
        const TelemetryTokenData links = token->snapshot();
        link_s += links.total / 1e3;
        if (is_kernel) {
            kernels_by_target[name.substr(name.find('.') + 1)] += links.count;
        } else {
            firmware += links.count;
        }
    }
    std::vector<std::string> by_target;
    uint32_t binaries = firmware;
    for (const auto& [target, count] : kernels_by_target) {
        binaries += count;
        by_target.push_back(fmt::format("{} {}", target, count));
    }
    if (firmware > 0) {
        by_target.push_back(fmt::format("firmware {}", firmware));
    }

    rows.emplace_back(
        "JIT build wall time (local builds running)",
        fmt::format(
            "{:.1f} s, {} binaries, {} repeat requests",
            busy_s,
            binaries,
            impl.jit_once_dedup_count.load(std::memory_order_acquire)));
    if (!by_target.empty()) {
        rows.emplace_back("  binaries per RISC", fmt::format("{}", fmt::join(by_target, ", ")));
    }
    // Average number of builds running at once while any build is running: four 10 s builds give 4 when they
    // all overlap and 1 when they run one after another. The peak can pass the CPU count: when every pool
    // thread is busy, a build runs on the thread that asked for it.
    rows.emplace_back(
        "builds running at once",
        fmt::format(
            "{:.1f} average, {} peak, on {} CPU threads, {:.1f} s per build",
            summed_build_s / busy_s,
            builds.peak,
            detail::EXECUTOR_NTHREADS,
            summed_build_s / builds.intervals));
    // Builds compile sources, shared by the RISCs they link for, then link one binary per RISC. Link time is
    // mostly LTO optimization and code generation.
    rows.emplace_back(
        "build steps (summed over builds)",
        fmt::format("compile {:.0f} s, link (LTO codegen) {:.0f} s", seconds_of("JitBuildState::compile"), link_s));
    return rows;
}

}  // namespace

void BuildCacheTelemetry::dump_metrics() const {
    if (!impl_) {
        return;
    }

    log_compile_summary();

    // Serialize dumps before reading the endpoints so an older snapshot cannot overwrite a
    // newer one. Replace the single window sample: appending it would count the same build
    // activity again on each explicit dump and once more at process exit.
    std::lock_guard lk(impl_->token_registry_mutex);
    const int64_t first_ns = impl_->build_window_first_ns.load(std::memory_order_relaxed);
    const int64_t last_ns = impl_->build_window_last_ns.load(std::memory_order_relaxed);
    if (build_window_token_ != nullptr && last_ns >= first_ns) {
        build_window_token_->set_single_sample(static_cast<double>(last_ns - first_ns) / 1e6);
    }
    log_info(tt::LogBuildKernels, "JIT telemetry: {} registered TelemetryTokens", impl_->registered_tokens.size());

    // One info line per token, and the per-target metrics register dozens of them, so every
    // tt-metal process would print a wall of output at exit. Off by default; opt in with
    // TT_METAL_LOG_JIT_TELEMETRY=1, mirroring the TT_METAL_LOG_KERNEL_COMPILE gate in build.cpp.
    static const bool log_tokens = tt::parse_env<bool>("TT_METAL_LOG_JIT_TELEMETRY", false);
    SummaryRows summary;
    {
        std::lock_guard summary_lk(impl_->summary_mutex);
        if (auto it = impl_->busy_clocks.find(kJitBuildBusy); it != impl_->busy_clocks.end()) {
            build_peak_in_flight_token_->set_single_sample(it->second.peak);
        }
        if (log_tokens) {
            summary = summarize(*impl_);
            for (const auto& [core_type, tightest] : impl_->tightest_programs) {
                summary.emplace_back(
                    fmt::format("{} config buffer, least headroom", core_type),
                    fmt::format(
                        "{} B left ({} of {} B used)",
                        tightest.headroom,
                        tightest.limit - tightest.headroom,
                        tightest.limit));
                for (const auto& [label, value] : tightest.breakdown) {
                    summary.emplace_back("  " + label, value);
                }
            }
        }
    }
    if (!log_tokens) {
        return;
    }
    for (const auto& [label, value] : summary) {
        log_info(tt::LogBuildKernels, "JIT telemetry summary | {:<42} {}", label, value);
    }

    for (const auto* token : impl_->registered_tokens) {
        const TelemetryTokenData snap = token->snapshot();
        if (snap.count == 0) {
            continue;
        }
        const double mean_val = snap.total / static_cast<double>(snap.count);
        // Byte counts are whole numbers; three decimals would just be trailing zeros.
        const int precision = token->unit() == "B" ? 0 : 3;
        log_info(
            tt::LogBuildKernels,
            "JIT telemetry [{}] ({}): count={}, total={:.{}f}, min={:.{}f}, max={:.{}f}, mean={:.{}f}",
            token->name(),
            token->unit(),
            snap.count,
            snap.total,
            precision,
            snap.min_val,
            precision,
            snap.max_val,
            precision,
            mean_val,
            precision);
    }
}

}  // namespace tt::tt_metal
