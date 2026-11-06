// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Terminal UI for the telemetry firmware. Primes the chips with --prime; otherwise
// loads the firmware on one tile per chip, polls each chip's table on a worker
// thread and shows it live. q, Esc or Ctrl-C quits and stops the firmware.

#include "telemetry_host_lib.hpp"

#include "x280/device/x280_runtime.hpp"

#include <tt-metalium/mesh_device.hpp>

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace x280;
using namespace ftxui;

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint64_t kRunValue = 1;
constexpr int kDefaultPollMs = 1000;
constexpr size_t kHistoryLength = 600;  // Samples kept for the graphs, 10 minutes at the default poll.
constexpr auto kHeartbeatStaleAfter = std::chrono::seconds(10);
constexpr const char* kDefaultLogFile = "x280_telemetry_tui.log";
constexpr const char* kArcPrefix = "TT_ARC_";

// Set by SIGINT while the firmware is loading, before the UI owns Ctrl-C.
std::atomic<bool> g_interrupted = false;

void on_sigint(int) { g_interrupted = true; }

// ---------------------------------------------------------------------------
// Decoding raw ARC values for display.
// ---------------------------------------------------------------------------

std::string format_double(const char* fmt, double value) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), fmt, value);
    return buf;
}

std::string hex32(uint32_t value) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "0x%08x", value);
    return buf;
}

// All four bytes, most significant first. For versions UMD doesn't decode.
std::string dotted_bytes(uint32_t value) {
    char buf[32];
    std::snprintf(
        buf, sizeof(buf), "%u.%u.%u.%u", (value >> 24) & 0xFF, (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF);
    return buf;
}

// The firmware names ARC metrics TT_ARC_<tag>; the decode tables use the bare tag.
std::string arc_tag(const std::string& name) {
    const std::string prefix = kArcPrefix;
    return name.rfind(prefix, 0) == 0 ? name.substr(prefix.size()) : name;
}

struct Decoded {
    std::string text;              // Value with its unit, for display.
    std::optional<double> number;  // Set when the value can be plotted or gauged.
};

// Follows UMD's Blackhole decode (device/src/firmware/firmware_info_provider_implementation.cpp).
// Tags UMD doesn't decode are shown raw.
Decoded decode_arc_value(const std::string& tag, uint32_t raw) {
    static const std::map<std::string, const char*> kUnits = {
        {"MAX_GDDR_TEMP", "°C"},  {"THM_LIMIT_SHUTDOWN", "°C"}, {"THM_LIMIT_THROTTLE", "°C"}, {"VCORE", "mV"},
        {"TDP", "W"},             {"TDP_LIMIT_MAX", "W"},       {"BOARD_POWER_LIMIT", "W"},  {"TDC", "A"},
        {"TDC_LIMIT_MAX", "A"},   {"AICLK", "MHz"},             {"AXICLK", "MHz"},           {"ARCCLK", "MHz"},
        {"L2CPUCLK0", "MHz"},     {"L2CPUCLK1", "MHz"},         {"L2CPUCLK2", "MHz"},        {"L2CPUCLK3", "MHz"},
        {"AICLK_LIMIT_MAX", "MHz"}, {"AICLK_ARB_MIN", "MHz"},   {"AICLK_ARB_MAX", "MHz"},    {"FAN_SPEED", "%"},
        {"FAN_RPM", "rpm"},
    };
    // Versions UMD decodes as major.minor.patch from the top three bytes (firmware_utils.cpp, semver.hpp).
    static const std::set<std::string> kTopByteVersions = {"FLASH_BUNDLE_VERSION", "CM_FW_VERSION", "DM_APP_FW_VERSION"};
    // Versions UMD doesn't decode on Blackhole; shown as their four bytes.
    static const std::set<std::string> kRawVersions = {"ETH_FW_VERSION", "DM_BL_FW_VERSION", "L2CPU_FW_VERSION"};
    static const std::set<std::string> kHex = {
        "BOARD_ID_HIGH",
        "BOARD_ID_LOW",
        "ASIC_ID",
        "ASIC_ID_HIGH",
        "ASIC_ID_LOW",
        "HARVESTING_STATE",
        "ETH_LIVE_STATUS",
        "GDDR_STATUS",
        "ENABLED_TENSIX_COL",
        "ENABLED_ETH",
        "ENABLED_GDDR",
        "ENABLED_L2CPU",
        "NOC_TRANSLATION",
        "GDDR_UNCORR_ERRS",
        "VDD_LIMITS",
    };

    // Signed 16.16 fixed point.
    if (tag == "ASIC_TEMPERATURE" || tag == "BOARD_TEMPERATURE") {
        const double celsius = static_cast<int32_t>(raw) / 65536.0;
        return {format_double("%.1f °C", celsius), celsius};
    }

    // Two GDDR modules per word, one byte each for the bottom and top sensor.
    if (tag.rfind("GDDR_", 0) == 0 && tag.size() > 5 && tag.substr(tag.size() - 5) == "_TEMP") {
        const uint32_t b0 = raw & 0xFF, t0 = (raw >> 8) & 0xFF, b1 = (raw >> 16) & 0xFF, t1 = (raw >> 24) & 0xFF;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%u/%u  %u/%u °C", b0, t0, b1, t1);
        return {buf, static_cast<double>(std::max({b0, t0, b1, t1}))};
    }

    if (tag == "FAN_RPM" && raw == 0xFFFFFFFF) {
        return {"n/a", std::nullopt};
    }

    char version[32];
    if (kTopByteVersions.contains(tag)) {
        std::snprintf(version, sizeof(version), "%u.%u.%u", raw >> 24, (raw >> 16) & 0xFF, (raw >> 8) & 0xFF);
        // The bundle's low byte is a pre-release number.
        if (tag == "FLASH_BUNDLE_VERSION" && (raw & 0xFF) != 0) {
            std::snprintf(version + std::strlen(version), sizeof(version) - std::strlen(version), "-rc%u", raw & 0xFF);
        }
        return {version, std::nullopt};
    }
    if (tag == "TT_FLASH_VERSION") {
        std::snprintf(version, sizeof(version), "%u.%u.%u", (raw >> 16) & 0xFF, (raw >> 8) & 0xFF, raw & 0xFF);
        return {version, std::nullopt};
    }
    if (tag == "GDDR_FW_VERSION") {
        std::snprintf(version, sizeof(version), "%u.%u", raw >> 16, raw & 0xFFFF);
        return {version, std::nullopt};
    }
    if (kRawVersions.contains(tag)) {
        return {dotted_bytes(raw), std::nullopt};
    }

    if (kHex.contains(tag)) {
        return {hex32(raw), std::nullopt};
    }

    auto unit = kUnits.find(tag);
    if (unit != kUnits.end()) {
        return {std::to_string(raw) + " " + unit->second, static_cast<double>(raw)};
    }

    return {std::to_string(raw), static_cast<double>(raw)};
}

Decoded decode_metric(const TelemetryMetric& metric) {
    if (metric.is_string) {
        return {metric.str_value, std::nullopt};
    }
    return decode_arc_value(arc_tag(metric.name), metric.num_value);
}

// ---------------------------------------------------------------------------
// What the UI shows, filled by the poller thread.
// ---------------------------------------------------------------------------

struct History {
    std::deque<double> samples;

    void add(std::optional<double> value) {
        if (!value) {
            return;
        }
        samples.push_back(*value);
        if (samples.size() > kHistoryLength) {
            samples.pop_front();
        }
    }
};

struct ChipModel {
    TileId tile_id;

    // The latest table that could be read, and why the most recent read failed (empty if it didn't).
    std::optional<TelemetrySnapshot> snapshot;
    std::string read_error;
    Clock::time_point last_good_read{};
    uint64_t good_reads = 0;
    uint64_t failed_reads = 0;

    // Metric names whose value changed in the latest read.
    std::set<std::string> changed;

    // Hart 0 runs the telemetry loop.
    bool hart_observed = false;
    uint64_t hart_phase = 0;
    uint32_t hart_stage = 0;

    // ARC bumps TIMER_HEARTBEAT and the firmware copies it every pass, so if it
    // stops changing, ARC or the firmware is stuck.
    std::optional<uint32_t> heartbeat;
    Clock::time_point heartbeat_changed{};

    History asic_temp;
    History power;
    History aiclk;
};

struct Model {
    std::mutex mutex;
    std::vector<ChipModel> chips;
};

const TelemetryMetric* find_metric(const TelemetrySnapshot& snapshot, const std::string& name) {
    for (const auto& metric : snapshot.metrics) {
        if (metric.name == name || arc_tag(metric.name) == name) {
            return &metric;
        }
    }
    return nullptr;
}

std::optional<double> metric_number(const TelemetrySnapshot& snapshot, const std::string& tag) {
    const auto* metric = find_metric(snapshot, tag);
    if (metric == nullptr || metric->is_string) {
        return std::nullopt;
    }
    return decode_metric(*metric).number;
}

void apply_sample(
    ChipModel& chip,
    const std::optional<HartState>& hart0,
    std::optional<TelemetrySnapshot> snapshot,
    const std::string& error) {
    const auto now = Clock::now();

    chip.hart_observed = hart0.has_value();
    if (hart0) {
        chip.hart_phase = hart0->state.phase;
        chip.hart_stage = hart0->state.stage;
    }

    if (!snapshot) {
        chip.read_error = error;
        ++chip.failed_reads;
        return;
    }

    chip.read_error.clear();
    ++chip.good_reads;
    chip.last_good_read = now;

    chip.changed.clear();
    if (chip.snapshot) {
        for (const auto& metric : snapshot->metrics) {
            const auto* previous = find_metric(*chip.snapshot, metric.name);
            if (previous != nullptr &&
                (previous->num_value != metric.num_value || previous->str_value != metric.str_value)) {
                chip.changed.insert(metric.name);
            }
        }
    }

    if (const auto* beat = find_metric(*snapshot, "TIMER_HEARTBEAT"); beat != nullptr && !beat->is_string) {
        if (!chip.heartbeat || *chip.heartbeat != beat->num_value) {
            chip.heartbeat = beat->num_value;
            chip.heartbeat_changed = now;
        }
    }

    if (snapshot->table_version != 0) {
        chip.asic_temp.add(metric_number(*snapshot, "ASIC_TEMPERATURE"));
        chip.power.add(metric_number(*snapshot, "TDP"));
        chip.aiclk.add(metric_number(*snapshot, "AICLK"));
    }

    chip.snapshot = std::move(snapshot);
}

// Reads every chip, then asks the screen to redraw. Runs until stop() is called.
// It is the only user of the runtime while the UI is up.
class Poller {
public:
    Poller(X280Runtime& runtime, Model& model, ScreenInteractive& screen, std::chrono::milliseconds interval) :
        runtime_(runtime), model_(model), screen_(screen), interval_(interval), thread_([this] { run(); }) {}

    ~Poller() { stop(); }

    void stop() {
        {
            std::lock_guard lock(wake_mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    void run() {
        std::vector<TileId> tile_ids;
        {
            std::lock_guard lock(model_.mutex);
            for (const auto& chip : model_.chips) {
                tile_ids.push_back(chip.tile_id);
            }
        }

        for (;;) {
            for (size_t i = 0; i < tile_ids.size(); ++i) {
                // Talk to the hardware without holding the model lock, so the UI never waits on it.
                auto& tile = runtime_.tile(tile_ids[i]);

                std::optional<HartState> hart0;
                try {
                    const auto state = tile.inspect();
                    if (state.observation_valid && state.harts[0].observation_valid) {
                        hart0 = state.harts[0];
                    }
                } catch (const std::exception&) {
                    // Shown as "no tile"; the table read below reports the reason.
                }

                std::string error;
                auto snapshot = read_telemetry_table(tile, error);

                std::lock_guard lock(model_.mutex);
                apply_sample(model_.chips[i], hart0, std::move(snapshot), error);
            }

            screen_.PostEvent(Event::Custom);

            std::unique_lock lock(wake_mutex_);
            if (wake_.wait_for(lock, interval_, [this] { return stopping_; })) {
                return;
            }
        }
    }

    X280Runtime& runtime_;
    Model& model_;
    ScreenInteractive& screen_;
    std::chrono::milliseconds interval_;

    std::mutex wake_mutex_;
    std::condition_variable wake_;
    bool stopping_ = false;

    // Last, so everything above is constructed before the thread starts.
    std::thread thread_;
};

// ---------------------------------------------------------------------------
// Rendering. Everything here runs on the UI thread with the model locked.
// ---------------------------------------------------------------------------

enum class ChipStatus { Live, Waiting, Stale, Stopped, NoTile, ReadError };

ChipStatus chip_status(const ChipModel& chip) {
    if (!chip.hart_observed) {
        return ChipStatus::NoTile;
    }
    if (chip.hart_phase != X280_HART_PHASE_RUNNING_ACTIVE_FW) {
        return ChipStatus::Stopped;
    }
    if (!chip.read_error.empty()) {
        return ChipStatus::ReadError;
    }
    if (!chip.snapshot || chip.snapshot->table_version == 0) {
        return ChipStatus::Waiting;
    }
    if (chip.heartbeat && Clock::now() - chip.heartbeat_changed > kHeartbeatStaleAfter) {
        return ChipStatus::Stale;
    }
    return ChipStatus::Live;
}

Element status_badge(ChipStatus status) {
    switch (status) {
        case ChipStatus::Live: return text("● LIVE") | color(Color::Green) | bold;
        case ChipStatus::Waiting: return text("◌ WAITING") | color(Color::Yellow);
        case ChipStatus::Stale: return text("◐ STALE") | color(Color::Yellow) | bold;
        case ChipStatus::Stopped: return text("■ STOPPED") | color(Color::Red) | bold;
        case ChipStatus::NoTile: return text("✕ NO TILE") | color(Color::Red) | bold;
        case ChipStatus::ReadError: return text("✕ READ ERROR") | color(Color::Red) | bold;
    }
    return text("?");
}

Color level_color(double fraction) {
    if (fraction < 0.7) {
        return Color::Green;
    }
    if (fraction < 0.9) {
        return Color::Yellow;
    }
    return Color::Red;
}

std::string format_duration(Clock::duration duration) {
    const auto total = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02lld:%02lld:%02lld", total / 3600LL, (total / 60LL) % 60LL, total % 60LL);
    return buf;
}

std::string phase_name(uint64_t phase) {
    switch (phase) {
        case X280_HART_PHASE_IDLE: return "IDLE";
        case X280_HART_PHASE_RUNNING_ACTIVE_FW: return "RUNNING";
        case X280_HART_PHASE_RETURNED_TO_IDLE: return "RETURNED_TO_IDLE";
        default: return hex32(static_cast<uint32_t>(phase));
    }
}

constexpr int kLabelWidth = 11;
constexpr int kValueWidth = 12;

// One dashboard line: label, value and, when there's a limit, a gauge against it.
Element meter(
    const std::string& label, const TelemetrySnapshot& snapshot, const std::string& tag, const std::string& limit_tag = "") {
    const auto* metric = find_metric(snapshot, tag);
    if (metric == nullptr) {
        return hbox(
            text(label) | size(WIDTH, EQUAL, kLabelWidth), text("—") | dim | size(WIDTH, EQUAL, kValueWidth), filler());
    }

    const auto value = decode_metric(*metric);
    Elements row = {
        text(label) | size(WIDTH, EQUAL, kLabelWidth),
        text(value.text) | bold | size(WIDTH, EQUAL, kValueWidth),
    };

    std::optional<double> limit;
    std::string limit_text;
    if (!limit_tag.empty()) {
        if (const auto* limit_metric = find_metric(snapshot, limit_tag); limit_metric != nullptr) {
            limit = decode_metric(*limit_metric).number;
            limit_text = decode_metric(*limit_metric).text;
        }
    } else if (tag == "FAN_SPEED") {
        limit = 100.0;
        limit_text = "100 %";
    }

    if (value.number && limit && *limit > 0) {
        const double fraction = std::clamp(*value.number / *limit, 0.0, 1.0);
        row.push_back(gauge(static_cast<float>(fraction)) | color(level_color(fraction)) | flex);
        row.push_back(text(" " + limit_text) | dim | size(WIDTH, EQUAL, 11));
    } else {
        row.push_back(filler());
    }

    return hbox(std::move(row));
}

// The four L2CPU clocks on one line, in MHz.
Element l2cpu_clocks(const TelemetrySnapshot& snapshot) {
    std::string values;
    for (int i = 0; i < 4; ++i) {
        const auto clock = metric_number(snapshot, "L2CPUCLK" + std::to_string(i));
        values += (clock ? std::to_string(static_cast<int>(*clock)) : std::string("—")) + (i < 3 ? " / " : " MHz");
    }
    return hbox(text("L2CPU 0-3") | size(WIDTH, EQUAL, kLabelWidth), text(values) | bold, filler());
}

Element panel(const std::string& title, Elements rows) {
    return window(text(" " + title + " ") | bold, vbox(std::move(rows))) | flex;
}

// A filled area graph of history, newest sample on the right. The y range is the
// samples' min to max, widened to at least min_span so sensor noise stays flat.
Element history_graph(
    const std::string& title,
    const char* unit_fmt,
    const History& history,
    double min_span,
    std::chrono::milliseconds poll_interval,
    Color graph_color) {
    if (history.samples.empty()) {
        return window(text(" " + title + " ") | bold, text("no samples yet") | dim | center) | flex;
    }

    const auto [lo_it, hi_it] = std::minmax_element(history.samples.begin(), history.samples.end());
    const double mid = (*lo_it + *hi_it) / 2;
    const double half_span = std::max(*hi_it - *lo_it, min_span) / 2;
    const double lo = mid - half_span;
    const double hi = mid + half_span;
    const auto window_length = poll_interval * static_cast<int>(history.samples.size());
    const std::string now = " " + format_double(unit_fmt, history.samples.back()) + " ";
    const std::string stats = "min " + format_double(unit_fmt, *lo_it) + "  max " + format_double(unit_fmt, *hi_it) +
                              "  last " + format_duration(window_length);

    // Copied, since the graph is drawn after the model lock is released.
    std::vector<double> samples(history.samples.begin(), history.samples.end());
    auto plot = [samples, lo, hi](int width, int height) {
        // -1 leaves a column empty; graph() draws something for 0.
        std::vector<int> out(width, -1);
        const double span = hi - lo;
        const size_t count = std::min<size_t>(samples.size(), static_cast<size_t>(width));
        for (size_t i = 0; i < count; ++i) {
            const double value = samples[samples.size() - count + i];
            const double fraction = span > 0 ? (value - lo) / span : 0.5;
            // 1 to height - 1, so the lowest sample still shows and the peak isn't clipped.
            out[width - count + i] = static_cast<int>(std::lround(fraction * (height - 2))) + 1;
        }
        return out;
    };

    return window(
               hbox(text(" " + title + " ") | bold, filler(), text(now) | bold | color(graph_color)),
               vbox(graph(std::move(plot)) | color(graph_color) | flex, text(stats) | dim)) |
           flex;
}

Element render_dashboard(const ChipModel& chip, std::chrono::milliseconds poll_interval) {
    const auto& snapshot = *chip.snapshot;

    auto thermals = panel(
        "Thermals",
        {
            meter("ASIC", snapshot, "ASIC_TEMPERATURE", "THM_LIMIT_THROTTLE"),
            meter("Board", snapshot, "BOARD_TEMPERATURE"),
            meter("GDDR max", snapshot, "MAX_GDDR_TEMP", "THM_LIMIT_THROTTLE"),
            meter("Fan", snapshot, "FAN_SPEED"),
            meter("Fan speed", snapshot, "FAN_RPM"),
            meter("Shutdown", snapshot, "THM_LIMIT_SHUTDOWN"),
        });

    auto power = panel(
        "Power",
        {
            meter("TDP", snapshot, "TDP", "BOARD_POWER_LIMIT"),
            meter("TDC", snapshot, "TDC", "TDC_LIMIT_MAX"),
            meter("VCORE", snapshot, "VCORE"),
        });

    auto clocks = panel(
        "Clocks",
        {
            meter("AICLK", snapshot, "AICLK", "AICLK_LIMIT_MAX"),
            meter("AXICLK", snapshot, "AXICLK"),
            meter("ARCCLK", snapshot, "ARCCLK"),
            l2cpu_clocks(snapshot),
        });

    auto graphs = hbox({
        history_graph("ASIC temperature", "%.1f °C", chip.asic_temp, 10.0, poll_interval, Color::RedLight),
        history_graph("Power (TDP)", "%.0f W", chip.power, 20.0, poll_interval, Color::YellowLight),
        history_graph("AICLK", "%.0f MHz", chip.aiclk, 200.0, poll_interval, Color::CyanLight),
    });

    auto string_value = [&](const std::string& name) {
        const auto* metric = find_metric(snapshot, name);
        return metric != nullptr ? decode_metric(*metric).text : std::string("—");
    };

    const auto now = Clock::now();
    std::string heartbeat = "—";
    if (chip.heartbeat) {
        heartbeat = std::to_string(*chip.heartbeat) + " (changed " +
                    std::to_string(std::chrono::duration_cast<std::chrono::seconds>(now - chip.heartbeat_changed).count()) +
                    "s ago)";
    }

    auto info_row = [](const std::string& label, const std::string& value) {
        return hbox(text(label) | dim | size(WIDTH, EQUAL, 16), text(value));
    };

    auto info = window(
        text(" Firmware ") | bold,
        hbox({
            vbox({
                info_row("X280 firmware", string_value("tt_firmware_version")),
                info_row("Author", string_value("tt_firmware_athor")),
                info_row("Table version", hex32(snapshot.table_version)),
                info_row("Metrics", std::to_string(snapshot.metrics.size())),
            }) | flex,
            separator(),
            vbox({
                info_row("Hart 0", phase_name(chip.hart_phase) + ", stage " + std::to_string(chip.hart_stage)),
                info_row("ARC heartbeat", heartbeat),
                info_row("Reads", std::to_string(chip.good_reads) + " ok, " + std::to_string(chip.failed_reads) + " failed"),
                info_row("Tile", "NoC (" + std::to_string(chip.tile_id.noc_x) + "," +
                                     std::to_string(chip.tile_id.noc_y) + "), L2CPU " +
                                     std::to_string(chip.tile_id.hw_index)),
            }) | flex,
            separator(),
            vbox({
                info_row("Flash bundle", string_value("FLASH_BUNDLE_VERSION")),
                info_row("CM firmware", string_value("CM_FW_VERSION")),
                info_row("ETH firmware", string_value("ETH_FW_VERSION")),
                info_row("L2CPU firmware", string_value("L2CPU_FW_VERSION")),
            }) | flex,
        }));

    return vbox({
        hbox({thermals, power, clocks}),
        graphs | size(HEIGHT, GREATER_THAN, 8) | flex,
        info,
    });
}

Element render_metrics(const ChipModel& chip, int& selected_row) {
    const auto& metrics = chip.snapshot->metrics;
    selected_row = std::clamp(selected_row, 0, std::max(0, static_cast<int>(metrics.size()) - 1));

    auto header = hbox({
                      text("Metric") | size(WIDTH, EQUAL, 34),
                      text("Value") | size(WIDTH, EQUAL, 30),
                      text("Raw"),
                  }) |
                  bold;

    Elements rows;
    for (int i = 0; i < static_cast<int>(metrics.size()); ++i) {
        const auto& metric = metrics[i];
        const auto value = decode_metric(metric);
        const std::string raw = metric.is_string ? "string" : hex32(metric.num_value);

        auto row = hbox({
            text(metric.name) | size(WIDTH, EQUAL, 34),
            text(value.text) | size(WIDTH, EQUAL, 30),
            text(raw) | dim,
            filler(),
        });

        if (chip.changed.contains(metric.name)) {
            row = row | color(Color::Yellow) | bold;
        }
        if (i == selected_row) {
            row = row | inverted | focus;
        }
        rows.push_back(row);
    }

    auto table = vbox({header, separator(), vbox(std::move(rows)) | vscroll_indicator | yframe | flex});
    return window(
        hbox(
            text(" All metrics ") | bold,
            filler(),
            text(" highlighted = changed in the last read ") | color(Color::Yellow) | dim),
        table);
}

// ---------------------------------------------------------------------------
// Startup: priming, opening and loading, all before the UI takes the terminal.
// ---------------------------------------------------------------------------

void show_chip_state(const ChipOpenResult& chip_result) {
    std::cout << "Chip: " << chip_result.chip_id;
    switch (chip_result.outcome) {
        case RuntimeOpenOutcome::Ready: std::cout << " is ready, you can load your active firmware.\n"; break;

        case RuntimeOpenOutcome::PreparationRequired: std::cout << " must be primed (run with --prime).\n"; break;

        case RuntimeOpenOutcome::ResetRequired:
            std::cout << " reset via tt-smi -r " << chip_result.chip_id << "\n";
            break;

        case RuntimeOpenOutcome::Failed:
            std::cout << " is in a failed state: " << to_string(chip_result.outcome) << ", " << chip_result.diagnostic
                      << "\n";
            break;

        default: std::cout << " is in an unknown state.\n"; break;
    }
}

int prime(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh) {
    // AutoPrime (the default) primes an unprimed chip and reports ResetRequired.
    auto result = X280Runtime::open(mesh);
    std::cout << "There are " << result.chips.size() << " chips in this system.\n";
    if (result.ready()) {
        std::cout << "All chips are primed.\n";
        return 0;
    }

    for (const auto& curr_outcome : result.chips) {
        show_chip_state(curr_outcome);
    }
    return 1;
}

int run_tui(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh,
    const std::string& image,
    std::chrono::milliseconds poll_interval,
    const std::string& log_file) {
    auto result = X280Runtime::open(mesh, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
    if (!result.ready()) {
        std::cout << "Not all chips are ready.\n";
        for (const auto& curr_outcome : result.chips) {
            show_chip_state(curr_outcome);
        }
        return 1;
    }
    auto& runtime = *result.runtime;

    auto loaded = load_telemetry_firmware(runtime, image, kRunValue, std::cout);
    if (!loaded) {
        return 1;
    }

    if (g_interrupted) {
        std::cout << "Interrupted, stopping the firmware.\n";
        return stop_telemetry_firmware(runtime, *loaded, std::cout) ? 0 : 1;
    }

    Model model;
    for (const auto& curr : *loaded) {
        model.chips.push_back(ChipModel{.tile_id = curr.tile_id});
    }

    const auto started = Clock::now();
    const std::string image_name = image.substr(image.find_last_of('/') + 1);
    int selected_chip = 0;
    bool show_metrics = false;
    int selected_row = 0;

    auto screen = ScreenInteractive::Fullscreen();

    auto renderer = Renderer([&] {
        std::lock_guard lock(model.mutex);
        const int num_chips = static_cast<int>(model.chips.size());

        // Header: title, then one tab per chip with its status.
        Elements tabs;
        for (int i = 0; i < num_chips; ++i) {
            auto tab = hbox({text(" Chip " + std::to_string(model.chips[i].tile_id.chip_id) + " "),
                             status_badge(chip_status(model.chips[i])),
                             text(" ")});
            tabs.push_back(i == selected_chip ? tab | inverted : tab);
            tabs.push_back(text(" "));
        }

        auto header = vbox({
            hbox({
                text(" X280 Telemetry ") | bold | color(Color::Black) | bgcolor(Color::CyanLight),
                text(" " + image_name + " ") | dim,
                filler(),
                text("poll " + std::to_string(poll_interval.count()) + " ms   up " +
                     format_duration(Clock::now() - started) + " "),
            }),
            hbox({hbox(std::move(tabs)),
                  filler(),
                  text(show_metrics ? "[2] All metrics " : "[1] Dashboard ") | bold}),
        });

        Element body;
        const auto& chip = model.chips[selected_chip];
        if (!chip.snapshot || chip.snapshot->table_version == 0) {
            std::string why = "Waiting for the firmware to publish its table...";
            if (!chip.read_error.empty()) {
                why = "Can't read the table: " + chip.read_error;
            }
            body = window(text(" Chip " + std::to_string(chip.tile_id.chip_id) + " ") | bold,
                          vbox({status_badge(chip_status(chip)) | center, text(why) | center}) | center) |
                   flex;
        } else {
            body = show_metrics ? render_metrics(chip, selected_row) : render_dashboard(chip, poll_interval);
            if (!chip.read_error.empty()) {
                body = vbox({text(" Last read failed, showing older data: " + chip.read_error) | color(Color::Red),
                             body | flex});
            }
        }

        auto footer = hbox({
            text(" ←/→ chip  1/2 view  ↑/↓ PgUp/PgDn scroll  q quit and stop the firmware ") | dim,
            filler(),
            text("Metal log: " + log_file + " ") | dim,
        });

        return vbox({header, separator(), body | flex, footer});
    });

    auto app = CatchEvent(renderer, [&](const Event& event) {
        const int num_chips = static_cast<int>(model.chips.size());
        if (event == Event::Character('q') || event == Event::Character('Q') || event == Event::Escape) {
            screen.Exit();
            return true;
        }
        if (event == Event::ArrowRight || event == Event::Tab || event == Event::Character('l')) {
            selected_chip = (selected_chip + 1) % num_chips;
            return true;
        }
        if (event == Event::ArrowLeft || event == Event::TabReverse || event == Event::Character('h')) {
            selected_chip = (selected_chip + num_chips - 1) % num_chips;
            return true;
        }
        if (event == Event::Character('1')) {
            show_metrics = false;
            return true;
        }
        if (event == Event::Character('2')) {
            show_metrics = true;
            return true;
        }
        if (event == Event::Character('v')) {
            show_metrics = !show_metrics;
            return true;
        }
        // render_metrics clamps the row, so these can overshoot.
        if (event == Event::ArrowDown || event == Event::Character('j')) {
            ++selected_row;
            return true;
        }
        if (event == Event::ArrowUp || event == Event::Character('k')) {
            selected_row = std::max(0, selected_row - 1);
            return true;
        }
        if (event == Event::PageDown) {
            selected_row += 20;
            return true;
        }
        if (event == Event::PageUp) {
            selected_row = std::max(0, selected_row - 20);
            return true;
        }
        if (event == Event::Home) {
            selected_row = 0;
            return true;
        }
        if (event == Event::End) {
            selected_row = 1 << 20;
            return true;
        }
        return false;
    });

    {
        Poller poller(runtime, model, screen, poll_interval);
        // Returns on q, Esc, Ctrl-C or SIGTERM; FTXUI turns those signals into an exit.
        screen.Loop(app);
        poller.stop();
    }

    std::cout << "Stopping the firmware...\n";
    if (!stop_telemetry_firmware(runtime, *loaded, std::cout)) {
        return 1;
    }
    std::cout << "All tiles returned to resident idle.\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    bool do_prime = false;
    std::string image = std::string(X280_IMAGE_DIR) + "/firmware-telemetry2.bin";
    int poll_ms = kDefaultPollMs;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--prime") {
            do_prime = true;
        } else if (arg == "--poll-ms" && i + 1 < argc) {
            poll_ms = std::max(100, std::atoi(argv[++i]));
        } else if (arg == "-h" || arg == "--help") {
            std::printf("usage: %s [--prime] [--poll-ms N] [image.bin]\n", argv[0]);
            return 0;
        } else {
            image = arg;
        }
    }

    // Metal and UMD log to stdout, which would scribble over the UI, so send their
    // logs to a file unless the user already chose one. Must happen before anything
    // logs: the logger reads this when it's first used.
    std::string log_file = "stdout";
    if (!do_prime) {
        if (const char* chosen = std::getenv("TT_LOGGER_FILE"); chosen != nullptr) {
            log_file = chosen;
        } else if (const char* chosen = std::getenv("TT_METAL_LOGGER_FILE"); chosen != nullptr) {
            log_file = chosen;
        } else {
            log_file = kDefaultLogFile;
            setenv("TT_LOGGER_FILE", kDefaultLogFile, 1);
        }
    }

    std::signal(SIGINT, on_sigint);

    std::cout << "Using image: " << image << "\n";
    if (!do_prime) {
        std::cout << "Opening the devices, Metal's log goes to " << log_file << "\n";
    }

    auto mesh =
        tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(std::nullopt));
    return do_prime ? prime(mesh) : run_tui(mesh, image, std::chrono::milliseconds(poll_ms), log_file);
}
