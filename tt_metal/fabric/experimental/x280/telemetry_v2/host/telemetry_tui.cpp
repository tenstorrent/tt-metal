// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Terminal UI for the standalone telemetry firmware.
//
// The firmware is booted by x280_telemetry_v2_loader --start and keeps running on its own, so this
// program only reads. It never opens X280Runtime and never stops or resets anything: it finds the
// tiles carrying the firmware's heartbeat magic, then polls each tile's heartbeat and telemetry
// table through the Metal transport on a worker thread and shows them live. q, Esc or Ctrl-C quits
// the UI; the firmware keeps running.
//
// usage: x280_telemetry_tui [--poll-ms N]

#include "x280_metal_transport.hpp"

#include "x280_address_map.h"
#include "common/telemetry_table_v2.h"

#include <tt-metalium/mesh_device.hpp>

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace ftxui;

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kDefaultPollMs = 1000;
constexpr size_t kHistoryLength = 600;  // Samples kept for the graphs, 10 minutes at the default poll.
constexpr size_t kMaxRows = 128;
// The firmware's own heartbeat ticks about once a millisecond but stalls while it walks the ARC
// table, so allow a generous gap before calling it stuck.
constexpr auto kFirmwareStaleAfter = std::chrono::seconds(5);
constexpr auto kArcStaleAfter = std::chrono::seconds(10);
constexpr const char* kDefaultLogFile = "x280_telemetry_tui.log";
constexpr const char* kArcPrefix = "TT_ARC_";

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

// Follows UMD's Blackhole decode. Tags UMD doesn't decode are shown raw.
Decoded decode_arc_value(const std::string& tag, uint32_t raw) {
    static const std::map<std::string, const char*> kUnits = {
        {"MAX_GDDR_TEMP", "°C"},      {"THM_LIMIT_SHUTDOWN", "°C"}, {"THM_LIMIT_THROTTLE", "°C"},
        {"VCORE", "mV"},              {"TDP", "W"},                 {"TDP_LIMIT_MAX", "W"},
        {"BOARD_POWER_LIMIT", "W"},   {"INPUT_POWER", "W"},         {"TDC", "A"},
        {"TDC_LIMIT_MAX", "A"},       {"AICLK", "MHz"},             {"AXICLK", "MHz"},
        {"ARCCLK", "MHz"},            {"L2CPUCLK0", "MHz"},         {"L2CPUCLK1", "MHz"},
        {"L2CPUCLK2", "MHz"},         {"L2CPUCLK3", "MHz"},         {"AICLK_LIMIT_MAX", "MHz"},
        {"AICLK_ARB_MIN", "MHz"},     {"AICLK_ARB_MAX", "MHz"},     {"FAN_SPEED", "%"},
        {"FAN_RPM", "rpm"},
    };
    // Versions UMD decodes as major.minor.patch from the top three bytes.
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

    char version[48];
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

// ---------------------------------------------------------------------------
// The table as read from LIM.
// ---------------------------------------------------------------------------

struct Metric {
    std::string name;
    bool is_string = false;
    uint32_t num_value = 0;
    std::string str_value;
};

struct Snapshot {
    uint32_t table_version = 0;
    std::vector<Metric> metrics;
};

// The firmware fills fixed-size char arrays and doesn't guarantee a NUL when the text uses the
// whole array, so never read past the end.
template <size_t N>
std::string fixed_string(const char (&chars)[N]) {
    return std::string(chars, strnlen(chars, N));
}

Decoded decode_metric(const Metric& metric) {
    if (metric.is_string) {
        return {metric.str_value, std::nullopt};
    }
    return decode_arc_value(arc_tag(metric.name), metric.num_value);
}

const Metric* find_metric(const Snapshot& snapshot, const std::string& name) {
    for (const auto& metric : snapshot.metrics) {
        if (metric.name == name || arc_tag(metric.name) == name) {
            return &metric;
        }
    }
    return nullptr;
}

std::optional<double> metric_number(const Snapshot& snapshot, const std::string& tag) {
    const auto* metric = find_metric(snapshot, tag);
    if (metric == nullptr || metric->is_string) {
        return std::nullopt;
    }
    return decode_metric(*metric).number;
}

struct Tile {
    int chip_id;
    tt::umd::CoreCoord core;
};

struct Heartbeat {
    uint64_t magic = 0;
    uint64_t counter = 0;
};

Heartbeat read_heartbeat(const x280::X280MetalTransport& transport, const Tile& tile) {
    Heartbeat beat;
    beat.magic = transport.read_u64(tile.chip_id, tile.core, X280_STANDALONE_HEARTBEAT_ADDR);
    if (beat.magic == static_cast<uint64_t>(X280_STANDALONE_HEARTBEAT_MAGIC)) {
        beat.counter = transport.read_u64(tile.chip_id, tile.core, X280_STANDALONE_HEARTBEAT_COUNTER_ADDR);
    }
    return beat;
}

// Reads the table once the firmware has published table_version. Returns a snapshot with
// table_version 0 while it hasn't. Throws if the transport fails.
Snapshot read_table(const x280::X280MetalTransport& transport, const Tile& tile) {
    Snapshot snapshot;
    snapshot.table_version = transport.read_reg32(tile.chip_id, tile.core, TABLE_ADDR);
    if (snapshot.table_version == 0) {
        return snapshot;
    }

    // MetricsTelemetryTable is about 15 KiB, so keep it off the poller thread's stack.
    auto table = std::make_unique<MetricsTelemetryTable>();
    transport.read_block(tile.chip_id, tile.core, TABLE_ADDR, table.get(), sizeof(MetricsTelemetryTable));
    snapshot.table_version = table->table_version;
    if (snapshot.table_version == 0) {
        return snapshot;
    }

    const size_t num_metrics = std::min<size_t>(table->num_metrics, kMaxRows);
    snapshot.metrics.reserve(num_metrics);
    for (size_t i = 0; i < num_metrics; ++i) {
        const auto& label = table->labels[i];
        Metric metric;
        metric.name = fixed_string(label.metric_name);
        metric.is_string = label.metric_type == 1;
        if (label.value_index < kMaxRows) {
            if (metric.is_string) {
                metric.str_value = fixed_string(table->str_values[label.value_index].str_value);
            } else {
                metric.num_value = table->num_values[label.value_index].num_value;
            }
        }
        snapshot.metrics.push_back(std::move(metric));
    }
    return snapshot;
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
    Tile tile;

    // The latest table that could be read, and why the most recent read failed (empty if it didn't).
    std::optional<Snapshot> snapshot;
    std::string read_error;
    uint64_t good_reads = 0;
    uint64_t failed_reads = 0;

    // Metric names whose value changed in the latest read.
    std::set<std::string> changed;

    // The firmware's own heartbeat: magic, and a counter that moves about once a millisecond.
    bool magic_ok = false;
    uint64_t fw_counter = 0;
    Clock::time_point fw_counter_changed{};
    double fw_ticks_per_s = 0;  // Counter rate over the last poll.
    Clock::time_point last_poll{};

    // ARC bumps TIMER_HEARTBEAT and the firmware copies it on each refresh, so if it stops
    // changing, ARC or the firmware's refresh is stuck.
    std::optional<uint32_t> arc_heartbeat;
    Clock::time_point arc_heartbeat_changed{};

    History asic_temp;
    History power;
    History aiclk;
};

struct Model {
    std::mutex mutex;
    std::vector<ChipModel> chips;
};

void apply_sample(
    ChipModel& chip, const Heartbeat& beat, std::optional<Snapshot> snapshot, const std::string& error) {
    const auto now = Clock::now();

    chip.magic_ok = beat.magic == static_cast<uint64_t>(X280_STANDALONE_HEARTBEAT_MAGIC);
    if (chip.magic_ok) {
        if (beat.counter != chip.fw_counter) {
            if (chip.last_poll != Clock::time_point{} && beat.counter > chip.fw_counter) {
                const double seconds = std::chrono::duration<double>(now - chip.last_poll).count();
                chip.fw_ticks_per_s = seconds > 0 ? static_cast<double>(beat.counter - chip.fw_counter) / seconds : 0;
            }
            chip.fw_counter_changed = now;
        } else {
            chip.fw_ticks_per_s = 0;
        }
        if (chip.fw_counter_changed == Clock::time_point{}) {
            chip.fw_counter_changed = now;
        }
        chip.fw_counter = beat.counter;
    }
    chip.last_poll = now;

    if (!snapshot) {
        chip.read_error = error;
        ++chip.failed_reads;
        return;
    }

    chip.read_error.clear();
    ++chip.good_reads;

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

    if (const auto* beat_metric = find_metric(*snapshot, "TIMER_HEARTBEAT");
        beat_metric != nullptr && !beat_metric->is_string) {
        if (!chip.arc_heartbeat || *chip.arc_heartbeat != beat_metric->num_value) {
            chip.arc_heartbeat = beat_metric->num_value;
            chip.arc_heartbeat_changed = now;
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
class Poller {
public:
    Poller(
        std::shared_ptr<x280::X280MetalTransport> transport,
        Model& model,
        ScreenInteractive& screen,
        std::chrono::milliseconds interval) :
        transport_(std::move(transport)),
        model_(model),
        screen_(screen),
        interval_(interval),
        thread_([this] { run(); }) {}

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
        std::vector<Tile> tiles;
        {
            std::lock_guard lock(model_.mutex);
            for (const auto& chip : model_.chips) {
                tiles.push_back(chip.tile);
            }
        }

        for (;;) {
            for (size_t i = 0; i < tiles.size(); ++i) {
                // Talk to the hardware without holding the model lock, so the UI never waits on it.
                Heartbeat beat;
                std::optional<Snapshot> snapshot;
                std::string error;
                try {
                    beat = read_heartbeat(*transport_, tiles[i]);
                    if (beat.magic == static_cast<uint64_t>(X280_STANDALONE_HEARTBEAT_MAGIC)) {
                        snapshot = read_table(*transport_, tiles[i]);
                    } else {
                        error = "heartbeat magic is gone (chip reset?)";
                    }
                } catch (const std::exception& e) {
                    error = e.what();
                    snapshot.reset();
                }

                std::lock_guard lock(model_.mutex);
                apply_sample(model_.chips[i], beat, std::move(snapshot), error);
            }

            screen_.PostEvent(Event::Custom);

            std::unique_lock lock(wake_mutex_);
            if (wake_.wait_for(lock, interval_, [this] { return stopping_; })) {
                return;
            }
        }
    }

    std::shared_ptr<x280::X280MetalTransport> transport_;
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

enum class ChipStatus { Live, Waiting, Stale, NoFirmware, ReadError };

ChipStatus chip_status(const ChipModel& chip) {
    if (!chip.magic_ok) {
        return ChipStatus::NoFirmware;
    }
    if (!chip.read_error.empty()) {
        return ChipStatus::ReadError;
    }
    if (Clock::now() - chip.fw_counter_changed > kFirmwareStaleAfter) {
        return ChipStatus::Stale;
    }
    if (!chip.snapshot || chip.snapshot->table_version == 0) {
        return ChipStatus::Waiting;
    }
    if (chip.arc_heartbeat && Clock::now() - chip.arc_heartbeat_changed > kArcStaleAfter) {
        return ChipStatus::Stale;
    }
    return ChipStatus::Live;
}

Element status_badge(ChipStatus status) {
    switch (status) {
        case ChipStatus::Live: return text("● LIVE") | color(Color::Green) | bold;
        case ChipStatus::Waiting: return text("◌ WAITING") | color(Color::Yellow);
        case ChipStatus::Stale: return text("◐ STALE") | color(Color::Yellow) | bold;
        case ChipStatus::NoFirmware: return text("■ NO FIRMWARE") | color(Color::Red) | bold;
        case ChipStatus::ReadError: return text("✕ READ ERROR") | color(Color::Red) | bold;
    }
    return text("?");
}

// The X280 firmware's heartbeat for the selected chip, shown in the header: a heart that is green
// while the counter is moving and red when it isn't, then the count and its rate.
Element heartbeat_indicator(const ChipModel& chip) {
    if (!chip.magic_ok) {
        return hbox({text("♡ ") | color(Color::Red), text("no heartbeat") | dim});
    }
    const bool beating = Clock::now() - chip.fw_counter_changed <= kFirmwareStaleAfter;
    return hbox({
        text(beating ? "♥ " : "♡ ") | color(beating ? Color::Green : Color::Red) | bold,
        text(std::to_string(chip.fw_counter)),
        text(" (" + format_double("%.0f", chip.fw_ticks_per_s) + "/s)") | dim,
    });
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

constexpr int kLabelWidth = 11;
constexpr int kValueWidth = 12;

// One dashboard line: label, value and, when there's a limit, a gauge against it.
Element meter(
    const std::string& label, const Snapshot& snapshot, const std::string& tag, const std::string& limit_tag = "") {
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
            const auto decoded = decode_metric(*limit_metric);
            limit = decoded.number;
            limit_text = decoded.text;
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
Element l2cpu_clocks(const Snapshot& snapshot) {
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

// A filled area graph of history, newest sample on the right. The y range is the samples' min to
// max, widened to at least min_span so sensor noise stays flat.
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
            meter("Input", snapshot, "INPUT_POWER"),
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
    const auto seconds_since = [&](Clock::time_point when) {
        return std::to_string(std::chrono::duration_cast<std::chrono::seconds>(now - when).count());
    };

    std::string fw_heartbeat = std::to_string(chip.fw_counter);
    fw_heartbeat += " (" + format_double("%.0f", chip.fw_ticks_per_s) + "/s)";

    std::string arc_heartbeat = "—";
    if (chip.arc_heartbeat) {
        arc_heartbeat = std::to_string(*chip.arc_heartbeat) + " (changed " + seconds_since(chip.arc_heartbeat_changed) +
                        "s ago)";
    }

    auto info_row = [](const std::string& label, const std::string& value) {
        return hbox(text(label) | dim | size(WIDTH, EQUAL, 16), text(value));
    };

    // The firmware names its author tag "tt_firmware_athor".
    const std::string author = find_metric(snapshot, "tt_firmware_author") != nullptr
                                   ? string_value("tt_firmware_author")
                                   : string_value("tt_firmware_athor");

    auto info = window(
        text(" Firmware ") | bold,
        hbox({
            vbox({
                info_row("X280 firmware", string_value("tt_firmware_version")),
                info_row("Author", author),
                info_row("Table version", hex32(snapshot.table_version)),
                info_row("Metrics", std::to_string(snapshot.metrics.size())),
            }) | flex,
            separator(),
            vbox({
                info_row("FW heartbeat", fw_heartbeat),
                info_row("ARC heartbeat", arc_heartbeat),
                info_row(
                    "Reads", std::to_string(chip.good_reads) + " ok, " + std::to_string(chip.failed_reads) + " failed"),
                info_row("Tile", "NoC (" + std::to_string(chip.tile.core.x) + "," + std::to_string(chip.tile.core.y) + ")"),
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

int run_tui(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh,
    std::chrono::milliseconds poll_interval,
    const std::string& log_file) {
    auto transport = x280::X280MetalTransport::bind(*mesh);

    // Every enabled L2CPU core on every chip reachable over PCIe. We don't know which tile the loader
    // started, so keep the ones that carry the firmware's magic.
    Model model;
    for (int chip_id : x280::sorted_usable_chip_ids(*mesh, *transport)) {
        if (!transport->is_mmio_local(chip_id)) {
            std::cout << "Chip " << chip_id << " is not PCIe-local, skipping.\n";
            continue;
        }
        for (const auto& core : transport->l2cpu_cores(chip_id)) {
            const Tile tile{.chip_id = chip_id, .core = core};
            try {
                if (read_heartbeat(*transport, tile).magic == static_cast<uint64_t>(X280_STANDALONE_HEARTBEAT_MAGIC)) {
                    model.chips.push_back(ChipModel{.tile = tile});
                }
            } catch (const std::exception& e) {
                std::cout << "Chip " << chip_id << " L2CPU (" << core.x << "," << core.y << "): " << e.what() << "\n";
            }
        }
    }

    if (model.chips.empty()) {
        std::cout << "No tile shows the standalone heartbeat magic. Run x280_telemetry_v2_loader --start first.\n";
        return 1;
    }

    const auto started = Clock::now();
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
            auto tab = hbox(
                {text(" Chip " + std::to_string(model.chips[i].tile.chip_id) + " "),
                 status_badge(chip_status(model.chips[i])),
                 text(" ")});
            tabs.push_back(i == selected_chip ? tab | inverted : tab);
            tabs.push_back(text(" "));
        }

        auto header = vbox({
            hbox({
                text(" X280 Telemetry ") | bold | color(Color::Black) | bgcolor(Color::CyanLight),
                text(" standalone firmware ") | dim,
                filler(),
                text("poll " + std::to_string(poll_interval.count()) + " ms   up " +
                     format_duration(Clock::now() - started) + " "),
            }),
            hbox({
                hbox(std::move(tabs)),
                filler(),
                heartbeat_indicator(model.chips[selected_chip]),
                text("  "),
                text(show_metrics ? "[2] All metrics " : "[1] Dashboard ") | bold,
            }),
        });

        Element body;
        const auto& chip = model.chips[selected_chip];
        if (!chip.snapshot || chip.snapshot->table_version == 0) {
            std::string why = "Waiting for the firmware to publish its table...";
            if (!chip.read_error.empty()) {
                why = "Can't read the table: " + chip.read_error;
            }
            body = window(
                       text(" Chip " + std::to_string(chip.tile.chip_id) + " ") | bold,
                       vbox({status_badge(chip_status(chip)) | center, text(why) | center}) | center) |
                   flex;
        } else {
            body = show_metrics ? render_metrics(chip, selected_row) : render_dashboard(chip, poll_interval);
            if (!chip.read_error.empty()) {
                body = vbox(
                    {text(" Last read failed, showing older data: " + chip.read_error) | color(Color::Red),
                     body | flex});
            }
        }

        auto footer = hbox({
            text(" ←/→ chip  1/2 view  ↑/↓ PgUp/PgDn scroll  q quit (the firmware keeps running) ") | dim,
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
        Poller poller(transport, model, screen, poll_interval);
        // Returns on q, Esc, Ctrl-C or SIGTERM; FTXUI turns those signals into an exit.
        screen.Loop(app);
        poller.stop();
    }

    std::cout << "Telemetry UI closed. The firmware is still running on the X280.\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    int poll_ms = kDefaultPollMs;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--poll-ms" && i + 1 < argc) {
            poll_ms = std::max(100, std::atoi(argv[++i]));
        } else if (arg == "-h" || arg == "--help") {
            std::printf("usage: %s [--poll-ms N]\n", argv[0]);
            return 0;
        } else {
            std::printf("usage: %s [--poll-ms N]\n", argv[0]);
            return 1;
        }
    }

    // Metal and UMD log to stdout, which would scribble over the UI, so send their logs to a file
    // unless the user already chose one. Must happen before anything logs: the logger reads this
    // when it's first used.
    std::string log_file = kDefaultLogFile;
    if (const char* chosen = std::getenv("TT_LOGGER_FILE"); chosen != nullptr) {
        log_file = chosen;
    } else if (const char* chosen = std::getenv("TT_METAL_LOGGER_FILE"); chosen != nullptr) {
        log_file = chosen;
    } else {
        setenv("TT_LOGGER_FILE", kDefaultLogFile, 1);
    }

    std::cout << "Opening the devices, Metal's log goes to " << log_file << "\n";
    auto mesh =
        tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(std::nullopt));
    return run_tui(mesh, std::chrono::milliseconds(poll_ms), log_file);
}
