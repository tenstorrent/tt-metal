// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "telemetry_host_lib.hpp"

#include <tt-metalium/mesh_device.hpp>
#include <common/telemetry_table_v2.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <memory>
#include <iostream>
#include <iomanip>
#include <thread>

namespace {

using namespace x280;

constexpr size_t kMaxRows = 128;

// The firmware fills fixed-size char arrays and doesn't guarantee a NUL when the
// text uses the whole array, so never read past the end.
template <size_t N>
std::string fixed_string(const char (&chars)[N]) {
    return std::string(chars, strnlen(chars, N));
}


} // namespace

std::optional<TelemetrySnapshot> read_telemetry_table(X280Tile& tile, std::string& error) {
    // MetricsTelemetryTable is about 16 KiB, so keep it off the stack of whichever thread calls us.
    auto table = std::make_unique<MetricsTelemetryTable>();

    try {
        auto bytes = tile.read_available_lim(TABLE_ADDR, sizeof(MetricsTelemetryTable));
        if (bytes.size() < sizeof(MetricsTelemetryTable)) {
            error = "short read: " + std::to_string(bytes.size()) + " of " +
                    std::to_string(sizeof(MetricsTelemetryTable)) + " bytes";
            return std::nullopt;
        }
        std::memcpy(table.get(), bytes.data(), sizeof(MetricsTelemetryTable));
    } catch (const std::exception& e) {
        // read_available_lim throws when the tile can't be observed, ECC isn't primed,
        // or the range is outside available LIM.
        error = e.what();
        return std::nullopt;
    }

    TelemetrySnapshot snapshot;
    snapshot.table_version = table->table_version;
    if (snapshot.table_version == 0) {
        return snapshot;
    }

    const size_t num_metrics = std::min<size_t>(table->num_metrics, kMaxRows);
    snapshot.metrics.reserve(num_metrics);
    for (size_t i = 0; i < num_metrics; ++i) {
        const auto& label = table->labels[i];
        TelemetryMetric metric;
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

bool get_telemetry_table(X280Tile& tile) {
    const auto state = tile.inspect();
    if (!state.observation_valid) {
        std::cerr << "inspect: " << state.invalid_reason.c_str() << "\n";
        return false;
    }

    // We keep reading until we get something in the table_version.
    uint32_t table_version = 0;
    while (table_version == 0) {
        auto bytes = tile.read_available_lim(TABLE_ADDR, sizeof(uint32_t));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::memcpy(&table_version, bytes.data(), sizeof(uint32_t));
    }

    std::string error;
    auto snapshot = read_telemetry_table(tile, error);
    if (!snapshot) {
        std::cerr << "read table: " << error << "\n";
        return false;
    }

    std::cout << "Version: " << std::hex << "0x" << snapshot->table_version << "\n";
    std::cout << "Num Metrics: " << snapshot->metrics.size() << "\n";
    std::cout << std::left;
    for (const auto& metric : snapshot->metrics) {
        std::cout << std::setw(32) << metric.name << " | ";
        if (metric.is_string) {
            std::cout << " string | ";
            std::cout << std::setw(64) << metric.str_value << " | ";
        } else {
            std::cout << " number | ";
            std::cout << "0x" << std::setw(62) << metric.num_value << " | ";
        }
        std::cout << "\n";
    }
    return true;
}


//bool get_telemetry_tables(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh) {
//    // RequirePrepared never primes; an unprimed chip is reported instead.
//    auto result = X280Runtime::open(mesh, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
//    if (!result.ready()) {
//        for (const auto& curr_chip_outcome : result.chips) {
//            show_chip_state(curr_chip_outcome);
//        }
//
//        return false;
//    }
//    auto& runtime = *result.runtime;
//
//    auto tile_ids = runtime.tile_ids();
//    if (tile_ids.empty()) {
//        std::cerr << "no enabled X280 tiles\n";
//        return false;
//    }
//
//    // We only load on one tile per chip, so remove all dup chips.
//    auto last = std::unique(std::begin(tile_ids), std::end(tile_ids), [](const auto& t1, const auto& t2) {
//        if (t1.chip_id == t2.chip_id) {
//            return true;
//        }
//
//        return false;
//    });
//
//    tile_ids.erase(last, tile_ids.end());
//
//    for (const auto& curr_tile_id : tile_ids) {
//        std::cout << "Table for chip: " << curr_tile_id.chip_id << "\n";
//        auto& tile = runtime.tile(curr_tile_id);
//        auto result = get_telemetry_table(tile);
//        if (!result) {
//            return false;
//        }
//    }
//
//    return true;
//}
