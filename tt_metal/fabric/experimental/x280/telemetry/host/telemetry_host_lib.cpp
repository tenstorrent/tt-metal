// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280/device/x280_runtime.hpp"

#include <tt-metalium/mesh_device.hpp>
#include <common/telemetry_table_v2.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <thread>

namespace {

using namespace x280;

//void show_chip_state(const ChipOpenResult& chip_result) {
//    std::cout << "Chip: " << chip_result.chip_id;
//    switch (chip_result.outcome) {
//        case RuntimeOpenOutcome::Ready: std::cout << " is ready, you can load your active firmware.\n"; break;
//
//        case RuntimeOpenOutcome::PreparationRequired: std::cout << " must be primed.\n"; break;
//
//        case RuntimeOpenOutcome::ResetRequired:
//            std::cout << " reset via tt-smi -r " << chip_result.chip_id << "\n";
//            break;
//
//        case RuntimeOpenOutcome::Failed:
//            std::cout << " is in a faileedd state.\n";
//            std::cout << " prime failed: " << to_string(chip_result.outcome) << ", " << chip_result.diagnostic << "\n";
//            break;
//
//        default: std::cout << " is in an unknown state.\n"; break;
//    }
//}

} // namespace

bool get_telemetry_table(X280Tile& tile) {
    const auto state = tile.inspect();
    if (!state.observation_valid) {
        std::cerr << "inspect: " << state.invalid_reason.c_str() << "\n";
        return false;
    }

    // We keep reading until we get something in the table_version.
    MetricsTelemetryTable table;
    std::memset(&table, 0, sizeof(MetricsTelemetryTable));

    while (table.table_version == 0) {
        auto bytes = tile.read_available_lim(TABLE_ADDR, sizeof(uint32_t));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::memcpy(&table, bytes.data(), sizeof(uint32_t));
    }
    auto bytes = tile.read_available_lim(TABLE_ADDR, sizeof(MetricsTelemetryTable));
    std::memcpy(&table, bytes.data(), sizeof(MetricsTelemetryTable));
    std::cout << "Version: " << std::hex << "0x" << table.table_version << "\n";
    std::cout << "Num Metrics: " << table.num_metrics << "\n";
    std::cout << std::left;
    for (int i = 0; i < table.num_metrics && i < 128; ++i) {
        std::cout << std::setw(32) << table.labels[i].metric_name << " | ";
        if (table.labels[i].metric_type == 1) {
            std::cout << " string | ";
            std::cout << std::setw(64) << table.str_values[table.labels[i].value_index].str_value << " | ";
        } else {
            std::cout << " number | ";
            std::cout << "0x" << std::setw(62) << table.num_values[table.labels[i].value_index].num_value << " | ";
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


