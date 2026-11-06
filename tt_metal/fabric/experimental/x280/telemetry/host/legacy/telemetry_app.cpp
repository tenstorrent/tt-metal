// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "x280/device/x280_runtime.hpp"

#include <tt-metalium/mesh_device.hpp>
#include <common/telemetry_table.h>

#include <iostream>
#include <iomanip>

namespace {

using namespace x280;

void show_chip_state(const ChipOpenResult& chip_result) {
    std::cout << "Chip: " << chip_result.chip_id;
    switch (chip_result.outcome) {
        case RuntimeOpenOutcome::Ready: std::cout << " is ready, you can load your active firmware.\n"; break;

        case RuntimeOpenOutcome::PreparationRequired: std::cout << " must be primed.\n"; break;

        case RuntimeOpenOutcome::ResetRequired:
           std::cout << " reset via tt-smi -r " << chip_result.chip_id << "\n";
        break;

        case RuntimeOpenOutcome::Failed:
            std::cout << " is in a faileedd state.\n";
            std::cout << " prime failed: " << to_string(chip_result.outcome) << ", " << chip_result.diagnostic
                      << "\n";
            break;

        default: std::cout << " is in an unknown state.\n"; break;
    }
}

bool get_telemetry_table(X280Tile& tile) {
//    auto& tile = runtime.tile(tile_id);

    const auto state = tile.inspect();
    if (!state.observation_valid) {
        std::cerr << "inspect: " << state.invalid_reason.c_str() << "\n";
        return false;
    }

    auto bytes = tile.read_available_lim(TABLE_ADDR, sizeof(TelemetryTable));
    std::cout << "Bytes Read: " << bytes.size() << "\n";

    TelemetryTable table;
    std::memcpy(&table, bytes.data(), sizeof(table));
    std::cout << "Version: " << std::hex << "0x" << table.header.table_version << "\n";
    std::cout << "Num rows: " << table.header.num_rows << "\n";
    std::cout << std::left;
    for (int i = 0; i < table.header.num_rows; ++i) {
        std::cout << std::setw(32) << table.rows[i].metric_name << " | ";
        if (table.rows[i].metric_type == MetricType::metric_type_string) {
            std::cout << " string | ";
            std::cout << std::setw(64) << table.rows[i].str_data << " | ";
        } else {
            std::cout << " number | ";
            std::cout << "0x" << std::setw(62) << table.rows[i].num_data << " | ";
        }
        std::cout << "\n";
    }
    return true;
}

bool get_telemetry_tables(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh) {
    // RequirePrepared never primes; an unprimed chip is reported instead.
    auto result = X280Runtime::open(mesh, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
    if (!result.ready()) {

        for(const auto& curr_chip_outcome: result.chips) {
            show_chip_state(curr_chip_outcome);
        }

        return false;
    }
    auto& runtime = *result.runtime;

    auto tile_ids = runtime.tile_ids();
    if (tile_ids.empty()) {
        std::cerr << "no enabled X280 tiles\n";
        return false;
    }

    // We only load on one tile per chip, so remove all dup chips.
    auto last = std::unique(std::begin(tile_ids), std::end(tile_ids), [](const auto& t1, const auto& t2) {
        if (t1.chip_id == t2.chip_id) {
            return true;
        }
    
        return false;
    });
    
    tile_ids.erase(last, tile_ids.end());

    for(const auto& curr_tile_id: tile_ids) {
        std::cout << "Table for chip: " << curr_tile_id.chip_id << "\n";
        auto& tile = runtime.tile(curr_tile_id);
        auto result = get_telemetry_table(tile);
        if (!result) {
            return false;
        }
    }

    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    for (int i = 0; i < argc; ++i) {
        std::cout << "argv[" << i << "] = " << argv[i] << "\n";
    }

    auto mesh =
        tt::tt_metal::distributed::MeshDevice::create(tt::tt_metal::distributed::MeshDeviceConfig(std::nullopt));
    auto result = get_telemetry_tables(mesh);

    if (!result) {
        return 1;
    }
    return 0;
}
