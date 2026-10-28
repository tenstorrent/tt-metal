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
bool get_telemetry_table(const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh) {
    // RequirePrepared never primes; an unprimed chip is reported instead.
    auto opened = X280Runtime::open(mesh, RuntimeOpenConfig{.priming_policy = LimEccPrimingPolicy::RequirePrepared});
    if (!opened.ready()) {
        std::fprintf(stderr, "open: %s: %s\n", to_string(opened.outcome), opened.diagnostic.c_str());
        if (opened.requires_preparation()) {
            std::cerr << "run with --prime first, then tt-smi -r 0\n";
        }
        return false;
    }
    auto& runtime = *opened.runtime;

    const auto ids = runtime.tile_ids();
    if (ids.empty()) {
        std::cerr << "no enabled X280 tiles\n";
        return false;
    }
    auto& tile = runtime.tile(ids.front());

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

}  // namespace

int main(int argc, char* argv[]) {
    for (int i = 0; i < argc; ++i) {
        std::cout << "argv[" << i << "] = " << argv[i] << "\n";
    }

    auto mesh = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(0);
    auto result = get_telemetry_table(mesh);

    if (!result) {
        return 1;
    }
    return 0;
}
