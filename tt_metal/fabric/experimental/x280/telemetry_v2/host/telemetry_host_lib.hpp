// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host side helpers shared by the telemetry programs: start the firmware on one
// tile per chip, and read the telemetry table back out of LIM.

#pragma once

#include "x280/device/x280_runtime.hpp"

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

// One row of the table, copied out of LIM into host types.
struct TelemetryMetric {
    std::string name;
    bool is_string = false;
    uint32_t num_value = 0;
    std::string str_value;
};

struct TelemetrySnapshot {
    // 0 means the firmware hasn't published the table yet; metrics is empty then.
    uint32_t table_version = 0;
    std::vector<TelemetryMetric> metrics;
};

// The firmware running on one chip.
struct LoadedTile {
    x280::TileId tile_id;
    x280::SessionId session;
};

std::optional<TelemetrySnapshot> read_telemetry_table(x280::X280Tile& tile, std::string& error);

bool get_telemetry_table(x280::X280Tile& tile);

