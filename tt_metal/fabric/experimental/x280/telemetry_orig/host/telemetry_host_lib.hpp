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

// Reads the whole table once, without waiting for the firmware to publish it.
// Returns nullopt and fills error if the tile can't be read.
std::optional<TelemetrySnapshot> read_telemetry_table(x280::X280Tile& tile, std::string& error);

// Waits for the table to be published, then prints it to std::cout.
bool get_telemetry_table(x280::X280Tile& tile);

// Starts image on the first tile of every chip, booting resident idle first where
// needed, and waits for each to be running. Progress and errors go to log.
// On failure, any firmware already started is stopped and nullopt is returned.
std::optional<std::vector<LoadedTile>> load_telemetry_firmware(
    x280::X280Runtime& runtime, const std::string& image, uint64_t run_value, std::ostream& log);

// Sends STOP to every session and waits for the tiles to return to resident idle.
// Returns false if any of them didn't.
bool stop_telemetry_firmware(x280::X280Runtime& runtime, const std::vector<LoadedTile>& loaded, std::ostream& log);
