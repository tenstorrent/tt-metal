// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// harness/inventory/inventory.hpp -- harvesting-aware tile enumeration and
// placement facts.
//
// The characterization suites do not need to reason about multi-chip
// placement: by default a case uses the first enabled X280 tile on chip 0 and
// logical Tensix worker (0, 0); paired cases may select an ordered tile pair.

#ifndef X280_HARNESS_INVENTORY_HPP
#define X280_HARNESS_INVENTORY_HPP

#include "x280/device/x280_runtime.hpp"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace x280::harness::inventory {

// Pick the first enabled X280 tile on the mesh, or nullopt if none.
inline std::optional<TileId> first_enabled_tile(x280::X280Runtime& runtime) {
    const auto ids = runtime.tile_ids();
    if (ids.empty()) {
        return std::nullopt;
    }
    return ids.front();
}

struct TilePairSpec {
    uint32_t first = 0;
    uint32_t second = 0;
};

// Parse an ordered tile-pair selection "a,b", where a and b are L2CPU
// hardware indices. Surrounding whitespace on either index is accepted.
//
// Throws std::invalid_argument for malformed fields, values exceeding u32, or
// duplicate indices.
inline TilePairSpec parse_tile_pair_spec(const std::string& spec) {
    const auto reject = [&spec](const std::string& why) {
        throw std::invalid_argument(
            "x280: tile pair selection \"" + spec + "\" " + why + "; expected two L2CPU hardware indices as \"a,b\"");
    };

    const auto comma = spec.find(',');
    if (comma == std::string::npos) {
        reject("has no comma");
    }
    if (spec.find(',', comma + 1) != std::string::npos) {
        reject("has more than one comma");
    }

    const auto index_of = [&](const std::string& field, const char* which) -> uint32_t {
        const auto begin = field.find_first_not_of(" \t");
        if (begin == std::string::npos) {
            reject(std::string("has an empty ") + which + " index");
        }
        const auto end = field.find_last_not_of(" \t");
        const std::string trimmed = field.substr(begin, end - begin + 1);
        if (trimmed.find_first_not_of("0123456789") != std::string::npos) {
            reject(std::string("has a non-numeric ") + which + " index \"" + trimmed + '"');
        }
        const unsigned long value = std::strtoul(trimmed.c_str(), nullptr, 10);
        if (value > 0xFFFFFFFFul) {
            reject(std::string("has a ") + which + " index \"" + trimmed + "\" exceeding u32");
        }
        return static_cast<uint32_t>(value);
    };

    TilePairSpec parsed{};
    parsed.first = index_of(spec.substr(0, comma), "first");
    parsed.second = index_of(spec.substr(comma + 1), "second");
    if (parsed.first == parsed.second) {
        reject("names L2CPU " + std::to_string(parsed.first) + " twice");
    }
    return parsed;
}

// Resolve an ordered tile pair. An empty spec selects the first two enabled
// tiles and returns nullopt if fewer than two exist. An explicit malformed or
// unavailable selection throws std::invalid_argument.
inline std::optional<std::pair<TileId, TileId>> selected_tile_pair(
    x280::X280Runtime& runtime, const std::string& spec) {
    const auto ids = runtime.tile_ids();
    if (spec.empty()) {
        if (ids.size() < 2) {
            return std::nullopt;
        }
        return std::make_pair(ids[0], ids[1]);
    }

    const auto wanted = parse_tile_pair_spec(spec);
    const auto find = [&](uint32_t hw_index) -> const TileId& {
        for (const auto& id : ids) {
            if (static_cast<uint32_t>(id.hw_index) == hw_index) {
                return id;
            }
        }
        std::string enabled;
        for (const auto& id : ids) {
            enabled += (enabled.empty() ? "" : ", ") + std::to_string(id.hw_index);
        }
        throw std::invalid_argument(
            "x280: tile pair selection \"" + spec + "\" names L2CPU " + std::to_string(hw_index) +
            ", which is not enabled; enabled: " + (enabled.empty() ? std::string("none") : enabled));
    };
    return std::make_pair(find(wanted.first), find(wanted.second));
}

// Default Tensix worker for the paired X280<->Tensix members. Worker (0, 0)
// in LOGICAL is what every member picks; centralised here so a future
// multi-worker case has one place to change.
struct TensixWorker {
    uint32_t x = 0;
    uint32_t y = 0;
};
inline TensixWorker default_worker() { return TensixWorker{0u, 0u}; }

// Blackhole physical NoC extent; rows and columns are rings.
// See blackhole_140_arch.yaml and firmware core_config.h.
constexpr uint32_t kNocGridX = 17;
constexpr uint32_t kNocGridY = 12;

enum class NocId : uint32_t { Noc0 = 0, Noc1 = 1 };

[[noreturn]] inline void reject_endpoint(const char* which, Noc0Coord coord) {
    throw std::out_of_range(
        "x280: hops() " + std::string(which) + " noc0 (" + std::to_string(coord.x) + "," + std::to_string(coord.y) +
        ") is outside the " + std::to_string(kNocGridX) + "x" + std::to_string(kNocGridY) + " NOC grid");
}

// Directional wraparound hops between physical NOC0 coordinates. Uses
// tt-metal's get_worker_noc_hop_distance ordering; throws std::out_of_range
// for off-grid endpoints.
inline uint32_t hops(Noc0Coord from, Noc0Coord to, NocId noc_id) {
    if (from.x >= kNocGridX || from.y >= kNocGridY) {
        reject_endpoint("from", from);
    }
    if (to.x >= kNocGridX || to.y >= kNocGridY) {
        reject_endpoint("to", to);
    }

    const auto ring_distance = [](uint32_t start, uint32_t end, uint32_t extent) {
        return (end + extent - start) % extent;
    };
    if (noc_id == NocId::Noc0) {
        return ring_distance(from.x, to.x, kNocGridX) + ring_distance(from.y, to.y, kNocGridY);
    }
    return ring_distance(to.x, from.x, kNocGridX) + ring_distance(to.y, from.y, kNocGridY);
}

}  // namespace x280::harness::inventory

#endif  // X280_HARNESS_INVENTORY_HPP
