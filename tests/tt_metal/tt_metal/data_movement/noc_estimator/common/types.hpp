// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <map>
#include <string>
#include <cstdint>

namespace tt::tt_metal::noc_estimator::common {

// Values match tt::ARCH from umd/device/types/arch.hpp
enum class Architecture { WORMHOLE_B0 = 2, BLACKHOLE = 3 };

enum class NocMechanism { UNICAST, MULTICAST, MULTICAST_LINKED };

enum class MemoryType { L1, DRAM_INTERLEAVED, DRAM_SHARDED };

enum class NocPattern {
    ONE_FROM_ONE,
    ONE_TO_ONE,
    ONE_FROM_ALL,
    ONE_TO_ALL,
    ALL_TO_ALL,
    ALL_FROM_ALL,
    ONE_TO_ROW,
    ROW_TO_ROW,
    ONE_TO_COLUMN,
    COLUMN_TO_COLUMN
};

// Standard transaction sizes (stored once in YAML header)
const std::vector<uint32_t> STANDARD_TRANSACTION_SIZES = {
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

struct LatencyData {
    std::vector<double> latencies;  // One per transaction size
};

// Default values for GroupKey fields (used by YAML serializer/loader)
constexpr int DEFAULT_MECHANISM = 0;  // UNICAST
constexpr int DEFAULT_PATTERN = 1;    // ONE_TO_ONE
constexpr int DEFAULT_MEMORY = 0;     // L1
constexpr int DEFAULT_ARCH = static_cast<int>(Architecture::WORMHOLE_B0);
constexpr uint32_t DEFAULT_NUM_TRANSACTIONS = 1;
constexpr uint32_t DEFAULT_NUM_SUBORDINATES = 1;
constexpr bool DEFAULT_SAME_AXIS = false;
constexpr bool DEFAULT_STATEFUL = false;
constexpr bool DEFAULT_LOOPBACK = false;
constexpr uint32_t DEFAULT_NOC_INDEX = 0;

// Key for grouping data points (all parameters except transaction_size)
struct GroupKey {
    NocMechanism mechanism;
    NocPattern pattern;
    MemoryType memory;
    Architecture arch;
    uint32_t num_transactions;
    uint32_t num_subordinates;
    bool same_axis;
    bool stateful;
    bool loopback;
    uint32_t noc_index;

    // Operators are needed for std::map containers
    bool operator<(const GroupKey& other) const {
        if (mechanism != other.mechanism) {
            return mechanism < other.mechanism;
        }
        if (pattern != other.pattern) {
            return pattern < other.pattern;
        }
        if (memory != other.memory) {
            return memory < other.memory;
        }
        if (arch != other.arch) {
            return arch < other.arch;
        }
        if (num_transactions != other.num_transactions) {
            return num_transactions < other.num_transactions;
        }
        if (num_subordinates != other.num_subordinates) {
            return num_subordinates < other.num_subordinates;
        }
        if (same_axis != other.same_axis) {
            return same_axis < other.same_axis;
        }
        if (stateful != other.stateful) {
            return stateful < other.stateful;
        }
        if (loopback != other.loopback) {
            return loopback < other.loopback;
        }
        return noc_index < other.noc_index;
    }

    bool operator==(const GroupKey& other) const {
        return mechanism == other.mechanism && pattern == other.pattern && memory == other.memory &&
               arch == other.arch && num_transactions == other.num_transactions &&
               num_subordinates == other.num_subordinates && same_axis == other.same_axis &&
               stateful == other.stateful && loopback == other.loopback && noc_index == other.noc_index;
    }

    // Check if non-numeric fields match (for interpolation)
    bool matches_non_numeric(const GroupKey& other) const {
        return mechanism == other.mechanism && pattern == other.pattern && memory == other.memory &&
               arch == other.arch && same_axis == other.same_axis && stateful == other.stateful &&
               loopback == other.loopback && noc_index == other.noc_index;
    }
};

// Generic numeric field accessors for interpolation
// To add a new numeric field: update extract() and with_values()
struct NumericFields {
    static std::map<std::string, uint32_t> extract(const GroupKey& key) {
        return {{"num_transactions", key.num_transactions}, {"num_subordinates", key.num_subordinates}};
    }

    static GroupKey with_values(const GroupKey& base, const std::map<std::string, uint32_t>& values) {
        GroupKey result = base;
        result.num_transactions = values.at("num_transactions");
        result.num_subordinates = values.at("num_subordinates");
        return result;
    }
};

}  // namespace tt::tt_metal::noc_estimator::common
