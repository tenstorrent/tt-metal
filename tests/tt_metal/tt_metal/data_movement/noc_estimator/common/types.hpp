// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cstdint>

namespace tt::noc_estimator::common {

enum class Architecture { WORMHOLE_B0, BLACKHOLE };

enum class NocMechanism { UNICAST, MULTICAST };

enum class MemoryType { L1, DRAM };

enum class NocPattern { ONE_FROM_ONE, ONE_TO_ONE, ONE_FROM_ALL, ONE_TO_ALL, ALL_TO_ALL, ALL_FROM_ALL };

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
constexpr int DEFAULT_ARCH = 0;       // WORMHOLE_B0
constexpr uint32_t DEFAULT_NUM_TRANSACTIONS = 1;
constexpr uint32_t DEFAULT_NUM_SUBORDINATES = 1;
constexpr bool DEFAULT_SAME_AXIS = false;
constexpr bool DEFAULT_LINKED = false;

// Key for grouping data points (all parameters except transaction_size)
struct GroupKey {
    NocMechanism mechanism;
    NocPattern pattern;
    MemoryType memory;
    Architecture arch;
    uint32_t num_transactions;
    uint32_t num_subordinates;
    bool same_axis;
    bool linked;

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
        return linked < other.linked;
    }

    bool operator==(const GroupKey& other) const {
        return mechanism == other.mechanism && pattern == other.pattern && memory == other.memory &&
               arch == other.arch && num_transactions == other.num_transactions &&
               num_subordinates == other.num_subordinates && same_axis == other.same_axis && linked == other.linked;
    }

    // Check if non-numeric fields match (for interpolation)
    bool matches_non_numeric(const GroupKey& other) const {
        return mechanism == other.mechanism && pattern == other.pattern && memory == other.memory &&
               arch == other.arch && same_axis == other.same_axis && linked == other.linked;
    }
};

// Generic numeric field accessors for interpolation
// To add a new numeric field: update extract() and with_values()
struct NumericFields {
    // Extract all numeric values from a GroupKey
    static std::vector<uint32_t> extract(const GroupKey& key) { return {key.num_transactions, key.num_subordinates}; }

    // Create a new key with different numeric values
    static GroupKey with_values(const GroupKey& base, const std::vector<uint32_t>& values) {
        GroupKey result = base;
        result.num_transactions = values[0];
        result.num_subordinates = values[1];
        return result;
    }

    static constexpr std::size_t count = 2;  // Number of numeric fields
};

}  // namespace tt::noc_estimator::common
