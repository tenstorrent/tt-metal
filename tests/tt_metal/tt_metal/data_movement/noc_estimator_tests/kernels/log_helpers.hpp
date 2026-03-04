// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Memory type
constexpr uint32_t MEMORY_TYPE_L1 = 0;
constexpr uint32_t MEMORY_TYPE_DRAM_INTERLEAVED = 1;
constexpr uint32_t MEMORY_TYPE_DRAM_SHARDED = 2;

// Mechanism
constexpr uint32_t MECHANISM_UNICAST = 0;
constexpr uint32_t MECHANISM_MULTICAST = 1;
constexpr uint32_t MECHANISM_MULTICAST_LINKED = 2;

// Pattern
constexpr uint32_t PATTERN_ONE_TO_ONE = 0;
constexpr uint32_t PATTERN_ONE_FROM_ONE = 1;
constexpr uint32_t PATTERN_ONE_TO_ALL = 2;
constexpr uint32_t PATTERN_ONE_FROM_ALL = 3;
constexpr uint32_t PATTERN_ALL_TO_ALL = 4;
constexpr uint32_t PATTERN_ALL_FROM_ALL = 5;
constexpr uint32_t PATTERN_ONE_TO_ROW = 6;
constexpr uint32_t PATTERN_ROW_TO_ROW = 7;
constexpr uint32_t PATTERN_ONE_TO_COLUMN = 8;
constexpr uint32_t PATTERN_COLUMN_TO_COLUMN = 9;

// Writer kernel modes
constexpr uint32_t WRITER_MODE_UNICAST_SINGLE = 0;
constexpr uint32_t WRITER_MODE_UNICAST_MULTI = 1;
constexpr uint32_t WRITER_MODE_MULTICAST = 2;
constexpr uint32_t WRITER_MODE_MULTICAST_LINKED = 3;

// Reader kernel modes
constexpr uint32_t READER_MODE_SINGLE = 0;
constexpr uint32_t READER_MODE_MULTI = 1;

FORCE_INLINE void log_estimator_metadata(
    uint32_t test_id,
    uint32_t noc_idx,
    uint32_t num_transactions,
    uint32_t transaction_size_bytes,
    uint32_t memory_type,
    uint32_t mechanism,
    uint32_t pattern,
    uint32_t num_subordinates,
    uint32_t same_axis,
    uint32_t stateful,
    uint32_t loopback) {
    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", num_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("NoC Index", noc_idx);
    DeviceTimestampedData("Memory type", memory_type);
    DeviceTimestampedData("Mechanism", mechanism);
    DeviceTimestampedData("Pattern", pattern);
    DeviceTimestampedData("Number of subordinates", num_subordinates);
    DeviceTimestampedData("Same axis", same_axis);
    DeviceTimestampedData("Stateful", stateful);
    DeviceTimestampedData("Loopback", loopback);
}
