// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// This file is shared by host and device ethernet microbenchmarks

#pragma once

#include <cstdint>

enum BenchmarkType : uint8_t {
    EthOnlyUniDir = 0,
    EthOnlyBiDir = 1,
    EthEthTensixUniDir = 2,
    EthEthTensixBiDir = 3,
    TensixPushEth = 4,
    EthMcastTensix = 5,
    EthToLocalEth = 6,
    EthToLocalEthAndMcastTensix = 7,
};

enum MeasurementType : uint8_t { Latency = 0, Bandwidth = 1 };

struct eth_buffer_slot_sync_t {
    volatile uint32_t bytes_sent;
    volatile uint32_t receiver_ack;
    volatile uint32_t src_id;
    uint32_t reserved_2;
};
