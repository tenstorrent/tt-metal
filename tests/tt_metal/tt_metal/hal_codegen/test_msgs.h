// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

struct DateTime {
    uint16_t year;
    uint8_t month;
    uint8_t day;
    uint8_t hour;
    uint8_t minute;
    uint8_t second;
    uint8_t timezone_id;
};

struct HopInfo {
    // struct in struct
    DateTime timestamp;
    // array of scalars
    uint8_t addr[4];
};

struct PacketInfo {
    // scalar in struct
    uint64_t len;
    // array of structs
    HopInfo trace_route[MAX_HOPS];
};
