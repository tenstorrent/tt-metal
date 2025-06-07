// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_fabric {

enum class Topology { Ring = 0, Linear = 1, Mesh = 2, Torus = 3 };

struct WorkerXY {
    uint16_t x;
    uint16_t y;

    constexpr WorkerXY(uint16_t x, uint16_t y) : x(x), y(y) {}

    constexpr uint32_t to_uint32() const { return (y << 16) | x; }
    static constexpr WorkerXY from_uint32(uint32_t v) { return WorkerXY(v & 0xFFFF, (v >> 16) & 0xFFFF); }

    constexpr bool operator==(const WorkerXY& rhs) const { return x == rhs.x && y == rhs.y; }
    constexpr bool operator!=(const WorkerXY& rhs) const { return !(*this == rhs); }
};

struct coord_t {
    coord_t(uint32_t x, uint32_t y) : x(x), y(y) {}
    uint32_t x;
    uint32_t y;
};

enum SendStatus : uint8_t {
    // Indicates that the sender was able to send the payload
    // but was not able to send the channel_sync_t at the end of the
    // buffer
    //
    // This enum should only ever be returned if we are sending less than
    // a full packet/buffer of data AND when we are trying to send the
    // channel_sync_t at the end of the buffer (which must be as a separate
    // command) but the eth_tx_cmd_q is busy for that second message
    //
    // Receiving this value indicates we
    // MUST:
    // - Eventually send the channel_sync_t before advancing to the next buffer
    // MUST NOT:
    // - Advance to the next buffer index
    // - Forward the other sender channel's data (if it has any)
    SENT_PAYLOAD_ONLY,

    // Indicates both the payload and the channel sync were sent successfully
    SENT_PAYLOAD_AND_SYNC,

    // Indicates no data was sent because the eth_tx_cmd_q was busy
    NOT_SENT,

    ERROR,
};

struct EDMChannelWorkerLocationInfo {
    uint32_t worker_semaphore_address;
    uint32_t align_pad_0;  // Padding added for safe reading over noc
    uint32_t align_pad_1;
    uint32_t align_pad_2;

    uint32_t worker_teardown_semaphore_address;
    uint32_t align_pad_3;  // Padding added for safe reading over noc
    uint32_t align_pad_4;
    uint32_t align_pad_5;

    WorkerXY worker_xy;
    uint32_t align_pad_6;  // Padding added for safe reading over noc
    uint32_t align_pad_7;
    uint32_t align_pad_8;

    uint32_t edm_read_counter = 0;
    uint32_t align_pad_9;  // Padding added for safe reading over noc
    uint32_t align_pad_10;
    uint32_t align_pad_11;
};

static_assert(sizeof(EDMChannelWorkerLocationInfo) <= 64);

}  // namespace tt::tt_fabric
