// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint64_t magic;  
    uint64_t counter;
} Heartbeat;

// _Static_assert(offsetof(Heartbeat, magic) == 0u, "Heartbeat.magic");
// _Static_assert(offsetof(Heartbeat, counter) == 8u, "Heartbeat.counter");
// _Static_assert(offsetof(Heartbeat, fifo_addr_lo) == 16u, "Heartbeat.fifo_addr_lo");
// _Static_assert(offsetof(Heartbeat, fifo_addr_hi) == 20u, "Heartbeat.fifo_addr_hi");
// _Static_assert(offsetof(Heartbeat, fifo_size) == 24u, "Heartbeat.fifo_size");
// _Static_assert(sizeof(Heartbeat) == 32u, "Heartbeat size");

